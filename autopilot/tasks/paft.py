"""This module defines the PAFT task

Multiple Child rpis running the "PAFT_Child" Task connect to this Parent.
The Parent chooses the correct stimulus and logs all events. It tells each
Child if it should start playing sounds and when it should stop. The Child
knows that a poke into the port that is currently playing sound should
be rewarded.

The Parent establishes a router Net_Node on port 5001. A dealer Net_Node
on each Child connects to it.

The Parent responds to the following message keys on port 5001:
* HELLO : This means the Child has booted the task.
    The value is dispatched to PAFT.recv_hello, with the following keys:
    'from' : string; the name of the child (e.g., rpi06)
* POKE : This means the Child has detected a poke.
    The value is dispatched to PAFT.recv_poke, with the following keys:
    'from' : string; the name of the child (e.g., rpi06)
    'poke' : one of {'L', 'R'}; the side that was poked

The Parent will not start the first trial until each Child in PAFT.CHILDREN
has sent the "HELLO" message. 

The Parent can send the following message keys:
* HELLO : not used
* PLAY : This tells the Child to start playing and be ready to reward
    The value has the following keys:
    'target' : one of {'L', 'R'}; the side that should play
* STOP : This tells the Child to stop playing sounds and not reward.
    The value is an empty dict.    
* END : This tells the Child the session is over.
    The value is an empty dict.    
"""

import threading
import itertools
import random
import datetime
import functools
from collections import OrderedDict as odict
import time
import queue
import tables
import numpy as np
import pandas
import autopilot.hardware.gpio
from autopilot.stim.sound import sounds
from autopilot.tasks.task import Task
from autopilot.networking import Net_Node
from autopilot import prefs
from autopilot.hardware import BCM_TO_BOARD
from autopilot.core.loggers import init_logger
from autopilot.stim.sound import jackclient

# The name of the task
# This declaration allows Subject to identify which class in this file 
# contains the task class. 
TASK = 'PAFT'


## Define the Task
class PAFT(Task):
    """The probabalistic auditory foraging task (PAFT).

    This passes through three stages and returns random data for each.
    
    To understand the stage progression logic, see:
    * autopilot.core.pilot.Pilot.run_task - the main loop
    * autopilot.tasks.task.Task.handle_trigger - set stage trigger
    
    To understand the data saving logic, see:
    * autopilot.core.terminal.Terminal.l_data - what happens when data is sent
    * autopilot.core.subject.Subject.data_thread - how data is saved

    Class attributes:
        PARAMS : collections.OrderedDict
            This defines the params we expect to receive from the terminal.
        TrialData : subclass of tables.IsDescription
            This defines how to set up the hdf5 file for the Subject with
            the returned data
        HARDWARE : dict of dicts
            Defines 'POKES', 'PORTS', and 'LEDS'
        CHILDREN : dict of dicts
            Defines the child pis that we'll connect to
        PLOT : dict of dicts
            Defines how to plot the results

    Attributes:
        trial_counter (:class:`itertools.count`): 
            Counts trials starting from current_trial specified as argument
        triggers (dict): 
            Dictionary mapping triggered pins to callable methods.
        num_stages (int): 
            number of stages in task (2)
        stages (:class:`itertools.cycle`): 
            iterator to cycle indefinitely through task stages.
    """
    
    
    ## Define the class attributes
    # This defines params we receive from terminal on session init
    # It also determines the params that are available to specify in the
    # Protocol creation GUI.
    # The params themselves are defined the protocol json.
    # Presently these can only be of type int, bool, enum (aka list), or sound
    # Defaults cannot be specified here or in the GUI, only in the corresponding
    # kwarg in __init__
    PARAMS = odict()
    PARAMS['reward'] = {
        'tag':'reward duration (ms)',
        'type':'int',
        }
    
    # This DataFrame indicates a bunch of parameters that each have
    # _max, _min, and _n_choices versions, to specify ranges
    helper_to_form_params = pandas.DataFrame.from_records([
        ('target_rate', 'rate of target sounds at goal (Hz)', 'float'),
        ('target_temporal_log_std', 
            'log(std(inter-target intervals [s]))', 'float'),
        
        ('target_spatial_extent', 
            'number of ports on each side of target that play sound', 'float'),
        
        ('distracter_rate', 'rate of distracter sounds (Hz)', 'float'),
        ('distracter_temporal_log_std', 
            'log(std(inter-distracter intervals [s]))', 'float'),
        
        ('target_center_freq', 'center freq of target sound (Hz)', 'float'),
        ('target_bandwidth', 
            'bandwidth (high-low) of target sound (Hz)', 'float'),
        ('target_log_amplitude', 
            'log(amplitude of target sound)', 'float'),
        
        ('distracter_center_freq', 
            'center freq of distracter sound (Hz)', 'float'),
        ('distracter_bandwidth', 
            'bandwidth (high-low) of distracter sound (Hz)', 'float'),
        ('distracter_log_amplitude', 
            'log(amplitude of distracter sound)', 'float'),
        ],
        columns=['key', 'tag', 'type'],
        )
    
    # This generates the actual PARAMS using the helper
    for param in helper_to_form_params.itertuples():
        # Add the min
        PARAMS['{}_min'.format(param.key)] = {
            'tag': 'min[{}]'.format(param.tag),
            'type': param.type,
            }

        # Add the max
        PARAMS['{}_max'.format(param.key)] = {
            'tag': 'max[{}]'.format(param.tag),
            'type': param.type,
            }    
        
        # Add the n_choices, which is always an int
        PARAMS['{}_n_choices'.format(param.key)] = {
            'tag': 'n_choices[{}]'.format(param.tag),
            'type': 'int',
            }    

    
    ## Set up TrialData and Continuous Data schema
    # Per https://docs.auto-pi-lot.com/en/latest/guide/task.html:
    # The `TrialData` object is used by the `Subject` class when a task
    # is assigned to create the data storage table
    # 'trial_num' and 'session_num' get added by the `Subject` class
    # 'session_num' is properly set by `Subject`, but 'trial_num' needs
    # to be set properly here.
    # If they are left unspecified on any given trial, they receive 
    # a default value, such as 0 for Int32Col.
    class TrialData(tables.IsDescription):
        # The trial within this session
        # Unambigously label this
        trial_in_session = tables.Int32Col()
        
        # If this isn't specified here, it will be added anyway
        trial_num = tables.Int32Col()
        
        # The rewarded_port
        # Must specify the max length of the string, we use 64 to be safe
        previously_rewarded_port = tables.StringCol(64)
        rewarded_port = tables.StringCol(64)
        
        # The timestamps
        timestamp_trial_start = tables.StringCol(64)
        timestamp_reward = tables.StringCol(64)
        
        # A bunch of stimulus parameters
        # TODO: generate this programmatically from helper
        # TODO: add "log" to the names of the log-spaced params
        stim_target_rate = tables.Float32Col()
        stim_target_temporal_std = tables.Float32Col()
        stim_target_spatial_extent = tables.Float32Col()
        stim_distracter_rate = tables.Float32Col()
        stim_distracter_temporal_std = tables.Float32Col()
        stim_target_center_freq = tables.Float32Col()
        stim_target_bandwidth = tables.Float32Col()
        stim_target_amplitude = tables.Float32Col()
        stim_distracter_center_freq = tables.Float32Col()
        stim_distracter_bandwidth = tables.Float32Col()
        stim_distracter_amplitude = tables.Float32Col()

    # Define continuous data
    # https://docs.auto-pi-lot.com/en/latest/guide/task.html
    # autopilot.core.subject.Subject.data_thread would like one of the
    # keys to be "timestamp"
    # Actually, no I think that is extracted automatically from the 
    # networked message, and should not be defined here
    class ContinuousData(tables.IsDescription):
        reward_timestamp = tables.StringCol(64)
        trial = tables.Int32Col()

    # Define chunk data
    # This is like ContinuousData, but each row is sent together, as a chunk
    class ChunkData_Sounds(tables.IsDescription):
        relative_time = tables.Float64Col()
        side = tables.StringCol(10)
        sound = tables.StringCol(10)
        pilot = tables.StringCol(20)
        locking_timestamp = tables.StringCol(50)        
        gap = tables.Float64Col()
        gap_chunks = tables.IntCol()
    
    class ChunkData_Pokes(tables.IsDescription):
        timestamp = tables.StringCol(64)
        poked_port = tables.StringCol(64)
        trial = tables.Int32Col()
        first_poke = tables.Int32Col()
        reward_delivered = tables.Int32Col()
        poke_rank = tables.Int32Col()
    
    CHUNKDATA_CLASSES = [ChunkData_Sounds, ChunkData_Pokes]
    
    ## Set up hardware and children
    # Per https://docs.auto-pi-lot.com/en/latest/guide/task.html:
    # The HARDWARE dictionary maps a hardware type (eg. POKES) and 
    # identifier (eg. 'L') to a Hardware object. The task uses the hardware 
    # parameterization in the prefs file (also see setup_pilot) to 
    # instantiate each of the hardware objects, so their naming system 
    # must match (ie. there must be a prefs.PINS['POKES']['L'] entry in 
    # prefs for a task that has a task.HARDWARE['POKES']['L'] object).
    HARDWARE = {
        'POKES':{
            'L': autopilot.hardware.gpio.Digital_In,
            'R': autopilot.hardware.gpio.Digital_In,
        },
        'LEDS':{
            'L': autopilot.hardware.gpio.LED_RGB,
            'R': autopilot.hardware.gpio.LED_RGB,
        },
        'PORTS':{
            'L': autopilot.hardware.gpio.Solenoid,
            'R': autopilot.hardware.gpio.Solenoid,
        }
    }
    
    # This defines the child rpis to connect to
    children_names = prefs.get('CHILDID')
    if children_names is None:
        # This happens on terminal
        children_names = []
    CHILDREN = {}
    for child in children_names:
        CHILDREN[child] = {'task_type': "PAFT_Child"}

    
    ## Define the class methods
    def __init__(self, stage_block, current_trial, step_name, task_type, 
        subject, step, session, pilot, graduation, reward, **task_params):
        """Initialize a new PAFT Task. 
        
        All arguments are provided by the Terminal.
        
        Note that this __init__ does not call the superclass __init__, 
        because that superclass Task includes functions for punish_block
        and so on that we don't want to use.
        
        Some params, such as `step_name` and `task_type`, are always required 
        to be specified in the json defining this protocol
        
        Other params, such as `reward`, are custom to this particular task.
        They should be described in the class attribute `PARAMS` above, and
        their values should be specified in the protocol json.
        
        These need to be defined in JSON now:
            target_rate_min
            target_rate_max
            target_rate_n_choices
            target_temporal_log_std_min
            target_temporal_log_std_max
            target_temporal_log_std_n_choices
            target_spatial_extent_min
            target_spatial_extent_max
            target_spatial_extent_n_choices
            distractor_rate_min
            distractor_rate_max
            distractor_rate_n_choices
            distractor_temporal_log_std_min
            distractor_temporal_log_std_max
            distractor_temporal_log_std_n_choices
            target_center_freq_min
            target_center_freq_max
            target_center_freq_n_choices
            target_bandwidth_min
            target_bandwidth_max
            target_bandwidth_n_choices
            target_log_amplitude_min
            target_log_amplitude_max
            target_log_amplitude_n_choices
            distractor_center_freq_min
            distractor_center_freq_max
            distractor_center_freq_n_choices
            distractor_bandwidth_min
            distractor_bandwidth_max
            distractor_bandwidth_n_choices
            distractor_log_amplitude_min
            distractor_log_amplitude_max
            distractor_log_amplitude_n_choices
        
        Arguments
        ---------
        stage_block (:class:`threading.Event`): 
            used to signal to the carrying Pilot that the current trial 
            stage is over
        current_trial (int): 
            If not zero, initial number of `trial_counter`
            This is set to be 1 greater than the last value of "trial_num"
            in the HDF5 file by autopilot.core.subject.Subject.prepare_run
            Or sometimes this is just zero, for some reason
        step_name : string
            This is passed from the "protocol" json
            Currently it is always "PAFT"
        task_type : string
            This is passed from the "protocol" json
            Currently it is always "PAFT"
        subject : string
            The name of the subject
        step : 0
            Index into the "protocol" json?
        session : int
            number of times it's been started
        pilot : string
            The name of this pilot
        graduation : dict
            Probably a dict of graduation criteria
        reward (int): 
            ms to open solenoids
            This is passed from the "protocol" json
        """    
        
        ## These are things that would normally be done in superclass __init__
        # Set up a logger first, so we can debug if anything goes wrong
        self.logger = init_logger(self)

        # This threading.Event is checked by Pilot.run_task before
        # advancing through stages. Clear it to wait for triggers; set
        # it to advance to the next stage.
        self.stage_block = stage_block
        
        # This is needed for sending Node messages
        self.subject = subject
        
        # This is used to count the trials for the "trial_num" HDF5 column
        self.counter_trials_across_sessions = int(current_trial)

        # This is used to count the trials for the "trial_in_session" HDF5 column
        self.counter_trials_in_session = 0

        # A dict of hardware triggers
        self.triggers = {}
        
        
        ## Store the stimulus parameters
        self.stim_choosing_params = {}
        
        # Form choice list for each param
        for param in self.helper_to_form_params.itertuples():
            # Shortcuts
            param_min = '{}_min'.format(param.key)
            param_max = '{}_max'.format(param.key)
            param_n_choices = '{}_n_choices'.format(param.key)
            
            # Depends on how many choices
            if task_params[param_n_choices] == 1:
                # If only 1, assert equal, and corresponding entry in 
                # self.stim_choosing_params is a list of length one
                assert (task_params[param_min] == task_params[param_max])
                self.stim_choosing_params[param.key] = [task_params[param_min]]
            else:
                # Otherwise, linspace between min and max                
                assert (task_params[param_min] < task_params[param_max])
                self.stim_choosing_params[param.key] = np.linspace(
                    task_params[param_min],
                    task_params[param_max],
                    task_params[param_n_choices])

        # Log
        self.logger.debug('received task_params:\n{}'.format(task_params))
        self.logger.debug('set self.stim_choosing_params:\n{}'.format(
            self.stim_choosing_params))

        
        ## Define the possible ports
        self.known_pilot_ports = []
        for child in prefs.get('CHILDID'):
            self.known_pilot_ports.append('{}_{}'.format(child, 'L'))
            self.known_pilot_ports.append('{}_{}'.format(child, 'R'))
        
    
        ## Define the stages
        # Stage list to iterate
        # Iterate through these three stages forever
        stage_list = [
            self.choose_stimulus, self.wait_for_response, 
            self.report_reward, self.end_of_trial,
            ]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)        
        
        # This is used to keep track of the rewarded port
        self.rewarded_port = None
        self.previously_rewarded_port = None
        
        # This is used to infer the first poke of each trial
        self.trial_of_last_poke = None
        
        # This is used to infer whether reward was delivered
        self.reward_delivered_on_this_trial = False
        
        # This is used to calculate performance metrics
        self.ports_poked_on_this_trial = []
        
        
        ## Init hardware -- this sets self.hardware, self.pin_id, and
        ## assigns self.handle_trigger to gpio callbacks
        self.init_hardware()


        ## Connect to children
        # This dict keeps track of which self.CHILDREN have connected
        self.child_connected = {}
        for child in self.CHILDREN.keys():
            self.child_connected[child] = False
        
        # Tell each child to start the task
        self.initiate_task_on_children(subject, reward)
        
        # Create a Net_Node for communicating with the children, and
        # wait until all children have connected
        self.create_inter_pi_communication_node()

    def initiate_task_on_children(self, subject, reward):
        """Defines a Net_Node and uses it to tell each child to start
        
        This Net_Node is saved as `self.node`. A 'CHILD' message is sent,
        I think to the Pilot_Node, which is handled by
        networking.station.Pilot_Node.l_child .
        
        That code broadcasts the 'START' message to each of the children
        specified in CHILDID in prefs.json, telling them to start the
        'PAFT_Child' task. That 'START' message also includes task 
        parameters specified here, such as subject name and reward value.
        
        The same `self.node` is used later to end the session on the children.
        """
        # With instance=True, I get a threading error about current event loop
        self.node = Net_Node(
            id="T_{}".format(prefs.get('NAME')),
            upstream=prefs.get('NAME'),
            port=prefs.get('MSGPORT'),
            listens={},
            instance=False,
            )

        # Construct a message to send to child
        # Specify the subjects for the child (twice)
        # These extra params end up in the __init__ for the child class
        value = {
            'child': {
                'parent': prefs.get('NAME'), 'subject': subject},
            'task_type': 'PAFT_Child',
            'subject': subject,
            'reward': reward,          
        }

        # send to the station object with a 'CHILD' key
        self.node.send(to=prefs.get('NAME'), key='CHILD', value=value)        

    def create_inter_pi_communication_node(self):
        """Defines a Net_Node to communicate with the children
        
        This is a second Net_Node that is used to directly exchange information
        with the children about pokes and sounds. Unlike the first Net_Node,
        for this one the parent is the "router" / server and the children
        are the "dealer" / clients .. ie many dealers, one router.
        
        Each child needs to create a corresponding Net_Node and connect to
        this one. This function will block until that happens.
        
        The Net_Node defined here also specifies "listens" (ie triggers)
        of functions to be called upon receiving specified messages
        from the children, such as "HELLO" or "POKE".
        
        This Net_Node is saved as `self.node2`.
        """
        ## Create a second node to communicate with the child
        # We (parent) are the "router"/server
        # The children are the "dealer"/clients
        # Many dealers, one router        
        self.node2 = Net_Node(
            id='parent_pi',
            upstream='',
            port=5000,
            router_port=5001,
            listens={
                'HELLO': self.recv_hello,
                'POKE': self.recv_poke,
                'REWARD': self.recv_reward,
                'CHUNK': self.recv_chunk,                
                },
            instance=False,
            )

        # Wait until the child connects!
        self.logger.debug("Waiting for child to connect")
        while True:
            stop_looping = True
            
            for child, is_conn in self.child_connected.items():
                if not is_conn:
                    stop_looping = False
            
            if stop_looping:
                break
        self.logger.debug(
            "All children have connected: {}".format(self.child_connected)
            )        

    def silence_all(self, left_punish, right_punish):
        """Tell all children to play no sound and punish all pokes"""
        for which_pi in prefs.get('CHILDID'):
            self.silence_pi(which_pi, left_punish, right_punish)

    def silence_pi(self, which_pi, left_punish, right_punish):
        """Silence `which_pi` by playing neither and punishing both"""
        self.logger.debug('silencing {}'.format(which_pi))
        self.node2.send(
            to=which_pi,
            key='PLAY',
            value={
                'left_on': False, 'right_on': False,
                'left_punish': left_punish, 'right_punish': right_punish,
                'left_reward': False, 'right_reward': False,
                'synchronization_flash': False,
                },
            )              

    def send_acoustic_params(self, port_params, stim_params_to_send, 
        synchronization_flash):
        """Send params to each pi
        
        port_params : DataFrame of port-specific params
        
        stim_params_to_send : dict of global params
        
        synchronization_flash : bool
            Whether to also send a synchronization request

        """
        # Iterate over pilots
        for which_pi, sub_df in port_params.groupby('pilot'):
            # Extract kwargs for this pilot
            sub_df = sub_df.set_index('side')
            kwargs = {
                'left_on': sub_df.loc['L', 'sound_on'],
                'left_target_rate': sub_df.loc['L', 'target_rate'],
                'left_distracter_rate': sub_df.loc['L', 'distracter_rate'],
                'left_punish': ~sub_df.loc['L', 'reward'],
                'left_reward': sub_df.loc['L', 'reward'],
                'right_on': sub_df.loc['R', 'sound_on'],
                'right_target_rate': sub_df.loc['R', 'target_rate'],
                'right_distracter_rate': sub_df.loc['R', 'distracter_rate'],
                'right_punish': ~sub_df.loc['R', 'reward'],
                'right_reward': sub_df.loc['R', 'reward'],
                }
            
            # Add on global params
            kwargs.update(stim_params_to_send)
            
            # Sync
            if synchronization_flash:
                kwargs['synchronization_flash'] = True
            else:
                kwargs['synchronization_flash'] = False

            # Send the message
            self.node2.send(to=which_pi, key='PLAY', value=kwargs)
    
    def choose_stimulus(self):
        """A stage that chooses the stimulus"""
        # Get timestamp
        timestamp_trial_start = datetime.datetime.now()
        
        # Pokes occuring *right here* will get the wrong trial number!
        
        # Increment trial now
        self.counter_trials_in_session += 1
        self.counter_trials_across_sessions += 1
        
        # Announce
        self.logger.debug(
            'choose_stimulus: entering stage at {}'.format(
            timestamp_trial_start.isoformat()))

        # self.rewarded_port on *previous* trial is now previously_rewarded_port
        self.previously_rewarded_port = self.rewarded_port
        
        # Exclude previously rewarded port
        choose_from = [
            kpp for kpp in self.known_pilot_ports 
            if kpp != self.previously_rewarded_port]
        
        # Choose stimulus randomly and update `self.rewarded_port`
        self.rewarded_port = random.choice(choose_from)
        self.logger.debug('choose_stimulus: chose {}'.format(self.rewarded_port))
        
        # This will be set at the time of reward
        self.timestamp_of_last_reward = None        
        
        # This is used to keep track of rank of each poke
        self.ports_poked_on_this_trial = []
        
        # This is used to keep track of whether a reward was delivered
        self.reward_delivered_on_this_trial = False
        
        
        ## Choose params
        # Each is taken from an entry in `self.stim_choosing_params`
        # Each matches a column in `TrialData`
        stim_target_rate = random.choice(
            self.stim_choosing_params['target_rate'])
        stim_target_temporal_log_std = random.choice(
            self.stim_choosing_params['target_temporal_log_std'])
        stim_target_spatial_extent = random.choice(
            self.stim_choosing_params['target_spatial_extent'])
        stim_distracter_rate = random.choice(
            self.stim_choosing_params['distracter_rate'])
        stim_distracter_temporal_log_std = random.choice(
            self.stim_choosing_params['distracter_temporal_log_std'])
        stim_target_center_freq = random.choice(
            self.stim_choosing_params['target_center_freq'])
        stim_target_bandwidth = random.choice(
            self.stim_choosing_params['target_bandwidth'])
        stim_target_log_amplitude = random.choice(
            self.stim_choosing_params['target_log_amplitude'])
        stim_distracter_center_freq = random.choice(
            self.stim_choosing_params['distracter_center_freq'])
        stim_distracter_bandwidth = random.choice(
            self.stim_choosing_params['distracter_bandwidth'])
        stim_distracter_log_amplitude = random.choice(
            self.stim_choosing_params['distracter_log_amplitude'])
        
        # Put the ones that don't vary with port in a dict
        # The ones that do vary with port are captured by port_params
        stim_params_to_send = {
            'stim_target_temporal_log_std': stim_target_temporal_log_std,
            'stim_distracter_temporal_log_std': stim_distracter_temporal_log_std,
            'stim_target_center_freq': stim_target_center_freq,
            'stim_target_bandwidth': stim_target_bandwidth,
            'stim_target_log_amplitude': stim_target_log_amplitude,
            'stim_distracter_center_freq': stim_distracter_center_freq,
            'stim_distracter_bandwidth': stim_distracter_bandwidth,
            'stim_distracter_log_amplitude': stim_distracter_log_amplitude,            
            }
        
        
        ## Generate port_params DataFrame
        port_params = pandas.Series(
            self.known_pilot_ports, name='port').to_frame()
        
        # Extract pilot and side
        port_params['pilot'] = port_params['port'].apply(
            lambda s: s.split('_')[0])
        port_params['side'] = port_params['port'].apply(
            lambda s: s.split('_')[1])

        # Find the rewarded row
        rewarded_idx = port_params.index[
            np.where(port_params['port'] == self.rewarded_port)[0][0]]
        
        # Only reward that one
        port_params['reward'] = False
        port_params.loc[rewarded_idx, 'reward'] = True

        # This is the distance from each port to the rewarded port
        half_dist = len(port_params) // 2
        port_params['absdist'] = np.abs(np.mod(
            port_params.index - rewarded_idx + half_dist, 
            len(port_params)) - half_dist)
        
        
        ## Use the acoustic params to set the port_params
        # These are the params that vary with distance from goal
        # They all have sound on
        port_params.loc[:, 'sound_on'] = True
        
        # Set rate of target sounds
        # Once port_params['absdist'] reaches 1 + stim_target_spatial_extent,
        # target rate falls to zero
        port_params.loc[:, 'target_rate'] = stim_target_rate * (
            (1 + stim_target_spatial_extent - port_params['absdist']) /
            (1 + stim_target_spatial_extent)
            )
        
        # Floor target rate at zero
        port_params.loc[port_params['target_rate'] < 0, 'target_rate'] = 0
        
        # Calculate distracter rate
        port_params.loc[:, 'distracter_rate'] = (
            stim_distracter_rate - port_params.loc[:, 'target_rate'])
        port_params.loc[
            port_params['distracter_rate'] < 0, 'distracter_rate'] = 0
        
        
        ## Send the play and silence messages
        # Debug
        self.logger.debug('chose port_params:\n{}'.format(port_params))
        self.logger.debug('chose stim_params_to_send:\n{}'.format(
            stim_params_to_send))
        
        # Send those parameters, and also request a synchronization flash
        self.send_acoustic_params(
            port_params, stim_params_to_send, synchronization_flash=True)
    

        ## Continue to the next stage
        # CLEAR means "wait for triggers"
        # SET means "advance anyway"
        self.stage_block.set()

        # Return data about chosen_stim so it will be added to HDF5
        # Because this is the first return of the incremented trial_num,
        # this will make a new row in the HDF5 file.
        # Even if we don't increment trial_num,
        # it will still make another row in the HDF5, but it might warn.
        # (TODO: Check this actually works)
        # (This happens in autopilot.core.subject.Subject.data_thread)
        if self.previously_rewarded_port is None:
            prp_to_send = ''
        else:
            prp_to_send = self.previously_rewarded_port
        
        return {
            'rewarded_port': self.rewarded_port,
            'previously_rewarded_port': prp_to_send,
            'timestamp_trial_start': timestamp_trial_start.isoformat(),
            'trial_num': self.counter_trials_across_sessions,
            'trial_in_session': self.counter_trials_in_session,
            'stim_target_rate': stim_target_rate,
            'stim_target_temporal_std': stim_target_temporal_log_std,
            'stim_target_spatial_extent': stim_target_spatial_extent,
            'stim_distracter_rate': stim_distracter_rate,
            'stim_distracter_temporal_std': stim_distracter_temporal_log_std,
            'stim_target_center_freq': stim_target_center_freq,
            'stim_target_bandwidth': stim_target_bandwidth,
            'stim_target_amplitude': stim_target_log_amplitude,
            'stim_distracter_center_freq': stim_distracter_center_freq,
            'stim_distracter_bandwidth': stim_distracter_bandwidth,
            'stim_distracter_amplitude': stim_distracter_log_amplitude,              
            }

    def wait_for_response(self):
        """A stage that waits for a response"""
        # Wait a little before doing anything
        self.logger.debug('wait_for_response: entering stage')

        # Do not continue until the stage_block is set, e.g. by a poke
        self.stage_block.clear()        
        
        # This is tested in recv_poke before advancing
        self.advance_on_port = self.rewarded_port
    
    def report_reward(self):
        """A stage that just reports reward timestamp"""
        # Silence all speakers, no punishment
        self.silence_all(left_punish=False, right_punish=False)

        # Immediately advance to next stage
        self.stage_block.set()
        
        # Return self.timestamp_of_last_reward, which was set at the time
        # of the reward.
        # Check for None just to be safe
        if self.timestamp_of_last_reward is not None:
            return {
                'timestamp_reward': self.timestamp_of_last_reward.isoformat(),        
                } 
        else:
            self.logger.debug(
                "error: self.timestamp_of_last_reward is None, "
                "this shouldn't happen")
    
    def end_of_trial(self):
        """A stage that ends the trial
        
        TODO: split out the ITI component into its own stage, so
        reward can be reported immediately
        """
        # Announce
        self.logger.debug('end_of_trial: entering stage')
        
        # 5 s ITI
        time.sleep(3)        

        # Continue to the next stage
        self.stage_block.set()        
        
        # Return TRIAL_END so the Terminal knows the trial is over, which
        # appends a row to the HDF5
        return {
            'TRIAL_END': True,
            }

    def init_hardware(self, *args, **kwargs):
        """Placeholder to init hardware
        
        This is here to remind me that init_hardware is implemented by the
        base class `Task`. This function could be removed if there is
        no hardware actually connected and/or defined in HARDWARE.
        """
        super().init_hardware(*args, **kwargs)

    def handle_trigger(self, pin, level=None, tick=None):
        """Handle a GPIO trigger, overriding superclass.
        
        This overrides the behavior in the superclass `Task`, most importantly
        by not changing the stage block or clearing the triggers. Therefore
        this function changes the way tasks proceed through stages, and
        should be included in any PAFT-like task to provide consistent
        stage progression logic. 
        
        All GPIO triggers call this function because the `init_hardware`
        function sets their callback to this function. (Possibly true only
        for pins in self.HARDWARE?) 
        
        When they do call, they provide these arguments:
            pin (int): BCM Pin number
            level (bool): True, False high/low
            tick (int): ticks since booting pigpio
        
        This function converts the BCM pin number to a board number using
        BCM_TO_BOARD, and then to a letter using `self.pin_id`.
        
        That letter is used to look up the relevant triggers in
        `self.triggers`, and calls each of them.
        
        `self.triggers` MUST be a list-like.
        
        This function does NOT clear the triggers or the stage block.
        """
        # Convert to BOARD_PIN
        board_pin = BCM_TO_BOARD[pin]
        
        # Convert to letter, e.g., 'C'
        pin_letter = self.pin_id[board_pin]

        # Log
        self.logger.debug(
            'trigger bcm {}; board {}; letter {}; level {}; tick {}'.format(
            pin, board_pin, pin_letter, level, tick))
        
        # TODO: acquire trigger_lock here?
        # Call any triggers that exist
        if pin_letter in self.triggers:
            trigger_l = self.triggers[pin_letter]
            for trigger in trigger_l:
                trigger()
        else:
            self.logger.debug(f"No trigger found for {pin}")

    def recv_hello(self, value):
        self.logger.debug(
            "received HELLO from child with value {}".format(value))
        
        # Set this flag
        self.child_connected[value['from']] = True

    def recv_chunk(self, value):
        """Forwards a chunk of data from a child to the terminal.
        
        value : dict, with keys:
            'payload' : 2d array
            'payload_columns' : list of strings
                These become the names of the columns of `payload`, so they
                should be the same length.
            'timestamp' : the time of the message
            'pilot' : the name of the child
        
        Any other items in `value` are ignored. 
        
        Those items in `value`, plus {'subject': self.subject, 'chunk': True},
        are put into a new Message and sent to _T with key 'DATA' and
        flags to disable printing and repeating.
        """
        # Log
        self.logger.debug(
            "received CHUNK from child, passing along"
            )
        
        # Pass along to terminal for saving
        # `value` should have keys pilot, payload, and timestamp
        value_to_send = {
            'payload': value['payload'],
            'payload_columns': value['payload_columns'],
            'timestamp': value['timestamp'],
            'pilot': value['pilot'], # required by something
            'subject': self.subject, # required by terminal.l_data            
            'chunkclass_name': value['chunkclass_name'], # which chunk
            }
        
        # Generate the Message
        msg = autopilot.networking.Message(
            to='_T', # send to terminal
            key='DATA', # choose listen
            value=value_to_send, # the value to send
            flags={
                'MINPRINT': True, # disable printing of value
                'NOREPEAT': True, # disable repeating
                },
            id="dummy_dst2", # does nothing (?), but required
            sender="dummy_src2", # does nothing (?), but required 
            )

        # Send to terminal
        self.node.send('_T', 'DATA', msg=msg)
    
    def recv_poke(self, value):
        """A poke was received. Send info to terminal.

            poked_port : which port was poked
            
            first_poke : True only on the very first poke of the trial
                (excluding previously rewarded port)
                This is used to determine if the trial was correct
                This happens exactly once per trial
            
            reward_delivered : True if a reward was delivered on this very
                poke. This is used to determine if the trial is over
                This happens exactly once per trial
            
            poke_rank : rank of the poked port on this trial
                (excluding previously rewarded port)
                This must be 0 if (first_poke and reward_delivered),
                which happens once on correct trials 
                and never on incorrect trials.                
                The value of this on the single poke per trial when 
                reward_delivered is True is used to calculate RCP and FC
        
        If poked_port == previously_rewarded_port:
            first_poke is False
            reward_delivered is False
            poke_rank is -1
        """
        # TODO: get the timestamp directly from the child rpi instead of 
        # inferring it here
        poke_timestamp = datetime.datetime.now()

        # Form poked_port
        poked_port = '{}_{}'.format(value['from'], value['poke'])

        # Special case previously rewarded port
        if poked_port == self.previously_rewarded_port:
            # Don't count these pokes
            this_is_first_poke = False
            this_is_rewarded_poke = False
            this_poke_rank = -1
        
        else:
            # Infer whether this is the first poke of the current trial
            if self.counter_trials_in_session != self.trial_of_last_poke:
                this_is_first_poke = True

                # It is the first poke of the trial, update the memory
                self.trial_of_last_poke = self.counter_trials_in_session
            else:
                this_is_first_poke = False
            
            # Infer whether reward delivered
            if (
                    poked_port == self.rewarded_port and 
                    not self.reward_delivered_on_this_trial
                    ):
                this_is_rewarded_poke = True
                
                # Setting this flag ensures consummation pokes are not counted
                # again
                self.reward_delivered_on_this_trial = True                
            else:
                this_is_rewarded_poke = False
            
            # Keep track of rank of poke on this trial
            this_poke_rank = len(self.ports_poked_on_this_trial)
            if poked_port not in self.ports_poked_on_this_trial:
                self.ports_poked_on_this_trial.append(poked_port)

        # Announce
        self.logger.debug(
            "[{}] received POKE from child with value {}".format(
            poke_timestamp.isoformat(), value))
        self.logger.debug(
            "[{}] {} poked; {}; {}; {}".format(
                poke_timestamp.isoformat(), 
                poked_port,
                'reward delivered' if this_is_rewarded_poke else 'unrewarded',
                'first of trial' if this_is_first_poke else 'not first',
                'rewarded' if this_is_rewarded_poke else 'not rewarded',
                value))

        
        ## Chunk and send to terminal for saving
        # Convert to Series
        payload_df = pandas.DataFrame.from_dict({
            'poked_port': [poked_port],
            'first_poke': [this_is_first_poke],
            'reward_delivered': [this_is_rewarded_poke],
            'poke_rank': [this_poke_rank],
            'timestamp': [poke_timestamp.isoformat()],
            'trial': [self.counter_trials_in_session],
            })
            
        # `value` should have keys pilot, payload, and timestamp
        value_to_send = {
            'payload': payload_df.values,
            'payload_columns': payload_df.columns.values,
            'timestamp': poke_timestamp.isoformat(),
            'pilot': prefs.get('name'), # required by something
            'subject': self.subject, # required by terminal.l_data            
            'chunkclass_name': 'ChunkData_Pokes', # which chunk
            }
        
        # Generate the Message
        msg = autopilot.networking.Message(
            to='_T', # send to terminal
            key='DATA', # choose listen
            value=value_to_send, # the value to send
            flags={
                'MINPRINT': True, # disable printing of value
                'NOREPEAT': True, # disable repeating
                },
            id="dummy_dst2", # does nothing (?), but required
            sender="dummy_src2", # does nothing (?), but required 
            )

        # Send to terminal
        self.node.send('_T', 'DATA', msg=msg)
    
    
        ## Also send to plot
        self.node.send(
            to='P_{}'.format(prefs.get('NAME')),
            key='DATA',
            value={
                'subject': self.subject,
                'pilot': prefs.get('NAME'),
                'timestamp': poke_timestamp.isoformat(),
                'poked_port': poked_port,
                'first_poke': this_is_first_poke,
                'reward_delivered': this_is_rewarded_poke,
                'poke_rank': this_poke_rank,
                },
            )            

    def recv_reward(self, value):
        """Log reward and advance stage block"""
        # TODO: get the timestamp directly from the child rpi instead of 
        # inferring it here
        reward_timestamp = datetime.datetime.now()

        # Announce
        self.logger.debug(
            "[{}] received REWARD from child with value {}".format(
            reward_timestamp.isoformat(), value))

        # Form poked_port
        poked_port = '{}_{}'.format(value['from'], value['poke'])
        
        # If the poked port was the rewarded port, then set the stage_block
        # This guard should not be necessary because the child pi should
        # only reward once per trial
        if poked_port == self.advance_on_port:
            # Null this flag so we can't somehow advance twice
            self.advance_on_port = None
            
            # Advance
            self.stage_block.set()
        else:
            self.logger.debug(
                "error: reward signal received from {}, ".format(poked_port) +
                "but advance_on_port was {}".format(self.advance_on_port)
                )

        # Store the time of the reward
        self.timestamp_of_last_reward = reward_timestamp   

        # Directly report continuous data to terminal (aka _T)
        # Otherwise it can be encoded in the returned data, but that is only
        # once per stage
        # subject is needed by core.terminal.Terminal.l_data
        # pilot is needed by networking.station.Terminal_Station.l_data
        # timestamp and continuous are needed by subject.Subject.data_thread
        # `trial` is for convenience, but note it will be wrong for pokes
        # that occur during the "choose_stimulus" function
        self.node.send(
            to='_T',
            key='DATA',
            value={
                'subject': self.subject,
                'pilot': prefs.get('NAME'),
                'continuous': True,
                'reward_timestamp': reward_timestamp,
                'trial': self.counter_trials_in_session,
                },
            )

    def end(self, *args, **kwargs):
        """Called when the task is ended by the user.
        
        The base class `Task` releases hardware objects here.
        This is a placeholder to remind me of that.
        """
        self.logger.debug('end: entering function')
        
        # Tell the child to end the task
        self.node.send(to=prefs.get('NAME'), key='CHILD', value={'KEY': 'STOP'})

        # Sometimes it seems like the children don't get the message,
        # maybe wait?
        time.sleep(1)

        # This sock.close seems to be necessary to be able to communicate again
        self.node.sock.close()
        self.node.release()

        # This router.close() prevents ZMQError on the next start
        self.node2.router.close()
        self.node2.release() 

        # Let the superclass end handle releasing hardware
        super().end(*args, **kwargs)

