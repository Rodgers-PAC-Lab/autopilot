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
        'tag':'Reward Duration (ms)',
        'type':'int',
        }

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
        rewarded_port = tables.StringCol(64)
        
        # The timestamps
        timestamp_trial_start = tables.StringCol(64)
        timestamp_reward = tables.StringCol(64)

    # Define continuous data
    # https://docs.auto-pi-lot.com/en/latest/guide/task.html
    # autopilot.core.subject.Subject.data_thread would like one of the
    # keys to be "timestamp"
    # Actually, no I think that is extracted automatically from the 
    # networked message, and should not be defined here
    class ContinuousData(tables.IsDescription):
        poked_port = tables.StringCol(64)

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
    children_names = prefs.get('childid')
    if children_names is None:
        # This happens on terminal
        children_names = []
    CHILDREN = {}
    for child in children_names:
        CHILDREN[child] = {'task_type': "PAFT_Child"}

    
    ## Define the class methods
    def __init__(self, stage_block, current_trial, step_name, task_type, 
        subject, step, session, pilot, graduation, reward):
        """Initialize a new PAFT Task. 
        
        
        --
        Plan
        
        The first version of this new task needs to:
        * Be able to play streams of sounds from two speakers (no children)
        
        Another version needs to
        * Connect to the children, without playing sounds
        
        Then these can be merged in a second version that can
        * Report pokes from children
        * Tell children to play sounds
        
        The third version needs to:
        * Incorporate advancing through stages
        * Return data about each trial
        * Test continuous data (pokes)
        * Test plot data
        
        The fourth version needs to:
        * Incorporate subject-specific training params (skip for now)
        
        Then bring together into one version that
        * Connects to children, each of which play sounds
        * Chooses stim and tells them to play which sound
        * Reports pokes as continuous data
        * Reports trial outcome as trial data
        * Plots        
        
        ---
        
        All arguments are provided by the Terminal.
        
        Note that this __init__ does not call the superclass __init__, 
        because that superclass Task inclues functions for punish_block
        and so on that we don't want to use.
        
        Some params, such as `step_name` and `task_type`, are always required 
        to be specified in the json defining this protocol
        
        Other params, such as `reward`, are custom to this particular task.
        They should be described in the class attribute `PARAMS` above, and
        their values should be specified in the protocol json.
        
        Arguments
        ---------
        stage_block (:class:`threading.Event`): 
            used to signal to the carrying Pilot that the current trial 
            stage is over
        current_trial (int): 
            If not zero, initial number of `trial_counter`
            This is set to be 1 greater than the last value of "trial_num"
            in the HDF5 file by autopilot.core.subject.Subject.prepare_run
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
        self.counter_trials_across_sessions = itertools.count(int(current_trial))        

        # This is used to count the trials for the "trial_in_session" HDF5 column
        self.counter_trials_in_session = itertools.count(0)

        # A dict of hardware triggers
        self.triggers = {}
        
        
        ## Define the possible ports
        self.known_pilot_ports = []
        for child in self.children_names:
            self.known_pilot_ports.append('{}_{}'.format(child, 'L'))
            self.known_pilot_ports.append('{}_{}'.format(child, 'R'))
        
    
        ## Define the stages
        # Stage list to iterate
        # Iterate through these three stages forever
        stage_list = [
            self.choose_stimulus, self.wait_for_response, self.end_of_trial]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)        
        
        
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

    def silence_all(self):
        """Tell all children to play no sound and punish all pokes"""
        for which_pi in self.children_names:
            self.silence_pi(which_pi)

    def silence_pi(self, which_pi):
        """Silence `which_pi` by playing neither and punishing both"""
        self.logger.debug('silencing {}'.format(which_pi))
        self.node2.send(
            to=which_pi,
            key='PLAY',
            value={
                'left_on': False, 'right_on': False,
                'left_punish': True, 'right_punish': True,
                },
            )              

    def reward_one(self, which_pi, which_side):
        """Tell one speaker to play and silence all others"""
        
        ## Tell `which_pi` to reward `which_side` (and not the other)
        # Construct kwargs
        if which_side in ['left', 'L']:
            kwargs = {
                'left_on': True, 'right_on': False,
                'left_punish': False, 'right_punish': True
                }
        elif which_side in ['right', 'R']:
            kwargs = {
                'left_on': False, 'right_on': True,
                'left_punish': True, 'right_punish': False
                }
        else:
            raise ValueError("unexpected which_side: {}".format(which_side))        
        
        # Send the message
        self.node2.send(to=which_pi, key='PLAY', value=kwargs)
        
        
        ## Tell all other children to reward neither
        for other_pi in self.children_names:
            if other_pi == which_pi:
                continue
            
            self.silence_pi(other_pi)      

    def send_acoustic_params(self, port_params):
        """Take a DataFrame of acoustic_params by pi and send them
        
           pilot side  reward  sound_on  mean_interval  var_interval
        0  rpi10    L   False      True           0.50          0.01
        1  rpi10    R    True      True           0.25          0.01
        2  rpi11    L   False      True           0.50          0.01
        3  rpi11    R   False     False           0.75          0.01
        4  rpi12    L   False     False           1.00          0.01
        5  rpi12    R   False     False           0.75          0.01
        """
        # Iterate over pilots
        for which_pi, sub_df in port_params.groupby('pilot'):
            # Extract kwargs for this pilot
            sub_df = sub_df.set_index('side')
            kwargs = {
                'left_on': sub_df.loc['L', 'sound_on'],
                'left_mean_interval': sub_df.loc['L', 'mean_interval'],
                'left_var_interval': sub_df.loc['L', 'var_interval'],
                'left_punish': ~sub_df.loc['L', 'reward'],
                'right_on': sub_df.loc['R', 'sound_on'],
                'right_mean_interval': sub_df.loc['R', 'mean_interval'],
                'right_var_interval': sub_df.loc['R', 'var_interval'],
                'right_punish': ~sub_df.loc['R', 'reward'],         
                }

            # Send the message
            self.node2.send(to=which_pi, key='PLAY', value=kwargs)
    
    def choose_stimulus(self):
        """A stage that chooses the stimulus"""
        # Get timestamp
        timestamp_trial_start = datetime.datetime.now()
        
        # Announce
        self.logger.debug(
            'choose_stimulus: entering stage at {}'.format(
            timestamp_trial_start.isoformat()))
        
        # Choose stimulus randomly
        rewarded_port = random.choice(self.known_pilot_ports)
        self.logger.debug('choose_stimulus: chose {}'.format(rewarded_port))
        
        
        ## Set acoustic params accordingly
        # Generate port_params DataFrame
        port_params = pandas.Series(
            self.known_pilot_ports, name='port').to_frame()
        
        # Extract pilot and side
        port_params['pilot'] = port_params['port'].apply(
            lambda s: s.split('_')[0])
        port_params['side'] = port_params['port'].apply(
            lambda s: s.split('_')[1])

        # Find the rewarded row
        rewarded_idx = port_params.index[
            np.where(port_params['port'] == rewarded_port)[0][0]]
        
        # Only reward that one
        port_params['reward'] = False
        port_params.loc[rewarded_idx, 'reward'] = True

        # This is the distance from each port to the rewarded port
        half_dist = len(port_params) // 2
        port_params['absdist'] = np.abs(np.mod(
            port_params.index - rewarded_idx + half_dist, 
            len(port_params)) - half_dist)
        
        # Only have sound on for some
        port_params.loc[:, 'sound_on'] = port_params['absdist'] <= 1
        
        # Choose mean_interval
        port_params.loc[:, 'mean_interval'] = .25 + port_params['absdist'] * .25
        
        # Choose var_interval
        port_params.loc[:, 'var_interval'] = .01
        
        
        ## Send the play and silence messages
        # Tell those to play
        self.logger.debug('using {}'.format(port_params))
        self.send_acoustic_params(port_params)
    

        ## Continue to the next stage
        # CLEAR means "wait for triggers"
        # SET means "advance anyway"
        self.stage_block.set()
        
        # Store this for the next stage
        self.rewarded_port = rewarded_port

        # Return data about chosen_stim so it will be added to HDF5
        # I think it's best to increment trial_num now, since this is the
        # first return from this trial. Even if we don't increment trial_num,
        # it will still make another row in the HDF5, but it might warn.
        # (This happens in autopilot.core.subject.Subject.data_thread)
        return {
            'rewarded_port': rewarded_port,
            'timestamp_trial_start': timestamp_trial_start.isoformat(),
            'trial_num': next(self.counter_trials_across_sessions),
            'trial_in_session': next(self.counter_trials_in_session),
            }

    def wait_for_response(self):
        """A stage that waits for a response"""
        # Wait a little before doing anything
        self.logger.debug('wait_for_response: entering stage')

        # Do not continue until the stage_block is set, e.g. by a poke
        self.stage_block.clear()        
        
        # This is tested in recv_poke before advancing
        self.advance_on_port = self.rewarded_port
    
    def end_of_trial(self):
        """A stage that ends the trial"""
        self.logger.debug('end_of_trial: entering stage')

        # Silence all of them for 5 s ITI
        self.silence_all()
        time.sleep(5)        
        
        # Cleanup logic could go here

        # Continue to the next stage
        self.stage_block.set()        
        
        # Return TRIAL_END so the Terminal knows the trial is over
        return {
            'TRIAL_END': True,
            }

    def init_hardware(self, *args, **kwargs):
        """Placeholder to init hardware
        
        This is here to remind me that init_hardware is implemented by the
        base class `Task`. This function could be removed if there is
        no hardware actually connected and/or defined in HARDWARE.
        """
        super(PAFT, self).init_hardware(*args, **kwargs)

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
    
    def recv_poke(self, value):
        # TODO: get the timestamp directly from the child rpi instead of 
        # inferring it here
        poke_timestamp = datetime.datetime.now()

        # Announce
        self.logger.debug(
            "[{}] received POKE from child with value {}".format(
            poke_timestamp.isoformat(), value))

        # Form poked_port
        poked_port = '{}_{}'.format(value['from'], value['poke'])
        
        # Directly report continuous data to terminal (aka _T)
        # Otherwise it can be encoded in the returned data, but that is only
        # once per stage
        # subject is needed by core.terminal.Terminal.l_data
        # pilot is needed by networking.station.Terminal_Station.l_data
        # timestamp and continuous are needed by subject.Subject.data_thread
        self.node.send(
            to='_T',
            key='DATA',
            value={
                'subject': self.subject,
                'pilot': prefs.get('NAME'),
                'continuous': True,
                'poked_port': poked_port,
                'timestamp': poke_timestamp.isoformat(),
                },
            )

        # Also send to plot
        self.node.send(
            to='P_{}'.format(prefs.get('NAME')),
            key='DATA',
            value={
                'subject': self.subject,
                'pilot': prefs.get('NAME'),
                'continuous': True,
                'poked_port': poked_port,
                'timestamp': poke_timestamp.isoformat(),
                },
            )  
        
        # If the poked port was the rewarded port, then set the stage_block
        if poked_port == self.advance_on_port:
            # Null this flag so we can't somehow advance twice
            self.advance_on_port = None
            
            # Advance
            self.stage_block.set()

    def end(self, *args, **kwargs):
        """Called when the task is ended by the user.
        
        The base class `Task` releases hardware objects here.
        This is a placeholder to remind me of that.
        """
        self.logger.debug('end: entering function')
        
        # Tell the child to end the task
        self.node.send(to=prefs.get('NAME'), key='CHILD', value={'KEY': 'STOP'})

        # This sock.close seems to be necessary to be able to communicate again
        self.node.sock.close()
        self.node.release()

        # This router.close() prevents ZMQError on the next start
        self.node2.router.close()
        self.node2.release() 

        # Let the superclass end handle releasing hardware
        super(PAFT, self).end(*args, **kwargs)

