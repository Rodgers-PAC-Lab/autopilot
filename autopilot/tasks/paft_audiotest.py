"""This module defines PAFT_With_Multisound.

All this does is play sounds from left and right speakers, with occasional
error sounds interspersed. This is to test / demonstrate a new way
to add sounds to jackclient directly instead of going through the Sound
class. This enables playing multiple queues of sounds at the same time.
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
TASK = 'PAFT_audiotest'


## Define the Task
class PAFT_audiotest(Task):
    """The probabalistic auditory foraging task (PAFT).
    
    To understand the stage progression logic, see:
    * autopilot.core.pilot.Pilot.run_task - the main loop
    * autopilot.tasks.task.Task.handle_trigger - set stage trigger
    
    To understand the data saving logic, see:
    * autopilot.core.terminal.Terminal.l_data - what happens when data is sent
    * autopilot.core.subject.Subject.data_thread - how data is saved

    Class attributes:
        PARAMS : collections.OrderedDict
            This defines the params we expect to receive from the terminal.
        DATA : dict of dicts
            This defines the kind of data we return to the terminal
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
    PARAMS = odict()
    PARAMS['reward'] = {
        'tag':'reward duration (ms)',
        'type':'int',
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
    

    
    ## Define the class methods
    def __init__(self, stage_block, current_trial, step_name, task_type, 
        subject, step, session, pilot, graduation, reward):
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
        
        Arguments
        ---------
        stage_block (:class:`threading.Event`): 
            used to signal to the carrying Pilot that the current trial 
            stage is over
        current_trial (int): 
            If not zero, initial number of `trial_counter`
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
        # Initialize to -1, because the first thing that happens is that
        # it is incremented upon choosing the first stimulus
        self.counter_trials_in_session = -1

    
        ## Define the stages
        # Stage list to iterate
        stage_list = [self.play]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)        
        
        
        ## Init hardware -- this sets self.hardware and self.pin_id
        self.triggers = {}
        self.init_hardware()
        


        
        
        
        ## Initialize sounds
        # Each block/frame is about 5 ms
        # Longer is more buffer against unexpected delays
        # Shorter is faster to empty and refill the queue
        self.target_qsize = 200

        # Some counters to keep track of how many sounds we've played
        self.n_frames = 0
        self.n_error_counter = 0        

        # Initialize these to None, in case they don't get defined later
        # (e.g., poketrain)
        self.left_error_sound = None
        self.right_error_sound = None

        # Fill the queue with empty frames
        # Sounds aren't initialized till the trial starts
        # Using False here should work even without sounds initialized yet
        self.set_sound_cycle(params={'left_on': False, 'right_on': False})

        # Use this to keep track of generated sounds
        self.current_audio_times_df = None

        # This is needed when sending messages about generated sounds
        self.n_messages_sent = 0


        ## Set up NET_Node to communicate with Parent
        # Do this after initializing the sounds, otherwise we won't be
        # ready to play yet
        self.create_inter_pi_communication_node()

    def initalize_sounds(self,             
        target_highpass, target_amplitude, target_lowpass,
        distracter_highpass, distracter_amplitude, distracter_lowpass,
        ):
        """Defines sounds that will be played during the task"""
        ## Define sounds
        # Left and right target noise bursts
        self.left_target_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=target_amplitude, channel=0, 
            lowpass=target_lowpass, highpass=target_highpass)       

        self.right_target_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=target_amplitude, channel=1, 
            lowpass=target_lowpass, highpass=target_highpass)        

        # Left and right distracter noise bursts
        self.left_distracter_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=distracter_amplitude, channel=0, 
            lowpass=distracter_lowpass, highpass=distracter_highpass)       

        self.right_distracter_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=distracter_amplitude, channel=1, 
            lowpass=distracter_lowpass, highpass=distracter_highpass)  
            
        # Left and right tritone error noises
        self.left_error_sound = autopilot.stim.sound.sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=0)

        self.right_error_sound = autopilot.stim.sound.sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=1)
        
        # Chunk the sounds into frames
        if not self.left_target_stim.chunks:
            self.left_target_stim.chunk()
        if not self.right_target_stim.chunks:
            self.right_target_stim.chunk()
        if not self.left_distracter_stim.chunks:
            self.left_distracter_stim.chunk()
        if not self.right_distracter_stim.chunks:
            self.right_distracter_stim.chunk()
        if not self.left_error_sound.chunks:
            self.left_error_sound.chunk()
        if not self.right_error_sound.chunks:
            self.right_error_sound.chunk()
    
    def set_sound_cycle(self, params):
        """Define self.sound_cycle, to go through sounds
        
        params : dict
            This comes from a message on the net node.
            Possible keys:
                left_on
                right_on
                left_mean_interval
                right_mean_interval
        """
        # Log
        self.logger.debug('set_sound_cycle: received params: {}'.format(params))
        
        # This is just a left sound, gap, then right sound, then gap
        # And use a cycle to repeat forever
        # But this could be made more complex
        self.sound_block = []

        # Helper function
        def append_gap(gap_chunk_size=30):
            """Append `gap_chunk_size` silent chunks to sound_block"""
            for n_blank_chunks in range(gap_chunk_size):
                self.sound_block.append(
                    np.zeros(autopilot.stim.sound.jackclient.BLOCKSIZE, 
                    dtype='float32'))

        # Extract params or use defaults
        left_on = params.get('left_on', False)
        right_on = params.get('right_on', False)
        left_target_rate = params.get('left_target_rate', 0)
        right_target_rate = params.get('right_target_rate', 0)
        left_distracter_rate = params.get('left_distracter_rate', 0)
        right_distracter_rate = params.get('right_distracter_rate', 0)
        
        # Global params
        target_temporal_std = 10 ** params.get(
            'stim_target_temporal_log_std', -2)
        distracter_temporal_std = 10 ** params.get(
            'stim_distracter_temporal_log_std', -2)
       
        
        ## Generate intervals 
        # left target
        if left_on and left_target_rate > 1e-3:
            # Change of basis
            mean_interval = 1 / left_target_rate
            var_interval = target_temporal_std ** 2

            # Change of basis
            gamma_shape = (mean_interval ** 2) / var_interval
            gamma_scale = var_interval / mean_interval

            # Draw
            left_target_intervals = np.random.gamma(
                gamma_shape, gamma_scale, 100)
        else:
            left_target_intervals = np.array([])

        # right target
        if right_on and right_target_rate > 1e-3:
            # Change of basis
            mean_interval = 1 / right_target_rate
            var_interval = target_temporal_std ** 2

            # Change of basis
            gamma_shape = (mean_interval ** 2) / var_interval
            gamma_scale = var_interval / mean_interval

            # Draw
            right_target_intervals = np.random.gamma(
                gamma_shape, gamma_scale, 100)
        else:
            right_target_intervals = np.array([])     

        # left distracter
        if left_on and left_distracter_rate > 1e-3:
            # Change of basis
            mean_interval = 1 / left_distracter_rate
            var_interval = distracter_temporal_std ** 2

            # Change of basis
            gamma_shape = (mean_interval ** 2) / var_interval
            gamma_scale = var_interval / mean_interval

            # Draw
            left_distracter_intervals = np.random.gamma(
                gamma_shape, gamma_scale, 100)
        else:
            left_distracter_intervals = np.array([])

        # right distracter
        if right_on and right_distracter_rate > 1e-3:
            # Change of basis
            mean_interval = 1 / right_distracter_rate
            var_interval = distracter_temporal_std ** 2

            # Change of basis
            gamma_shape = (mean_interval ** 2) / var_interval
            gamma_scale = var_interval / mean_interval

            # Draw
            right_distracter_intervals = np.random.gamma(
                gamma_shape, gamma_scale, 100)
        else:
            right_distracter_intervals = np.array([])               
        
        
        ## Sort all the drawn intervals together
        # Turn into series
        left_target_df = pandas.DataFrame.from_dict({
            'time': np.cumsum(left_target_intervals),
            'side': ['left'] * len(left_target_intervals),
            'sound': ['target'] * len(left_target_intervals),
            })
        right_target_df = pandas.DataFrame.from_dict({
            'time': np.cumsum(right_target_intervals),
            'side': ['right'] * len(right_target_intervals),
            'sound': ['target'] * len(right_target_intervals),
            })
        left_distracter_df = pandas.DataFrame.from_dict({
            'time': np.cumsum(left_distracter_intervals),
            'side': ['left'] * len(left_distracter_intervals),
            'sound': ['distracter'] * len(left_distracter_intervals),
            })
        right_distracter_df = pandas.DataFrame.from_dict({
            'time': np.cumsum(right_distracter_intervals),
            'side': ['right'] * len(right_distracter_intervals),
            'sound': ['distracter'] * len(right_distracter_intervals),
            })
        
        # Concatenate them all together and resort by time
        both_df = pandas.concat([
            left_target_df, right_target_df, 
            left_distracter_df, right_distracter_df,
            ], axis=0).sort_values('time')

        # Calculate the gap between sounds
        both_df['gap'] = both_df['time'].diff().shift(-1)
        
        # Drop the last row which has a null gap
        both_df = both_df.loc[~both_df['gap'].isnull()].copy()

        # Keep only those below the sound cycle length
        both_df = both_df.loc[both_df['time'] < 10].copy()
        
        # Nothing should be null
        assert not both_df.isnull().any().any() 

        # Calculate gap size in chunks
        both_df['gap_chunks'] = (both_df['gap'] *
            autopilot.stim.sound.jackclient.FS / 
            autopilot.stim.sound.jackclient.BLOCKSIZE)
        both_df['gap_chunks'] = both_df['gap_chunks'].round().astype(np.int)
        
        # Floor gap_chunks at 1 chunk, the minimal gap size
        # This is to avoid distortion
        both_df.loc[both_df['gap_chunks'] < 1, 'gap_chunks'] = 1
        
        # Log
        self.logger.debug("generated both_df: {}".format(both_df))
        
        # Save
        self.current_audio_times_df = both_df.copy()
        self.current_audio_times_df = self.current_audio_times_df.rename(
            columns={'time': 'relative_time'})

        
        ## Depends on how long both_df is
        # If both_df has a nonzero but short length, results will be weird,
        # because it might just be one noise burst repeating every ten seconds
        # This only happens with low rates ~0.1Hz
        if len(both_df) == 0:
            # If no sound, then just put gaps
            append_gap(100)
        else:
            # Iterate through the rows, adding the sound and the gap
            # TODO: the gap should be shorter by the duration of the sound,
            # and simultaneous sounds should be possible
            for bdrow in both_df.itertuples():
                # Append the sound
                if bdrow.side == 'left' and bdrow.sound == 'target':
                    for frame in self.left_target_stim.chunks:
                        self.sound_block.append(frame) 
                elif bdrow.side == 'left' and bdrow.sound == 'distracter':
                    for frame in self.left_distracter_stim.chunks:
                        self.sound_block.append(frame)                         
                elif bdrow.side == 'right' and bdrow.sound == 'target':
                    for frame in self.right_target_stim.chunks:
                        self.sound_block.append(frame) 
                elif bdrow.side == 'right' and bdrow.sound == 'distracter':
                    for frame in self.right_distracter_stim.chunks:
                        self.sound_block.append(frame)       
                else:
                    raise ValueError(
                        "unrecognized side and sound: {} {}".format(
                        bdrow.side, bdrow.sound))
                
                # Append the gap
                append_gap(bdrow.gap_chunks)
        
        
        ## Cycle so it can repeat forever
        self.sound_cycle = itertools.cycle(self.sound_block)        

    def play(self):
        """A single stage that repeats forever and plays sound.
        
        On each call, jackclient.QUEUE is loaded to a target size with
        the current data in self.sound_cycle.
        
        On occasion, jackclient.QUEUE2 is loaded with an error sound.
        """
        ## Keep the stimulus queue minimum this length
        # Each block/frame is about 5 ms, so this is about 5 s of data
        # Longer is more buffer against unexpected delays
        target_qsize = 1000

        
        ## Load queue with stimulus, if needed
        print("before loading: {}".format(jackclient.QUEUE.qsize()))            
        
        # Add frames until target size reached
        qsize = jackclient.QUEUE.qsize()
        while qsize < target_qsize:
            with jackclient.Q_LOCK:
                # Add a frame from the sound cycle
                frame = next(self.sound_cycle)
                jackclient.QUEUE.put_nowait(frame)
                
                # Keep track of how many frames played
                self.n_frames = self.n_frames + 1
                
            qsize = jackclient.QUEUE.qsize()
        print("after loading: {}".format(jackclient.QUEUE.qsize()))

        
        ## Loade queue2 with error sound, if needed
        # Only do this after an initial pause
        if self.n_frames > 3000:
            # Only do this on every third call to this function
            if np.mod(self.n_error_counter, 3) == 0:
                print("before loading 2: {}".format(jackclient.QUEUE2.qsize()))            
                with jackclient.Q2_LOCK:
                    # Add frames from error sound
                    for frame in self.left_error_sound.chunks:
                        jackclient.QUEUE2.put_nowait(frame)
                qsize = jackclient.QUEUE2.qsize()
                print("after loading 2: {}".format(jackclient.QUEUE2.qsize()))        
            
            # Keep track of how many calls to this function
            self.n_error_counter = self.n_error_counter + 1

        # Start it playing
        # The play event is cleared if it ever runs out of sound, which
        # ideally doesn't happen
        jackclient.PLAY.set()
        
        # Sleep so we don't go crazy
        time.sleep(1)
        
        # Continue to the next stage (which is this one again)
        self.stage_block.set()

    def init_hardware(self, *args, **kwargs):
        """Placeholder to init hardware
        
        This is here to remind me that init_hardware is implemented by the
        base class `Task`. This function could be removed if there is
        no hardware actually connected and/or defined in HARDWARE.
        """
        super().init_hardware(*args, **kwargs)

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

