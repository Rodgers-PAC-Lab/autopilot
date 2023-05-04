"""This module defines PAFT_audiotest.

This is used to play sounds from the pi, for testing the hifiberry.
No children are used.
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
from pydantic import Field
from autopilot.data.models.protocol import Trial_Data
from autopilot.stim.sound import sounds
from autopilot.tasks.task import Task
from autopilot.networking import Net_Node
from autopilot import prefs
from autopilot.hardware import BCM_TO_BOARD
from autopilot.utils.loggers import init_logger
from autopilot.stim.sound import jackclient

# The name of the task
# This declaration allows Subject to identify which class in this file 
# contains the task class, and its human-readable task name.
TASK = 'PAFT_audiotest'


## Define the Task
class PAFT_audiotest(Task):
    """The probabalistic auditory foraging task (PAFT).
    
    To understand the stage progression logic, see:
    * autopilot.agents.pilot.Pilot.run_task - the main loop
    * autopilot.tasks.task.Task.handle_trigger - set stage trigger
    
    To understand the data saving logic, see:
    * autopilot.agents.terminal.Terminal.l_data - what happens when data is sent
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


    
    ## Set up TrialData and Continuous Data schema
    # Per https://docs.auto-pi-lot.com/en/latest/guide/task.html:
    # The `TrialData` object is used by the `Subject` class when a task
    # is assigned to create the data storage table
    # 'trial_num' and 'session_num' get added by the `Subject` class
    # 'session_num' is properly set by `Subject`, but 'trial_num' needs
    # to be set properly here.
    # If they are left unspecified on any given trial, they receive 
    # a default value, such as 0 for Int32Col.
    #
    # An updated version using pydantic
    class TrialData(Trial_Data):
        # The trial within this session
        # Unambigously label this
        trial_in_session: int = Field(
            description='The 0-based trial number within the session')
        
        # If this isn't specified here, it will be added anyway
        trial_num: int = Field(
            description='The trial number aggregating over sessions')

    # This is irrelevant for this task but kept because required
    # Define continuous data
    # https://docs.auto-pi-lot.com/en/latest/guide/task.html
    # autopilot.core.subject.Subject.data_thread would like one of the
    # keys to be "timestamp"
    # Actually, no I think that is extracted automatically from the 
    # networked message, and should not be defined here
    class ContinuousData(tables.IsDescription):
        reward_timestamp = tables.StringCol(64)
        trial = tables.Int32Col()

    # This is irrelevant for this task but kept because required
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
    
    # This is irrelevant for this task but kept because required
    class ChunkData_Pokes(tables.IsDescription):
        timestamp = tables.StringCol(64)
        poked_port = tables.StringCol(64)
        trial = tables.Int32Col()
        first_poke = tables.Int32Col()
        reward_delivered = tables.Int32Col()
        poke_rank = tables.Int32Col()
    
    # This defines the classes that act like ChunkData
    # See Subject.data_thread
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
        
    
        ## Define the stages
        # Stage list to iterate
        stage_list = [self.play]
        self.stages = itertools.cycle(stage_list)        
        
        
        ## Init hardware -- this sets self.hardware, self.pin_id, and
        ## assigns self.handle_trigger to gpio callbacks
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

        # Initialize sounds
        self.initalize_sounds()

        # Fill the queue with empty frames
        self.set_sound_cycle()

    def initalize_sounds(self):
        """Defines sounds that will be played during the task"""
        # Left and right target noise bursts
        self.noise_bursts = [
            # This is the unattenuated version
            #~ autopilot.stim.sound.sounds.Noise(
                #~ duration=2000, amplitude=.01, channel=0, 
                #~ lowpass=None, highpass=None, 
                #~ ),
        
            autopilot.stim.sound.sounds.Noise(
                duration=2000, amplitude=.01, channel=0, 
                lowpass=None, highpass=None, 
                attenuation_file='/home/pi/attenuation.csv',
                ),                            
            ] 
    
    def set_sound_cycle(self):
        """Define self.sound_cycle, to go through sounds

        """
        ## Generate self.sound_block
        # This is where sounds go
        self.sound_block = []

        # Helper function
        def append_gap(gap_chunk_size=30):
            """Append `gap_chunk_size` silent chunks to sound_block"""
            for n_blank_chunks in range(gap_chunk_size):
                self.sound_block.append(
                    np.zeros(autopilot.stim.sound.jackclient.BLOCKSIZE, 
                    dtype='float32'))

        # Append long noise burst
        for noise_burst in self.noise_bursts:
            for frame in noise_burst.chunks:
                self.sound_block.append(frame) 

            # Append gap
            append_gap(50)        

        
        ## Cycle so it can repeat forever
        self.sound_cycle = itertools.cycle(self.sound_block)        

    def play(self):
        """A single stage"""
        # Don't want to do a "while True" here, because we need to exit
        # this method eventually, so that it can respond to END
        # But also don't want to change stage too frequently or the debug
        # messages are overwhelming
        for n in range(10):
            # Add stimulus sounds to queue 1 as needed
            self.append_sound_to_queue1_as_needed()

            # Don't want to iterate too quickly, but rather add chunks
            # in a controlled fashion every so often
            time.sleep(.1)

        # Continue to the next stage (which is this one again)
        # If it is cleared, then nothing happens until the next message
        # from the Parent (not sure why)
        # If we never end this function, then it won't respond to END
        self.stage_block.set()

    def append_sound_to_queue1_as_needed(self):
        """Dump frames from `self.sound_cycle` into queue

        The queue is filled until it reaches `self.target_qsize`

        This function should be called often enough that the queue is never
        empty.
        """        
        # TODO: as a figure of merit, keep track of how empty the queue gets
        # between calls. If it's getting too close to zero, then target_qsize
        # needs to be increased.
        # Get the size of QUEUE1 now
        qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
        if qsize == 0:
            self.logger.debug('warning: queue1 was empty')

        # Add frames until target size reached
        while qsize < self.target_qsize:
            with autopilot.stim.sound.jackclient.Q_LOCK:
                # Add a frame from the sound cycle
                frame = next(self.sound_cycle)
                autopilot.stim.sound.jackclient.QUEUE.put_nowait(frame)
                
                # Keep track of how many frames played
                self.n_frames = self.n_frames + 1
            
            # Update qsize
            qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()

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
        
        # Let the superclass end handle releasing hardware
        super().end(*args, **kwargs)

