"""This module defines PAFT_startle.

This just plays a loud noise burst every couple of minutes
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
TASK = 'PAFT_startle'


## Define the Task
class PAFT_startle(Task):
    """The startle test"""
    
    
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

    # For SoundsPlayed
    class ChunkData_SoundsPlayed(tables.IsDescription):
        hash = tables.IntCol()
        last_frame_time = tables.IntCol()
        frames_since_cycle_start = tables.IntCol()
        equiv_dt = tables.StringCol(64)
        pilot = tables.StringCol(20)
    
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
        'LEDS':{
            'L': autopilot.hardware.gpio.LED_RGB,
            'R': autopilot.hardware.gpio.LED_RGB
        },
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
        # Store my name
        # This is used for reporting sounds to the terminal
        self.name = prefs.get('NAME')
        
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
        self.set_sound_cycle(with_sound=False)
        
        # Keep track of when the last noise burst was played
        self.time_of_last_sound = None

    def initalize_sounds(self):
        """Defines sounds that will be played during the task"""
        # One loud-ish noise burst
        self.noise_bursts = [
            autopilot.stim.sound.sounds.Noise(
                duration=100, amplitude=.03, channel=0, 
                lowpass=None, highpass=None, 
                attenuation_file='/home/pi/attenuation.csv',
                ),                             
            ] 
    
    def set_sound_cycle(self, with_sound=False):
        """Define self.sound_cycle, to go through sounds

        with_sound : bool
            If True, it plays a noise burst, and then at least 5 s of silence
            If False, it plays silence only
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
                if with_sound:
                    self.sound_block.append(frame) 
                else:
                    self.sound_block.append(frame * 0)

            # Append gap -- this needs to be long enough that we don't
            # hear the sound again
            append_gap(1000)        


        ## Cycle so it can repeat forever
        self.sound_cycle = itertools.cycle(self.sound_block)        

    def play(self):
        """A single stage"""
        
        # Don't want to do a "while True" here, because we need to exit
        # this method eventually, so that it can respond to END
        # But also don't want to change stage too frequently or the debug
        # messages are overwhelming
        for n in range(10):
            # Get current time
            current_time = datetime.datetime.now()
            
            # Set the sound cycle as needed
            if (
                    self.time_of_last_sound is None or 
                    current_time >= self.time_of_last_sound + datetime.timedelta(seconds=30)
                    ):
                # If it's been long enough, play a noise burst
                # Add the noise burst
                self.set_sound_cycle(with_sound=True)
                self.empty_queue1()
                self.append_sound_to_queue1_as_needed()
                
                # Flag that it hasn't been silenced
                self.sound_has_been_silenced = False
                
                # Set time to now
                self.time_of_last_sound = current_time
                
                # Turn LED on
                self.hardware['LEDS']['L'].set((255, 0, 0))
                self.hardware['LEDS']['R'].set((255, 0, 0))

            elif current_time >= self.time_of_last_sound + datetime.timedelta(seconds=1):
                # if it's been long enough for the burst to finish, silence it
                if not self.sound_has_been_silenced:
                    # Silence it
                    self.set_sound_cycle(with_sound=False)
                    
                    # Flag that it has been silenced
                    self.sound_has_been_silenced = True
                    
                    # Turn LED off
                    self.hardware['LEDS']['L'].set((0, 0, 0))
                    self.hardware['LEDS']['R'].set((0, 0, 0))
            
            # Add stimulus sounds to queue 1 as needed
            self.append_sound_to_queue1_as_needed()
            
            # Extract any recently played sound info
            sound_data_l = []
            with autopilot.stim.sound.jackclient.QUEUE_NONZERO_BLOCKS_LOCK:
                try:
                    data = autopilot.stim.sound.jackclient.QUEUE_NONZERO_BLOCKS.get_nowait()
                except queue.Empty:
                    break
                sound_data_l.append(data)
            
            if len(sound_data_l) > 0:
                # DataFrame it
                # This has to match code in jackclient.py
                # And it also has to match task_class.ChunkData_SoundsPlayed
                payload = pandas.DataFrame.from_records(
                    sound_data_l,
                    columns=['hash', 'last_frame_time', 'frames_since_cycle_start', 'equiv_dt'],
                    )
                self.send_chunk_of_sound_played_data(payload)

            # Don't want to iterate too quickly, but rather add chunks
            # in a controlled fashion every so often
            time.sleep(.1)

        # Continue to the next stage (which is this one again)
        # If it is cleared, then nothing happens until the next message
        # from the Parent (not sure why)
        # If we never end this function, then it won't respond to END
        self.stage_block.set()

    def send_chunk_of_sound_played_data(self, payload):
        """Report metadata about sounds played directly to terminal
        
        This is adapted from send_chunk_of_sound_data in paft_child
        Here we send data from jackclient about the nonzero frames
        TODO: add this to paft_child, alongside send_chunk_of_sound_data
        """
        ## Create a serialized message
        # Adapted from the bandwidth test
        
        # Time of sending this message
        timestamp = datetime.datetime.now().isoformat()
        
        # Only send data if there are rows of data
        # Otherwise, no sound playing, and nothing to report
        if len(payload) > 0:
            # Store these additional values, which are the same for all rows
            payload['pilot'] = self.name
            
            # This is the value to send
            # Must be serializable
            # Definitely include payload (the data), some kind of locking
            # timestamp, and the origin (our name). 
            # When this message is repeated by the Parent to the terminal,
            # there are additional constraints based on what save_data expects
            value = {
                'pilot': self.name,
                'payload': payload.values,
                'payload_columns': payload.columns.values,
                'chunkclass_name': 'ChunkData_SoundsPlayed', 
                'timestamp': timestamp,
            }        
            
            # Construct the message
            msg = autopilot.networking.Message(
                id="{}-{}".format(self.name, self.n_messages_sent), # must be unique
                sender="dummy_src", # required but I don't think it matters
                key='CHUNK', # this selects listen method. required for encoding
                to="dummy_dst", # required but I don't think it matters
                value=value, # the 'value' to send
                flags={
                    'MINPRINT': True, # disable printing of value
                    'NOREPEAT': True, # disable repeating
                    },
                )
            
            # Sending it will automatically serialize it, which in turn will
            # automatically compress numpy using blosc
            # See Node.send and Message.serialize
            self.node2.send('parent_pi', msg=msg)

            # Increment this counter to keep the message id unique
            self.n_messages_sent = self.n_messages_sent + 1     

    def empty_queue1(self, tosize=0):
        """Empty queue1"""
        while True:
            # I think it's important to keep the lock for a short period
            # (ie not throughout the emptying)
            # in case the `process` function needs it to play sounds
            # (though if this does happen, there will be an artefact because
            # we just skipped over a bunch of frames)
            with autopilot.stim.sound.jackclient.Q_LOCK:
                try:
                    data = autopilot.stim.sound.jackclient.QUEUE.get_nowait()
                except queue.Empty:
                    break
            
            # Stop if we're at or below the target size
            qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
            if qsize < tosize:
                break
        
        qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
        self.logger.debug('empty_queue1: new size {}'.format(qsize))

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

