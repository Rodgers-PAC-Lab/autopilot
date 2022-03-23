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


## Set box-specific params
# TODO: Figure out some cleaner way of doing this
# But right now vars like MY_PI2 are needed just to initiate a PAFT object

# Figure out which box we're in
MY_NAME = prefs.get('NAME')

if MY_NAME in ['rpi01', 'rpi02', 'rpi03', 'rpi04']:
    MY_BOX = 'Box1'
    MY_PARENTS_NAME = 'rpi01'
    MY_PI1 = 'rpi01'
    MY_PI2 = 'rpi02'
    MY_PI3 = 'rpi03'
    MY_PI4 = 'rpi04'

elif MY_NAME in ['rpi05', 'rpi06', 'rpi07', 'rpi08']:
    MY_BOX = 'Box2'
    MY_PARENTS_NAME = 'rpi05'
    MY_PI1 = 'rpi05'
    MY_PI2 = 'rpi06'
    MY_PI3 = 'rpi07'
    MY_PI4 = 'rpi08'

elif MY_NAME in ['rpi09', 'rpi10', 'rpi11', 'rpi12']:
    MY_BOX = 'Box2'
    MY_PARENTS_NAME = 'rpi05'
    MY_PI1 = 'rpi05'
    MY_PI2 = 'rpi06'
    MY_PI3 = 'rpi07'
    MY_PI4 = 'rpi08'

else:
    # This happens on the Terminal, for instance
    MY_BOX = 'NoBox'
    MY_PARENTS_NAME = 'NoParent'
    MY_PI1 = 'NoPi1'
    MY_PI2 = 'NoPi2'
    MY_PI3 = 'NoPi3'
    MY_PI4 = 'NoPi4'


## Define the Task
class PAFT(Task):
    """The probabalistic auditory foraging task (PAFT).
    
    This task chooses a port at random, lights up the LED for that port,
    plays sounds through the associated speaker, and then dispenses water 
    once the subject pokes there.

    Stage list:
    * waiting for the response
    * reporting the response

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
        target ('L', 'C', 'R'): The correct port
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
        'tag':'Reward Duration (ms)',
        'type':'int',
        }

    # This defines the data we return after each trial
    DATA = {
        'trial': {'type':'i32'},
        'trials_total': {'type': 'i32'},
        'rpi': {'type': 'S10'},
        'side': {'type': 'S10'},
        'light': {'type': 'S10'},
        'sound': {'type': 'S10'},
        'timestamp': {'type':'S26'},
    }

    # This is used by the terminal to build an HDF5 file of data for each trial 
    class TrialData(tables.IsDescription):
        # The trial within this session
        trial = tables.Int32Col()
        
        # The trial number accumulated over all sessions
        trials_total = tables.Int32Col()
        
        # The target
        rpi = tables.StringCol(10)
        side = tables.StringCol(10)
        light = tables.StringCol(10)
        sound = tables.StringCol(10)
        
        # The timestamp
        timestamp = tables.StringCol(26)

    # This defines the hardware that is required
    HARDWARE = {
        'POKES':{
            'L': autopilot.hardware.gpio.Digital_In,
            'R': autopilot.hardware.gpio.Digital_In
        },
        'LEDS':{
            'L': autopilot.hardware.gpio.LED_RGB,
            'R': autopilot.hardware.gpio.LED_RGB
        },
        'PORTS':{
            'L': autopilot.hardware.gpio.Solenoid,
            'R': autopilot.hardware.gpio.Solenoid
        }
    }
    
    # This defines the child rpis to connect to
    CHILDREN = {
        MY_PI2: {
            'task_type': "PAFT_Child",
        },
        MY_PI3: {
            'task_type': "PAFT_Child",
        },
        MY_PI4: {
            'task_type': "PAFT_Child",
        },
    }
    
    # This is used by the terminal to plot the results of each trial
    PLOT = {
        'data': {
            'target': 'point'
        }
    }
    
    
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
        
        The fourth version needs to:
        * Incorporate subject-specific training params
        
        
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

        # Use the provided threading.Event to handle stage progression
        self.stage_block = stage_block
        
        # This is used to count the trials, it is initialized by
        # the Terminal to wherever we are in the Protocol graduation
        self.trial_counter = itertools.count(int(current_trial))        

        # A dict of hardware triggers
        self.triggers = {}
        
    
        ## Define the stages
        # Stage list to iterate
        stage_list = [self.play]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)        
        
        
        ## Init hardware -- this sets self.hardware and self.pin_id
        self.init_hardware()
        
        
        ## Define sounds
        self.left_stim = sounds.Noise(
            duration=10, amplitude=.01, channel=0, 
            highpass=5000)       

        self.right_stim = sounds.Noise(
            duration=10, amplitude=.01, channel=1, 
            highpass=5000)        
        
        self.left_error_sound = sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=0)

        self.right_error_sound = sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=1)
        
        # Chunk the sounds into frames
        if not self.left_stim.chunks:
            self.left_stim.chunk()
        if not self.right_stim.chunks:
            self.right_stim.chunk()
        if not self.left_error_sound.chunks:
            self.left_error_sound.chunk()
        if not self.right_error_sound.chunks:
            self.right_error_sound.chunk()
        
        
        ## Generate the stimulus sound block
        # This is just a left sound, gap, then right sound, then gap
        # And use a cycle to repeat forever
        # But this could be made more complex
        self.sound_block = []
        for frame in self.left_stim.chunks:
            self.sound_block.append(frame)
        for n_blank_chunks in range(30):
            self.sound_block.append(np.zeros(jackclient.BLOCKSIZE, dtype='float32'))
        for frame in self.right_stim.chunks:
            self.sound_block.append(frame)
        for n_blank_chunks in range(30):
            self.sound_block.append(np.zeros(jackclient.BLOCKSIZE, dtype='float32'))
        
        # Cycle so it can repeat forever
        self.sound_cycle = itertools.cycle(self.sound_block)

        # Some counters to keep track of how many sounds we've played
        self.n_frames = 0
        self.n_error_counter = 0

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

    def init_hardware(self):
        """
        Use the HARDWARE dict that specifies what we need to run the task
        alongside the HARDWARE subdict in :mod:`prefs` to tell us how
        they're plugged in to the pi

        Instantiate the hardware, assign it :meth:`.Task.handle_trigger`
        as a callback if it is a trigger.
        
        Sets the following:
            self.hardware
            self.pin_id
        """
        # We use the HARDWARE dict that specifies what we need to run the task
        # alongside the HARDWARE subdict in the prefs structure to tell us 
        # how they're plugged in to the pi
        self.hardware = {}
