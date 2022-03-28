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
    # I think that the `DATA` parameter is no longer necessary here
    class TrialData(tables.IsDescription):
        # The trial within this session
        trial = tables.Int32Col()
        
        # The rewarded port
        # Must specify the max length of the string, we use 64 to be safe
        rewarded_port = tables.StringCol(64)
        
        # The timestamp
        trial_start_timestamp = tables.StringCol(64)

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
        stage_list = [self.play, self.wait]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)        
        
        
        ## Init hardware -- this sets self.hardware and self.pin_id
        self.init_hardware()
    
    def play(self):
        # Wait a little before doing anything
        self.logger.debug('entering play state')
        time.sleep(3)
        
        # Continue to the next stage
        # CLEAR means "wait for triggers"
        # SET means "advance anyway"
        self.stage_block.set()

    def wait(self):
        # Wait a little before doing anything
        self.logger.debug('entering wait state')
        time.sleep(3)
        
        # Continue to the next stage
        self.stage_block.set()        

    def init_hardware(self, *args, **kwargs):
        """Placeholder to init hardware
        
        This is here to remind me that init_hardware is implemented by the
        base class `Task`. This function could be removed if there is
        no hardware actually connected and/or defined in HARDWARE.
        """
        super(PAFT, self).init_hardware(*args, **kwargs)

    def end(self):
        """Called when the task is ended by the user.
        
        The base class `Task` (TODO start here)
        """
        self.logger.debug('inside self.end')
        
        super(PAFT, self).init_hardware(*args, **kwargs)
