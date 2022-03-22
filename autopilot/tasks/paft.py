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

# The name of the task
# This declaration allows Subject to identify which class in this file 
# contains the task class. 
TASK = 'PAFT'

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
        subject, step, session, pilot, reward):
        """Initialize a new PAFT Task. 
        
        All arguments are provided by the Terminal.
        
        Some of these arguments are handled and/or required by the
        superclass named `Task`. (TODO: Doc which ones.)
        
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
        reward (int): 
            ms to open solenoids
            This is passed from the "protocol" json
        """    