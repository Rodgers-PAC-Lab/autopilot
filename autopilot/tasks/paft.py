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
    MY_PARENTS_NAME = 'rpi09'
    MY_PI1 = 'rpi09'
    MY_PI2 = 'rpi10'
    MY_PI3 = 'rpi11'
    MY_PI4 = 'rpi12'

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
        #~ MY_PI3: {
            #~ 'task_type': "PAFT_Child",
        #~ },
        #~ MY_PI4: {
            #~ 'task_type': "PAFT_Child",
        #~ },
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


        ## Initialize net node for communications with child
        # This dict keeps track of which self.CHILDREN have connected
        self.child_connected = {}
        for child in self.CHILDREN.keys():
            self.child_connected[child] = False
        
        # With instance=True, I get a threading error about current event loop
        self.node = Net_Node(id="T_{}".format(prefs.get('NAME')),
            upstream=prefs.get('NAME'),
            port=prefs.get('MSGPORT'),
            listens={},
            #~ instance=False,
            )

        # Construct a message to send to child
        # Specify the subjects for the child (twice)
        self.subject = subject
        value = {
            'child': {
                'parent': prefs.get('NAME'), 'subject': subject},
            'task_type': 'PAFT_Child',
            'subject': subject,
            'reward': reward,
        }

        # send to the station object with a 'CHILD' key
        self.node.send(to=prefs.get('NAME'), key='CHILD', value=value)

        
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
            #~ instance=False,
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
            "All children have connected: {}".format(self.child_connected))

    def play(self):
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

    def recv_hello(self, value):
        self.logger.debug(
            "received HELLO from child with value {}".format(value))
        
        # Set this flag
        self.child_connected[value['from']] = True
    
    def recv_poke(self, value):
        self.logger.debug(
            "received POKE from child with value {}".format(value))

    def end(self):
        """
        When shutting down, release all hardware objects and turn LEDs off.
        """
        self.logger.debug('inside self.end')
        
        # Tell each child to END
        for child_name in self.CHILDREN.keys():
            # Tell child what the target is
            self.node2.send(
                to=child_name,
                key='END',
                value={},
                )    

        self.node.release()
        self.node2.release()
