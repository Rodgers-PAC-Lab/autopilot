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
        
        # The chosens stimulus and response
        # Must specify the max length of the string, we use 64 to be safe
        chosen_stimulus = tables.StringCol(64)
        chosen_response = tables.StringCol(64)
        
        # The timestamps
        timestamp_trial_start = tables.StringCol(64)
        timestamp_response = tables.StringCol(64)

    # Definie continuous data
    # https://docs.auto-pi-lot.com/en/latest/guide/task.html
    # autopilot.core.subject.Subject.data_thread would like one of the
    # keys to be "timestamp"
    # Actually, no I think that is extracted automatically from the 
    # networked message, and should not be defined here
    class ContinuousData(tables.IsDescription):
        poked_pilot = tables.StringCol(64)
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
        
        # Announce
        self.logger.debug(
            '__init__: received current_trial {}'.format(current_trial))

    
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
        
        
        ## For reporting'
        # With instance=True, I get a threading error about current event loop
        self.node = Net_Node(
            id="T_{}".format(prefs.get('NAME')),
            upstream=prefs.get('NAME'),
            port=prefs.get('MSGPORT'),
            listens={},
            instance=False,
            )        
    
    def choose_stimulus(self):
        """A stage that chooses the stimulus"""
        # Get timestamp
        timestamp_trial_start = datetime.datetime.now()
        
        # Wait a little before doing anything
        self.logger.debug(
            'choose_stimulus: entering stage at {}'.format(
            timestamp_trial_start.isoformat()))
        time.sleep(3)
        
        # Choose stimulus randomly
        chosen_stimulus = random.choice(['stim0', 'stim1', 'stim2'])
        self.logger.debug('choose_stimulus: chose {}'.format(chosen_stimulus))
        
        # Continue to the next stage
        # CLEAR means "wait for triggers"
        # SET means "advance anyway"
        self.stage_block.set()

        # Return data about chosen_stim so it will be added to HDF5
        # I think it's best to increment trial_num now, since this is the
        # first return from this trial. Even if we don't increment trial_num,
        # it will still make another row in the HDF5, but it might warn.
        # (This hapepns in autopilot.core.subject.Subject.data_thread)
        return {
            'chosen_stimulus': chosen_stimulus,
            'timestamp_trial_start': timestamp_trial_start.isoformat(),
            'trial_num': next(self.counter_trials_across_sessions),
            'trial_in_session': next(self.counter_trials_in_session),
            }

    def wait_for_response(self):
        """A stage that waits for a response"""
        # Wait a little before doing anything
        self.logger.debug('wait_for_response: entering stage')
        time.sleep(3)
        
        # Choose response randomly
        chosen_response = random.choice(['choice0', 'choice1'])
    
        # Get timestamp of response
        timestamp_response = datetime.datetime.now()
        self.logger.debug('wait_for_response: chose {} at {}'.format(
            chosen_response, timestamp_response.isoformat()))

        # subject and pilot are needed to avoid exceptions
        # timestamp is needed to add the atom
        self.node.send(
            to='_T',
            key='DATA',
            value={
                'subject': self.subject,
                'pilot': prefs.get('NAME'),
                'continuous': True,
                'poked_port': 'L',
                'poked_pilot': 'rpi03',
                'timestamp': datetime.datetime.now().isoformat(),
                },
            )

        # Continue to the next stage
        self.stage_block.set()        
        
        # Return data about chosen_stim so it will be added to HDF5
        return {
            #~ 'continuous': True,
            #~ 'timestamp': timestamp_response.isoformat(),
            #~ 'poked_port': 'L',
            #~ 'poked_pilot': 'rpi03',
            'chosen_response': chosen_response,
            'timestamp_response': timestamp_response.isoformat(),
            #~ 'new_data': timestamp_response.isoformat()[:4] + 'asdf',
            }        
    
    def end_of_trial(self):
        """A stage that ends the trial"""
        # Wait a little before doing anything
        self.logger.debug('end_of_trial: entering stage')
        time.sleep(3)
        
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

    def end(self, *args, **kwargs):
        """Called when the task is ended by the user.
        
        The base class `Task` releases hardware objects here.
        This is a placeholder to remind me of that.
        """
        self.logger.debug('end: entering function')
        super(PAFT, self).end(*args, **kwargs)
