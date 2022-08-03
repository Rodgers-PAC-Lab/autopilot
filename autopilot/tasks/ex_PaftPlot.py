"""This module defines an example PaftPlot task that plots data.

The Pilot generates spurious pokes and trial structure, reporting the pokes
as continuous data and the trials as trial data.

The Plot plots them.
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
from autopilot.utils.loggers import init_logger
from autopilot.stim.sound import jackclient

# The name of the task
# This declaration allows Subject to identify which class in this file 
# contains the task class. 
TASK = 'ex_PaftPlot'


## Define the Task
class ex_PaftPlot(Task):
    """The probabalistic auditory foraging task (PAFT).

    This passes through three stages and returns random data for each.
    Continuous data is returned for pokes.
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

    # Definie continuous data
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
    
    ## Define the class methods
    def __init__(self, stage_block, current_trial, step_name, task_type, 
        subject, step, session, pilot, graduation, reward):
        """Initialize a new ex_PaftPlot Task. 
        
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


        ## This is used to report fake pokes
        self.known_pilot_ports = [
            'rpi09_L',
            'rpi09_R',
            'rpi10_L',
            'rpi10_R',            
            'rpi11_L',
            'rpi11_R',
            'rpi12_L',
            'rpi12_R',            
            ]
        self.poked_port_cycle = itertools.cycle(self.known_pilot_ports)

        
        ## For reporting data to the Terminal and plots
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
        rewarded_port = random.choice(self.known_pilot_ports)
        self.logger.debug('choose_stimulus: chose {}'.format(rewarded_port))
        
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
            'rewarded_port': rewarded_port,
            'timestamp_trial_start': timestamp_trial_start.isoformat(),
            'trial_num': next(self.counter_trials_across_sessions),
            'trial_in_session': next(self.counter_trials_in_session),
            }

    def wait_for_response(self):
        """A stage that waits for a response"""
        # Wait a little before doing anything
        self.logger.debug('wait_for_response: entering stage')
        time.sleep(3)
        
        # Directly report continuous data to terminal (aka _T)
        # Otherwise it can be encoded in the returned data, but that is only
        # once per stage
        # subject is needed by core.terminal.Terminal.l_data
        # pilot is needed by networking.station.Terminal_Station.l_data
        # timestamp and continuous are needed by subject.Subject.data_thread
        timestamp_response = datetime.datetime.now()
        poked_port = next(self.poked_port_cycle)
        self.node.send(
            to='_T',
            key='DATA',
            value={
                'subject': self.subject,
                'pilot': prefs.get('NAME'),
                'continuous': True,
                'poked_port': poked_port,
                'timestamp': timestamp_response.isoformat(),
                },
            )
        self.node.send(
            to='P_rpi09',
            key='DATA',
            value={
                'subject': self.subject,
                'pilot': prefs.get('NAME'),
                'continuous': True,
                'poked_port': poked_port,
                'timestamp': timestamp_response.isoformat(),
                },
            )            

        # Continue to the next stage
        self.stage_block.set()        
        
        # Return data about chosen_stim so it will be added to HDF5
        # Could also return continuous data here
        return {
            'timestamp_reward': timestamp_response.isoformat(),
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
        super(ex_PaftPlot, self).init_hardware(*args, **kwargs)

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
        
        # This sock.close seems to be necessary to be able to communicate again
        self.node.sock.close()
        self.node.release()
        
        super(ex_PaftPlot, self).end(*args, **kwargs)

