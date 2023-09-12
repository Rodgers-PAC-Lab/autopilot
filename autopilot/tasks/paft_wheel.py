"""This module defines the PAFT_Wheel task

Take input from a rotary encoder
"""
import time
import threading
import itertools
import random
import datetime
import functools
from collections import OrderedDict as odict
import pigpio
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

# The name of the task
# This declaration allows Subject to identify which class in this file 
# contains the task class. 
TASK = 'PAFT_Wheel'


## Set box-specific params
# TODO: Figure out some cleaner way of doing this
# But right now vars like MY_PI2 are needed just to initiate a PAFT object


## Define the Task
class PAFT_Wheel(Task):

    ## Define the class attributes
    # This defines params we receive from terminal on session init
    PARAMS = odict()
    PARAMS['reward'] = {
        'tag':'Reward Duration (ms)',
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
        pass
    
    class ContinuousData(tables.IsDescription):
        pass

    # This defines the classes that act like ChunkData
    # See Subject.data_thread
    CHUNKDATA_CLASSES = []
    
    ## Define the class methods
    def __init__(self, stage_block, current_trial, step_name, task_type, 
        subject, step, session, pilot, graduation, reward):
        """Initialize a new MinimalPAFT Task. 
        
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
        stage_list = [self.do_nothing]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)        
        
        
        ## Init hardware -- this sets self.hardware and self.pin_id
        self.init_hardware()
        
        
        ## Keep track of pigpio.pi
        self.pi = pigpio.pi()
        
        
        ## Add a callback
        # Use BCM pin number
        self.position = 0
        self.event_log = []
        self.state_log = []
        self.a_state = 0
        self.b_state = 0
        self.pi.callback(16, pigpio.RISING_EDGE, self.pulseA_detected)
        self.pi.callback(12, pigpio.RISING_EDGE, self.pulseB_detected)
        self.pi.callback(16, pigpio.FALLING_EDGE, self.pulseA_down)
        self.pi.callback(12, pigpio.FALLING_EDGE, self.pulseB_down)
        
        """
        States
        00 10
        01 11
        When A goes up, move right. When A goes down, move left.
        When B goes up, move down. When B goes down, move up.
        When the state moves clockwise, increment position.
        When the state moves counter-clockwise, decrement position.
        
        A cute trick might be to take the state's value in binary, subtract
        0.6, and take absolute value. If this result is increasing, increment
        position, otherwise decrement position. That's probably not any faster
        though.
        """

    def pulseA_detected(self, pin, level, tick):
        self.event_log.append('A')
        self.a_state = 1
        if self.b_state == 0:
            self.position += 1
        else:
            self.position -= 1
        self.state_log.append(
            '{}{}_{}'.format(self.a_state, self.b_state, self.position))

    def pulseB_detected(self, pin, level, tick):
        self.event_log.append('B')
        self.b_state = 1
        if self.a_state == 0:
            self.position -= 1
        else:
            self.position += 1
        self.state_log.append(
            '{}{}_{}'.format(self.a_state, self.b_state, self.position))
    
    def pulseA_down(self, pin, level, tick):
        self.event_log.append('a')
        self.a_state = 0
        if self.b_state == 0:
            self.position -= 1
        else:
            self.position += 1
        self.state_log.append(
            '{}{}_{}'.format(self.a_state, self.b_state, self.position))
    
    def pulseB_down(self, pin, level, tick):
        self.event_log.append('b')
        self.b_state = 0
        if self.a_state == 0:
            self.position += 1
        else:
            self.position -= 1
        self.state_log.append(
            '{}{}_{}'.format(self.a_state, self.b_state, self.position))
    
    def do_nothing(self):
        print("current position: {}".format(self.position))
        print(''.join(self.event_log[-60:]))
        print('\t'.join(self.state_log[-4:]))
        time.sleep(1)
        
        self.stage_block.set()

    def init_hardware(self):
        self.hardware = {}
