"""This module defines the ex_serialize Task.

This is an example showing how to send serialized data from the Child
to the Parent to the Terminal.
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
TASK = 'ex_serialize'


## Define the Task
class ex_serialize(Task):
    
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
        

    # Define continuous data
    # https://docs.auto-pi-lot.com/en/latest/guide/task.html
    # autopilot.core.subject.Subject.data_thread would like one of the
    # keys to be "timestamp"
    # Actually, no I think that is extracted automatically from the 
    # networked message, and should not be defined here
    class ContinuousData(tables.IsDescription):
        poked_port = tables.StringCol(64)
        trial = tables.Int32Col()


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
    
    # This defines the child rpis to connect to
    children_names = prefs.get('CHILDID')
    if children_names is None:
        # This happens on terminal
        children_names = []
    CHILDREN = {}
    for child in children_names:
        CHILDREN[child] = {'task_type': "PAFT_Child"}

    
    ## Define the class methods
    def __init__(self, stage_block, current_trial, step_name, task_type, 
        subject, step, session, pilot, graduation, reward, **task_params):
        """Initialize a new ex_serialize Task. 
        
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
            This is set to be 1 greater than the last value of "trial_num"
            in the HDF5 file by autopilot.core.subject.Subject.prepare_run
            Or sometimes this is just zero, for some reason
        step_name : string
            This is passed from the "protocol" json
        task_type : string
            This is passed from the "protocol" json
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
        self.counter_trials_in_session = 0

        # A dict of hardware triggers
        self.triggers = {}

        
        ## Define the possible ports
        self.known_pilot_ports = []
        for child in prefs.get('CHILDID'):
            self.known_pilot_ports.append('{}_{}'.format(child, 'L'))
            self.known_pilot_ports.append('{}_{}'.format(child, 'R'))
        
    
        ## Define the stages
        # Stage list to iterate
        stage_list = [self.play]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)        
        
        
        ## Init hardware -- this sets self.hardware, self.pin_id, and
        ## assigns self.handle_trigger to gpio callbacks
        self.init_hardware()


        ## Connect to children
        # This dict keeps track of which self.CHILDREN have connected
        self.child_connected = {}
        for child in self.CHILDREN.keys():
            self.child_connected[child] = False
        
        # Tell each child to start the task
        self.initiate_task_on_children(subject, reward)
        
        # Create a Net_Node for communicating with the children, and
        # wait until all children have connected
        self.create_inter_pi_communication_node()

    def initiate_task_on_children(self, subject, reward):
        """Defines a Net_Node and uses it to tell each child to start
        
        This Net_Node is saved as `self.node`. A 'CHILD' message is sent,
        I think to the Pilot_Node, which is handled by
        networking.station.Pilot_Node.l_child .
        
        That code broadcasts the 'START' message to each of the children
        specified in CHILDID in prefs.json, telling them to start the
        'PAFT_Child' task. That 'START' message also includes task 
        parameters specified here, such as subject name and reward value.
        
        The same `self.node` is used later to end the session on the children.
        """
        # With instance=True, I get a threading error about current event loop
        self.node = Net_Node(
            id="T_{}".format(prefs.get('NAME')),
            upstream=prefs.get('NAME'),
            port=prefs.get('MSGPORT'),
            listens={},
            instance=False,
            )

        # Construct a message to send to child
        # Specify the subjects for the child (twice)
        # These extra params end up in the __init__ for the child class
        value = {
            'child': {
                'parent': prefs.get('NAME'), 'subject': subject},
            'task_type': 'ex_serialize_child',
            'subject': subject,
            'reward': reward,          
        }

        # send to the station object with a 'CHILD' key
        self.node.send(to=prefs.get('NAME'), key='CHILD', value=value)        

    def create_inter_pi_communication_node(self):
        """Defines a Net_Node to communicate with the children
        
        This is a second Net_Node that is used to directly exchange information
        with the children about pokes and sounds. Unlike the first Net_Node,
        for this one the parent is the "router" / server and the children
        are the "dealer" / clients .. ie many dealers, one router.
        
        Each child needs to create a corresponding Net_Node and connect to
        this one. This function will block until that happens.
        
        The Net_Node defined here also specifies "listens" (ie triggers)
        of functions to be called upon receiving specified messages
        from the children, such as "HELLO" or "POKE".
        
        This Net_Node is saved as `self.node2`.
        """
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
                },
            instance=False,
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
            "All children have connected: {}".format(self.child_connected)
            )        

    def play(self):
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

    def recv_hello(self, value):
        self.logger.debug(
            "received HELLO from child with value {}".format(value))
        
        # Set this flag
        self.child_connected[value['from']] = True
    
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

