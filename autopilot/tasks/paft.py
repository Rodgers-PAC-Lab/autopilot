"""This module defines the PAFT task


Multiple Child rpis running the "PAFT Child" Task connect to this Parent.
The Parent chooses the correct stimulus and logs all events. It tells each
Child if it should start playing sounds and when it should stop. The Child
knows that a poke into the port that is currently playing sound should
be rewarded.

The Parent establishes a router Net_Node on port 5001. A dealer Net_Node
on each Child connects to it.

The Parent responds to the following message keys on port 5001:
* HELLO : This means the Child has booted the task.
    The value is dispatched to PAFT.recv_hello, with the following keys:
    'from' : string; the name of the child (e.g., rpi02)
* POKE : This means the Child has detected a poke.
    The value is dispatched to PAFT.recv_poke, with the following keys:
    'from' : string; the name of the child (e.g., rpi02)
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

class PAFT():
    """The probabalistic auditory foraging task (PAFT).
    
    This task chooses a port at random, lights up the LED for that port,
    plays sounds through the associated speaker, and then dispenses water 
    once the subject pokes there.

    Stage list:
    * waiting for the response
    * reporting the response

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
    ## Params
    PARAMS = odict()
    PARAMS['reward'] = {
        'tag':'Reward Duration (ms)',
        'type':'int',
        }


    ## Returned data
    DATA = {
        'trial_num': {'type':'i32'},
        'target': {'type':'S1', 'plot':'target'},
        'timestamp': {'type':'S26'}, # only one timestamp, since next trial instant
    }

    # TODO: This should be generated from DATA above. 
    # Perhaps parsimoniously by using tables types rather than string descriptors
    class TrialData(tables.IsDescription):
        trial_num = tables.Int32Col()
        target    = tables.StringCol(1)
        timestamp = tables.StringCol(26)


    ## Required hardware
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
    
    
    ## The child rpi that handles the other ports
    CHILDREN = {
        'rpi02': {
            'task_type': "PAFT Child",
        },
        'rpi03': {
            'task_type': "PAFT Child",
        },
        'rpi04': {
            'task_type': "PAFT Child",
        },
    }

    
    ## Plot parameters
    PLOT = {
        'data': {
            'target': 'point'
        }
    }

    
    ## Methods
    def __init__(self, stage_block=None, stim=None, current_trial=0,
        reward=150, step_name=None, task_type=None, subject=None, step=None,
        session=None, pilot=None, child=None):
        """Initialize a new PAFT Task
        
        Arguments
        ---------
        stage_block (:class:`threading.Event`): 
            used to signal to the carrying Pilot that the current trial 
            stage is over
        current_trial (int): 
            If not zero, initial number of `trial_counter`
        reward (int): 
            ms to open solenoids
        **kwargs:
        """
        ## Task management
        # a threading.Event used by the pilot to manage stage transitions
        # Who provides this?
        self.stage_block = stage_block  
        
        # Set up a logger
        self.logger = init_logger(self)

        # This dict keeps track of which self.CHILDREN have connected
        self.child_connected = {}
        for child in self.CHILDREN.keys():
            self.child_connected[child] = False
        
        # This is used for the current trial target
        self.target = None
        
        # This keeps track of the current stim
        self.stim = None
        
        # This is used to count the trials, it is initialized by
        # something to wherever we are in the Protocol graduation
        self.trial_counter = itertools.count(int(current_trial))
        
        # This is a trial counter that always starts at zero
        self.n_trials = 0
        
        # A dict of hardware triggers
        self.triggers = {}

        # Stage list to iterate
        stage_list = [self.water, self.response]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)

        # Init hardware -- this sets self.hardware and self.pin_id
        self.init_hardware()

        # Set reward values for solenoids
        for port_name, port in self.hardware['PORTS'].items():
            self.logger.debug(
                "setting reward for {} to {}".format(port_name, reward))
            port.duration = float(reward)

        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)
        
        
        ## Initialize net node for communications with child
        # With instance=True, I get a threading error about current event loop
        self.node = Net_Node(id="T_{}".format(prefs.get('NAME')),
            upstream=prefs.get('NAME'),
            port=prefs.get('MSGPORT'),
            listens={},
            instance=False)

        # Construct a message to send to child
        # Specify the subjects for the child (twice)
        self.subject = subject
        value = {
            'child': {
                'parent': prefs.get('NAME'), 'subject': subject},
            'task_type': 'PAFT Child',
            'subject': subject,
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
            "All children have connected: {}".format(self.child_connected))

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
        self.pin_id = {} # Reverse dict to identify pokes
        pin_numbers = prefs.get('HARDWARE')

        # We first iterate through the types of hardware we need
        for type, values in self.HARDWARE.items():
            self.hardware[type] = {}
            # then iterate through each pin and handler of this type
            for pin, handler in values.items():
                try:
                    hw_args = pin_numbers[type][pin]
                    if isinstance(hw_args, dict):
                        if 'name' not in hw_args.keys():
                            hw_args['name'] = "{}_{}".format(type, pin)
                        hw = handler(**hw_args)
                    else:
                        hw_name = "{}_{}".format(type, pin)
                        hw = handler(hw_args, name=hw_name)

                    # if a pin is a trigger pin (event-based input), 
                    # give it the trigger handler
                    if hw.is_trigger:
                        hw.assign_cb(self.handle_trigger)

                    # add to forward and backwards pin dicts
                    self.hardware[type][pin] = hw
                    if isinstance(hw_args, int) or isinstance(hw_args, str):
                        self.pin_id[hw_args] = pin
                    elif isinstance(hw_args, list):
                        for p in hw_args:
                            self.pin_id[p] = pin
                    elif isinstance(hw_args, dict):
                        if 'pin' in hw_args.keys():
                            self.pin_id[hw_args['pin']] = pin 

                except:
                    self.logger.exception(
                        "Pin could not be instantiated - Type: "
                        "{}, Pin: {}".format(type, pin))

    def handle_trigger(self, pin, level=None, tick=None):
        """Handle a GPIO trigger.
        
        All GPIO triggers call this function with the pin number, 
        level (high, low), and ticks since booting pigpio.

        Args:
            pin (int): BCM Pin number
            level (bool): True, False high/low
            tick (int): ticks since booting pigpio
        
        This converts the BCM pin number to a board number using
        BCM_TO_BOARD and then a letter using `self.pin_id`.
        
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

        # Call any triggers that exist
        if pin_letter in self.triggers:
            trigger_l = self.triggers[pin_letter]
            for trigger in trigger_l:
                trigger()
        else:
            self.logger.debug(f"No trigger found for {pin}")
            return

    def end(self):
        """
        Release all hardware objects
        """
        for k, v in self.hardware.items():
            for pin, obj in v.items():
                obj.release()

    def recv_poke(self, value):
        # Log it
        self.log_poke_from_child(value)
        
        # Identify target
        if '_' in self.target:
            target_child_name, target_side = self.target.split('_')
        else:
            target_child_name = 'rpi01'
            target_side = self.target
        
        # Identify poked
        child_name = value['from']
        poke_name = value['poke']
        
        # Compare target to poked
        if target_child_name == child_name and target_side == poke_name:
            self.logger.debug('correct poke {}; target was {}'.format(value, self.target))
            self.stage_block.set()
        else:
            self.logger.debug('incorrect poke {}; target was {}'.format(value, self.target))
        
    def log_poke_from_child(self, value):
        child_name = value['from']
        poke_name = value['poke']
        self.log_poke('{}_{}'.format(child_name, poke_name))
    
    def log_poke(self, port):
        self.logger.debug('{} {} poke'.format(
            datetime.datetime.now().isoformat(),
            port,
            ))
    
    def set_poke_triggers(self):
        """"Set triggers for poke entry
        
        For each poke, sets these triggers:
            self.log_poke (write to own debugger)
            self.report_poke (report to parent)
        
        The C-poke doesn't really exist, but this is useful for debugging.
        """
        for poke in ['L', 'C', 'R']:
            self.triggers[poke] = [
                functools.partial(self.log_poke, poke),
                ]        
    
    def water(self, *args, **kwargs):
        """
        First stage of task - open a port if it's poked.

        Returns:
            dict: Data dictionary containing::

                'target': ('L', 'C', 'R') - correct response
                'timestamp': isoformatted timestamp
                'trial_num': number of current trial
        """
        ## Prevents moving to next stage
        self.stage_block.clear()

        
        ## Choose targets
        # Identify possible targets
        all_possible_targets = [
            'L', 'R', 
            'rpi02_L', 'rpi02_R', 
            'rpi03_L', 'rpi03_R',
            'rpi04_L', 'rpi04_R',
            ]
        excluding_previous = [
            t for t in all_possible_targets if t != self.target]
        
        # Choose
        meth = 'RANDOM'
        if meth == 'CYCLE':
            self.target = all_possible_targets[
                np.mod(self.n_trials, len(all_possible_targets))]
            self.n_trials += 1
        elif meth == 'RANDOM':
            self.target = random.choice(excluding_previous)
        else:
            raise ValueError("unknown trial choosing meth: {}".format(meth))

        # Print debug
        self.logger.debug("The chosen target is {}".format(self.target))
        
        
        ## Set poke triggers (for logging)
        self.set_poke_triggers()
        

        ## Set stim and LEDs by target
        if self.target == 'L':
            self.stim = sounds.Noise(
                duration=100, amplitude=.003, channel=0, nsamples=19456)
            
            # Turn on green led
            self.hardware['LEDS']['L'].set(r=0, g=255, b=0)
            self.hardware['LEDS']['R'].set(r=0, g=0, b=0)
            
            # Add a trigger to open the port
            self.triggers['L'].append(self.hardware['PORTS']['L'].open)
            self.triggers['L'].append(self.stage_block.set)
            
        elif self.target == 'R':
            self.stim = sounds.Noise(
                duration=100, amplitude=.003, channel=1, nsamples=19456)

            self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
            self.hardware['LEDS']['R'].set(r=0, g=255, b=0)
            
            # Add a trigger to open the port
            self.triggers['R'].append(self.hardware['PORTS']['R'].open)
            self.triggers['R'].append(self.stage_block.set)
        
        elif self.target.startswith('rpi'):
            # It's a child target
            child_name, side = self.target.split('_')

            # No LED or stim
            self.stim = None
            self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
            self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

            # Tell child what the target is
            self.node2.send(
                to=child_name,
                key='PLAY',
                value={'target': side},
                )        

        else:
            raise ValueError("unknown target: {}".format(target))
        
        
        ## Set the stim to repeat
        if self.stim is not None:
            # Set the trigger to call function when stim is over
            self.stim.set_trigger(self.delayed_play_again)
            
            # Buffer the stim and start playing it after a delay
            self.stim.buffer()
            threading.Timer(.75, self.stim.play).start()        
        
        
        ## Return data
        data = {
            'target': self.target,
            'timestamp': datetime.datetime.now().isoformat(),
            'trial_num' : next(self.trial_counter)
        }
        return data

    def recv_hello(self, value):
        self.logger.debug("received HELLO from child with value {}".format(value))
        self.child_connected[value['from']] = True

    def response(self):
        """
        Just have to alert the Terminal that the current trial has ended
        and turn off any lights.
        """
        # Turn off the "stim_end" trigger so it doesn't keep playing
        if self.stim is not None:
            self.stim.set_trigger(self.done_playing)
        
        # Turn off any LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

        # Tell the child
        if self.target.startswith('rpi'):
            child_name, side = self.target.split('_')
            
            # Tell child what the target is
            self.node2.send(
                to=child_name,
                key='STOP',
                value={},
                )                

        # Tell the Terminal the trial has ended
        return {'TRIAL_END':True}

    def end(self):
        """
        When shutting down, release all hardware objects and turn LEDs off.
        """
        # Tell each child to END
        for child_name in self.CHILDREN.keys():
            # Tell child what the target is
            self.node2.send(
                to=child_name,
                key='END',
                value={},
                )    
        
        # Stop playing sound
        if self.stim is not None:
            self.stim.set_trigger(self.done_playing)
    
        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

        # Release all hardware
        for k, v in self.hardware.items():
            for pin, obj in v.items():
                obj.release()

    def delayed_play_again(self):
        """Called when stim over
        """
        # Play it again, after a delay
        self.stim.buffer()
        threading.Timer(.75, self.stim.play).start()
        
    def done_playing(self):
        # This is called when the last stim of the trial has finished playing
        pass
