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

else:
    # This happens on the Terminal, for instance
    MY_BOX = 'NoBox'
    MY_PARENTS_NAME = 'NoParent'
    MY_PI1 = 'NoPi1'
    MY_PI2 = 'NoPi2'
    MY_PI3 = 'NoPi3'
    MY_PI4 = 'NoPi4'

# Duration of the ITI
ITI_DURATION_SEC = 1
STIM_AMPLITUDE = .01
STIM_HP_FILT = 5000
INTER_STIM_INTERVAL_FLOOR = .15
STIM_DURATION_MS = 10

# Define a stimulus set to use
method = 'sound_and_not_light'
if method == 'sound_xor_light':
    stimulus_set = pandas.DataFrame.from_records([
        (MY_PI1, 'L', True, False),
        (MY_PI1, 'R', True, False),
        (MY_PI2, 'L', True, False),
        (MY_PI2, 'R', True, False),
        (MY_PI3, 'L', True, False),
        (MY_PI3, 'R', True, False),
        (MY_PI4, 'L', True, False),
        (MY_PI4, 'R', True, False),
        (MY_PI1, 'L', False, True),
        (MY_PI1, 'R', False, True),
        (MY_PI2, 'L', False, True),
        (MY_PI2, 'R', False, True),
        (MY_PI3, 'L', False, True),
        (MY_PI3, 'R', False, True),
        (MY_PI4, 'L', False, True),
        (MY_PI4, 'R', False, True),
        ], columns=['rpi', 'side', 'sound', 'light'],
        )
elif method == 'sound_and_not_light':
    stimulus_set = pandas.DataFrame.from_records([
        (MY_PI1, 'L', True, False),
        (MY_PI1, 'R', True, False),
        (MY_PI2, 'L', True, False),
        (MY_PI2, 'R', True, False),
        (MY_PI3, 'L', True, False),
        (MY_PI3, 'R', True, False),
        (MY_PI4, 'L', True, False),
        (MY_PI4, 'R', True, False),
        ], columns=['rpi', 'side', 'sound', 'light'],
        )        
elif method == 'sound_and_maybe_light':
    stimulus_set = pandas.DataFrame.from_records([
        (MY_PI1, 'L', True, False),
        (MY_PI1, 'R', True, False),
        (MY_PI2, 'L', True, False),
        (MY_PI2, 'R', True, False),
        (MY_PI3, 'L', True, False),
        (MY_PI3, 'R', True, False),
        (MY_PI4, 'L', True, False),
        (MY_PI4, 'R', True, False),
        (MY_PI1, 'L', True, True),
        (MY_PI1, 'R', True, True),
        (MY_PI2, 'L', True, True),
        (MY_PI2, 'R', True, True),
        (MY_PI3, 'L', True, True),
        (MY_PI3, 'R', True, True),
        (MY_PI4, 'L', True, True),
        (MY_PI4, 'R', True, True),
        ], columns=['rpi', 'side', 'sound', 'light'],
        )
elif method == 'sound_or_light':
    stimulus_set = pandas.DataFrame.from_records([
        (MY_PI1, 'L', True, False),
        (MY_PI1, 'R', True, False),
        (MY_PI2, 'L', True, False),
        (MY_PI2, 'R', True, False),
        (MY_PI3, 'L', True, False),
        (MY_PI3, 'R', True, False),
        (MY_PI4, 'L', True, False),
        (MY_PI4, 'R', True, False),
        (MY_PI1, 'L', False, True),
        (MY_PI1, 'R', False, True),
        (MY_PI2, 'L', False, True),
        (MY_PI2, 'R', False, True),
        (MY_PI3, 'L', False, True),
        (MY_PI3, 'R', False, True),
        (MY_PI4, 'L', False, True),
        (MY_PI4, 'R', False, True),
        (MY_PI1, 'L', True, True),
        (MY_PI1, 'R', True, True),
        (MY_PI2, 'L', True, True),
        (MY_PI2, 'R', True, True),
        (MY_PI3, 'L', True, True),
        (MY_PI3, 'R', True, True),
        (MY_PI4, 'L', True, True),
        (MY_PI4, 'R', True, True),
        ], columns=['rpi', 'side', 'sound', 'light'],
        )
elif method == 'sound_and_light':
    stimulus_set = pandas.DataFrame.from_records([
        (MY_PI1, 'L', True, True),
        (MY_PI1, 'R', True, True),
        (MY_PI2, 'L', True, True),
        (MY_PI2, 'R', True, True),
        (MY_PI3, 'L', True, True),
        (MY_PI3, 'R', True, True),
        (MY_PI4, 'L', True, True),
        (MY_PI4, 'R', True, True),
        ], columns=['rpi', 'side', 'sound', 'light'],
        )
else:
    raise ValueError('unrecognized method: {}'.format(method))

class PAFT(Task):
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
        'trial': {'type':'i32'},
        'trials_total': {'type': 'i32'},
        'rpi': {'type': 'S10'},
        'side': {'type': 'S10'},
        'light': {'type': 'S10'},
        'sound': {'type': 'S10'},
        'timestamp': {'type':'S26'},
    }

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

    
    ## Plot parameters
    PLOT = {
        'data': {
            'target': 'point'
        }
    }

    
    ## Methods
    def __init__(self, stage_block, current_trial, step_name, task_type, 
        subject, step, session, pilot, reward):
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
            This is passed from the "protocol" json
        step_name : 'PAFT'
            This is passed from the "protocol" json
        task_type : 'PAFT'
            This is passed from the "protocol" json
        subject : taken from Terminal
        step : 0
            Index into the "protocol" json?
        session : number of times it's been started
        pilot : name of pilot
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
        
        # This keeps track of the current stim
        self.stim = None
        self.stim_index = None
        self.stim_params = None
        
        # This is used to count the trials, it is initialized by
        # something to wherever we are in the Protocol graduation
        self.trial_counter = itertools.count(int(current_trial))
        
        # This is a trial counter that always starts at zero
        self.n_trials = 0
        
        # A dict of hardware triggers
        self.triggers = {}

        # This is used in the ITI stage
        self.iti_is_over = False

        # Stage list to iterate
        stage_list = [self.ITI_start, self.ITI_wait, self.water, self.response]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)


        ## Subject-specific params (requires self.logger)
        self.subject_params = {}
        if subject in [
            'tstPAFT', 'Female2_0903', 'Female3_0903', 'Female4_0903',
            'Male3_0720', 'Male4_0720', 'Male5_0720',
            '3276-2', '3276-7',
            '3279-2', '3279-9',
            '3279-3',
            '3277-1', '3277-3',
            ]:
            # Irregular
            self.subject_params['gamma_scale'] = 0.15
        
        elif subject in [
            'Cage3276F', 'Cage3277F',
            'Cage3279F', 'Cage3279M', 'Cage3277M',
            '3276-1', 
            '3279-4',
            '3277-4', '3277-5',
            '3279-8',
            '3277-2',
            ]:
            # Regular
            self.subject_params['gamma_scale'] = 0.001
        
        else:
            # Default (but warn, because this should be specified)
            self.logger.debug("warning: unknown subject {}".format(subject))
            self.subject_params['gamma_scale'] = 0.001


        ## Init hardware -- this sets self.hardware and self.pin_id
        self.init_hardware()

        # Set reward values for solenoids
        for port_name, port in self.hardware['PORTS'].items():
            self.logger.debug(
                "setting reward for {} to {}".format(port_name, reward))
            port.duration = float(reward)

        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)


        ## This is used for error pokes
        self.left_error_sound = sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=0)
        self.right_error_sound = sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=1)
        
        # init sound
        self.init_sound = sounds.Noise(duration=100, amplitude=.001, channel=0)

        
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
        
        # Play init sound
        # This is just because there's often weird audio garbling until the
        # first sound is played, not sure why, not sure which of these lines
        # helps
        self.init_sound.buffer()
        self.init_sound.set_trigger(self.do_nothing)
        threading.Timer(.75, self.init_sound.play).start()
    
    def do_nothing(self):
        pass

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
        
        # Identify poked
        child_name = value['from']
        poke_name = value['poke']
        
        # Compare target to poked
        if (
                self.stim_params['rpi'] == child_name and 
                self.stim_params['side'] == poke_name):
            self.logger.debug('correct poke {}; target was {}'.format(
                value, 
                self.stim_params['rpi'] + '_' + self.stim_params['side']))
            self.stage_block.set()
        else:
            self.logger.debug('incorrect poke {}; target was {}'.format(
                value, 
                self.stim_params['rpi'] + '_' + self.stim_params['side']))
        
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

        # Append error sound to each
        self.triggers['L'].append(self.left_error_sound.play)
        self.triggers['R'].append(self.right_error_sound.play)
    
    def ITI_start(self, *args, **kwargs):
        """A state that initiates an ITI timer"""
        # Set poke triggers (for logging)
        # Make sure this doesn't depend on self.stim which hasn't been
        # chosen yet!
        self.set_poke_triggers()
        
        # This flag is set after the timer is over
        self.iti_is_over = False
        
        # Start the timer
        threading.Timer(ITI_DURATION_SEC, self.ITI_stop).start()
        
        # Continue to next stage (self.ITI_wait)
        self.stage_block.set()

    def ITI_stop(self):
        """Helper function just to set flag iti_is_over"""
        self.iti_is_over = True
        
    def ITI_wait(self, *args, **kwargs):
        """A state that waits for the ITI to be over"""
        # Wait until the ITI is over
        while not self.iti_is_over:
            pass
        
        # Set the stage block
        self.stage_block.set()
    
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


        ## Set poke triggers (for logging)
        # Make sure this doesn't depend on self.stim which hasn't been
        # chosen yet!
        self.set_poke_triggers()
        
        
        ## Choose target
        # Identify possible stim (those that do not repeat the reward port)
        excluding_previous = []
        for idx in stimulus_set.index:
            if self.stim_params is not None:
                if (
                    stimulus_set.loc[idx, 'side'] == self.stim_params['side'] 
                    and stimulus_set.loc[idx, 'rpi'] == self.stim_params['rpi'] 
                    ):
                    continue
            excluding_previous.append(idx)
        
        # Choose
        self.stim_index = random.choice(excluding_previous)
        self.stim_params = stimulus_set.loc[self.stim_index]
        stringy_stim_params = self.stim_params.to_string().replace('\n', '; ')
        self.logger.debug("Chosen stim params: {}".format(stringy_stim_params))
        
        
        ## Set stim
        if self.stim_params['rpi'] == MY_PI1:
            ## This rpi controls the target port
            # Set channel and other_side variables, used below
            if self.stim_params['side'] == 'L':
                channel = 0
                other_side = 'R'
            else:
                channel = 1
                other_side = 'L'
            
            # Set sound on or off
            if self.stim_params['sound']:
                amplitude = STIM_AMPLITUDE
            else:
                amplitude = 0
            
            # Set light on or off
            if self.stim_params['light']:
                self.hardware['LEDS'][self.stim_params['side']].set(
                    r=0, g=255, b=0)
                self.hardware['LEDS'][other_side].set(
                    r=0, g=0, b=0)
            
            # Generate the sound
            self.stim = sounds.Noise(
                duration=STIM_DURATION_MS, amplitude=amplitude, channel=channel, 
                highpass=STIM_HP_FILT)

            # Remove the error sound (should be the last one)
            popped = self.triggers[self.stim_params['side']].pop()
            assert popped in [
                self.left_error_sound.play, self.right_error_sound.play]
            
            # Add a trigger to open the port
            self.triggers[self.stim_params['side']].append(
                self.hardware['PORTS'][self.stim_params['side']].open)
            self.triggers[self.stim_params['side']].append(
                self.stage_block.set)

        else:
            ## A child rpi controls the target port
            # No stim
            self.stim = None

            # Tell child what the target is
            self.node2.send(
                to=self.stim_params['rpi'],
                key='PLAY',
                value={
                    'side': self.stim_params['side'],
                    'light': self.stim_params['light'],
                    'sound': self.stim_params['sound'],
                    },
                )


        ## Set the stim to repeat
        if self.stim is not None:
            # Set the trigger to call function when stim is over
            self.stim.set_trigger(self.delayed_play_again)
            
            # Buffer the stim and start playing it after a delay
            self.stim.buffer()
            threading.Timer(.75, self.stim.play).start()        
        
        
        ## Return data
        data = {
            'rpi': self.stim_params['rpi'],
            'side': self.stim_params['side'],
            'light': str(self.stim_params['light']),
            'sound': str(self.stim_params['sound']),
            'timestamp': datetime.datetime.now().isoformat(),
            'trial': self.n_trials,
            'trials_total' : next(self.trial_counter)
        }
        self.n_trials = self.n_trials + 1

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

        # Tell the child to stop
        if self.stim_params['rpi'] != MY_PI1:
            self.node2.send(
                to=self.stim_params['rpi'],
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

        # Release
        self.node2.release()

        # Release all hardware
        for k, v in self.hardware.items():
            for pin, obj in v.items():
                obj.release()

    def delayed_play_again(self):
        """Called when stim over
        """
        # Play it again, after a delay
        self.stim.buffer()
        
        # Draw the interval
        interval = np.random.gamma(3, self.subject_params['gamma_scale'])
        
        # Hard floor
        if interval < INTER_STIM_INTERVAL_FLOOR:
            interval = INTER_STIM_INTERVAL_FLOOR
        
        threading.Timer(interval, self.stim.play).start()
        
    def done_playing(self):
        # This is called when the last stim of the trial has finished playing
        pass
