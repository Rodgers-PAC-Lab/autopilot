"""
Sub-tasks that serve as children to other tasks.

.. note::

    The Child agent will be formalized in an upcoming release, until then these classes
    remain relatively undocumented as their design will likely change.

"""


import threading
import itertools
from itertools import cycle
import random
import datetime
from time import sleep
import functools
from collections import OrderedDict as odict
from collections import deque
from queue import Empty, LifoQueue
import tables
import numpy as np
import autopilot.hardware.gpio
from autopilot.stim.sound import sounds
from autopilot.tasks.task import Task
from autopilot.networking import Net_Node
from autopilot import prefs
from autopilot.hardware import BCM_TO_BOARD
from autopilot.core.loggers import init_logger
import autopilot.transform
from autopilot.hardware.gpio import Digital_Out
from autopilot.hardware.usb import Wheel
from autopilot.hardware import cameras
from autopilot.transform import transforms

STIM_AMPLITUDE = .01
STIM_HP_FILT = 5000
INTER_STIM_INTERVAL_FLOOR = .15
STIM_DURATION_MS = 10

# Figure out which box we're in
MY_NAME = prefs.get('NAME')
if MY_NAME in ['rpi01', 'rpi02', 'rpi03', 'rpi04']:
    MY_BOX = 'Box1'
elif MY_NAME in ['rpi05', 'rpi06', 'rpi07', 'rpi08']:
    MY_BOX = 'Box2'
else:
    # This happens on the Terminal, for instance
    MY_BOX = 'NoBox'
    
class Child(object):
    """Just a placeholder class for now to work with :func:`autopilot.get`"""

class PAFT_Child(Child):
    # PARAMS to accept
    PARAMS = odict()

    # HARDWARE to init
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

    def __init__(self, stage_block, task_type, subject, child, reward):
        """Initialize a new PAFT_Child
        
        task_type : 'PAFT Child'
        subject : from Terminal
        child : {'parent': parent's name, 'subject' from Terminal}
        reward : from value of START/CHILD message
        """
        ## Init
        # Store my name
        # This is used for reporting pokes to the parent
        self.name = prefs.get('NAME')

        # This keeps track of the current stim
        self.stim = None

        # Set up a logger
        self.logger = init_logger(self)
        
        
        ## Subject-specific params (requires self.logger)
        self.subject_params = {}
        if subject in [
            'tstPAFT', 'Female2_0903', 'Female3_0903', 'Female4_0903',
            'Male3_0720', 'Male4_0720', 'Male5_0720',
            ]:
            # Irregular
            self.subject_params['gamma_scale'] = 0.15
        
        elif subject in [
            'Cage3276F', 'Cage3277F',
            'Cage3279F', 'Cage3279M', 'Cage3277M',
            ]:
            # Regular
            self.subject_params['gamma_scale'] = 0.001
        
        else:
            # Default (but warn, because this should be specified)
            self.logger.debug("warning: unknown subject {}".format(subject))
            self.subject_params['gamma_scale'] = 0.001
        

        ## Hardware
        self.init_hardware()

        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)
        
        # Rewards
        for port_name, port in self.hardware['PORTS'].items():
            port.duration = float(reward)


        ## This is used for error pokes
        self.left_error_sound = sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=0)
        self.right_error_sound = sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=1)

        # init sound
        self.init_sound = sounds.Noise(duration=100, amplitude=.001, channel=0)


        ## Triggers
        self.triggers = {}
        self.set_poke_triggers()

        
        ## Stages
        # Only one stage
        self.stages = cycle([self.noop])
        self.stage_block = stage_block

        
        ## Networking
        self.node2 = Net_Node(
            id=self.name,
            upstream='parent_pi',
            port=5001,
            upstream_ip=prefs.get('PARENTIP'),
            listens={
                'HELLO': self.recv_hello,
                'PLAY': self.recv_play,
                'STOP': self.recv_stop,
                'END': self.recv_end,
                },
            instance=False,
            )        
        
        # Send
        self.node2.send(
            'parent_pi', 'HELLO', {'from': self.name})
        
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

    def set_poke_triggers(self):
        """"Set triggers for poke entry
        
        For each poke, sets these triggers:
            self.log_poke (write to own debugger)
            self.report_poke (report to parent)
        """
        for poke in ['L', 'R']:
            self.triggers[poke] = [
                functools.partial(self.log_poke, poke),
                functools.partial(self.report_poke, poke),
                ]        
        
        # Append error sound to each
        self.triggers['L'].append(self.left_error_sound.play)
        self.triggers['R'].append(self.right_error_sound.play)

    def log_poke(self, poke):
        """Write in the logger that the poke happened"""
        self.logger.debug('{} {} poke'.format(
            datetime.datetime.now().isoformat(),
            poke,
            ))

    def report_poke(self, poke):
        """Tell the parent that the poke happened"""
        self.node2.send(
            'parent_pi', 'POKE', {'from': self.name, 'poke': poke},
            )

    def recv_hello(self, value):
        self.logger.debug("received HELLO from parent")

    def noop(self):
        """The noop stage"""
        # Set these triggers again, in case they were just unset by 
        # handle_trigger
        self.set_poke_triggers()
        
        # Prevent moving to the next stage, because there's only one stage
        # anyway
        self.stage_block.clear()
        
        # Return no data because this call will fly by very quickly
        return {}

    def end(self):
        self.stage_block.set()

    def recv_play(self, value):
        self.logger.debug("recv_play with value: {}".format(value))
        
        # Set target
        side = value['side']
        use_light = value['light']
        use_sound = value['sound']
        
        # Set channel and other_side variables
        if side == 'L':
            channel = 0
            other_side = 'R'
        else:
            channel = 1
            other_side = 'L'
        
        # Set sound on or off
        if use_sound:
            amplitude = STIM_AMPLITUDE
        else:
            amplitude = 0
        
        # Set light on or off
        if use_light:
            other_side = 'R' if side == 'L' else 'L'
            self.hardware['LEDS'][side].set(
                r=0, g=255, b=0)
            self.hardware['LEDS'][other_side].set(
                r=0, g=0, b=0)
        
        # Generate the sound
        self.stim = sounds.Noise(
            duration=STIM_DURATION_MS, amplitude=amplitude, channel=channel, 
            highpass=STIM_HP_FILT)
        
        # Remove the error sound (should be the last one)
        popped = self.triggers[side].pop()
        assert popped in [
            self.left_error_sound.play, self.right_error_sound.play]
        
        # Add a trigger to open the port
        self.triggers[side].append(
            self.hardware['PORTS'][side].open)

        # Immediately after opening, reset the poke triggers
        # Kind of weird to modify self.triggers while we're iterating
        # over it, but should be okay since this is the last one
        self.triggers[side].append(
            self.set_poke_triggers)


        ## Set the stim to repeat
        # Set the trigger to call function when stim is over
        self.stim.set_trigger(self.delayed_play_again)
        
        # Buffer the stim and start playing it after a delay
        self.stim.buffer()
        threading.Timer(.75, self.stim.play).start()        

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

    def recv_stop(self, value):
        # debug
        self.logger.debug("recv_stop with value: {}".format(value))
        
        # Stop playing sound
        if self.stim is not None:
            self.stim.set_trigger(self.done_playing)
    
        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

    def recv_end(self, value):
        # debug
        self.logger.debug("recv_end with value: {}".format(value))
        
        # Stop playing sound
        if self.stim is not None:
            self.stim.set_trigger(self.done_playing)
    
        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)
        
        # Release Net_Node
        self.node2.release()
        
        # Release all hardware
        for k, v in self.hardware.items():
            for pin, obj in v.items():
                obj.release()

    def done_playing(self):
        # This is called when the last stim of the trial has finished playing
        pass

class PokeTrain_Child(Child):
    # PARAMS to accept
    PARAMS = odict()

    # HARDWARE to init
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

    def __init__(self, stage_block, task_type, subject, child, reward):
        """Initialize a new PokeTrain_Child
        
        task_type : 'PokeTrain Child'
        subject : from Terminal
        child : {'parent': parent's name, 'subject' from Terminal}
        reward : from value of START/CHILD message
        """
        ## Init
        # Store my name
        # This is used for reporting pokes to the parent
        self.name = prefs.get('NAME')

        # This keeps track of the current stim
        self.stim = None

        # Set up a logger
        self.logger = init_logger(self)
        
        
        ## Subject-specific params (requires self.logger)
        self.subject_params = {}
        if subject in [
            'tstPAFT', 'Female2_0903', 'Female3_0903', 'Female4_0903',
            'Male3_0720', 'Male4_0720', 'Male5_0720',
            ]:
            # Irregular
            self.subject_params['gamma_scale'] = 0.15
        
        elif subject in [
            'Cage3276F', 'Cage3277F',
            'Cage3279F', 'Cage3279M', 'Cage3277M',
            ]:
            # Regular
            self.subject_params['gamma_scale'] = 0.001
        
        else:
            # Default (but warn, because this should be specified)
            self.logger.debug("warning: unknown subject {}".format(subject))
            self.subject_params['gamma_scale'] = 0.001
        

        ## Hardware
        self.init_hardware()

        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)
        
        # Rewards
        for port_name, port in self.hardware['PORTS'].items():
            port.duration = float(reward)


        ## This is used for error pokes
        self.left_error_sound = sounds.Tritone(
            frequency=8000, duration=250, amplitude=.00003, channel=0)
        self.right_error_sound = sounds.Tritone(
            frequency=8000, duration=250, amplitude=.00003, channel=1)

        # init sound
        self.init_sound = sounds.Noise(duration=100, amplitude=.00001, channel=0)


        ## Triggers
        self.triggers = {}
        self.set_poke_triggers()

        
        ## Stages
        # Only one stage
        self.stages = cycle([self.noop])
        self.stage_block = stage_block

        
        ## Networking
        self.node2 = Net_Node(
            id=self.name,
            upstream='parent_pi',
            port=5001,
            upstream_ip=prefs.get('PARENTIP'),
            listens={
                'HELLO': self.recv_hello,
                'PLAY': self.recv_play,
                'STOP': self.recv_stop,
                'END': self.recv_end,
                },
            instance=False,
            )        
        
        # Send
        self.node2.send(
            'parent_pi', 'HELLO', {'from': self.name})
        
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

    def set_poke_triggers(self):
        """"Set triggers for poke entry
        
        For each poke, sets these triggers:
            self.log_poke (write to own debugger)
            self.report_poke (report to parent)
        """
        for poke in ['L', 'R']:
            self.triggers[poke] = [
                functools.partial(self.log_poke, poke),
                functools.partial(self.report_poke, poke),
                ]        
        
        # Append open to each
        self.triggers['L'].append(self.hardware['PORTS']['L'].open)
        self.triggers['R'].append(self.hardware['PORTS']['R'].open)

        # Immediately after opening, reset the poke triggers
        # Kind of weird to modify self.triggers while we're iterating
        # over it, but should be okay since this is the last one
        self.triggers['L'].append(self.set_poke_triggers)
        self.triggers['R'].append(self.set_poke_triggers)

    def log_poke(self, poke):
        """Write in the logger that the poke happened"""
        self.logger.debug('{} {} poke'.format(
            datetime.datetime.now().isoformat(),
            poke,
            ))

    def report_poke(self, poke):
        """Tell the parent that the poke happened"""
        self.node2.send(
            'parent_pi', 'POKE', {'from': self.name, 'poke': poke},
            )

    def recv_hello(self, value):
        self.logger.debug("received HELLO from parent")

    def noop(self):
        """The noop stage"""
        # Set these triggers again, in case they were just unset by 
        # handle_trigger
        self.set_poke_triggers()
        
        # Prevent moving to the next stage, because there's only one stage
        # anyway
        self.stage_block.clear()
        
        # Return no data because this call will fly by very quickly
        return {}

    def end(self):
        self.stage_block.set()

    def recv_play(self, value):
        self.logger.debug("recv_play with value: {}".format(value))
        
        # Set target
        side = value['side']
        use_light = value['light']
        use_sound = value['sound']
        
        # Set channel and other_side variables
        if side == 'L':
            channel = 0
            other_side = 'R'
        else:
            channel = 1
            other_side = 'L'
        
        # Set sound on or off
        if use_sound:
            amplitude = 0.00001 #STIM_AMPLITUDE
        else:
            amplitude = 0
        
        # Set light on or off
        if use_light:
            other_side = 'R' if side == 'L' else 'L'
            self.hardware['LEDS'][side].set(
                r=0, g=255, b=0)
            self.hardware['LEDS'][other_side].set(
                r=0, g=0, b=0)
        
        # Generate the sound
        self.stim = sounds.Noise(
            duration=STIM_DURATION_MS, amplitude=amplitude, channel=channel, 
            highpass=STIM_HP_FILT)
        
        # Pop the last two events from the blocked port
        popped = self.triggers[side].pop()
        assert popped == self.set_poke_triggers

        popped = self.triggers[side].pop()
        assert popped == self.hardware['PORTS'][side].open


        ## Set the stim to repeat
        # Set the trigger to call function when stim is over
        self.stim.set_trigger(self.delayed_play_again)
        
        # Buffer the stim and start playing it after a delay
        self.stim.buffer()
        threading.Timer(.75, self.stim.play).start()        

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

    def recv_stop(self, value):
        # debug
        self.logger.debug("recv_stop with value: {}".format(value))
        
        # Stop playing sound
        if self.stim is not None:
            self.stim.set_trigger(self.done_playing)
    
        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

    def recv_end(self, value):
        # debug
        self.logger.debug("recv_end with value: {}".format(value))
        
        # Stop playing sound
        if self.stim is not None:
            self.stim.set_trigger(self.done_playing)
    
        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)
        
        # Release Net_Node
        self.node2.release()
        
        # Release all hardware
        for k, v in self.hardware.items():
            for pin, obj in v.items():
                obj.release()

    def done_playing(self):
        # This is called when the last stim of the trial has finished playing
        pass


class Wheel_Child(Child):
    STAGE_NAMES = ['collect']

    PARAMS = odict()
    PARAMS['fs'] = {'tag': 'Velocity Reporting Rate (Hz)',
                    'type': 'int'}
    PARAMS['thresh'] = {'tag': 'Distance Threshold',
                        'type': 'int'}

    HARDWARE = {
        "OUTPUT": Digital_Out,
        "WHEEL":  Wheel
    }



    def __init__(self, stage_block=None, fs=10, thresh=100, **kwargs):
        super(Wheel_Child, self).__init__(**kwargs)
        self.fs = fs
        self.thresh = thresh

        self.hardware = {}
        self.hardware['OUTPUT'] = Digital_Out(prefs.get('HARDWARE')['OUTPUT'])
        self.hardware['WHEEL'] = Wheel(digi_out = self.hardware['OUTPUT'],
                                       fs       = self.fs,
                                       thresh   = self.thresh,
                                       mode     = "steady")
        self.stages = cycle([self.noop])
        self.stage_block = stage_block

    def noop(self):
        # just fitting in with the task structure.
        self.stage_block.clear()
        return {}

    def end(self):
        self.hardware['WHEEL'].release()
        self.stage_block.set()


class Video_Child(Child):
    PARAMS = odict()
    PARAMS['cams'] = {'tag': 'Dictionary of camera params, or list of dicts',
                      'type': ('dict', 'list')}

    def __init__(self, cams=None, stage_block = None, start_now=True, **kwargs):
        """
        Args:
            cams (dict, list): Should be a dictionary of camera parameters or a list of dicts. Dicts should have, at least::

                {
                    'type': 'string_of_camera_class',
                    'name': 'name_of_camera_in_task',
                    'param1': 'first_param'
                }
        """
        super(Video_Child, self).__init__(**kwargs)

        if cams is None:
            Exception('Need to give us a cams dictionary!')

        self.cams = {}

        self.start_now = start_now


        if isinstance(cams, dict):

            try:
                cam_class = getattr(cameras, cams['type'])
                self.cams[cams['name']] = cam_class(**cams)
                # if start:
                #     self.cams[cams['name']].capture()
            except AttributeError:
                AttributeError("Camera type {} not found!".format(cams['type']))

        elif isinstance(cams, list):
            for cam in cams:
                try:
                    cam_class = getattr(cameras, cam['type'])
                    self.cams[cam['name']] = cam_class(**cam)
                    # if start:
                    #     self.cams[cam['name']].capture()
                except AttributeError:
                    AttributeError("Camera type {} not found!".format(cam['type']))

        self.stages = cycle([self.noop])
        self.stage_block = stage_block


        if self.start_now:
            self.start()
        # self.thread = threading.Thread(target=self._stream)
        # self.thread.daemon = True
        # self.thread.start()

    def start(self):
        for cam in self.cams.values():
            cam.capture()

    def stop(self):
        for cam_name, cam in self.cams.items():
            try:
                cam.release()
            except Exception as e:
                Warning('Couldnt release camera {},\n{}'.format(cam_name, e))



    def _stream(self):
        self.node = Net_Node(
            "T_CHILD",
            upstream=prefs.get('NAME'),
            port=prefs.get('MSGPORT'),
            listens = {},
            instance=True
        )

        while True:
            for name, cam in self.cams.items():
                try:
                    frame, timestamp = cam.q.get_nowait()
                    self.node.send(key='CONTINUOUS',
                                   value={cam.name:frame,
                                          'timestamp':timestamp},
                                   repeat=False,
                                   flags={'MINPRINT':True})
                except Empty:
                    pass



    def noop(self):
        # just fitting in with the task structure.
        self.stage_block.clear()
        return {}

    # def start(self):
    #     for cam in self.cams.values():
    #         cam.capture()
    #
    # def stop(self):
    #     for cam in self.cams.values():
    #         cam.release()

class Transformer(Child):

    def __init__(self, transform,
                 operation: str ="trigger",
                 return_id = 'T',
                 return_key = None,
                 stage_block = None,
                 value_subset=None,
                 **kwargs):
        """

        Args:
            transform:
            operation (str): either

                * "trigger", where the last transform is a :class:`~autopilot.transform.transforms.Condition`
                and a trigger is returned to sender only when the return value of the transformation changes, or
                * "stream", where each result of the transformation is returned to sender

            return_id:
            stage_block:
            value_subset (str): Optional - subset a value from from a dict/list sent to :meth:`.l_process`
            **kwargs:
        """
        super(Transformer, self).__init__(**kwargs)
        assert operation in ('trigger', 'stream', 'debug')
        self.operation = operation
        self._last_result = None

        if return_key is None:
            self.return_key = self.operation.upper()
        else:
            self.return_key = return_key

        self.return_id = return_id
        self.stage_block = stage_block
        self.stages = cycle([self.noop])
        # self.input_q = LifoQueue()
        self.input_q = deque(maxlen=1)
        self.value_subset = value_subset

        self.logger = init_logger(self)

        self.process_thread = threading.Thread(target=self._process, args=(transform,))
        self.process_thread.daemon = True
        self.process_thread.start()

    def noop(self):
        # just fitting in with the task structure.
        self.stage_block.clear()
        return {}



    def _process(self, transform):

        self.transform = autopilot.transform.make_transform(transform)

        self.node = Net_Node(
            f"{prefs.get('NAME')}_TRANSFORMER",
            upstream=prefs.get('NAME'),
            port=prefs.get('MSGPORT'),
            listens = {
                'CONTINUOUS': self.l_process
            },
            instance=False
        )

        self.node.send(self.return_id, 'STATE', value='READY')

        while True:
            try:
                # value = self.input_q.get_nowait()
                value = self.input_q.popleft()
            # except Empty:
            except IndexError:
                sleep(0.001)
                continue
            result = self.transform.process(value)

            self.node.logger.debug(f'Processed frame, result: {result}')


            if self.operation == "trigger":
                if result != self._last_result:
                    self.node.send(self.return_id, self.return_key, result)
                    self._last_result = result

            elif self.operation == 'stream':
                # FIXME: Another key that's not TRIGGER
                self.node.send(self.return_id, self.return_key, result)

            elif self.operation == 'debug':
                pass


    def l_process(self, value):
        # get array out of value

        # FIXME hack for dlc
        self.node.logger.debug('Received and queued processing!')
        # self.input_q.put_nowait(value['MAIN'])
        if self.value_subset:
            value = value[self.value_subset]
        self.input_q.append(value)










