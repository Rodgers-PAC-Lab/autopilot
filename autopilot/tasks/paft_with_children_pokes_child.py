from collections import OrderedDict as odict
import time
import functools
import datetime
import autopilot
from autopilot import prefs
from itertools import cycle
from . import children

class Paft_With_Children_Pokes_Child(children.Child):
    """Define the child task associated with Paft_With_Children_Pokes"""
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
        """Initialize a new Paft_With_Children_Pokes_Child
        
        task_type : 'Paft_With_Children_Pokes_Child'
        subject : from Terminal
        child : {'parent': parent's name, 'subject' from Terminal}
        reward : from value of START/CHILD message
        """
        ## Init
        # Store my name
        # This is used for reporting pokes to the parent
        self.name = prefs.get('NAME')

        # Set up a logger
        self.logger = autopilot.utils.loggers.init_logger(self)


        ## Hardware
        self.triggers = {}
        self.init_hardware()
        self.set_poke_triggers()


        ## Stages
        # Only one stage
        self.stages = cycle([self.play])
        self.stage_block = stage_block
        
        
        ## Set up NET_Node to communicate with Parent
        self.create_inter_pi_communication_node()

    def create_inter_pi_communication_node(self):
        """Defines a Net_Node to communicate with the Parent
        
        This is a Net_Node that is used to directly exchange information
        with the parent about pokes and sounds. The parent is the 
        "router" / server and the children are the "dealer" / clients .. 
        ie many dealers, one router.
        
        The Parent will be blocked until each Child sends a "HELLO" message
        which happens in this function.
        
        The Net_Node defined here also specifies "listens" (ie triggers)
        of functions to be called upon receiving specified messages
        from the parrent, such as "HELLO" or "END".
        
        This Net_Node is saved as `self.node2`.
        """
        self.node2 = autopilot.networking.Net_Node(
            id=self.name,
            upstream='parent_pi',
            port=5001,
            upstream_ip=prefs.get('PARENTIP'),
            listens={
                'HELLO': self.recv_hello,
                },
            instance=False,
            )        
        
        # Send HELLO so that Parent knows we are here
        self.node2.send('parent_pi', 'HELLO', {'from': self.name})        

    def init_hardware(self):
        """Placeholder"""
        self.hardware = {}        

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
        board_pin = autopilot.hardware.BCM_TO_BOARD[pin]
        
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
    
    def play(self):
        """A single stage"""
        self.logger.debug("Starting the play stage")
        
        # Sleep so we don't go crazy
        time.sleep(3)

        # Continue to the next stage (which is this one again)
        self.stage_block.set()

    def recv_hello(self, value):
        """This is probably unnecessary"""
        self.logger.debug(
            "received HELLO from parent with value {}".format(value))

    def end(self):
        """This is called when the STOP signal is received from the parent"""
        self.logger.debug("Inside the self.end function")

        # Explicitly close the socket (helps with restarting cleanly)
        self.node2.sock.close()
        self.node2.release()
