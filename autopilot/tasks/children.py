"""
Sub-tasks that serve as children to other tasks.

.. note::

    The Child agent will be formalized in an upcoming release, until then these classes
    remain relatively undocumented as their design will likely change.

"""

from collections import OrderedDict as odict
from collections import deque
import time

import autopilot.transform
from autopilot import prefs
from autopilot.hardware.gpio import Digital_Out
from autopilot.hardware.usb import Wheel
from autopilot.hardware import cameras
from autopilot.networking import Net_Node
from autopilot.core.loggers import init_logger
from autopilot.transform import transforms
from autopilot.hardware.i2c import I2C_9DOF
from autopilot.hardware.cameras import PiCamera
from autopilot.tasks import Task
from itertools import cycle
from queue import Empty, LifoQueue
import threading
import logging
from time import sleep

class Child(object):
    """Just a placeholder class for now to work with :func:`autopilot.get`"""

class PAFT_Child_Simple(Child):
    """Define the child task associated with PAFT_Parent_Child"""
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
        """Initialize a new PAFT_Child_Simple
        
        task_type : 'PAFT_Child_Simple'
        subject : from Terminal
        child : {'parent': parent's name, 'subject' from Terminal}
        reward : from value of START/CHILD message
        """
        ## Init
        # Store my name
        # This is used for reporting pokes to the parent
        self.name = prefs.get('NAME')

        # Set up a logger
        self.logger = init_logger(self)


        ## Hardware
        self.init_hardware()


        ## Stages
        # Only one stage
        self.stages = cycle([self.play])
        self.stage_block = stage_block
        
        
        ## Networking
        self.node2 = Net_Node(
            id=self.name,
            upstream='parent_pi',
            port=5001,
            upstream_ip=prefs.get('PARENTIP'),
            listens={
                'HELLO': self.recv_hello,
                'END': self.recv_end,
                },
            instance=False,
            )        
        
        # Send
        self.node2.send(
            'parent_pi', 'HELLO', {'from': self.name})

    def init_hardware(self):
        """Placeholder"""
        self.hardware = {}        

    def play(self):
        """A single stage"""
        self.logger.debug("Starting the play stage")
        
        # Sleep so we don't go crazy
        time.sleep(1)

        # Continue to the next stage (which is this one again)
        self.stage_block.set()

    def recv_hello(self, value):
        self.logger.debug(
            "received HELLO from parent with value {}".format(value))

    def recv_end(self, value):
        self.logger.debug("recv_end with value: {}".format(value))
        
        # Here would be a good place to release hardware, although
        # could also just do this in regular end

    def end(self):
        """This is called when the STOP signal is received from the parent"""
        self.logger.debug("Inside the self.end function")

        # Explicitly close the socket (helps with restarting cleanly)
        self.node2.sock.close()
        self.node2.release()
        
        # Do this so it stops cycling through stages
        self.stop_running = True
        self.stage_block.clear()

class PAFT_Child(Child):
    """Define the child task associated with PAFT"""
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

        # Set up a logger
        self.logger = init_logger(self)


        ## Hardware
        self.init_hardware()


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
        self.node2 = Net_Node(
            id=self.name,
            upstream='parent_pi',
            port=5001,
            upstream_ip=prefs.get('PARENTIP'),
            listens={
                'HELLO': self.recv_hello,
                #~ 'PLAY': self.recv_play,
                #~ 'STOP': self.recv_stop,
                #~ 'END': self.recv_end,
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
        time.sleep(1)

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
                 node_id = None,
                 return_id = 'T',
                 return_ip = None,
                 return_port = None,
                 return_key = None,
                 router_port = None,
                 stage_block = None,
                 value_subset=None,
                 forward_id=None,
                 forward_ip=None,
                 forward_port=None,
                 forward_key=None,
                 forward_what='both',
                 **kwargs):
        """

        Args:
            transform:
            operation (str): either

                * "trigger", where the last transform is a :class:`~autopilot.transform.transforms.Condition`
                and a trigger is returned to sender only when the return value of the transformation changes, or
                * "stream", where each result of the transformation is returned to sender

            return_id:
            return_ip:
            return_port:
            return_key:
            router_port (None, int): If not ``None`` (default), spawn the node with a route port to receieve
            stage_block:
            value_subset (str): Optional - subset a value from from a dict/list sent to :meth:`.l_process`
            forward_what (str): one of 'input', 'output', or 'both' (default) that determines what is forwarded
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
        self.return_ip = return_ip
        self.return_port = return_port
        if self.return_port is None:
            self.return_port = prefs.get('MSGPORT')
        if node_id is None:
            self.node_id = f"{prefs.get('NAME')}_TRANSFORMER"
        else:
            self.node_id = node_id
        self.router_port = router_port

        self.forward_id = forward_id
        self.forward_ip = forward_ip
        self.forward_port = forward_port
        self.forward_key = forward_key
        self.forward_node = None
        self.forward_what = forward_what

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
            self.node_id,
            upstream=self.return_id,
            upstream_ip=self.return_ip,
            port=self.return_port,
            router_port=self.router_port,
            listens = {
                'CONTINUOUS': self.l_process
            },
            instance=False
        )

        if all([x is not None for x in
                (self.forward_id,
                 self.forward_ip,
                 self.forward_key,
                 self.forward_port)]):
            self.forward_node = Net_Node(
                id=self.node_id,
                upstream=self.forward_id,
                upstream_ip=self.forward_ip,
                port=self.forward_port,
                listens={}
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
                    if self.forward_node is not None:
                        self.forward(value, result)
                    self._last_result = result

            elif self.operation == 'stream':
                # FIXME: Another key that's not TRIGGER
                self.node.send(self.return_id, self.return_key, result)
                if self.forward_node is not None:
                    self.forward(value, result)

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

    def forward(self, input=None, output=None):
        if self.forward_what == 'both':
            self.forward_node.send(self.forward_id, self.forward_key, {'input':input,'output':output},flags={'MINPRINT':True,'NOREPEAT':True})
        elif self.forward_what == 'input':
            self.forward_node.send(self.forward_id, self.forward_key, input,flags={'MINPRINT':True,'NOREPEAT':True})
        elif self.forward_what == 'output':
            self.forward_node.send(self.forward_id, self.forward_key, output,flags={'MINPRINT':True,'NOREPEAT':True})
        else:
            raise ValueError("forward_what must be one of 'input', 'output', or 'both'")












