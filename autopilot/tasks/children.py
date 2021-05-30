"""
Sub-tasks that serve as children to other tasks.

.. note::

    The Child agent will be formalized in an upcoming release, until then these classes
    remain relatively undocumented as their design will likely change.

"""

from collections import OrderedDict as odict
from collections import deque

import autopilot.transform
from autopilot import prefs
from autopilot.hardware.gpio import Digital_Out
from autopilot.hardware.usb import Wheel
from autopilot.hardware import cameras
from autopilot.networking import Net_Node
from autopilot.core.loggers import init_logger
from autopilot.transform import transforms
from autopilot.stim.sound import sounds

from itertools import cycle
from queue import Empty, LifoQueue
import threading
import logging
from time import sleep
import datetime
import functools
from autopilot.tasks.task import Task


class PAFT_Child(Task):
    # Just one stage?
    STAGE_NAMES = ['noop']

    # Init PARAMS
    PARAMS = odict()
    #~ PARAMS['fs'] = {'tag': 'Velocity Reporting Rate (Hz)',
                    #~ 'type': 'int'}

    # Init HARDWARE
    HARDWARE = {
        'POKES':{
            'L': autopilot.hardware.gpio.Digital_In,
            #~ 'C': autopilot.hardware.gpio.Digital_In,
            'R': autopilot.hardware.gpio.Digital_In
        },
        'LEDS':{
            # TODO: use LEDs, RGB vs. white LED option in init
            'L': autopilot.hardware.gpio.LED_RGB,
            #~ 'C': autopilot.hardware.gpio.LED_RGB,
            'R': autopilot.hardware.gpio.LED_RGB
        },
        'PORTS':{
            'L': autopilot.hardware.gpio.Solenoid,
            #~ 'C': autopilot.hardware.gpio.Solenoid,
            'R': autopilot.hardware.gpio.Solenoid
        }
    }

    def __init__(self, stage_block=None, start=True, **kwargs):
        super(PAFT_Child, self).__init__()
        
        ## Store my name
        # This is used for reporting pokes to the parent
        self.name = prefs.get('NAME')
        
        
        ## Hardware
        self.init_hardware()
        
        # Only one stage
        self.stages = cycle([self.noop])
        self.stage_block = stage_block

        # Networking
        self.node2 = Net_Node(
            id='child_pi',
            upstream='parent_pi',
            port=5001,
            upstream_ip='192.168.11.201',
            listens={
                'HELLO': self.recv_hello,
                'PLAY': self.recv_play,
                'STOP': self.recv_stop,
                },
            instance=False,
            )        
        
        # Send
        self.node2.send(
            'parent_pi', 'HELLO', 'my name is child_pi')

        
        ## Initialize poke triggers
        self.set_poke_triggers()

        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

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
                functools.partial(self.report_poke, poke),
                ]        

    def log_poke(self, poke):
        """Write in the logger that the poke happened"""
        print("poke detected: {}".format(poke))
        self.logger.debug('{} {} poke'.format(
            datetime.datetime.now().isoformat(),
            poke,
            ))

    def report_poke(self, poke):
        """Tell the parent that the poke happened"""
        self.node2.send(
            'parent_pi', 'POKE', {'name': 'child0', 'poke': poke},
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
        target = value['target']
        if target == 'child_L':
            self.stim = sounds.Noise(
                duration=100, amplitude=.003, channel=0, nsamples=19456)
            
            # Turn on green led
            self.hardware['LEDS']['L'].set(r=0, g=255, b=0)
            self.hardware['LEDS']['R'].set(r=0, g=0, b=0)
            
            # Add a trigger to open the port
            self.triggers['L'].append(self.hardware['PORTS']['L'].open)
            
        elif target == 'child_R':
            self.stim = sounds.Noise(
                duration=100, amplitude=.003, channel=1, nsamples=19456)

            self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
            self.hardware['LEDS']['R'].set(r=0, g=255, b=0)
            
            # Add a trigger to open the port
            self.triggers['R'].append(self.hardware['PORTS']['R'].open)
        
        else:
            self.logger.debug("ignoring target {}".format(target))


        ## Set the stim to repeat
        if self.stim is not None:
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
        threading.Timer(.75, self.stim.play).start()

    def recv_stop(self, value):
        self.logger.debug("recv_stop with value: {}".format(value))
        
        if self.stim is not None:
            self.stim.set_trigger(self.done_playing)
    
        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

    def done_playing(self):
        # This is called when the last stim of the trial has finished playing
        pass

class Wheel_Child(object):
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


class Video_Child(object):
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

class Transformer(object):

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










