from collections import OrderedDict as odict
import time
import functools
import datetime
import itertools
import numpy as np
import autopilot
from autopilot import prefs
from . import children

class PAFT_Child(children.Child):
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
        self.logger = autopilot.core.loggers.init_logger(self)


        ## Hardware
        self.triggers = {}
        self.init_hardware()
        self.set_poke_triggers()


        ## Stages
        # Only one stage
        self.stages = itertools.cycle([self.play])
        self.stage_block = stage_block
        
        
        ## Set up NET_Node to communicate with Parent
        self.create_inter_pi_communication_node()


        ## Initialize sounds
        # Define the sounds that will be used in the cycle
        self.initalize_sounds()
        
        # Define a cycle of those sounds
        self.set_sound_cycle()

        # Each block/frame is about 5 ms, so this is about 5 s of data
        # Longer is more buffer against unexpected delays
        self.target_qsize = 1000

        # Some counters to keep track of how many sounds we've played
        self.n_frames = 0
        self.n_error_counter = 0        

    def initalize_sounds(self):
        """Defines sounds that will be played during the task"""
        ## Define sounds
        # Left and right noise bursts
        self.left_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=.01, channel=0, 
            highpass=5000)       

        self.right_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=.01, channel=1, 
            highpass=5000)        
        
        # Left and right tritone error noises
        self.left_error_sound = autopilot.stim.sound.sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=0)

        self.right_error_sound = autopilot.stim.sound.sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=1)
        
        # Chunk the sounds into frames
        if not self.left_stim.chunks:
            self.left_stim.chunk()
        if not self.right_stim.chunks:
            self.right_stim.chunk()
        if not self.left_error_sound.chunks:
            self.left_error_sound.chunk()
        if not self.right_error_sound.chunks:
            self.right_error_sound.chunk()
    
    def set_sound_cycle(self, left_on=False, right_on=False):
        """Define self.sound_cycle, to go through sounds"""
        # This is just a left sound, gap, then right sound, then gap
        # And use a cycle to repeat forever
        # But this could be made more complex
        self.sound_block = []

        # Helper function
        def append_gap(gap_chunk_size=30):
            """Append `gap_chunk_size` silent chunks to sound_block"""
            for n_blank_chunks in range(gap_chunk_size):
                self.sound_block.append(
                    np.zeros(autopilot.stim.sound.jackclient.BLOCKSIZE, 
                    dtype='float32'))

        # If left_on, append left sound and gap
        if left_on:
            for frame in self.left_stim.chunks:
                self.sound_block.append(frame)
            append_gap()
        
        # If right_on, append right sound and gap
        if right_on:
            for frame in self.right_stim.chunks:
                self.sound_block.append(frame)
            append_gap()
        
        # If nothing else, append gap (so it's not empty)
        if len(self.sound_block) == 0:
            append_gap()
        
        # Cycle so it can repeat forever
        self.sound_cycle = itertools.cycle(self.sound_block)        

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
                'PLAY': self.recv_play,
                'STOP': self.recv_stop,
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

    def set_poke_triggers(self, left_punish=False, right_punish=False):
        """"Set triggers for poke entry
        
        For each poke, sets these triggers:
            self.log_poke (write to own debugger)
            self.report_poke (report to parent)
        """
        for poke in ['L', 'R']:
            # Always trigger reporting pokes
            self.triggers[poke] = [
                functools.partial(self.log_poke, poke),
                functools.partial(self.report_poke, poke),
                ]       
            
            # Trigger punish sounds
            if left_punish:
                self.triggers['L'].append(functools.partial(
                    self.append_error_sound_to_queue2, 'left')))
            elif right_punish:
                self.triggers['R'].append(functools.partial(
                    self.append_error_sound_to_queue2, 'right'))
        
            # TODO: also trigger rewards here

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

        
        ## Add sounds to queues
        # Add stimulus sounds to queue 1 as needed
        self.append_sound_to_queue1_as_needed()
        

        ## Start it playing
        # The play event is cleared if it ever runs out of sound, which
        # ideally doesn't happen
        autopilot.stim.sound.jackclient.PLAY.set()


        ## Sleep so we don't go crazy
        time.sleep(1)

        # Continue to the next stage (which is this one again)
        self.stage_block.set()

    def append_sound_to_queue1_as_needed(self):
        # Get the size of QUEUE1 now
        qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
        self.logger.debug(
            'append_sound_to_queue1_as_needed: qsize before = {}'.format(qsize))

        # Add frames until target size reached
        while qsize < self.target_qsize:
            with autopilot.stim.sound.jackclient.Q_LOCK:
                # Add a frame from the sound cycle
                frame = next(self.sound_cycle)
                autopilot.stim.sound.jackclient.QUEUE.put_nowait(frame)
                
                # Keep track of how many frames played
                self.n_frames = self.n_frames + 1
            
            # Update qsize
            qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
        
        # Get the size of QUEUE1 now
        qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
        self.logger.debug(
            'append_sound_to_queue1_as_needed: qsize after = {}'.format(qsize))        

    def append_error_sound_to_queue2(self, which):
        """Dump frames from error sound into queue2
        
        Since queue2 is only sporadically used, they will likely be
        played immediately.
        
        TODO: empty queue2 before adding error sound .. no reason to 
        accumulate error sounds over time.
        """
        # Get the size of QUEUE2 now
        qsize = autopilot.stim.sound.jackclient.QUEUE2.qsize()
        self.logger.debug('play_error_sound: qsize before = {}'.format(qsize))

        # Add sounds from the appropriate error sound to QUEUE2
        with autopilot.stim.sound.jackclient.Q2_LOCK:
            if which == 'left':
                for frame in self.left_error_sound.chunks:
                    autopilot.stim.sound.jackclient.QUEUE2.put_nowait(frame)
            elif which == 'right':
                for frame in self.right_error_sound.chunks:
                    autopilot.stim.sound.jackclient.QUEUE2.put_nowait(frame)
            else:
                raise ValueError("unrecognized which: {}".format(which))

        # Get the size of QUEUE2 afterward
        qsize = autopilot.stim.sound.jackclient.QUEUE2.qsize()
        self.logger.debug('play_error_sound: qsize after = {}'.format(qsize))
    
    def recv_hello(self, value):
        """This is probably unnecessary"""
        self.logger.debug(
            "received HELLO from parent with value {}".format(value))

    def recv_play(self, value):
        self.logger.debug("recv_play with value: {}".format(value))
        
        # Extract which pokes are punished
        left_punish = value['left_punish']
        right_punish = value['right_punish']
        
        # Use this to set triggers
        self.set_poke_triggers(
            left_punish=left_punish, right_punish=right_punish)
        
        # Extract which speakers are active
        left_on = value['left_on']
        right_on = value['right_on']
        
        # Use this to update the sound cycle
        self.set_sound_cycle(left_on=left_on, right_on=right_on)
        
    def recv_stop(self, value):
        # debug
        self.logger.debug("recv_stop with value: {}".format(value))

    def end(self):
        """This is called when the STOP signal is received from the parent"""
        self.logger.debug("Inside the self.end function")

        # Explicitly close the socket (helps with restarting cleanly)
        self.node2.sock.close()
        self.node2.release()
