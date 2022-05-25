from collections import OrderedDict as odict
import time
import functools
import datetime
import itertools
import queue
import pandas
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

    def __init__(self, stage_block, task_type, subject, child, reward,
        ):
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
        
        # Set initial poke triggers
        self.set_poke_triggers(left_punish=False, right_punish=False,
            left_reward=False, right_reward=False)

        # Set reward values for solenoids
        for port_name, port in self.hardware['PORTS'].items():
            self.logger.debug(
                "setting reward for {} to {}".format(port_name, reward))
            port.duration = float(reward)


        ## Stages
        # Only one stage
        self.stages = itertools.cycle([self.play])
        self.stage_block = stage_block
        
        # This is used to ensure only one reward per trial
        # It is set to False as soon as the "port open" trigger is added
        # And it is set to True as soon as the port is opened
        # The port will not open if the flag is True
        # As written, this wouldn't allow rewarding multiple ports
        self.reward_already_dispensed_on_this_trial = True
        
        
        ## Initialize sounds
        # Each block/frame is about 5 ms
        # Longer is more buffer against unexpected delays
        # Shorter is faster to empty and refill the queue
        self.target_qsize = 200

        # Some counters to keep track of how many sounds we've played
        self.n_frames = 0
        self.n_error_counter = 0        

        # Initialize these to None, in case they don't get defined later
        # (e.g., poketrain)
        self.left_error_sound = None
        self.right_error_sound = None

        # Fill the queue with empty frames
        # Sounds aren't initialized till the trial starts
        # Using False here should work even without sounds initialized yet
        self.set_sound_cycle(params={'left_on': False, 'right_on': False})


        ## Set up NET_Node to communicate with Parent
        # Do this after initializing the sounds, otherwise we won't be
        # ready to play yet
        self.create_inter_pi_communication_node()

    def initalize_sounds(self,             
        target_highpass, target_amplitude, target_lowpass,
        distracter_highpass, distracter_amplitude, distracter_lowpass,
        ):
        """Defines sounds that will be played during the task"""
        ## Define sounds
        # Left and right target noise bursts
        self.left_target_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=target_amplitude, channel=0, 
            lowpass=target_lowpass, highpass=target_highpass)       

        self.right_target_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=target_amplitude, channel=1, 
            lowpass=target_lowpass, highpass=target_highpass)        

        # Left and right distracter noise bursts
        self.left_distracter_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=distracter_amplitude, channel=0, 
            lowpass=distracter_lowpass, highpass=distracter_highpass)       

        self.right_distracter_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=distracter_amplitude, channel=1, 
            lowpass=distracter_lowpass, highpass=distracter_highpass)  
            
        # Left and right tritone error noises
        self.left_error_sound = autopilot.stim.sound.sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=0)

        self.right_error_sound = autopilot.stim.sound.sounds.Tritone(
            frequency=8000, duration=250, amplitude=.003, channel=1)
        
        # Chunk the sounds into frames
        if not self.left_target_stim.chunks:
            self.left_target_stim.chunk()
        if not self.right_target_stim.chunks:
            self.right_target_stim.chunk()
        if not self.left_distracter_stim.chunks:
            self.left_distracter_stim.chunk()
        if not self.right_distracter_stim.chunks:
            self.right_distracter_stim.chunk()
        if not self.left_error_sound.chunks:
            self.left_error_sound.chunk()
        if not self.right_error_sound.chunks:
            self.right_error_sound.chunk()
    
    def set_sound_cycle(self, params):
        """Define self.sound_cycle, to go through sounds
        
        params : dict
            This comes from a message on the net node.
            Possible keys:
                left_on
                right_on
                left_mean_interval
                right_mean_interval
        """
        print(params)
        
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

        # Extract params or use defaults
        left_on = params.get('left_on', False)
        right_on = params.get('right_on', False)
        left_target_rate = params.get('left_target_rate', 0)
        right_target_rate = params.get('right_target_rate', 0)
        left_distracter_rate = params.get('left_distracter_rate', 0)
        right_distracter_rate = params.get('right_distracter_rate', 0)
        
        # Global params
        target_temporal_std = 10 ** params.get(
            'stim_target_temporal_log_std', -2)
        distracter_temporal_std = 10 ** params.get(
            'stim_distracter_temporal_log_std', -2)
       
        
        ## Generate intervals 
        # left target
        if left_on and left_target_rate > 1e-3:
            # Change of basis
            mean_interval = 1 / left_target_rate
            var_interval = target_temporal_std ** 2

            # Change of basis
            gamma_shape = (mean_interval ** 2) / var_interval
            gamma_scale = var_interval / mean_interval

            # Draw
            left_target_intervals = np.random.gamma(
                gamma_shape, gamma_scale, 100)
        else:
            left_target_intervals = np.array([])

        # right target
        if right_on and right_target_rate > 1e-3:
            # Change of basis
            mean_interval = 1 / right_target_rate
            var_interval = target_temporal_std ** 2

            # Change of basis
            gamma_shape = (mean_interval ** 2) / var_interval
            gamma_scale = var_interval / mean_interval

            # Draw
            right_target_intervals = np.random.gamma(
                gamma_shape, gamma_scale, 100)
        else:
            right_target_intervals = np.array([])     

        # left distracter
        if left_on and left_distracter_rate > 1e-3:
            # Change of basis
            mean_interval = 1 / left_distracter_rate
            var_interval = distracter_temporal_std ** 2

            # Change of basis
            gamma_shape = (mean_interval ** 2) / var_interval
            gamma_scale = var_interval / mean_interval

            # Draw
            left_distracter_intervals = np.random.gamma(
                gamma_shape, gamma_scale, 100)
        else:
            left_distracter_intervals = np.array([])

        # right distracter
        if right_on and right_distracter_rate > 1e-3:
            # Change of basis
            mean_interval = 1 / right_distracter_rate
            var_interval = distracter_temporal_std ** 2

            # Change of basis
            gamma_shape = (mean_interval ** 2) / var_interval
            gamma_scale = var_interval / mean_interval

            # Draw
            right_distracter_intervals = np.random.gamma(
                gamma_shape, gamma_scale, 100)
        else:
            right_distracter_intervals = np.array([])               
        
        
        ## Sort all the drawn intervals together
        # Turn into series
        left_target_df = pandas.DataFrame.from_dict({
            'time': np.cumsum(left_target_intervals),
            'side': ['left'] * len(left_target_intervals),
            'sound': ['target'] * len(left_target_intervals),
            })
        right_target_df = pandas.DataFrame.from_dict({
            'time': np.cumsum(right_target_intervals),
            'side': ['right'] * len(right_target_intervals),
            'sound': ['target'] * len(right_target_intervals),
            })
        left_distracter_df = pandas.DataFrame.from_dict({
            'time': np.cumsum(left_distracter_intervals),
            'side': ['left'] * len(left_distracter_intervals),
            'sound': ['distracter'] * len(left_distracter_intervals),
            })
        right_distracter_df = pandas.DataFrame.from_dict({
            'time': np.cumsum(right_distracter_intervals),
            'side': ['right'] * len(right_distracter_intervals),
            'sound': ['distracter'] * len(right_distracter_intervals),
            })
        
        # Concatenate them all together and resort by time
        both_df = pandas.concat([
            left_target_df, right_target_df, 
            left_distracter_df, right_distracter_df,
            ], axis=0).sort_values('time')

        # Calculate the gap between sounds
        both_df['gap'] = both_df['time'].diff().shift(-1)
        
        # Drop the last row which has a null gap
        both_df = both_df.loc[~both_df['gap'].isnull()].copy()

        # Keep only those below the sound cycle length
        both_df = both_df.loc[both_df['time'] < 10].copy()
        
        # Nothing should be null
        assert not both_df.isnull().any().any() 

        # Calculate gap size in chunks
        both_df['gap_chunks'] = (both_df['gap'] *
            autopilot.stim.sound.jackclient.FS / 
            autopilot.stim.sound.jackclient.BLOCKSIZE)
        both_df['gap_chunks'] = both_df['gap_chunks'].round().astype(np.int)
        
        # Floor gap_chunks at 1 chunk, the minimal gap size
        # This is to avoid distortion
        both_df.loc[both_df['gap_chunks'] < 1, 'gap_chunks'] = 1
        
        # Log
        self.logger.debug("generated both_df: {}".format(both_df))
        
        
        ## Depends on how long both_df is
        # If both_df has a nonzero but short length, results will be weird,
        # because it might just be one noise burst repeating every ten seconds
        # This only happens with low rates ~0.1Hz
        if len(both_df) == 0:
            # If no sound, then just put gaps
            append_gap(100)
        else:
            # Iterate through the rows, adding the sound and the gap
            # TODO: the gap should be shorter by the duration of the sound,
            # and simultaneous sounds should be possible
            for bdrow in both_df.itertuples():
                # Append the sound
                if bdrow.side == 'left' and bdrow.sound == 'target':
                    for frame in self.left_target_stim.chunks:
                        self.sound_block.append(frame) 
                elif bdrow.side == 'left' and bdrow.sound == 'distracter':
                    for frame in self.left_distracter_stim.chunks:
                        self.sound_block.append(frame)                         
                elif bdrow.side == 'right' and bdrow.sound == 'target':
                    for frame in self.right_target_stim.chunks:
                        self.sound_block.append(frame) 
                elif bdrow.side == 'right' and bdrow.sound == 'distracter':
                    for frame in self.right_distracter_stim.chunks:
                        self.sound_block.append(frame)       
                else:
                    raise ValueError(
                        "unrecognized side and sound: {} {}".format(
                        bdrow.side, bdrow.sound))
                
                # Append the gap
                append_gap(bdrow.gap_chunks)
        
        
        ## Cycle so it can repeat forever
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

    def set_poke_triggers(self, left_punish, right_punish, 
        left_reward, right_reward):
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
            
        if left_punish:
            # Punish left
            self.triggers['L'].append(functools.partial(
                self.append_error_sound_to_queue2, 'left'))
        
        if left_reward:
            # Reward left
            self.reward_already_dispensed_on_this_trial = False
            self.triggers['L'].append(self.reward_left_once)
        
        if right_punish:
            # Punish right
            self.triggers['R'].append(functools.partial(
                self.append_error_sound_to_queue2, 'right'))
        
        if right_reward:
            # Reward right
            self.reward_already_dispensed_on_this_trial = False
            self.triggers['R'].append(self.reward_right_once)

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
    
    def reward_left_once(self):
        """Reward left port. Set flag so we don't reward again till next trial
        
        """
        if not self.reward_already_dispensed_on_this_trial:
            # Get the time
            reward_timestamp = datetime.datetime.now()
            
            # Set the flag so we don't reward twice
            self.reward_already_dispensed_on_this_trial = True
            
            # Log
            self.logger.debug('[{}] rewarding left port'.format(
                datetime.datetime.now().isoformat()))
            
            # Send to the parent
            self.node2.send(
                'parent_pi', 'REWARD', {'from': self.name, 'poke': 'L'},
                )         
            
            # Open the port
            self.hardware['PORTS']['L'].open()
    
    def reward_right_once(self):
        """Reward right port. Set flag so we don't reward again till next trial
        
        """
        if not self.reward_already_dispensed_on_this_trial:
            # Get the time
            reward_timestamp = datetime.datetime.now()
            
            # Set the flag so we don't reward twice
            self.reward_already_dispensed_on_this_trial = True
            
            # Log
            self.logger.debug('[{}] rewarding right port'.format(
                datetime.datetime.now().isoformat()))
            
            # Send to the parent
            self.node2.send(
                'parent_pi', 'REWARD', {'from': self.name, 'poke': 'R'},
                )         
            
            # Open the port
            self.hardware['PORTS']['R'].open()
    
    def play(self):
        """A single stage"""
        # Don't want to do a "while True" here, because we need to exit
        # this method eventually, so that it can respond to END
        # But also don't want to change stage too frequently or the debug
        # messages are overwhelming
        for n in range(10):
            # Add stimulus sounds to queue 1 as needed
            self.append_sound_to_queue1_as_needed()

            # Don't want to iterate too quickly, but rather add chunks
            # in a controlled fashion every so often
            time.sleep(.1)

        # Continue to the next stage (which is this one again)
        # If it is cleared, then nothing happens until the next message
        # from the Parent (not sure why)
        # If we never end this function, then it won't respond to END
        self.stage_block.set()

    def empty_queue1(self, tosize=0):
        """Empty queue1"""
        while True:
            # I think it's important to keep the lock for a short period
            # (ie not throughout the emptying)
            # in case the `process` function needs it to play sounds
            # (though if this does happen, there will be an artefact because
            # we just skipped over a bunch of frames)
            with autopilot.stim.sound.jackclient.Q_LOCK:
                try:
                    data = autopilot.stim.sound.jackclient.QUEUE.get_nowait()
                except queue.Empty:
                    break
            
            # Stop if we're at or below the target size
            qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
            if qsize < tosize:
                break
        
        qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
        self.logger.debug('empty_queue1: new size {}'.format(qsize))

    def empty_queue2(self):
        """Empty queue2"""
        while True:
            with autopilot.stim.sound.jackclient.Q2_LOCK:
                try:
                    data = autopilot.stim.sound.jackclient.QUEUE2.get_nowait()
                except queue.Empty:
                    break
    
    def append_sound_to_queue1_as_needed(self):
        """Dump frames from `self.sound_cycle` into queue

        The queue is filled until it reaches `self.target_qsize`

        This function should be called often enough that the queue is never
        empty.
        """        
        # TODO: as a figure of merit, keep track of how empty the queue gets
        # between calls. If it's getting too close to zero, then target_qsize
        # needs to be increased.
        # Get the size of QUEUE1 now
        qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
        if qsize == 0:
            self.logger.debug('warning: queue1 was empty')
        #~ self.logger.debug(
            #~ 'append_sound_to_queue1_as_needed: qsize before = {}'.format(qsize))

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
        
        #~ # Get the size of QUEUE1 now
        #~ qsize = autopilot.stim.sound.jackclient.QUEUE.qsize()
        #~ self.logger.debug(
            #~ 'append_sound_to_queue1_as_needed: qsize after = {}'.format(qsize))        

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

        # Empty queue2, because there's no reason to accumulate error sounds
        # over time
        self.empty_queue2()
        self.logger.debug('play_error_sound: emptied queue2')
        
        # Add sounds from the appropriate error sound to QUEUE2
        # The guards here for None are in case no error sound was defined
        # (e.g., poketrain)
        with autopilot.stim.sound.jackclient.Q2_LOCK:
            if which == 'left':
                if self.left_error_sound is not None:
                    for frame in self.left_error_sound.chunks:
                        autopilot.stim.sound.jackclient.QUEUE2.put_nowait(frame)
            elif which == 'right':
                if self.right_error_sound is not None:
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
        """On receiving a PLAY command, set sounds and fill queues"""
        # Log 
        self.logger.debug("recv_play with value: {}".format(value))
        
        # Pop out the punish and reward values
        left_punish = value.pop('left_punish')
        right_punish = value.pop('right_punish')
        left_reward = value.pop('left_reward')
        right_reward = value.pop('right_reward')        

        # Only get these params if sound is supposed to play
        # silence_pi doesn't include them
        if value['left_on'] or value['right_on']:
            # Pop out the sound definition values
            target_center_freq = value.pop('stim_target_center_freq')
            target_bandwidth = value.pop('stim_target_bandwidth')
            target_amplitude = 10 ** value.pop('stim_target_log_amplitude')
            distracter_center_freq = value.pop('stim_distracter_center_freq')
            distracter_bandwidth = value.pop('stim_distracter_bandwidth')
            distracter_amplitude = 10 ** value.pop('stim_distracter_log_amplitude')

            # Convert center and bandwidth to lowpass and highpass
            target_highpass = target_center_freq - target_bandwidth / 2
            target_lowpass = target_center_freq + target_bandwidth / 2
            distracter_highpass = distracter_center_freq - distracter_bandwidth / 2
            distracter_lowpass = distracter_center_freq + distracter_bandwidth / 2

            # Define the sounds that will be used in the cycle
            self.initalize_sounds(        
                target_highpass, target_amplitude, target_lowpass,
                distracter_highpass, distracter_amplitude, distracter_lowpass,
                )
        
        # Use left_punish and right_punish to set triggers
        self.set_poke_triggers(
            left_punish=left_punish, right_punish=right_punish,
            left_reward=left_reward, right_reward=right_reward)
        
        # Use the remaining params to update the sound cycle
        self.set_sound_cycle(value)
        
        # Empty queue1 and refill
        self.empty_queue1()
        self.append_sound_to_queue1_as_needed()
        
    def recv_stop(self, value):
        # debug
        self.logger.debug("recv_stop with value: {}".format(value))

    def end(self, *args, **kwargs):
        """This is called when the STOP signal is received from the parent"""
        self.logger.debug("Inside the self.end function")

        # Remove all triggers
        # Otherwise pokes can still trigger network events on closed sockets
        #~ self.triggers = {}

        # Explicitly close the socket (helps with restarting cleanly)
        self.node2.sock.close()
        self.node2.release()
        
        # Release hardware. No superclass to do this for us.
        for k, v in self.hardware.items():
            for pin, obj in v.items():
                obj.release()
