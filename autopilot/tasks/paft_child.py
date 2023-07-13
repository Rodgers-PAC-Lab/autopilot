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
import pigpio

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
        self.logger = autopilot.utils.loggers.init_logger(self)
        
        # This is needed when sending messages
        self.n_messages_sent = 0

        # This is set by create_inter_pi_communication_node
        # Initialize to None now, so that if pokes happen before
        # the node is set up, we can catch that and issue a warning
        self.node2 = None
        
        # This is used to keep track of how long we've been running,
        # to estimate the true sample rate
        self.dt_start = datetime.datetime.now()
       
        
        ## Hardware
        self.triggers = {}
        self.init_hardware()
        
        # Set initial poke triggers
        # As soon as this command is run, then pokes will trigger
        # calls to `log_poke` and `report_poke`. So we have to make sure
        # that those don't rely on any of the code in the rest of __init__,
        # which hasn't run yet. 
        self.set_poke_triggers(
            left_punish=False, right_punish=False,
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
        self.frame_rate_warning_already_issued = False
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

        # Use this to keep track of generated sounds
        self.current_audio_times_df = None

        # This is needed when sending messages about generated sounds
        self.n_messages_sent = 0


        ## Set up NET_Node to communicate with Parent
        # Do this after initializing the sounds, otherwise we won't be
        # ready to play yet, and we could receive a PLAY command from
        # the parent as soon as this node is set up. That PLAY command
        # will also add other callbacks that are triggered by pokes, such
        # as reward or punish. 
        self.create_inter_pi_communication_node()
        
        
        ## Keep a link to pigpio.pi() and to the hardware dict
        self.pi = pigpio.pi()
        self.prefs_hardware = prefs.get('HARDWARE')

    def initalize_sounds(self,             
        target_highpass, target_amplitude, target_lowpass,
        distracter_highpass, distracter_amplitude, distracter_lowpass,
        ):
        """Defines sounds that will be played during the task"""
        ## Define sounds
        # Left and right target noise bursts
        self.left_target_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=target_amplitude, channel=0, 
            lowpass=target_lowpass, highpass=target_highpass,
            attenuation_file='/home/pi/attenuation.csv',
            )       

        self.right_target_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=target_amplitude, channel=1, 
            lowpass=target_lowpass, highpass=target_highpass,
            attenuation_file='/home/pi/attenuation.csv',
            )

        # Left and right distracter noise bursts
        self.left_distracter_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=distracter_amplitude, channel=0, 
            lowpass=distracter_lowpass, highpass=distracter_highpass,
            attenuation_file='/home/pi/attenuation.csv',
            )       

        self.right_distracter_stim = autopilot.stim.sound.sounds.Noise(
            duration=10, amplitude=distracter_amplitude, channel=1, 
            lowpass=distracter_lowpass, highpass=distracter_highpass,
            attenuation_file='/home/pi/attenuation.csv',
            )  
            
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
        # Log
        self.logger.debug('set_sound_cycle: received params: {}'.format(params))
        
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
        
        # Save
        self.current_audio_times_df = both_df.copy()
        self.current_audio_times_df = self.current_audio_times_df.rename(
            columns={'time': 'relative_time'})

        
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
        """
        Use the HARDWARE dict that specifies what we need to run the task
        alongside the HARDWARE subdict in :mod:`prefs` to tell us how
        they're plugged in to the pi

        Instantiate the hardware, assign it :meth:`.Task.handle_trigger`
        as a callback if it is a trigger.
        """
        # We use the HARDWARE dict that specifies what we need to run the task
        # alongside the HARDWARE subdict in the prefs structure to tell us how they're plugged in to the pi
        self.hardware = {}
        self.pin_id = {} # Reverse dict to identify pokes
        pin_numbers = prefs.get('HARDWARE')

        # We first iterate through the types of hardware we need
        for type, values in self.HARDWARE.items():
            self.hardware[type] = {}
            # then iterate through each pin and handler of this type
            for pin, handler in values.items():
                # if the hardware is specified as a string, try and get it from registry
                if isinstance(handler, str):
                    handler = autopilot.get_hardware(handler)

                try:
                    hw_args = pin_numbers[type][pin]
                    if isinstance(hw_args, dict):
                        if 'name' not in hw_args.keys():
                            hw_args['name'] = "{}_{}".format(type, pin)
                        hw = handler(**hw_args)
                    else:
                        hw_name = "{}_{}".format(type, pin)
                        hw = handler(hw_args, name=hw_name)

                    # if a pin is a trigger pin (event-based input), give it the trigger handler
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

                except Exception as e:
                    self.logger.exception("Pin could not be instantiated - Type: {}, Pin: {}\nGot exception:{}".format(type, pin, e))

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
        if self.node2 is None:
            # This happens for pokes when we're still in __init__
            self.logger.debug("warning: could not report poke "
                "{} because node2 was not initialized yet".format(poke))
        else:
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

        ## Extract any recently played sound info
        sound_data_l = []
        with autopilot.stim.sound.jackclient.QUEUE_NONZERO_BLOCKS_LOCK:
            while True:
                try:
                    data = autopilot.stim.sound.jackclient.QUEUE_NONZERO_BLOCKS.get_nowait()
                except queue.Empty:
                    break
                sound_data_l.append(data)
        
        if len(sound_data_l) > 0:
            # DataFrame it
            # This has to match code in jackclient.py
            # And it also has to match task_class.ChunkData_SoundsPlayed
            payload = pandas.DataFrame.from_records(
                sound_data_l,
                columns=['hash', 'last_frame_time', 'frames_since_cycle_start', 'equiv_dt'],
                )
            self.send_chunk_of_sound_played_data(payload)
        
        # Estimate how fast we're playing sounds
        # This should be about 192000 / 1024 = 187.5 frames / s, 
        # although it will be a bit more because some the queue itself holds
        # 200 frames, and those are all deleted at the beginning of each trial
        # without being played. 
        # However, if there is the sample rate bug, this will be more like
        # 31 (empirically), about 6x less, not sure what this corresponds to,
        # maybe a true sample rate of 32 kHz?
        time_so_far = (datetime.datetime.now() - self.dt_start).total_seconds()
        frame_rate = self.n_frames / time_so_far
        self.logger.debug("info: "
            "added {} frames in {:.1f} s for a rate of {:.2f} frames/s".format(
            self.n_frames, time_so_far, frame_rate))
        
        # Warn if this is happening (but just once per session)
        # Really I would prefer that it inform the user to restart the pi
        # TODO: send some kind of "help/error" message to the plot
        if frame_rate < 150 and not self.frame_rate_warning_already_issued:
            self.logger.debug("error: frame rate seems to be far too low")
            self.frame_rate_warning_already_issued = True


        ## Continue to the next stage (which is this one again)
        # If it is cleared, then nothing happens until the next message
        # from the Parent (not sure why)
        # If we never end this function, then it won't respond to END
        self.stage_block.set()

    def send_chunk_of_sound_played_data(self, payload):
        """Report metadata about sounds played to parent
        
        This is adapted from send_chunk_of_sound_data in paft_child
        Here we send data from jackclient about the nonzero frames
        """
        ## Create a serialized message
        # Adapted from the bandwidth test
        
        # Time of sending this message
        timestamp = datetime.datetime.now().isoformat()
        
        # Only send data if there are rows of data
        # Otherwise, no sound playing, and nothing to report
        if len(payload) > 0:
            # Store these additional values, which are the same for all rows
            payload['pilot'] = self.name
            
            # This is the value to send
            # Must be serializable
            # Definitely include payload (the data), some kind of locking
            # timestamp, and the origin (our name). 
            # When this message is repeated by the Parent to the terminal,
            # there are additional constraints based on what save_data expects
            value = {
                'pilot': self.name,
                'payload': payload.values,
                'payload_columns': payload.columns.values,
                'chunkclass_name': 'ChunkData_SoundsPlayed', 
                'timestamp': timestamp,
            }        
            
            # Construct the message
            msg = autopilot.networking.Message(
                id="dummy_dst2", # does nothing (?), but required
                sender="dummy_src2", # does nothing (?), but required 
                key='CHUNK', # this selects listen method. required for encoding
                to="dummy_dst", # required but I don't think it matters
                value=value, # the 'value' to send
                flags={
                    'MINPRINT': True, # disable printing of value
                    'NOREPEAT': True, # disable repeating
                    },
                )
            
            # Sending it will automatically serialize it, which in turn will
            # automatically compress numpy using blosc
            # See Node.send and Message.serialize
            self.node2.send('parent_pi', msg=msg)

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
        # Use this to determine when the flash was done in local timebase
        timestamp = datetime.datetime.now().isoformat()

        # Whether to do a synchronization flash
        synchronization_flash = value.pop('synchronization_flash', False)

        # Blink an LED to serve as synchronization cue
        # Do this BEFORE processing any sounds, otherwise the latency varies
        # with the number of sounds to play
        if synchronization_flash:
            # get pin numbers
            left_red = autopilot.hardware.BOARD_TO_BCM[ 
                self.prefs_hardware['LEDS']['L']['pins'][0]]
            right_red = autopilot.hardware.BOARD_TO_BCM[ 
                self.prefs_hardware['LEDS']['R']['pins'][0]]
            
            # Turn left-red and right-red to full PWM
            self.pi.set_PWM_dutycycle(left_red, 255)
            self.pi.set_PWM_dutycycle(right_red, 255)
            
            # Wait 100 ms
            time.sleep(.100)
            
            # Turn left-red and right-red to zero PWM
            self.pi.set_PWM_dutycycle(left_red, 0)
            self.pi.set_PWM_dutycycle(right_red, 0)

            # Send to the parent
            self.node2.send(
                'parent_pi', 'FLASH', {
                    'from': self.name, 
                    'dt_flash_received': timestamp,
                    },
                )      
        
        # Log the time of the flash
        # Do this after the flash itself so that we don't jitter
        self.logger.debug(
            "[{}] synchronization_flash; ".format(timestamp) +
            "recv_play with value: {}".format(value)
            )
    
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
        
        # Inform terminal
        self.send_chunk_of_sound_data()

    def send_chunk_of_sound_data(self):
        ## Create a serialized message
        # Adapted from the bandwidth test
        
        # Use this as locking_timestamp, to which all `audio_time` timestamps
        # are relative
        timestamp = datetime.datetime.now().isoformat()
        
        # Get the payload, which was saved when the audio times were generated
        # This has to match task_class.ChunkData
        payload = self.current_audio_times_df
        
        # Only send data if there are rows of data
        # Otherwise, no sound playing, and nothing to report
        if len(payload) > 0:
            # Store these additional values, which are the same for all rows
            payload['pilot'] = self.name
            payload['locking_timestamp'] = timestamp
            
            # This is the value to send
            # Must be serializable
            # Definitely include payload (the data), some kind of locking
            # timestamp, and the origin (our name). 
            # When this message is repeated by the Parent to the terminal,
            # there are additional constraints based on what save_data expects
            value = {
                'pilot': self.name,
                'payload': payload.values,
                'payload_columns': payload.columns.values,
                'chunkclass_name': 'ChunkData_Sounds', 
                'timestamp': timestamp,
            }        
            
            # Construct the message
            msg = autopilot.networking.Message(
                id="{}-{}".format(self.name, self.n_messages_sent), # must be unique
                sender="dummy_src", # required but I don't think it matters
                key='CHUNK', # this selects listen method. required for encoding
                to="dummy_dst", # required but I don't think it matters
                value=value, # the 'value' to send
                flags={
                    'MINPRINT': True, # disable printing of value
                    'NOREPEAT': True, # disable repeating
                    },
                )
            
            # Sending it will automatically serialize it, which in turn will
            # automatically compress numpy using blosc
            # See Node.send and Message.serialize
            self.node2.send('parent_pi', msg=msg)

            # Increment this counter to keep the message id unique
            self.n_messages_sent = self.n_messages_sent + 1        
        
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
