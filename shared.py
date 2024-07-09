import datetime
import jack
import pigpio
import threading

class SoundPlayer(object):
    """Object to play sounds"""
    def __init__(self, name='jack_client', audio_cycle=None):
        """Initialize a new JackClient

        This object contains a jack.Client object that actually plays audio.
        It provides methods to send sound to its jack.Client, notably a 
        `process` function which is called every 5 ms or so.
        
        name : str
            Required by jack.Client
        
        audio_cycle : iter
            Should produce a frame of audio on request
        
        This object should focus only on playing sound as precisely as
        possible.
        """
        ## Store provided parameters
        self.name = name
        
        ## Acoustic parameters of the sound
        # TODO: define these elsewhere -- these should not be properties of
        # this object, because this object should be able to play many sounds
        
        # Lock for thread-safe set_channel() updates
        self.lock = threading.Lock()  
        
        
        ## Create the contained jack.Client
        # Creating a jack client
        self.client = jack.Client(self.name)

        # Pull these values from the initialized client
        # These come from the jackd daemon
        # `blocksize` is the number of samples to provide on each `process`
        # call
        self.blocksize = self.client.blocksize
        
        # `fs` is the sampling rate
        self.fs = self.client.samplerate
        
        # Debug message
        # TODO: add control over verbosity of debug messages
        print("Received blocksize {} and fs {}".format(self.blocksize, self.fs))

        
        ## Set up outchannels
        self.client.outports.register('out_0')
        self.client.outports.register('out_1')


        ## Set up the process callback
        # This will be called on every block and must provide data
        self.client.set_process_callback(self.process)

        
        ## Activate the client
        self.client.activate()


        ## Hook up the outports (data sinks) to physical ports
        # Get the actual physical ports that can play sound
        target_ports = self.client.get_ports(
            is_physical=True, is_input=True, is_audio=True)
        assert len(target_ports) == 2

        # Connect virtual outport to physical channel
        self.client.outports[0].connect(target_ports[0])
        self.client.outports[1].connect(target_ports[1])

    def process(self, frames):
        """Process callback function (used to play sound)
        
        TODO: reimplement this to use a queue instead
        The current implementation uses time.time(), but we need to be more
        precise.
        """
        # Making process() thread-safe (so that multiple calls don't try to
        # write to the outports at the same time)
        with self.lock: 
            # Get data from cycle
            data = next(audio_cycle)

            # Error check
            assert data.shape[1] == 2

            # Write one column to each channel
            for n_outport, outport in enumerate(self.client.outports):
                buff = outport.get_array()
                buff[:] = data[:, n_outport]

class WheelListener(object):
    def __init__(self, pi):
        # Global variables
        self.pi = pi
        self.position = 0
        self.event_log = []
        self.state_log = []
        self.a_state = 0
        self.b_state = 0
        
        self.pi.callback(17, pigpio.RISING_EDGE, self.pulseA_detected)
        self.pi.callback(27, pigpio.RISING_EDGE, self.pulseB_detected)
        self.pi.callback(17, pigpio.FALLING_EDGE, self.pulseA_down)
        self.pi.callback(27, pigpio.FALLING_EDGE, self.pulseB_down)
        
    def pulseA_detected(self, pin, level, tick):
        self.event_log.append('A')
        self.a_state = 1
        if self.b_state == 0:
            self.position += 1
        else:
            self.position -= 1
        self.state_log.append(
            '{}{}_{}'.format(self.a_state, self.b_state, self.position))

    def pulseB_detected(self, pin, level, tick):
        self.event_log.append('B')
        self.b_state = 1
        if self.a_state == 0:
            self.position -= 1
        else:
            self.position += 1
        self.state_log.append(
            '{}{}_{}'.format(self.a_state, self.b_state, self.position))

    def pulseA_down(self, pin, level, tick):
        self.event_log.append('a')
        self.a_state = 0
        if self.b_state == 0:
            self.position -= 1
        else:
            self.position += 1
        self.state_log.append(
            '{}{}_{}'.format(self.a_state, self.b_state, self.position))

    def pulseB_down(self, pin, level, tick):
        self.event_log.append('b')
        self.b_state = 0
        if self.a_state == 0:
            self.position += 1
        else:
            self.position -= 1
        self.state_log.append(
            '{}{}_{}'.format(self.a_state, self.b_state, self.position))

    def do_nothing(self):
        print("current position: {}".format(self.position))
        print('events: ' + ''.join(self.event_log[-60:]))
        print('states: ' + '\t'.join(self.state_log[-4:]))

class TouchListener(object):
    def __init__(self, pi, debug_print=False):
        """Initialize a new TouchListener
        
        pi : pigpio.pi
        debug_print : bool
            If True, then print messages on every touch
        """
        # Store pi and debug_print
        self.pi = pi
        self.debug_print = debug_print
        self.touch_trigger = None
        self.touch_trigger_refractory_period = 1
        self.touch_trigger_last_time = datetime.datetime.now()
        
        self.last_touch = datetime.datetime.now()
        self.touch_state = False

        self.pi.set_mode(16, pigpio.INPUT)
        self.pi.callback(16, pigpio.RISING_EDGE, self.touch_happened)
        self.pi.callback(16, pigpio.FALLING_EDGE, self.touch_stopped)

    def touch_happened(self, pin, level, tick):
        """A touch started
        
        Sets self.last_touch to now, and self.touch_state to True
        """
        touch_time = datetime.datetime.now()
        self.last_touch = touch_time
        self.touch_state = True

        # Call the trigger if it has been set
        if self.touch_trigger is not None:
            if (
                touch_time - self.touch_trigger_last_time > 
                datetime.timedelta(seconds=self.touch_trigger_refractory_period)
                ):
        
                # Call the trigger
                self.touch_trigger()
                
                # Set the time
                self.touch_trigger_last_time = touch_time

        # Debug
        if self.debug_print:
            print('touch start tick={} dt={}'.format(tick, touch_time))
    
    def touch_stopped(self, pin, level, tick):
        """A touch stopped
        
        Sets self.last_touch to now, and self.touch_state to False
        """        
        touch_time = datetime.datetime.now()
        self.last_touch = touch_time
        self.touch_state = False

        # Debug
        if self.debug_print:
            print('touch stop  tick={} dt={}'.format(tick, touch_time))

    def report(self):
        print("touch state={}; last_touch={}".format(
            self.touch_state, self.last_touch))
