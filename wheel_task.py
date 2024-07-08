import pigpio
import time
import datetime
import jack

class JackClient(object):
    def __init__(self, name='jack_client', outchannels=None):
        self.name = name

        # Create jack client
        self.client = jack.Client(self.name)

        # Pull these values from the initialized client
        # These comes from the jackd daemon
        self.blocksize = self.client.blocksize
        self.fs = self.client.samplerate
        print("received blocksize {} and fs {}".format(self.blocksize, self.fs))

        # Set the number of output channels
        if outchannels is None:
            self.outchannels = [0, 1]
        else:
            self.outchannels = outchannels

        # Set mono_output
        if len(self.outchannels) == 1:
            self.mono_output = True
        else:
            self.mono_output = False

        # Register outports
        if self.mono_output:
            # One single outport
            self.client.outports.register('out_0') #include this
        else:
            # One outport per provided outchannel
            for n in range(len(self.outchannels)):
                self.client.outports.register('out_{}'.format(n))

        # Process callback to self.process
        self.client.set_process_callback(self.process)

        # Activate the client
        self.client.activate()

        ## Hook up the outports (data sinks) to physical ports
        # Get the actual physical ports that can play sound
        target_ports = self.client.get_ports(
            is_physical=True, is_input=True, is_audio=True)

        # Depends on whether we're in mono mode
        if self.mono_output:
            ## Mono mode
            # Hook up one outport to all channels
            for target_port in target_ports:
                self.client.outports[0].connect(target_port)
        
        else:
            ## Not mono mode
            # Error check
            if len(self.outchannels) > len(target_ports):
                raise ValueError(
                    "cannot connect {} ports, only {} available".format(
                    len(self.outchannels),
                    len(target_ports),))
            
            # Hook up one outport to each channel
            for n in range(len(self.outchannels)):
                # This is the channel number the user provided in OUTCHANNELS
                index_of_physical_channel = self.outchannels[n]
                
                # This is the corresponding physical channel
                # I think this will always be the same as index_of_physical_channel
                physical_channel = target_ports[index_of_physical_channel]
                
                # Connect virtual outport to physical channel
                self.client.outports[n].connect(physical_channel)

    def process(self, frames):
        # Generate some fake data
        # In the future this will be pulled from the queue
        data = np.random.uniform(-1, 1, self.blocksize) # Generating a random white noise signal
        self.table = np.zeros((self.blocksize, 2)) # Creating a table of zeros with 2 columns
        self.table[:, 0] = data # Assigning the random white noise signal to a channel in (0,1)
        amplitude = 0.001 
        self.table = self.table * amplitude # Scaling the signal by amplitude
        self.table = self.table.astype(np.float32) # Converting the table to float32
        #data = np.zeros(self.blocksize, dtype='float32')
        #print("data shape:", data.shape)

        # Write
        self.write_to_outports(self.table)

    def write_to_outports(self, data):
        data = data.squeeze()
        if data.ndim == 1:
            ## 1-dimensional sound provided
            # Write the same data to each channel
            for outport in self.client.outports:
                buff = outport.get_array()
                buff[:] = data

        elif data.ndim == 2:
            # Error check
            if data.shape[1] != len(self.client.outports):
                raise ValueError(
                    "data has {} channels "
                    "but only {} outports in pref OUTCHANNELS".format(
                    data.shape[1], len(self.client.outports)))

            # Write one column to each channel
            for n_outport, outport in enumerate(self.client.outports):
                buff = outport.get_array()
                buff[:] = data[:, n_outport]

        else:
            raise ValueError("data must be 1D or 2D")

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
        print(''.join(self.event_log[-60:]))
        print('\t'.join(self.state_log[-4:]))

class TouchListener(object):
    def __init__(self, pi):
        # Global variables
        self.pi = pi
        self.last_touch = datetime.datetime.now()
        self.touch_state = False

        self.pi.set_mode(16, pigpio.INPUT)
        self.pi.callback(16, pigpio.RISING_EDGE, self.touch_happened)
        self.pi.callback(16, pigpio.FALLING_EDGE, self.touch_stopped)

    def touch_happened(self, pin, level, tick):
        touch_time = datetime.datetime.now()
        if touch_time - self.last_touch > datetime.timedelta(seconds=1):
            print('touch start received tick={} dt={}'.format(tick, touch_time))
            self.last_touch = touch_time
            self.touch_state = True
        else:
            print('touch start ignored tick={} dt={}'.format(tick, touch_time))
    
    def touch_stopped(self, pin, level, tick):
        touch_time = datetime.datetime.now()
        if touch_time - self.last_touch > datetime.timedelta(seconds=1):
            print('touch stop  received tick={} dt={}'.format(tick, touch_time))
            self.last_touch = touch_time
            self.touch_state = False
        else:
            print('touch stop  ignored tick={} dt={}'.format(tick, touch_time))    

    def report(self):
        print("touch state={}; last_touch={}".format(self.touch_state, self.last_touch))


## Keep track of pigpio.pi
pi = pigpio.pi()
wl = WheelListener(pi)
tl = TouchListener(pi)

# Define a client to play sounds
#~ jack_client = JackClient(name='jack_client')

# Solenoid
pi.set_mode(26, pigpio.OUTPUT)
pi.write(26, 0)

"""
States
00 10
01 11
When A goes up, move right. When A goes down, move left.
When B goes up, move down. When B goes down, move up.
When the state moves clockwise, increment position.
When the state moves counter-clockwise, decrement position.

A cute trick might be to take the state's value in binary, subtract
0.6, and take absolute value. If this result is increasing, increment
position, otherwise decrement position. That's probably not any faster
though.
"""

while True:
    # Print out the wheel status
    wl.do_nothing()

    # Print out the touch status
    tl.report()

    time.sleep(1)