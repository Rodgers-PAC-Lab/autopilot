import pigpio
import time


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
        time.sleep(1)
    

## Keep track of pigpio.pi
pi = pigpio.pi()
wl = WheelListener(pi)

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
    wl.do_nothing()
