# First run this:
#   source ~/.venv/py3/bin/activate
#
# Then run this script in ipython
#
# To install jack the autopilot way:
# git clone https://github.com/jackaudio/jack2 --depth 1
# cd jack2
# ./waf configure --alsa=yes --libdir=/usr/lib/arm-linux-gnueabihf/
# ./waf build -j6
# sudo ./waf install
# sudo ldconfig
# sudo sh -c "echo @audio - memlock 256000 >> /etc/security/limits.conf"
# sudo sh -c "echo @audio - rtprio 75 >> /etc/security/limits.conf"
# cd ..
# rm -rf ./jack2
#
# To set up hifiberry
# sudo adduser pi i2c
# sudo sed -i 's/^dtparam=audio=on/#dtparam=audio=on/g' /boot/config.txt
# sudo sed -i '$s/$/\ndtoverlay=hifiberry-dacplus\ndtoverlay=i2s-mmap\ndtoverlay=i2c-mmap\ndtparam=i2c1=on\ndtparam=i2c_arm=on/' /boot/config.txt
# echo -e 'pcm.!default {\n type hw card 0\n}\nctl.!default {\n type hw card 0\n}' | sudo tee /etc/asound.conf
#
# The first sed doesn't seem to do anything
# The second adds these lines
# dtoverlay=hifiberry-dacplus
# dtoverlay=i2s-mmap
# dtoverlay=i2c-mmap
# dtparam=i2c1=on
# dtparam=i2c_arm=on
# The final one echoes to asound.conf, which didn't formerly exist

import pigpio
import time
import datetime
import jack
import numpy as np
import shared
import os
import itertools
import importlib
importlib.reload(shared)

## Killing previous pigpiod and jackd background processes
os.system('sudo killall pigpiod')
os.system('sudo killall jackd')

# Wait long enough to make sure they are killed
time.sleep(1)


## Starting pigpiod and jackd background processes
# Start pigpiod
# -t 0 : use PWM clock (otherwise messes with audio)
# -l : disable remote socket interface (not sure why)
# -x : mask the GPIO which can be updated (not sure why; taken from autopilot)
# Runs in background by default (no need for &)
os.system('sudo pigpiod -t 0 -l -x 1111110000111111111111110000')
time.sleep(1)

# Start jackd
# https://linux.die.net/man/1/jackd
# -P75 : set realtime priority to 75 (why?)
# -p16 : Set the number of frames between process() calls. Must be power of 2.
#   Lower values will lower latency but increase probability of xruns.
#   Or is this --port-max?
# -t2000 : client timeout limit in milliseconds
# -dalsa : driver ALSA
#
# ALSA backend options:
# -dhw:sndrpihifiberry : device to use
# -P : provide only playback ports (why?)
# -r192000 : set sample rate to 192000
# -n3 : set the number of periods of playback latency to 3
# -s : softmode, ignore xruns reported by the ALSA driver
# & : run in background
# TODO: document these parameters
# TODO: Use subprocess to keep track of these background processes
#~ os.system(
    #~ 'jackd -P75 -p16 -t2000 -dalsa -dhw:sndrpihifiberry -P -r192000 -n3 -s &')
#~ time.sleep(1)


## Define audio to play
audio_cycle = itertools.cycle([
    0.01 * np.random.uniform(-1, 1, (1024, 2)),
    0.00 * np.random.uniform(-1, 1, (1024, 2)),
    ])


## Keep track of pigpio.pi
pi = pigpio.pi()

# Define object for listening to wheel
wl = shared.WheelListener(pi)

# Define object for listening to touches
#~ tl = shared.TouchListener(pi, debug_print=True)

# Define a client to play sounds
#~ sound_player = shared.SoundPlayer(audio_cycle=audio_cycle)

# Solenoid
pi.set_mode(26, pigpio.OUTPUT)
pi.write(26, 0)

def reward():
    # Activate solenoid
    pi.write(26, 1)
    time.sleep(0.1)
    pi.write(26, 0)    

#~ tl.touch_trigger = reward

## Loop forever
wheel_reward_thresh = 1000
last_rewarded_position = 0
last_reported_time = datetime.datetime.now()
report_interval = 5

# Loop forever
while True:
    # Get the current time
    current_time = datetime.datetime.now()
    
    # Report if it's been long enough
    if current_time - last_reported_time > datetime.timedelta(seconds=report_interval):
        # Print out the wheel status
        #~ wl.report()

        # Print out the touch status
        #~ tl.report()
        
        last_reported_time = current_time
    
    # See how far the wheel has moved
    current_wheel_position = wl.position
    if np.abs(current_wheel_position - last_rewarded_position) > wheel_reward_thresh:
        # Set last rewarded position to current position
        last_rewarded_position = current_wheel_position
        
        # Reward
        reward()
    
    time.sleep(.1)