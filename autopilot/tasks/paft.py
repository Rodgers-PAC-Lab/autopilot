"""This module defines the PAFT task"""

from collections import OrderedDict as odict
import tables
import itertools
import random
import datetime
import autopilot.hardware.gpio
from autopilot.stim.sound import sounds
from autopilot.tasks.task import Task
import time
import functools
from autopilot.networking import Net_Node
from autopilot import prefs
import threading

# The name of the task
# This declaration allows Subject to identify which class in this file 
# contains the task class. 
TASK = 'PAFT'

class PAFT(Task):
    """The probabalistic auditory foraging task (PAFT).
    
    This task chooses a port at random, lights up the LED for that port,
    plays sounds through the associated speaker, and then dispenses water 
    once the subject pokes there.

    Stage list:
    * waiting for the response
    * reporting the response

    Attributes:
        target ('L', 'C', 'R'): The correct port
        trial_counter (:class:`itertools.count`): 
            Counts trials starting from current_trial specified as argument
        triggers (dict): 
            Dictionary mapping triggered pins to callable methods.
        num_stages (int): 
            number of stages in task (2)
        stages (:class:`itertools.cycle`): 
            iterator to cycle indefinitely through task stages.
    """
    ## List of stages
    # These correspond to methods in this Class, I think (CR)
    STAGE_NAMES = ["water", "response"]

    
    ## Params
    PARAMS = odict()
    PARAMS['reward'] = {'tag':'Reward Duration (ms)',
                        'type':'int'}
    PARAMS['allow_repeat'] = {'tag':'Allow Repeated Ports?',
                              'type':'bool'}
    
    # CR: Added for sounds
    PARAMS['stim']           = {'tag':'Sounds',
                                'type':'sounds'}

    ## Returned data
    DATA = {
        'trial_num': {'type':'i32'},
        'target': {'type':'S1', 'plot':'target'},
        'timestamp': {'type':'S26'}, # only one timestamp, since next trial instant
    }

    # TODO: This should be generated from DATA above. 
    # Perhaps parsimoniously by using tables types rather than string descriptors
    class TrialData(tables.IsDescription):
        trial_num = tables.Int32Col()
        target    = tables.StringCol(1)
        timestamp = tables.StringCol(26)


    ## Required hardware
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
    
    
    ## The child rpi that handles the other ports
    CHILDREN = {
        'rpi02': {
            'task_type': "PAFT Child",
        }
    }

    
    ## Plot parameters
    PLOT = {
        'data': {
            'target': 'point'
        }
    }

    
    ## Methods
    def __init__(self, stage_block=None, stim=None, current_trial=0,
        reward=150, allow_repeat=False, **kwargs):
        """Initialize a new PAFT Task
        
        Arguments
        ---------
        stage_block (:class:`threading.Event`): 
            used to signal to the carrying Pilot that the current trial 
            stage is over
        current_trial (int): 
            If not zero, initial number of `trial_counter`
        reward (int): 
            ms to open solenoids
        allow_repeat (bool): 
            Whether the correct port is allowed to repeat between trials
        **kwargs:
        """
        super(PAFT, self).__init__()

        # stage_block
        if not stage_block:
            raise Warning('No stage_block Event() was passed, youll need'
                ' to handle stage progression on your own')
        else:
            self.stage_block = stage_block

        # Fixed parameters
        print("Reward is: {}".format(reward))
        if isinstance(reward, dict):
            self.reward = reward
        else:
            self.reward         = {'type':'duration',
                                   'value': float(reward)}

        # Variable parameters
        self.target = random.choice(['L', 'R'])
        self.trial_counter = itertools.count(int(current_trial))
        self.triggers = {}

        # Stage list to iterate
        stage_list = [self.water, self.response]
        self.num_stages = len(stage_list)
        self.stages = itertools.cycle(stage_list)

        # Init hardware
        self.hardware = {}
        self.pin_id = {} # Inverse pin dictionary
        self.init_hardware()

        # Set reward values for solenoids
        # TODO: Super inelegant, implement better with reward manager
        if self.reward['type'] == "volume":
            self.set_reward(vol=self.reward['value'])
        else:
            print("setting reward to {}".format(self.reward['value']))
            self.set_reward(duration=self.reward['value'])

        
        
        ## Initialize net node for communications with child
        # With instance=True, I get a threading error about current event loop
        self.node = Net_Node(id="T_{}".format(prefs.get('NAME')),
                             upstream=prefs.get('NAME'),
                             port=prefs.get('MSGPORT'),
                             listens={},
                             instance=False)

        # Construct a message to send to child
        # Why do we need to save self.subject here?
        self.subject = kwargs['subject']
        value = {
            'child': {
                'parent': prefs.get('NAME'), 'subject': kwargs['subject']},
            'task_type': self.CHILDREN['rpi02']['task_type'],
            'subject': kwargs['subject'],
        }

        # send to the station object with a 'CHILD' key
        self.node.send(to=prefs.get('NAME'), key='CHILD', value=value)

        # If we aren't passed an event handler
        # (used to signal that a trigger has been tripped),
        # we should warn whoever called us that things could get a little screwy
        if not stage_block:
            raise Warning(
                'No stage_block Event() was passed, youll need to '
                'handle stage progression on your own'
                )
        else:
            self.stage_block = stage_block


        # allow_repeat
        self.allow_repeat = bool(allow_repeat)
    
    def log_poke(self, port):
        self.logger.debug('{} {} poke'.format(
            datetime.datetime.now().isoformat(),
            port,
            ))

    def water(self, *args, **kwargs):
        """
        First stage of task - open a port if it's poked.

        Returns:
            dict: Data dictionary containing::

                'target': ('L', 'C', 'R') - correct response
                'timestamp': isoformatted timestamp
                'trial_num': number of current trial
        """
        self.stop_playing = False
        self.stage_block.clear()

        # Choose random port
        if self.allow_repeat:
            self.target = random.choice(['L', 'R'])
        else:
            other_ports = [t for t in ['L', 'R'] if t is not self.target]
            self.target = random.choice(other_ports)

        # Set triggers for target poke entry
        self.triggers[self.target] = [
            functools.partial(self.log_poke, self.target),
            self.hardware['PORTS'][self.target].open,
            ]
        
        # Set target LED to green
        self.set_leds({self.target: [0, 255, 0]})


        ## Choose target and generate stim
        print("The chosen target is {}".format(self.target))
        if self.target == 'L':
            self.stim = sounds.Noise(
                duration=100, amplitude=.003, channel=0, nsamples=19456)
            
        elif self.target == 'R':
            self.stim = sounds.Noise(
                duration=100, amplitude=.003, channel=1, nsamples=19456)
            
        else:
            raise ValueError("unknown target: {}".format(target))
        
        # Set the trigger to call function when stim is over
        self.stim.set_trigger(self.delayed_play_again)
        
        # Buffer the stim and start playing it after a delay
        self.stim.buffer()
        threading.Timer(.75, self.stim.play).start()        
        
        
        ## Message child
        self.node.send(
            to=prefs.get('NAME'),
            key='CHILD',
            value={'foo': 'bar', 'subject': self.subject, 'KEY': 'WAIT', 'keys': 'WAIT'},
            )        
        

        ## Return data
        data = {
            'target': self.target,
            'timestamp': datetime.datetime.now().isoformat(),
            'trial_num' : next(self.trial_counter)
        }
        return data

    def response(self):
        """
        Just have to alert the Terminal that the current trial has ended
        and turn off any lights.
        """
        time.sleep(.5)
        
        # Turn off the "stim_end" trigger so it doesn't keep playing
        self.stim.set_trigger(self.done_playing)
        
        # we just have to tell the Terminal that this trial has ended

        # mebs also turn the light off rl quick
        self.set_leds()

        return {'TRIAL_END':True}

    def end(self):
        """
        When shutting down, release all hardware objects and turn LEDs off.
        """
        for k, v in self.hardware.items():
            for pin, obj in v.items():
                obj.release()

    def delayed_play_again(self):
        """Called when stim over
        """
        # Play it again, after a delay
        self.stim.buffer()
        threading.Timer(.75, self.stim.play).start()
        
    def done_playing(self):
        # This is called when the last stim of the trial has finished playing
        pass
