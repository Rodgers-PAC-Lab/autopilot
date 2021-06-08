"""This module defines the PAFT task


Multiple Child rpis running the "PAFT Child" Task connect to this Parent.
The Parent chooses the correct stimulus and logs all events. It tells each
Child if it should start playing sounds and when it should stop. The Child
knows that a poke into the port that is currently playing sound should
be rewarded.

The Parent establishes a router Net_Node on port 5001. A dealer Net_Node
on each Child connects to it.

The Parent responds to the following message keys on port 5001:
* HELLO : This means the Child has booted the task.
    The value is dispatched to PAFT.recv_hello, with the following keys:
    'from' : string; the name of the child (e.g., rpi02)
* POKE : This means the Child has detected a poke.
    The value is dispatched to PAFT.recv_poke, with the following keys:
    'from' : string; the name of the child (e.g., rpi02)
    'poke' : one of {'L', 'R'}; the side that was poked

The Parent will not start the first trial until each Child in PAFT.CHILDREN
has sent the "HELLO" message. 

The Parent can send the following message keys:
* HELLO : not used
* PLAY : This tells the Child to start playing and be ready to reward
    The value has the following keys:
    'target' : one of {'L', 'R'}; the side that should play
* STOP : This tells the Child to stop playing sounds and not reward.
    The value is an empty dict.    
* END : This tells the Child the session is over.
    The value is an empty dict.    
"""

from collections import OrderedDict as odict
import tables
import itertools
import random
import datetime
import numpy as np
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
    
    
    ## The child rpi that handles the other ports
    CHILDREN = {
        'rpi02': {
            'task_type': "PAFT Child",
        },
        'rpi03': {
            'task_type': "PAFT Child",
        },
        'rpi04': {
            'task_type': "PAFT Child",
        },
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
        self.logger.debug("Reward is: {}".format(reward))
        if isinstance(reward, dict):
            self.reward = reward
        else:
            self.reward         = {'type':'duration',
                                   'value': float(reward)}

        # Variable parameters
        self.child_connected = {}
        for child in self.CHILDREN.keys():
            self.child_connected[child] = False
        self.target = random.choice(['L', 'R'])
        self.trial_counter = itertools.count(int(current_trial))
        self.n_trials = 0 # always start at zero
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
            self.logger.debug("setting reward to {}".format(self.reward['value']))
            self.set_reward(duration=self.reward['value'])

        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)
        
        
        ## Initialize net node for communications with child
        # With instance=True, I get a threading error about current event loop
        self.node = Net_Node(id="T_{}".format(prefs.get('NAME')),
                             upstream=prefs.get('NAME'),
                             port=prefs.get('MSGPORT'),
                             listens={},
                             instance=False)

        # Construct a message to send to child
        # Specify the subjects for the child (twice)
        self.subject = kwargs['subject']
        value = {
            'child': {
                'parent': prefs.get('NAME'), 'subject': kwargs['subject']},
            'task_type': 'PAFT Child',
            'subject': kwargs['subject'],
        }

        # send to the station object with a 'CHILD' key
        self.node.send(to=prefs.get('NAME'), key='CHILD', value=value)

        
        ## Create a second node to communicate with the child
        # We (parent) are the "router"/server
        # The children are the "dealer"/clients
        # Many dealers, one router        
        self.node2 = Net_Node(
            id='parent_pi',
            upstream='',
            port=5000,
            router_port=5001,
            listens={
                'HELLO': self.recv_hello,
                'POKE': self.recv_poke,
                },
            instance=False,
            )

        # Wait until the child connects!
        self.logger.debug("Waiting for child to connect")
        while True:
            stop_looping = True
            
            for child, is_conn in self.child_connected.items():
                if not is_conn:
                    stop_looping = False
            
            if stop_looping:
                break
        self.logger.debug(
            "All children have connected: {}".format(self.child_connected))

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
    
    def recv_poke(self, value):
        # Log it
        self.log_poke_from_child(value)
        
        # Mark as complete if correct
        target_child_name, target_side = self.target.split('_')
        child_name = value['from']
        poke_name = value['poke']
        
        if target_child_name == child_name and target_side == poke_name:
            self.logger.debug('correct poke {}; target was {}'.format(value, self.target))
            self.stage_block.set()
        else:
            self.logger.debug('incorrect poke {}; target was {}'.format(value, self.target))
        
    def log_poke_from_child(self, value):
        child_name = value['from']
        poke_name = value['poke']
        self.log_poke('{}_{}'.format(child_name, poke_name))
    
    def log_poke(self, port):
        self.logger.debug('{} {} poke'.format(
            datetime.datetime.now().isoformat(),
            port,
            ))
    
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
                #~ functools.partial(self.report_poke, poke),
                ]        
    
    def water(self, *args, **kwargs):
        """
        First stage of task - open a port if it's poked.

        Returns:
            dict: Data dictionary containing::

                'target': ('L', 'C', 'R') - correct response
                'timestamp': isoformatted timestamp
                'trial_num': number of current trial
        """
        ## What does this do? Anything?
        self.stop_playing = False
        
        
        ## Prevents moving to next stage
        self.stage_block.clear()

        
        ## Choose targets
        # Identify possible targets
        all_possible_targets = [
            'L', 'R', 
            'rpi02_L', 'rpi02_R', 
            'rpi03_L', 'rpi03_R',
            'rpi04_L', 'rpi04_R',
            ]
        excluding_previous = [
            t for t in all_possible_targets if t != self.target]
        
        # Choose
        meth = 'RANDOM'
        if meth == 'CYCLE':
            self.target = all_possible_targets[
                np.mod(self.n_trials, len(all_possible_targets))]
            self.n_trials += 1
        elif meth == 'RANDOM':
            self.target = random.choice(excluding_previous)
        else:
            raise ValueError("unknown trial choosing meth: {}".format(meth))

        # Print debug
        self.logger.debug("The chosen target is {}".format(self.target))
        
        
        ## Set poke triggers (for logging)
        self.set_poke_triggers()
        

        ## Set stim and LEDs by target
        if self.target == 'L':
            self.stim = sounds.Noise(
                duration=100, amplitude=.003, channel=0, nsamples=19456)
            
            # Turn on green led
            self.hardware['LEDS']['L'].set(r=0, g=255, b=0)
            self.hardware['LEDS']['R'].set(r=0, g=0, b=0)
            
            # Add a trigger to open the port
            self.triggers['L'].append(self.hardware['PORTS']['L'].open)
            
        elif self.target == 'R':
            self.stim = sounds.Noise(
                duration=100, amplitude=.003, channel=1, nsamples=19456)

            self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
            self.hardware['LEDS']['R'].set(r=0, g=255, b=0)
            
            # Add a trigger to open the port
            self.triggers['R'].append(self.hardware['PORTS']['R'].open)
        
        elif self.target.startswith('rpi'):
            # It's a child target
            child_name, side = self.target.split('_')

            # No LED or stim
            self.stim = None
            self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
            self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

            # Tell child what the target is
            self.node2.send(
                to=child_name,
                key='PLAY',
                value={'target': side},
                )        

        else:
            raise ValueError("unknown target: {}".format(target))
        
        
        ## Set the stim to repeat
        if self.stim is not None:
            # Set the trigger to call function when stim is over
            self.stim.set_trigger(self.delayed_play_again)
            
            # Buffer the stim and start playing it after a delay
            self.stim.buffer()
            threading.Timer(.75, self.stim.play).start()        
        
        
        ## Return data
        data = {
            'target': self.target,
            'timestamp': datetime.datetime.now().isoformat(),
            'trial_num' : next(self.trial_counter)
        }
        return data

    def recv_hello(self, value):
        self.logger.debug("received HELLO from child with value {}".format(value))
        self.child_connected[value['from']] = True

    def response(self):
        """
        Just have to alert the Terminal that the current trial has ended
        and turn off any lights.
        """
        # Turn off the "stim_end" trigger so it doesn't keep playing
        if self.stim is not None:
            self.stim.set_trigger(self.done_playing)
        
        # Turn off any LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

        # Tell the child
        if self.target.startswith('rpi'):
            child_name, side = self.target.split('_')
            
            # Tell child what the target is
            self.node2.send(
                to=child_name,
                key='STOP',
                value={},
                )                

        # Tell the Terminal the trial has ended
        return {'TRIAL_END':True}

    def end(self):
        """
        When shutting down, release all hardware objects and turn LEDs off.
        """
        # Tell each child to END
        for child_name in self.CHILDREN.keys():
            # Tell child what the target is
            self.node2.send(
                to=child_name,
                key='END',
                value={},
                )    
        
        # Stop playing sound
        if self.stim is not None:
            self.stim.set_trigger(self.done_playing)
    
        # Turn off LEDs
        self.hardware['LEDS']['L'].set(r=0, g=0, b=0)
        self.hardware['LEDS']['R'].set(r=0, g=0, b=0)

        # Release all hardware
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
