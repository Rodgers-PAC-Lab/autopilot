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

import numpy as np
import autopilot.networking

class ex_serialize_child(children.Child):
    """Define the child task associated with ex_serialize"""
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
        """Initialize a new ex_serialize_child
        
        task_type : 'ex_serialize_child'
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
        
        # Save subject
        # This is needed when sending messages
        self.subject = subject


        ## Hardware
        self.triggers = {}
        self.init_hardware()
        

        ## Stages
        # Only one stage
        self.stages = itertools.cycle([self.play])
        self.stage_block = stage_block
        
        
        ## Set up NET_Node to communicate with Parent
        self.create_inter_pi_communication_node()

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
                },
            instance=False,
            )        
        
        # Send HELLO so that Parent knows we are here
        self.node2.send('parent_pi', 'HELLO', {'from': self.name})        

    def init_hardware(self):
        """Placeholder"""
        self.hardware = {}        

    def play(self):
        """A single stage"""
        self.logger.debug("Starting the play stage")
        
        # Sleep so we don't go crazy
        time.sleep(3)

        # Create a serialized message
        # Adapted from the bandwidth test
        payload = np.arange(128, dtype=np.float64)
        message = {
            'pilot': self.name,
            'payload': payload,
            'timestamp': datetime.datetime.now().isoformat(),
            'subject': 'test_subject',
        }        
        test_msg = autopilot.networking.Message(
            to='rpiparent03', key='ARRDAT', value=message,
            flags={'MINPRINT':True},
            id="test_message", sender="test_sender", 
            subject=self.subject,
            blosc=True)
        
        test_msg.serialize()
        
        self.node2.send(to='parent_pi', 'ARRDAT', msg=test_msg)


        # Continue to the next stage (which is this one again)
        self.stage_block.set()

    def recv_hello(self, value):
        """This is probably unnecessary"""
        self.logger.debug(
            "received HELLO from parent with value {}".format(value))

    def end(self):
        """This is called when the STOP signal is received from the parent"""
        self.logger.debug("Inside the self.end function")

        # Explicitly close the socket (helps with restarting cleanly)
        self.node2.sock.close()
        self.node2.release()
        
        # Release hardware. No superclass to do this for us.
        for k, v in self.hardware.items():
            for pin, obj in v.items():
                obj.release()
