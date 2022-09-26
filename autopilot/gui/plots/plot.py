"""Classes to plot data in the GUI."""
from functools import wraps
from itertools import count

# CR
import datetime
import time
import logging
import os

# Upstream
import numpy as np
import pyqtgraph as pg
from PySide2 import QtCore, QtWidgets

import autopilot
from autopilot import prefs
from autopilot.utils.loggers import init_logger
from autopilot.gui.plots.video import Video
from autopilot.gui.plots.info import Timer
from autopilot.gui.plots.geom import Roll_Mean, HLine, PLOT_LIST
from autopilot.networking import Net_Node
from autopilot.utils.invoker import get_invoker, InvokeEvent

# CR: check this no longer needed
# pg config
# pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')

def gui_event(fn):
    """
    Wrapper/decorator around an event that posts GUI events back to the main
    thread that our window is running in.

    Args:
        fn (callable): a function that does something to the GUI
    """
    @wraps(fn)
    def wrapper_gui_event(*args, **kwargs):
        # type: (object, object) -> None
        """

        Args:
            *args ():
            **kwargs ():
        """
        QtCore.QCoreApplication.postEvent(get_invoker(), InvokeEvent(fn, *args, **kwargs))
    return wrapper_gui_event


class Plot_Widget(QtWidgets.QWidget):
    """
    Main plot widget that holds plots for all pilots

    Essentially just a container to give plots a layout and handle any
    logic that should apply to all plots.

    Attributes:
        logger (`logging.Logger`): The 'main' logger
        plots (dict): mapping from pilot name to :class:`.Plot`
    """
    # Widget that frames multiple plots
    def __init__(self):
        # type: () -> None
        QtWidgets.QWidget.__init__(self)

        self.logger = init_logger(self)


        # We should get passed a list of pilots to keep ourselves in order after initing
        self.pilots = None

        # Dict to store handles to plot windows by pilot
        self.plots = {}

        # Main Layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)

        # Plot Selection Buttons
        # TODO: Each plot bar should have an option panel, because different tasks have different plots
        #self.plot_select = self.create_plot_buttons()

        # Create empty plot container
        self.plot_layout = QtWidgets.QVBoxLayout()

        # Assemble buttons and plots
        #self.layout.addWidget(self.plot_select)
        self.layout.addLayout(self.plot_layout)

        self.setLayout(self.layout)

        self.setContentsMargins(0, 0, 0, 0)

    def init_plots(self, pilot_list):
        """
        For each pilot, instantiate a :class:`.Plot` and add to layout.

        Args:
            pilot_list (list): the keys from :attr:`.Terminal.pilots`
        """
        self.pilots = pilot_list

        # Make a plot for each pilot.
        for p in self.pilots:
            plot = Plot(pilot=p, parent=self)
            self.plot_layout.addWidget(plot)
            #~ self.plot_layout.addWidget(HLine())
            self.plots[p] = plot


class Plot(QtWidgets.QWidget):
    """Displays data for a single Pilot, within the overall Terminal.
    
    This version has been heavily customized and only works for PAFT
    tasks. TODO: allow different tasks to define their own Plot.
    
    This object inherits from QWidget. It lives inside a QVBoxLayout
    rendering the other Pilots. This is handled by the class Plot_Widget.
    
    This object contains other Widgets, corresponding to (for example)
    an infobox and one or more graphs.

    This object contains a Net_Node that listens for messages. The
    following messages are accepted:

    +-------------+------------------------+-------------------------+
    | Key         | Method                 | Description             |
    +=============+========================+=========================+
    | **'START'** | :meth:`~.Plot.l_start` | starting a new task     |
    +-------------+------------------------+-------------------------+
    | **'DATA'**  | :meth:`~.Plot.l_data`  | getting a new datapoint |
    +-------------+------------------------+-------------------------+
    | **'STOP'**  | :meth:`~.Plot.l_stop`  | stop the task           |
    +-------------+------------------------+-------------------------+
    | **'PARAM'** | :meth:`~.Plot.l_param` | change some parameter   |
    +-------------+------------------------+-------------------------+

    **Plot Parameters**

    The plot is built from the ``PLOT={data:plot_element}`` mappings described in the :class:`~autopilot.tasks.task.Task` class.
    Additional parameters can be specified in the ``PLOT`` dictionary. Currently:

    * **continuous** (bool): whether the data should be plotted against the trial number (False or NA) or against time (True)
    * **chance_bar** (bool): Whether to draw a red horizontal line at chance level (default: 0.5)
    * **chance_level** (float): The position in the y-axis at which the ``chance_bar`` should be drawn
    * **roll_window** (int): The number of trials :class:`~.Roll_Mean` take the average over.

    Attributes:
        pilot (str): The name of our pilot, used to set the identity of our socket, specifically::

            'P_{pilot}'

        infobox (:class:`QtWidgets.QFormLayout`): Box to plot basic task information like trial number, etc.
        info (dict): Widgets in infobox:

            * 'N Trials': :class:`QtWidgets.QLabel`,
            * 'Runtime' : :class:`.Timer`,
            * 'Session' : :class:`QtWidgets.QLabel`,
            * 'Protocol': :class:`QtWidgets.QLabel`,
            * 'Step'    : :class:`QtWidgets.QLabel`

        plot (:class:`pyqtgraph.PlotWidget`): The widget where we draw our plots
        plot_params (dict): A dictionary of plot parameters we receive from the Task class
        data (dict): A dictionary of the data we've received
        plots (dict): The collection of plots we instantiate based on `plot_params`
        node (:class:`.Net_Node`): Our local net node where we listen for data.
        state (str): state of the pilot, used to keep plot synchronized.
    """

    def __init__(self, pilot, parent=None):
        """Initialize a new Plot for a single pilot.
        
        Arguments:
            pilot (str): The name of the corresponding pilot.
                Our Net_Node will be named P_{}.format(pilot).
                Messages to this Net_Node will be handled by this object.
            
            parent (:class: `Plot_Widget`):
                The `Plot_Widget` in which we live.
                I don't think this is used by anything.
        """
        # Superclass init (Qt stuff)
        super(Plot, self).__init__()
        
        # Init logger
        self.logger = init_logger(self)

        # Capture these arguments
        self.parent = parent
        self.pilot = pilot
        
        # Keep track of our `state`. This can be IDLE, INITIALIZING, or RUNNING
        # Used to disregard messages when we're not able to handle them yet.
        self.state = "IDLE"
        
        # Qt magic?
        self.invoker = get_invoker()

        
        ## Task specific stuff
        # These are the possible ports to display
        # TODO: receive these from the Pilot? Or how to handle multiple boxes?
        # These will be rendered clockwise in the box plot
        # And from top down in the raster plot
        if pilot == 'rpi_parent01':
            self.known_pilot_ports = [
                'rpi09_L',
                'rpi09_R',
                'rpi10_L',
                'rpi10_R',            
                'rpi11_L',
                'rpi11_R',
                'rpi12_L',
                'rpi12_R', 
                ]
            
            # The first entry in the list above will be located at
            # the gui rotation offset (typically, straight up, or pi / 2).
            # Each subsequent port is 45 degrees clockwise from there.
            self.gui_rotation_offset = np.pi / 2
        
        elif pilot == 'rpi_parent02':
            self.known_pilot_ports = [
                'rpi07_L',
                'rpi07_R',
                'rpi08_L',
                'rpi08_R', 
                'rpi05_L',
                'rpi05_R',
                'rpi06_L',
                'rpi06_R',
                ]
            self.gui_rotation_offset = np.pi / 2
        
        elif pilot == 'rpiparent03':
            self.known_pilot_ports = [
                'rpi01_L',
                'rpi01_R',
                'rpi02_L',
                'rpi02_R',            
                'rpi03_L',
                'rpi03_R',
                'rpi04_L',
                'rpi04_R', 
                ]
            self.gui_rotation_offset = -3 * np.pi / 4
        
        elif pilot == 'rpiparent04':
            self.known_pilot_ports = [
                'rpi18_L',
                'rpi18_R',
                'rpi19_L',
                'rpi19_R',            
                'rpi20_L',
                'rpi20_R',
                'rpi21_L',
                'rpi21_R', ]
            self.gui_rotation_offset = 0
        
        else:
            # This can happen if another pilot is in pilot_db for whatever reason
            self.known_pilot_ports = []
            self.gui_rotation_offset = np.pi / 2
            self.logger.debug("plot: unrecognized parent name: {}".format(pilot))
        
        # These are used to store data we receive over time
        self.known_pilot_ports_poke_data = [
            [] for kpp in self.known_pilot_ports]
        self.known_pilot_ports_reward_data = [
            [] for kpp in self.known_pilot_ports]
        self.known_pilot_ports_correct_reward_data = [
            [] for kpp in self.known_pilot_ports]
        self.rank_of_poke_by_trial = []
        
        # These are used to store handles to different graph traces
        self.known_pilot_ports_poke_plot = []
        self.known_pilot_ports_reward_plot = []
        self.known_pilot_ports_correct_reward_plot = []

        
        ## Init the plots and handles
        self.init_plots()

        
        ## Station
        # Start the listener, subscribes to terminal_networking that will broadcast data
        self.listens = {
            'START' : self.l_start, # Receiving a new task
            'DATA' : self.l_data, # Receiving a new datapoint
            'CONTINUOUS': self.l_data,
            'STOP' : self.l_stop,
            'PARAM': self.l_param, # changing some param
            'STATE': self.l_state
        }
        
        # Start the Net_Node
        self.node = autopilot.networking.Net_Node(
            id='P_{}'.format(self.pilot),
            upstream="T",
            port=prefs.get('MSGPORT'),
            listens=self.listens,
            instance=True)

    @gui_event
    def init_plots(self):
        """Initalize our contained Widgets and graphs.
        
        This creates the following Widgets in an QHBoxLayout:
            infobox (QFormLayout) :
                Lists text results, such as n_trials
            plot_octagon (pg.PlotWidget) :
                Plot of the current status of the octagon
                A separate circle displays each port
            plot_timecourse (pg.PlotWidget) :
                Plot of the pokes over time
        """
        ## Ceates a horizontal box layout for all the sub-widgets
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(2,2,2,2)
        self.setLayout(self.layout)
    
        
        ## Widget 1: Infobox
        # Create the first widget: an infobox for n_trials, etc
        self.infobox = QtWidgets.QFormLayout()
        self.infobox_items = {
            'N Trials': QtWidgets.QLabel(),
            'N Correct Trials': QtWidgets.QLabel(),
            'N Rewards': QtWidgets.QLabel(),
            'FC': QtWidgets.QLabel(),
            'RCP': QtWidgets.QLabel(),
            'Runtime' : Timer(),
            'Last poke': Timer(),
        }
        
        # Keep track of N Rewards
        self.n_rewards = 0
        self.infobox_items['N Rewards'].setText('')

        # Keep track of N Trials
        self.n_trials = 0
        self.infobox_items['N Trials'].setText('')
        
        # Keep track of N Correct Trials
        self.n_correct_trials = 0
        self.infobox_items['N Correct Trials'].setText('')

        # This is for calculating rcp
        self.rank_of_poke_by_trial = []

        # Add rows to infobox
        for k, v in self.infobox_items.items():
            self.infobox.addRow(k, v)
        
        # Add to layout
        self.layout.addLayout(self.infobox, 2)

        
        ## Widget 2: Octagon plot
        # Create
        self.plot_octagon = pg.PlotWidget()

        # Within self.plot_octagon, add a circle representing each port
        self.octagon_port_plot_l = []
        
        # This if statement allows proceeding if pilot isn't actually a parent
        if len(self.known_pilot_ports) > 0:
            for n_port in range(8):
                # Determine location
                # The first entry in the list will be located at
                # the gui rotation offset (typically, straight up, or pi / 2).
                # Each port after that is 45 degrees clockwise (-pi/4)
                theta = self.gui_rotation_offset - n_port * np.pi / 4
                x_pos = np.cos(theta)
                y_pos = np.sin(theta)
                
                # Plot the circle
                port_plot = self.plot_octagon.plot(
                    x=[x_pos], y=[y_pos],
                    pen=None, symbolBrush=(255, 0, 0), symbolPen=None, symbol='o',
                    )
                
                # Store the handle to the plot
                self.octagon_port_plot_l.append(port_plot)
                
                # Text label for each port
                txt = pg.TextItem(self.known_pilot_ports[n_port],
                    color='white', anchor=(0.5, 0.5))
                txt.setPos(x_pos * .8, y_pos * .8)
                txt.setAngle(np.mod(theta * 180 / np.pi, 180) - 90)
                self.plot_octagon.addItem(txt)
        
        # Set ranges
        self.plot_octagon.setRange(xRange=(-1, 1), yRange=(-1, 1))
        self.plot_octagon.setFixedWidth(175)
        self.plot_octagon.setFixedHeight(200)
        
        # Add to layout
        self.layout.addWidget(self.plot_octagon, 8)
        
        
        ## Widget 3: Timecourse plot
        ## Create
        self.timecourse_plot = pg.PlotWidget()
        self.timecourse_plot.setContentsMargins(0,0,0,0)
        
        # Set xlim
        self.timecourse_plot.setRange(xRange=[0, 25 * 60], yRange=[0, 7])
        self.timecourse_plot.getViewBox().invertY(True)
       
        # Add a vertical line indicating the current time
        # This will shift over throughout the session
        self.line_of_current_time = self.timecourse_plot.plot(
            x=[0, 0], y=[-1, 8], pen='white')

        # Within self.timecourse_plot, add a trace for pokes made into
        # each port
        ticks_l = []
        for n_row in range(len(self.known_pilot_ports)):
            # Create a plot handle for pokes
            poke_plot = self.timecourse_plot.plot(
                x=[],
                y=np.array([]),
                pen=None, symbolBrush=(255, 0, 0), 
                symbolPen=None, symbol='arrow_down',
                )

            # Store
            self.known_pilot_ports_poke_plot.append(poke_plot)

            # Create a plot handle for rewards
            # This will be blue, because water was given
            reward_plot = self.timecourse_plot.plot(
                x=[],
                y=np.array([]),
                pen=None, symbolBrush=(0, 0, 255), 
                symbolPen=None, symbol='arrow_up',
                )

            # Store
            self.known_pilot_ports_reward_plot.append(reward_plot)

            # Create a plot handle for "correct rewards", which is if the
            # rewarded port was poked first
            # This will be green, because it was correct
            reward_plot = self.timecourse_plot.plot(
                x=[],
                y=np.array([]),
                pen=None, symbolBrush=(0, 255, 0), 
                symbolPen=None, symbol='arrow_up',
                )

            # Store
            self.known_pilot_ports_correct_reward_plot.append(reward_plot)
            
            # Also keep track of yticks
            ticks_l.append((n_row, self.known_pilot_ports[n_row]))

        # Set ticks
        self.timecourse_plot.getAxis('left').setTicks([ticks_l])

        # Add to layout
        self.layout.addWidget(self.timecourse_plot, 8)

    @gui_event
    def l_start(self, value):
        """Start a new session"""
        # If we're already running, log a warning, something didn't shut down
        if self.state in ("RUNNING", "INITIALIZING"):
            self.logger.debug(
                'Plot was told to start but the state is '
                'already {}'.format(self.state))
            return
        
        self.logger.debug('PLOT L_START')
        
        # set infobox stuff
        self.infobox_items['Runtime'].start_timer()
        self.infobox_items['Last poke'].start_timer()

        # Set state
        self.state = 'RUNNING'
        
        # Set reward counter to 0
        self.n_rewards = 0
        self.n_trials = 0
        self.n_correct_trials = 0
        self.rank_of_poke_by_trial = []

        # Update the infobox
        self.infobox_items['N Rewards'].setText('')
        self.infobox_items['N Trials'].setText('')
        self.infobox_items['N Correct Trials'].setText('')
        self.infobox_items['FC'].setText('')
        self.infobox_items['RCP'].setText('')
        self.infobox_items['Runtime'].setText('')
        self.infobox_items['Last poke'].setText('')

        # Set time
        self.start_time = None
        self.local_start_time = None
        
        # Remove residual pokes from previous session
        self.known_pilot_ports_poke_data = [
            [] for kpp in self.known_pilot_ports]
        self.known_pilot_ports_reward_data = [
            [] for kpp in self.known_pilot_ports]
        self.known_pilot_ports_correct_reward_data = [
            [] for kpp in self.known_pilot_ports]
        
        # Update each poke plot to start empty
        for poke_plot in self.known_pilot_ports_poke_plot:
            poke_plot.setData(x=[], y=[])
        for poke_plot in self.known_pilot_ports_reward_plot:
            poke_plot.setData(x=[], y=[])
        for poke_plot in self.known_pilot_ports_correct_reward_plot:
            poke_plot.setData(x=[], y=[])
        
        # Update every so often
        self.update_timer = pg.QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_time_bar)
        self.update_timer.start(50)        
    
    @gui_event
    def update_time_bar(self):
        """Use current time to approximately update timebar"""
        if self.local_start_time is not None:
            current_time = datetime.datetime.now()
            approx_time_in_session = (
                current_time - self.local_start_time).total_seconds()
            self.line_of_current_time.setData(
                x=[approx_time_in_session, approx_time_in_session], y=[-1, 9])

    @gui_event
    def l_data(self, value):
        """Receive data from a running task.

        Args:
            value (dict): Value field of a data message sent during a task.
        """
        self.logger.debug('plots : l_data : received value {}'.format(value))
        
        # Return if we're not ready to take data
        if self.state in ["INITIALIZING", "IDLE"]:
            self.logger.debug(
                'l_data returning because state is {}'.format(self.state))
            return
        
        # Use the start time of the first trial to define `self.start_time`
        # This is time zero on the graph
        if 'timestamp_trial_start' in value and self.start_time is None:
            self.logger.debug(
                'setting start time '
                'to {}'.format(value['timestamp_trial_start']))
            self.start_time = datetime.datetime.fromisoformat(
                value['timestamp_trial_start'])
            
            # Also store approx local start time
            self.local_start_time = datetime.datetime.now()

        # Get the timestamp of this message
        if 'timestamp' in value:
            timestamp_dt = datetime.datetime.fromisoformat(value['timestamp'])
            timestamp_sec = (timestamp_dt - self.start_time).total_seconds()
            
            # Update the current time line
            self.line_of_current_time.setData(
                x=[timestamp_sec, timestamp_sec], y=[-1, 9])
        
        # A new "rewarded_port" was just chosen. Mark it purple.
        # This means it is the beginning of a new trial.
        if 'rewarded_port' in value.keys():
            self.handle_new_trial_start(value)
        
        # A port was just poked
        if 'poked_port' in value.keys():
            # Log this, mark the port red or blue, plot the poke time 
            self.handle_poked_port(value, timestamp_sec)

    def handle_new_trial_start(self, value):
        """A new trial was just started, update plot
        
        Mark rewarded port as purple, previously rewarded as black,
        and all others as white.
        """
        # Use this flag to keep track of whether reward has been delivered
        # on this trial yet or not
        self.reward_delivered_on_this_trial = False
        
        # Find the matching kpp_idx for the rewarded_port
        try:
            rp_kpp_idx = self.known_pilot_ports.index(
                value['rewarded_port'])
        except ValueError:
            # This shouldn't happen
            rp_kpp_idx = None
        
        # Find the matching kpp_idx for the previously_rewarded_port
        try:
            prp_kpp_idx = self.known_pilot_ports.index(
                value['previously_rewarded_port'])
        except (KeyError, ValueError):
            # This shouldn't happen in PAFT
            # This happens in poketrain because 'rewarded_port' doesn't
            # actually indicate a trial start, and there is no
            # 'previously_rewarded_port' in that message
            # In any case, do nothing
            prp_kpp_idx = None
            
        # Make all ports white, except rewarded port purple, and
        # previously rewarded port black
        for opp_idx, opp in enumerate(self.octagon_port_plot_l):
            if opp_idx == rp_kpp_idx:
                opp.setSymbolBrush('purple')
            elif opp_idx == prp_kpp_idx:
                opp.setSymbolBrush('black')
            else:
                opp.setSymbolBrush('w')        

    def handle_poked_port(self, value, timestamp_sec):
        """A port was poked, update the plots accordingly
        
        Determines which port was poked
        Calls handle_rewarded_poke or handle_unrewarded_poke, depending
        """
        # Reset poke timer
        self.infobox_items['Last poke'].start_time = time.time()
       
        # Extract data
        poked_port = value['poked_port']
        
        # Find which pilot this is
        try:
            kpp_idx = self.known_pilot_ports.index(poked_port)
        except ValueError:
            self.logger.debug(
                'unknown poke received: {}'.format(poked_port))
            kpp_idx = None
        
        # Store the time and update the plot
        if kpp_idx is not None:
            # If reward_delivered, then this poke ended the trial
            #    If first_poke, then the trial was correct
            #       Plot as green tick and turn circle green
            #    Else, then the trial was incorrect
            #       Plot as blue tick and turn circle blue
            # Else, then the trial is not over
            #    Plot as red tick and turn circle red (unless it is
            #    already blue or green)
            
            ## Test whether this poke ended the trial
            if value['reward_delivered']:
                self.handle_rewarded_poke(value, kpp_idx, timestamp_sec)
            else:
                self.handle_unrewarded_poke(value, kpp_idx, timestamp_sec)

    def handle_rewarded_poke(self, value, kpp_idx, timestamp_sec):
        """Handle a rewarded poke
        
        Update infoboxes
        Plot ticks in green or blue
        Turn circle green or blue
        """
        # This poke ended the trial by delivering a reward
        self.reward_delivered_on_this_trial = True

        # Increment rewards and trials
        self.n_rewards += 1
        self.n_trials += 1
        self.infobox_items['N Rewards'].setText(str(self.n_rewards))
        self.infobox_items['N Trials'].setText(str(self.n_trials))
        
        # Store the rank
        self.rank_of_poke_by_trial.append(value['poke_rank'])

        # Test whether it was the first poke of the trial
        if value['first_poke']:
            # This poke was the first one
            # So this was the poke that made it a correct trial
            
            # Increment counter
            self.n_correct_trials += 1
            self.infobox_items['N Correct Trials'].setText(
                str(self.n_correct_trials))                
            
            # Store the time in the GREEN trace (correct trial)
            kpp_data = self.known_pilot_ports_correct_reward_data[kpp_idx]
            kpp_data.append(timestamp_sec)
            
            # Update the plot
            self.known_pilot_ports_correct_reward_plot[kpp_idx].setData(
                x=kpp_data,
                y=np.array([kpp_idx] * len(kpp_data)),
                )                

            # Turn the correspond poke circle GREEN (correct trial)
            self.octagon_port_plot_l[kpp_idx].setSymbolBrush('g')
        
        else:
            # This was not the first poke
            # So this poke was correct, but a mistake was made
            # on this trial
            
            # Store the time in the BLUE trace (water given)
            kpp_data = self.known_pilot_ports_reward_data[kpp_idx]
            kpp_data.append(timestamp_sec)
            
            # Update the plot
            self.known_pilot_ports_reward_plot[kpp_idx].setData(
                x=kpp_data,
                y=np.array([kpp_idx] * len(kpp_data)),
                )                
            
            # Turn the correspond poke circle BLUE (water given)
            self.octagon_port_plot_l[kpp_idx].setSymbolBrush('b')                        

        # Update FC and RCP
        if self.n_trials > 0:
            # FC
            self.infobox_items['FC'].setText(
                '{:0.3f}'.format(
                self.n_correct_trials / self.n_trials))
            
            # RCP
            self.infobox_items['RCP'].setText(
                '{:0.3f}'.format(
                np.mean(self.rank_of_poke_by_trial)))        

    def handle_unrewarded_poke(self, value, kpp_idx, timestamp_sec):
        """Handle an unrewarded poke
        
        Plot a red tick and turn the circle red
        Pokes to previously rewarded port are ignored
        """
        # This poke was unrewarded and did not end the trial
        # Either it was incorrect, or the reward was already given
        
        # TODO: Set a timer to decide if this is a consummation lick
        # from the previous trial, or it's been long enough and we should 
        # just treat this as another kind of error
        # For now, just plot these pokes as red ticks even though they
        # might be consummation licks
        
        #~ # Test whether it was a previously_rewarded_port
        #~ if value['poke_rank'] == -1:
            #~ # This was probably a consummation lick from the
            #~ # previous trial. Do nothing
            #~ pass
        
        #~ else:

        if True:
            # Store the time in the RED trace
            kpp_data = self.known_pilot_ports_poke_data[kpp_idx]
            kpp_data.append(timestamp_sec)
            
            # Update the plot
            self.known_pilot_ports_poke_plot[kpp_idx].setData(
                x=kpp_data,
                y=np.array([kpp_idx] * len(kpp_data)),
                )
            
            # Turn the correspond poke circle red,
            # unless reward has already been delivered, in which
            # case this is almost certainly consummation
            if not self.reward_delivered_on_this_trial:
                self.octagon_port_plot_l[kpp_idx].setSymbolBrush(
                    'r')

    @gui_event
    def l_stop(self, value):
        """Set all contained objects back to defaults before the next session

        """
        # Stop the timer
        self.infobox_items['Runtime'].stop_timer()
        self.infobox_items['Last poke'].stop_timer()
        self.update_timer.stop()
        
        # Don't close the Net_Node socket now or we can't receive again
        # Although find a way to close it when the user closes the Terminal

        self.state = 'IDLE'

    def l_param(self, value):
        """
        Warning:
            Not implemented

        Args:
            value:
        """
        pass

    def l_state(self, value):
        """
        Pilot letting us know its state has changed. Mostly for the case where
        we think we're running but the pi doesn't.

        Args:
            value (:attr:`.Pilot.state`): the state of our pilot

        """

        if (value in ('STOPPING', 'IDLE')) and self.state == 'RUNNING':
            #self.l_stop({})
            pass
