"""Classes to plot data in the GUI."""

# Classes for plots
import datetime
import time
import logging
import os
import numpy as np
import PySide2 # have to import to tell pyqtgraph to use it
from PySide2 import QtCore
from PySide2 import QtWidgets
import pyqtgraph as pg
from functools import wraps
import autopilot
from ..utils.invoker import InvokeEvent, get_invoker

# pg config
pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')

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

        self.logger = autopilot.core.loggers.init_logger(self)


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
    | **'STATE'** | :meth:`~.Plot.l_state` | TBD                     |
    +-------------+------------------------+-------------------------+
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
        self.logger = autopilot.core.loggers.init_logger(self)

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
        # These will be rendered clockwise from northwest in the box plot
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
        else:
            raise ValueError("unrecognized parent name: {}".format(pilot))
            
        # These are used to store data we receive over time
        self.known_pilot_ports_poke_data = [
            [] for kpp in self.known_pilot_ports]
        self.known_pilot_ports_reward_data = [
            [] for kpp in self.known_pilot_ports]
        self.known_pilot_ports_correct_reward_data = [
            [] for kpp in self.known_pilot_ports]
        
        # These are used to store handles to different graph traces
        self.known_pilot_ports_poke_plot = []
        self.known_pilot_ports_reward_plot = []
        self.known_pilot_ports_correct_reward_plot = []

        
        ## Init the plots and handles
        self.init_plots()

        
        ## Station
        # Define listens to be called on each message
        self.listens = {
            'START' : self.l_start,
            'DATA' : self.l_data,
            'CONTINUOUS': self.l_data,
            'STOP' : self.l_stop,
            #'PARAM': self.l_param,
            'STATE': self.l_state
        }
        
        # Start the Net_Node
        self.node = autopilot.networking.Net_Node(
            id='P_{}'.format(self.pilot),
            upstream="T",
            port=autopilot.prefs.get('MSGPORT'),
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
            'N Rewards': QtWidgets.QLabel(),
            'Runtime' : Timer(),
            'Last poke': Timer(),
        }
        
        # This is a counter for N Rewards
        self.n_rewards = 0
        self.infobox_items['N Rewards'].setText(str(self.n_rewards))
        
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
        for n_port in range(8):
            # Determine location
            theta = np.pi / 2 - n_port / 8 * 2 * np.pi + np.pi / 4
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
        self.plot_octagon.setFixedWidth(275)
        self.plot_octagon.setFixedHeight(300)
        
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
            reward_plot = self.timecourse_plot.plot(
                x=[],
                y=np.array([]),
                pen=None, symbolBrush=(0, 255, 0), 
                symbolPen=None, symbol='arrow_up',
                )

            # Store
            self.known_pilot_ports_reward_plot.append(reward_plot)

            # Create a plot handle for "correct rewards", which is if the
            # rewarded port was poked first
            reward_plot = self.timecourse_plot.plot(
                x=[],
                y=np.array([]),
                pen=None, symbolBrush=(0, 0, 255), 
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
        
        # Set time
        self.start_time = None
        self.local_start_time = None
        
        # Update each poke plot, mostly to remove residual pokes from
        # previous session
        for poke_plot in self.known_pilot_ports_poke_plot:
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
        
        # A new "rewarded_port" was just chosen. Mark it green.
        # This means it is the beginning of a new trial.
        if 'rewarded_port' in value.keys():
            # Extract data
            poked_port = value['rewarded_port']
            
            # Find the matching kpp_idx
            try:
                kpp_idx = self.known_pilot_ports.index(poked_port)
            except ValueError:
                self.logger.debug(
                    'unknown poke received: {}'.format(poked_port))
                kpp_idx = None
            
            # Make all ports white, except rewarded port green
            for opp_idx, opp in enumerate(self.octagon_port_plot_l):
                if opp_idx == kpp_idx:
                    opp.setSymbolBrush('purple')
                else:
                    opp.setSymbolBrush('w')
        
        # Handle a reward by incrementing N_Rewards
        if 'timestamp_reward' in value.keys():
            self.n_rewards += 1
            self.infobox_items['N Rewards'].setText(str(self.n_rewards))
        
        # A port was just poked
        # Log this, mark the port red or blue, plot the poke time 
        if 'poked_port' in value.keys():
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
                # If correct_trial, plot as blue tick
                # elif reward_delivered, plot as green tick
                # else plot as red tick
                if value['trial_correct']:
                    # This only happens if it was correct and the first poke
                    # of the trial
                    # Store the time in the BLUE trace
                    kpp_data = self.known_pilot_ports_correct_reward_data[kpp_idx]
                    kpp_data.append(timestamp_sec)
                    
                    # Update the plot
                    self.known_pilot_ports_correct_reward_plot[kpp_idx].setData(
                        x=kpp_data,
                        y=np.array([kpp_idx] * len(kpp_data)),
                        )                
                    
                    # Turn the correspond poke circle blue
                    self.octagon_port_plot_l[kpp_idx].setSymbolBrush('b')                    
                    
                elif value['reward_delivered']:
                    # It was rewarded but it was not a correct trial, so
                    # they must have poked the wrong port earlier
                    # Store the time in the GREEN trace
                    kpp_data = self.known_pilot_ports_reward_data[kpp_idx]
                    kpp_data.append(timestamp_sec)
                    
                    # Update the plot
                    self.known_pilot_ports_reward_plot[kpp_idx].setData(
                        x=kpp_data,
                        y=np.array([kpp_idx] * len(kpp_data)),
                        )                
                    
                    # Turn the correspond poke circle green
                    self.octagon_port_plot_l[kpp_idx].setSymbolBrush('g')
                
                else:
                    # Incorrect poke
                    # Store the time
                    kpp_data = self.known_pilot_ports_poke_data[kpp_idx]
                    kpp_data.append(timestamp_sec)
                    
                    # Update the plot
                    self.known_pilot_ports_poke_plot[kpp_idx].setData(
                        x=kpp_data,
                        y=np.array([kpp_idx] * len(kpp_data)),
                        )
                    
                    # Turn the correspond poke circle red
                    self.octagon_port_plot_l[kpp_idx].setSymbolBrush('r')

        # If we received a trial_in_session, then update the N_trials counter
        if 'trial_in_session' in value.keys():
            # Set the textbox
            self.infobox_items['N Trials'].setText(
                str(value['trial_in_session']))

    @gui_event
    def l_stop(self, value):
        """Set all contained objects back to defaults before the next session

        """
        # Clear the plots
        #~ self.plot_octagon.clear()
        #~ self.timecourse_plot.clear()
        
        # Stop the timer
        self.infobox_items['Runtime'].stop_timer()
        self.infobox_items['Last poke'].stop_timer()
        self.update_timer.stop()
        
        # Clear the data
        # Otherwise the next session will be using the same ones
        #~ self.known_pilot_ports_poke_plot = []
        
        # Clear the data
        self.known_pilot_ports_poke_data = [
            [] for kpp in self.known_pilot_ports]
        
        # Don't close the Net_Node socket now or we can't receive again
        # Although find a way to close it when the user closes the Terminal

        self.state = 'IDLE'

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

class Timer(QtWidgets.QLabel):
    """
    A simple timer that counts... time...

    Uses a :class:`QtCore.QTimer` connected to :meth:`.Timer.update_time` .
    """
    def __init__(self):
        # type: () -> None
        super(Timer, self).__init__()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_time)

        self.start_time = None

    def start_timer(self, update_interval=1000):
        """
        Args:
            update_interval (float): How often (in ms) the timer should be updated.
        """
        self.start_time = time.time()
        self.timer.start(update_interval)

    def stop_timer(self):
        """
        you can read the sign ya punk
        """
        self.timer.stop()
        self.setText("")

    def update_time(self):
        """
        Called every (update_interval) milliseconds to set the text of the timer.

        """
        secs_elapsed = int(np.floor(time.time()-self.start_time))
        self.setText("{:02d}:{:02d}:{:02d}".format(
            int(secs_elapsed/3600), int((secs_elapsed/60))%60, secs_elapsed%60))

class Video(QtWidgets.QWidget):
    def __init__(self, videos, fps=None):
        """
        Display Video data as it is collected.

        Uses the :class:`ImageItem_TimedUpdate` class to do timed frame updates.

        Args:
            videos (list, tuple): Names of video streams that will be displayed
            fps (int): if None, draw according to ``prefs.get('DRAWFPS')``. Otherwise frequency of widget update

        Attributes:
            videos (list, tuple): Names of video streams that will be displayed
            fps (int): if None, draw according to ``prefs.get('DRAWFPS')``. Otherwise frequency of widget update
            ifps (int): 1/fps, duration of frame in s
            qs (dict): Dictionary of :class:`~queue.Queue`s in which frames will be dumped
            quitting (:class:`threading.Event`): Signal to quit drawing
            update_thread (:class:`threading.Thread`): Thread with target=:meth:`~.Video._update_frame`
            layout (:class:`PySide2.QtWidgets.QGridLayout`): Widget layout
            vid_widgets (dict): dict containing widgets for each of the individual video streams.
        """
        super(Video, self).__init__()

        self.videos = videos

        if fps is None:
            if prefs.get( 'DRAWFPS'):
                self.fps = prefs.get('DRAWFPS')
            else:
                self.fps = 10
        else:
            self.fps = fps

        self.ifps = 1.0/self.fps

        self.layout = None
        self.vid_widgets = {}


        #self.q = Queue(maxsize=1)
        self.qs = {}
        self.quitting = Event()
        self.quitting.clear()


        self.init_gui()

        self.update_thread = Thread(target=self._update_frame)
        self.update_thread.setDaemon(True)
        self.update_thread.start()

    def init_gui(self):
        self.layout = QtWidgets.QGridLayout()
        self.vid_widgets = {}


        for i, vid in enumerate(self.videos):
            vid_label = QtWidgets.QLabel(vid)

            # https://github.com/pyqtgraph/pyqtgraph/blob/3d3d0a24590a59097b6906d34b7a43d54305368d/examples/VideoSpeedTest.py#L51
            graphicsView= pg.GraphicsView(self)
            vb = pg.ViewBox()
            graphicsView.setCentralItem(vb)
            vb.setAspectLocked()
            #img = pg.ImageItem()
            img = ImageItem_TimedUpdate()
            vb.addItem(img)

            self.vid_widgets[vid] = (graphicsView, vb, img)

            # 3 videos in a row
            row = np.floor(i/3.)*2
            col = i%3

            self.layout.addWidget(vid_label, row,col, 1,1)
            self.layout.addWidget(self.vid_widgets[vid][0],row+1,col,5,1)

            # make queue for vid
            self.qs[vid] = Queue(maxsize=1)



        self.setLayout(self.layout)
        self.resize(600,700)
        self.show()

    def _update_frame(self):
        """
        Pulls frames from :attr:`.Video.qs` and feeds them to the video widgets.

        Internal method, run in thread.
        """
        last_time = 0
        this_time = 0
        while not self.quitting.is_set():

            for vid, q in self.qs.items():
                data = None
                try:
                    data = q.get_nowait()
                    self.vid_widgets[vid][2].setImage(data)

                except Empty:
                    pass
                except KeyError:
                    pass

            this_time = time()
            sleep(max(self.ifps-(this_time-last_time), 0))
            last_time = this_time

    def update_frame(self, video, data):
        """
        Put a frame for a video stream into its queue.

        If there is a waiting frame, pull it from the queue first -- it's old now.

        Args:
            video (str): name of video stream
            data (:class:`numpy.ndarray`): video frame
        """
        #pdb.set_trace()
        # cur_time = time()

        try:
            # if there's a waiting frame, it's old now so pull it.
            _ = self.qs[video].get_nowait()
        except Empty:
            pass

        try:
            # put the new frame in there.
            self.qs[video].put_nowait(data)
        except Full:
            return
        except KeyError:
            return

    def release(self):
        self.quitting.set()

    