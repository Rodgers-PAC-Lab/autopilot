"""
Classes to plot data in the GUI.

.. todo::

    Add all possible plot objects and options in list.

Note:
    Plot objects need to be added to :data:`~.plots.PLOT_LIST` in order to be reachable.


"""

# Classes for plots
import datetime
import logging
import os
from collections import deque
import numpy as np
import PySide2 # have to import to tell pyqtgraph to use it
import pandas as pd
from PySide2 import QtCore
from PySide2 import QtWidgets
import pyqtgraph as pg
from time import time, sleep
from itertools import count
from functools import wraps
from threading import Event, Thread
from queue import Queue, Empty, Full
#import cv2
pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')

# from pyqtgraph.widgets.RawImageWidget import RawImageWidget, RawImageGLWidget

import autopilot
from autopilot import prefs
from autopilot.core import styles
from ..utils.invoker import InvokeEvent, Invoker, get_invoker
from autopilot.networking import Net_Node
from autopilot.core.loggers import init_logger


############
# Plot list at the bottom!
###########

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
    """
    Widget that hosts a :class:`pyqtgraph.PlotWidget` and manages
    graphical objects for one pilot depending on the task.

    **listens**

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

    def __init__(self, pilot, x_width=50, parent=None):
        """
        Args:
            pilot (str): The name of our pilot
            x_width (int): How many trials in the past should we plot?
        """
        super(Plot, self).__init__()
        
        # Init logger
        self.logger = init_logger(self)
        self.logger.debug('inside __init__')

        # Init some variables
        self.parent = parent
        self.session_trials = 0
        self.info = {}
        self.state = "IDLE"
        self.invoker = get_invoker()

        # The name of our pilot, used to listen for events
        self.pilot = pilot

        # For storing data that we receive
        self.chosen_stimulus_l = []
        self.chosen_response_l = []
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
        
        # These are created in init_plots
        self.known_pilot_ports_poke_data = []
        self.known_pilot_ports_poke_plot = []
        
        
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

        self.node = Net_Node(
            id='P_{}'.format(self.pilot),
            upstream="T",
            port=prefs.get('MSGPORT'),
            listens=self.listens,
            instance=True)

    @gui_event
    def init_plots(self):
        """
        Make pre-task GUI objects and set basic visual parameters of plot widgets
        """
        # Announce
        self.logger.debug('inside init_plots')
        
        # This creates a horizontal box layout for all the plot widgets,
        # such as the info box and the plot.
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(2,2,2,2)
        self.setLayout(self.layout)
    
        
        ## Widget 1: Infobox
        # Create the first widget: an infobox for n_trials, etc
        self.infobox = QtWidgets.QFormLayout()
        self.n_trials = count()
        self.session_trials = 0
        self.info = {
            'N Trials': QtWidgets.QLabel(),
            'Runtime' : Timer(),
            'Session' : QtWidgets.QLabel(),
            'Protocol': QtWidgets.QLabel(),
            'Step'    : QtWidgets.QLabel()
        }
        
        # Add rows to infobox
        for k, v in self.info.items():
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
            theta = np.pi / 2 - n_port / 8 * 2 * np.pi
            x_pos = np.cos(theta)
            y_pos = np.sin(theta)
            
            # Plot the circle
            port_plot = self.plot_octagon.plot(
                x=[x_pos], y=[y_pos],
                pen=None, symbolBrush=(255, 0, 0), symbolPen=None, symbol='o',
                )
            
            # Text
            txt = pg.TextItem(self.known_pilot_ports[n_port],
                color='white', anchor=(0.5, 0.5))
            txt.setPos(x_pos * .8, y_pos * .8)
            txt.setAngle(np.mod(theta * 180 / np.pi, 180) - 90)
            self.plot_octagon.addItem(txt)
            
            # Store the handle
            self.octagon_port_plot_l.append(port_plot)
        
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
        self.timecourse_plot.setRange(xRange=[0, 3 * 60], yRange=[0, 7])
        self.timecourse_plot.getViewBox().invertY(True)
       
        # Add a vertical line indicating the current time
        # This will shift over throughout the session
        self.line_of_current_time = self.timecourse_plot.plot(
            x=[0, 0], y=[-1, 8], pen='white')

        # Within self.timecourse_plot, add a trace for pokes made into
        # each port
        ticks_l = []
        for n_row in range(len(self.known_pilot_ports)):
            # Create the plot handle
            poke_plot = self.timecourse_plot.plot(
                x=[0],
                y=np.array([n_row]),
                pen=None, symbolBrush=(255, 0, 0), 
                symbolPen=None, symbol='arrow_down',
                )
            
            # Store
            self.known_pilot_ports_poke_plot.append(poke_plot)
            
            # Also use this list to store the times of the pokes
            self.known_pilot_ports_poke_data.append([])
            
            # Also keep track of yticks
            ticks_l.append((n_row, self.known_pilot_ports[n_row]))

        # Set ticks
        self.timecourse_plot.getAxis('left').setTicks([ticks_l])

        # Add to layout
        self.layout.addWidget(self.timecourse_plot, 8)


    @gui_event
    def l_start(self, value):

        self.logger.debug('inside l_start')

        if self.state in ("RUNNING", "INITIALIZING"):
            self.logger.debug('returning from l_start, already running')
            return

        self.state = "INITIALIZING"

        # set infobox stuff
        self.n_trials = count()
        self.session_trials = 0
        self.info['N Trials'].setText(str(value['current_trial']))
        self.info['Runtime'].start_timer()
        self.info['Step'].setText(str(value['step']))
        self.info['Session'].setText(str(value['session']))
        self.info['Protocol'].setText(value['step_name'])
        self.state = 'RUNNING'
        
        self.start_time = None
        self.local_start_time = None

        
        self.update_timer = pg.QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_time_bar)
        self.update_timer.start(50)        
    
    def update_time_bar(self):
        # Use current time to approximately update timebar
        if self.local_start_time is not None:
            current_time = datetime.datetime.now()
            approx_time_in_session = (
                current_time - self.local_start_time).total_seconds()
            self.line_of_current_time.setData(
                x=[approx_time_in_session, approx_time_in_session], y=[-1, 9])

    @gui_event
    def l_data(self, value):
        """
        Receive some data, if we were told to plot it, stash the data
        and update the assigned plot.

        Args:
            value (dict): Value field of a data message sent during a task.
        """
        # Announce
        self.logger.debug('Plot.l_data: received {}'.format(value))
        
        # Return if we're not ready to take data
        if self.state == "INITIALIZING":
            self.logger.debug('returning from l_data, still initializing')
            return
        if self.state == "IDLE":
            self.logger.debug('returning from l_data, now idle')
            return
        
        # Use the start time of the first trial
        if 'timestamp_trial_start' in value and self.start_time is None:
            self.logger.debug(
                'setting start time to {}'.format(value['timestamp_trial_start']))
            self.start_time = datetime.datetime.fromisoformat(
                value['timestamp_trial_start'])
            
            # Also store approx local start time
            self.local_start_time = datetime.datetime.now()

        # Get the timestamp of this message
        if 'timestamp' in value:
            timestamp_dt = datetime.datetime.fromisoformat(value['timestamp'])
            timestamp_sec = (timestamp_dt - self.start_time).total_seconds()
            self.line_of_current_time.setData(
                x=[timestamp_sec, timestamp_sec], y=[-1, 9])
        
        # Store the data received
        if 'rewarded_port' in value.keys():
            # Extract data
            poked_port = value['rewarded_port']
            
            # Turn the corresponding port white
            try:
                kpp_idx = self.known_pilot_ports.index(poked_port)
            except ValueError:
                self.logger.debug(
                    'unknown poke received: {}'.format(poked_port))
                kpp_idx = None
            
            # Make all ports white, except rewarded port green
            for opp_idx, opp in enumerate(self.octagon_port_plot_l):
                if opp_idx == kpp_idx:
                    opp.setSymbolBrush('g')
                else:
                    opp.setSymbolBrush('w')
        
        if 'timestamp_reward' in value.keys():
            #self.timestamp_reward_l.append(value['timestamp_reward'])
            pass
        
        # Store the time of the poke
        if 'poked_port' in value.keys():
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
                # Store the time
                kpp_data = self.known_pilot_ports_poke_data[kpp_idx]
                kpp_data.append(timestamp_sec)
                
                # Update the plot
                self.known_pilot_ports_poke_plot[kpp_idx].setData(
                    x=kpp_data,
                    y=np.array([kpp_idx] * len(kpp_data)),
                    )
                
                # Turn the correspond poke circle green
                self.octagon_port_plot_l[kpp_idx].setSymbolBrush('r')

        # If we received a trial_num, then update the N_trials counter
        if 'trial_num' in value.keys():
            # Store this as last_trial
            self.last_trial = value.pop('trial_num')
            
            # Set the textbox
            self.info['N Trials'].setText("{}".format(self.last_trial))
        

    @gui_event
    def l_stop(self, value):
        """
        Clean up the plot objects.

        Args:
            value (dict): if "graduation" is a key, don't stop the timer.
        """
        self.data = {}
        self.plots = {}
        self.timecourse_plot.clear()
        try:
            if isinstance(value, str) or ('graduation' not in value.keys()):
                self.info['Runtime'].stop_timer()
        except:
            self.info['Runtime'].stop_timer()



        self.info['N Trials'].setText('')
        self.info['Step'].setText('')
        self.info['Session'].setText('')
        self.info['Protocol'].setText('')

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





###################################
# Curve subclasses
class Point(pg.PlotDataItem):
    """
    A simple point.

    Attributes:
        brush (:class:`QtWidgets.QBrush`)
        pen (:class:`QtWidgets.QPen`)
    """

    def __init__(self, color=(255,0,0), size=5, **kwargs):
        """
        Args:
            color (tuple): RGB color of points
            size (int): width in px.
        """
        super(Point, self).__init__()

        self.continuous = False
        #~ if 'continuous' in kwargs.keys():
            #~ self.continuous = kwargs['continuous']

        self.brush = pg.mkBrush(color)
        self.pen   = pg.mkPen(color, width=size)
        self.size  = size

    def update(self, data):
        """
        Args:
            data (:class:`numpy.ndarray`): an x_width x 2 array where
                column 0 is trial number and column 1 is the value,
                where value can be "L", "C", "R" or a float.
        """
        # data should come in as an n x 2 array,
        # 0th column - trial number (x), 1st - (y) value
        data[data=="R"] = 1
        data[data=="C"] = 0.5
        data[data=="L"] = 0
        data = data.astype(np.float)
        print(data)
        print(data[..., 0])
        print(data[..., 1])

        #~ self.scatter.setData(x=data[...,0], y=data[...,1], size=self.size,
                             #~ brush=self.brush, symbol='o', pen=self.pen)
        
        print(self.scatter)
        self.scatter.setData(
            x=np.array([0., 10.,20.,30.]), y=np.array([0., 1., 2., 3.]), 
            size=12, symbol='o', brush=self.brush, pen=self.pen)

class Line(pg.PlotDataItem):
    """
    A simple line
    """

    def __init__(self, color=(0,0,0), size=1, **kwargs):
        super(Line, self).__init__(**kwargs)

        self.brush = pg.mkBrush(color)
        self.pen = pg.mkPen(color, width=size)
        self.size = size

    def update(self, data):
        data[data=="R"] = 1
        data[data=="L"] = 0
        data[data=="C"] = 0.5
        data = data.astype(np.float)

        self.curve.setData(data[...,0], data[...,1])



class Segment(pg.PlotDataItem):
    """
    A line segment that draws from 0.5 to some endpoint.
    """
    def __init__(self, **kwargs):
        # type: () -> None
        super(Segment, self).__init__(**kwargs)

    def update(self, data):
        """
        data is doubled and then every other value is set to 0.5,
        then :meth:`~pyqtgraph.PlotDataItem.curve.setData` is used with
        `connect='pairs'` to make line segments.

        Args:
            data (:class:`numpy.ndarray`): an x_width x 2 array where
                column 0 is trial number and column 1 is the value,
                where value can be "L", "C", "R" or a float.
        """
        # data should come in as an n x 2 array,
        # 0th column - trial number (x), 1st - (y) value
        data[data=="R"] = 1
        data[data=="L"] = 0
        data[data=="C"] = 0.5
        data = data.astype(np.float)

        xs = np.repeat(data[...,0],2)
        ys = np.repeat(data[...,1],2)
        ys[::2] = 0.5

        self.curve.setData(xs, ys, connect='pairs', pen='k')


class Roll_Mean(pg.PlotDataItem):
    """
    Shaded area underneath a rolling average.

    Typically used as a rolling mean of corrects, so area above and below 0.5 is drawn.
    """
    def __init__(self, winsize=10, **kwargs):
        # type: (int) -> None
        """
        Args:
            winsize (int): number of trials in the past to take a rolling mean of
        """
        super(Roll_Mean, self).__init__()

        self.winsize = winsize

        self.setFillLevel(0.5)

        self.series = pd.Series()

        self.brush = pg.mkBrush((0,0,0,100))
        self.setBrush(self.brush)

    def update(self, data):
        """
        Args:
            data (:class:`numpy.ndarray`): an x_width x 2 array where
                column 0 is trial number and column 1 is the value.
        """
        # data should come in as an n x 2 array,
        # 0th column - trial number (x), 1st - (y) value
        data = data.astype(np.float)

        self.series = pd.Series(data[...,1])
        ys = self.series.rolling(self.winsize, min_periods=0).mean().to_numpy()

        #print(ys)

        self.curve.setData(data[...,0], ys, fillLevel=0.5)

class Shaded(pg.PlotDataItem):
    """
    Shaded area for a continuous plot
    """

    def __init__(self, **kwargs):
        super(Shaded, self).__init__()

        #self.dur = float(dur) # duration of time to display points in seconds
        self.setFillLevel(0)
        self.series = pd.Series()

        self.getBoundingParents()


        self.brush = pg.mkBrush((0,0,0,100))
        self.setBrush(self.brush)

        self.max_num = 0


    def update(self, data):
        """
        Args:
            data (:class:`numpy.ndarray`): an x_width x 2 array where
                column 0 is time and column 1 is the value.
        """
        # data should come in as an n x 2 array,
        # 0th column - trial number (x), 1st - (y) value
        data = data.astype(np.float)

        self.max_num = float(np.abs(np.max(data[:,1])))

        if self.max_num > 1.0:
            data[:,1] = (data[:,1]/(self.max_num*2.0))+0.5
        #print(ys)

        self.curve.setData(data[...,0], data[...,1], fillLevel=0)




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
        self.start_time = time()
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
        secs_elapsed = int(np.floor(time()-self.start_time))
        self.setText("{:02d}:{:02d}:{:02d}".format(int(secs_elapsed/3600), int((secs_elapsed/60))%60, secs_elapsed%60))


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
            self.qs[vid] = deque(maxlen=1)



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
                    data = q.popleft()
                    self.vid_widgets[vid][2].setImage(data)

                except IndexError:
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
            # put the new frame in there.
            self.qs[video].append(data)
        except KeyError:
            return

    def release(self):
        self.quitting.set()


class HLine(QtWidgets.QFrame):
    """
    A Horizontal line.
    """
    def __init__(self):
        # type: () -> None
        super(HLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)

VIDEO_TIMER = None

class ImageItem_TimedUpdate(pg.ImageItem):
    """
    Reclass of :class:`pyqtgraph.ImageItem` to update with a fixed fps.

    Rather than calling :meth:`~pyqtgraph.ImageItem.update` every time a frame is updated,
    call it according to the timer.

    fps is set according to ``prefs.get('DRAWFPS')``, if not available, draw at 10fps

    Attributes:
        timer (:class:`~PySide2.QtCore.QTimer`): Timer held in ``globals()`` that synchronizes frame updates across
            image items


    """

    def __init__(self, *args, **kwargs):
        super(ImageItem_TimedUpdate, self).__init__(*args, **kwargs)

        if globals()['VIDEO_TIMER'] is None:
            globals()['VIDEO_TIMER'] = QtCore.QTimer()


        self.timer = globals()['VIDEO_TIMER']
        self.timer.stop()
        self.timer.timeout.connect(self.update_img)
        if prefs.get( 'DRAWFPS'):
            self.fps = prefs.get('DRAWFPS')
        else:
            self.fps = 10.
        self.timer.start(1./self.fps)




    def setImage(self, image=None, autoLevels=None, **kargs):
        #profile = debug.Profiler()

        gotNewData = False
        if image is None:
            if self.image is None:
                return
        else:
            gotNewData = True
            shapeChanged = (self.image is None or image.shape != self.image.shape)
            image = image.view(np.ndarray)
            if self.image is None or image.dtype != self.image.dtype:
                self._effectiveLut = None
            self.image = image
            if self.image.shape[0] > 2 ** 15 - 1 or self.image.shape[1] > 2 ** 15 - 1:
                if 'autoDownsample' not in kargs:
                    kargs['autoDownsample'] = True
            if shapeChanged:
                self.prepareGeometryChange()
                self.informViewBoundsChanged()

        #profile()

        if autoLevels is None:
            if 'levels' in kargs:
                autoLevels = False
            else:
                autoLevels = True
        if autoLevels:
            img = self.image
            while img.size > 2 ** 16:
                img = img[::2, ::2]
            mn, mx = np.nanmin(img), np.nanmax(img)
            # mn and mx can still be NaN if the data is all-NaN
            if mn == mx or np.isnan(mn) or np.isnan(mx):
                mn = 0
                mx = 255
            kargs['levels'] = [mn, mx]


        self.setOpts(update=False, **kargs)

        self.qimage = None

        if gotNewData:
            self.sigImageChanged.emit()

    def update_img(self):
        """
        Call :meth:`~ImageItem_TimedUpdate.update`
        """
        self.update()

    def __del__(self):
        super(ImageItem_TimedUpdate,self).__del__()
        self.timer.stop()




PLOT_LIST = {
    'point':Point,
    'segment':Segment,
    'rollmean':Roll_Mean,
    'shaded':Shaded,
    'line': Line
    # 'highlight':Highlight
}
"""
A dictionary connecting plot keys to objects.

TODO:
    Just reference the plot objects.
"""