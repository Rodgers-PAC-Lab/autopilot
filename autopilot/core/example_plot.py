import PySide2
import pyqtgraph as pg
import numpy as np

from pyqtgraph.Qt import QtCore
from time import perf_counter

poke_data = np.random.uniform(low=0, high=600, size=(8, 100))


start_time = perf_counter()

app = pg.mkQApp("Plotting Example")

win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
win.resize(1000,300)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)


## Ocatgon plot
plot_octagon = win.addPlot(title="current trial")
port_plot_l = []
for n_port in range(8):
    theta = n_port / 8 * 2 * np.pi
    x_pos = np.cos(theta)
    y_pos = np.sin(theta)
    port_plot = plot_octagon.plot(
        x=[x_pos], y=[y_pos],
        pen=None, symbolBrush=(255, 0, 0), symbolPen=None, symbol='o',
        )
    port_plot_l.append(port_plot)
plot_octagon.setRange(xRange=(-1, 1), yRange=(-1, 1))
plot_octagon.setFixedWidth(225)
plot_octagon.setFixedHeight(250)

txt = pg.TextItem('asdf', color='white', anchor=(0.5, 0.5))
txt.setPos(0, .25)
plot_octagon.addItem(txt)

## Timecourse plot
p3 = win.addPlot(title="Drawing with points")
p3.setRange(xRange=[0, 25*60])

line_of_current_time = p3.plot(
    x=[600, 600], y=[-1, 8], pen='white')


poke_plot_list = []
for n_row in range(len(poke_data)):
    poke_plot = p3.plot(
        x=[0],
        y=np.array([n_row]),
        pen=None, symbolBrush=(255, 0, 0), symbolPen=None, symbol='arrow_down',
        )
    poke_plot_list.append(poke_plot)
    

def update():
    current_time = perf_counter() - start_time
    for n_row in range(len(poke_data)):
        this_poke_data = np.sort(poke_data[n_row])
        this_poke_data = this_poke_data[this_poke_data < current_time]

        poke_plot = poke_plot_list[n_row]
        poke_plot.setData(
            x=this_poke_data,
            y=np.array([n_row] * len(this_poke_data))
            )

    line_of_current_time.setData(x=[current_time, current_time], y=[-1, 8])
    
    if current_time > 1:
        port_plot_l[0].setSymbolBrush('w')
    

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

pg.exec()