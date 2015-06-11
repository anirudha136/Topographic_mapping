__author__ = 'anirudha'

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import Rbf
import math



def pol2cart(rho, phi):
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    return(x, y)

def updatefig(*args):
    global i
    rbf = Rbf(x_chan, y_chan, data[i,:], function='linear')
    ZI = rbf(grid_y,grid_x)
    ZI = channel_plot(points,ZI)
    ZI = plot_head(ZI,x_c,y_c)
    ZI = plot_nose(ZI,x6,y6,x7,y7)
    t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12 = update_text()
    im.set_data(ZI)
    i=i+1
    print i
    return im,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12

def channel_plot(points,ZI):
    points = points.astype(int)
    ZI[points[:,1],points[:,0]] = -2.
    ZI[points[:,1]+1,points[:,0]] = -2.
    ZI[points[:,1]+1,points[:,0]+1] = -2.
    ZI[points[:,1]+1,points[:,0]-1] = -2.
    ZI[points[:,1]-1,points[:,0]+1] = -2.
    ZI[points[:,1]-1,points[:,0]-1] = -2.
    ZI[points[:,1]-1,points[:,0]] = -2.
    ZI[points[:,1],points[:,0]+1] = -2.
    ZI[points[:,1],points[:,0]-1] = -2.
    return ZI

def plot_head(ZI,x_c,y_c):
    ZI[x_c,y_c] = -2
    return ZI

def plot_nose(ZI,x6,y6,x7,y7):
    ZI[y6,x6] = -2
    ZI[y7,x7] = -2
    return ZI

def plot_ear(ZI):
    return ZI

def init():
    return line,annotation

def update_text(*args):
    # This is not working i 1.2.1
    # annotation.set_position((newData[0][0], newData[1][0]))
    #annotation.xyann = (points[0][0], points[1][0])
    return annotation,annotation1,annotation2,annotation3,annotation4,annotation5,annotation6,annotation7,\
           annotation8,annotation9,annotation10,annotation11

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

data1 = pd.read_csv('data.csv')
fs =128.

data2 = data1.values
data = (data2 - np.mean(data2))/np.std(data2)

print data.shape
chan2 = np.zeros((1,2))
with open('channellocs1412.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        chan2 = np.vstack((chan2,row))

chan = chan2[1:,:]
x_chan = np.zeros((chan.shape[0]))
y_chan = np.zeros((chan.shape[0]))
for i in range(chan.shape[0]):
    x_chan[i],y_chan[i] = pol2cart(float(chan[i,1]),float(chan[i,0]))

# Plot of head
g=150.
m1=(5./6.)*g
b1=0.00395*g
radius = 70.
theta = np.linspace(0,2*np.pi,1000)
x_c = (radius*np.cos(theta)+g/2).astype(int)
y_c = (radius*np.sin(theta)+g/2).astype(int)

#Plot of nose
width_nose=5
height_nose=2.5
delta_theta= math.acos((width_nose/2)/radius)
x1=radius*math.cos(delta_theta)+g/2
y1=radius*math.sin(delta_theta)+g/2
x2=x1-width_nose
y2=y1
x3=0+g/2
y3=radius+height_nose+g/2;
x6=(np.linspace(x2,g/2,200)).astype(int)
x7=(np.linspace(g/2,x2,200)).astype(int)
y6=(((y3-y2)/(width_nose/2))*(x6-x2)+y2).astype(int)
y7=(-((y3-y2)/(width_nose/2))*(x7-x3)+y3).astype(int)

# Plot of ear


# Plot Chan no


points = np.transpose([m1*(x_chan+b1),m1*(y_chan+b1)])
grid_x, grid_y = np.mgrid[-0.6:0.6:152j, -0.6:0.6:152j]
fig = plt.figure()
ax = plt.axes(xlim=(0, 152), ylim=(0, 152))

#im = plt.imshow(grid_y,cmap=plt.get_cmap('jet'),extent=(0,2,0,2), origin='lower')

plt.title("Scalp activity.")

rbf = Rbf(x_chan, y_chan, data[0], function='linear')
ZI = rbf(grid_x,grid_y)
im = plt.imshow(ZI,aspect='auto',origin='lower')
plt.colorbar()

line, = ax.plot([], [], 'r-')

annotation = ax.annotate('O1', xy=(points[0,0]+1, points[0,1]-2),size=10)
annotation1 = ax.annotate('FC6', xy=(points[1,0]+1, points[1,1]-2),size=10)
annotation2 = ax.annotate('F3', xy=(points[2,0]+1, points[2,1]-2),size=10)
annotation3 = ax.annotate('T7', xy=(points[3,0]+1, points[3,1]-2),size=10)
annotation4 = ax.annotate('F8', xy=(points[4,0]+1, points[4,1]-2),size=10)
annotation5 = ax.annotate('P7', xy=(points[5,0]+1, points[5,1]-2),size=10)
annotation6 = ax.annotate('P8', xy=(points[6,0]+1, points[6,1]-2),size=10)
annotation7 = ax.annotate('F7', xy=(points[7,0]+1, points[7,1]-2),size=10)
annotation8 = ax.annotate('T8', xy=(points[8,0]+1, points[8,1]-2),size=10)
annotation9 = ax.annotate('F4', xy=(points[9,0]+1, points[9,1]-2),size=10)
annotation10 = ax.annotate('FC5', xy=(points[10,0]+1, points[10,1]-2),size=10)
annotation11 = ax.annotate('O2', xy=(points[11,0]+1, points[11,1]-2),size=10)



annotation.set_animated(True)
annotation1.set_animated(True)
annotation2.set_animated(True)
annotation3.set_animated(True)
annotation4.set_animated(True)
annotation5.set_animated(True)
annotation6.set_animated(True)
annotation7.set_animated(True)
annotation8.set_animated(True)
annotation9.set_animated(True)
annotation10.set_animated(True)
annotation11.set_animated(True)

i=0
ani = animation.FuncAnimation(fig, updatefig, interval=np.round(1/fs), blit=True,init_func=init)
#ani.save('topoplot.mp4', writer=writer)
plt.show()
