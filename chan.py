__author__ = 'anirudha'

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata,interpn


def pol2cart(rho, phi):
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    return(x, y)



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


x1 = np.zeros((chan.shape[0]))
y1 = np.zeros((chan.shape[0]))
for i in range(chan.shape[0]):
    x1[i],y1[i] = pol2cart(float(chan[i,1]),float(chan[i,0]))
points = np.transpose([x1+0.5,y1+0.5])
print data[0].shape
grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]

print grid_y.shape

fig = plt.figure()
im = plt.imshow(grid_y,cmap=plt.get_cmap('jet'),extent=(0,2,0,2), origin='lower')
plt.colorbar()
plt.title("Scalp activity.")
i=0
def updatefig(*args):
    global i
    z = griddata(points,data[i,:],(grid_x,grid_y),method='cubic')
    im.set_array(z)

    i=i+1
    print i
    return im,
ani = animation.FuncAnimation(fig, updatefig, interval=np.round(1/fs), blit=True)
plt.show()