from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import re
import numpy
import math
import pandas as pd

file1 = open('int.dat')
data = file1.read()
data = data.split("\n")

X = []
Y = []
Z = []


for item in data:
    item2 = re.sub('\s+',' ',item).strip()
    item2 = item2.split(" ")
    #item2.pop(0)
    #item2.pop(-1)
    if len(item2) == 3:
        X.append(float(item2[0]))
        Y.append(float(item2[1]))
        Z.append(float(item2[2]))

l = math.sqrt(len(X))
l = int(l)
fig = pyplot.figure()
ax = fig.gca(projection='3d')
X = numpy.array(X)
Y = numpy.array(Y)
Z = numpy.array(Z)
Z = Z.reshape(l,l)
X = X.reshape(l,l)
Y = Y.reshape(l,l)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=False)

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(1, 2.5)


ax.set_xlabel('$x$')
ax.set_ylabel('$y$');
pyplot.show()
