import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D ##library for 3d projection plots
from jinja2 import Template


import pycuda.autoinit
import pycuda.driver as drv
from pycuda import driver, compiler, gpuarray, tools

from pycuda.compiler import SourceModule


# CUDA Data

dev = pycuda.autoinit.device

MAX_BLOCK_DIM_X = dev.get_attribute(pycuda.driver.device_attribute.MAX_BLOCK_DIM_X)
MAX_BLOCK_DIM_Y = dev.get_attribute(pycuda.driver.device_attribute.MAX_BLOCK_DIM_Y)
MAX_BLOCK_DIM_Z = dev.get_attribute(pycuda.driver.device_attribute.MAX_BLOCK_DIM_Z)
MAX_GRID_DIM_X = dev.get_attribute(pycuda.driver.device_attribute.MAX_GRID_DIM_X)
MAX_GRID_DIM_Y = dev.get_attribute(pycuda.driver.device_attribute.MAX_GRID_DIM_Y)
MAX_GRID_DIM_Z = dev.get_attribute(pycuda.driver.device_attribute.MAX_GRID_DIM_Z)
TOTAL_CONSTANT_MEMORY = dev.get_attribute(pycuda.driver.device_attribute.TOTAL_CONSTANT_MEMORY)
MAX_THREADS_PER_BLOCK = dev.get_attribute(pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK)

print(MAX_THREADS_PER_BLOCK)


###variable declarations
nx = 4096
ny = 4096
nt = 10000
nu = .05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu
r = (nu*dt)/dx**2

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

X, Y = np.meshgrid(x, y)

u_old = np.ones((ny, nx))  

# u_old[1,1] = 10000
# print(u_old)

u_new = u_old.copy()

u_new[int(1.5 / dy):int(1 / dy + 1),int(1.5 / dx):int(1 / dx + 1)] = 2  

XDir = u_new.shape[0]
YDir = u_new.shape[1]

tpl = Template("""

        __global__ void heatCalculate(double *u_new, double *u_old)
        {
    
            int i = (blockIdx.x)* blockDim.x + threadIdx.x;
            int j = (blockIdx.y)* blockDim.y + threadIdx.y;

            int width = {{ POINT_WIDTH }};

            double r = {{ DELTA }};

            double u1 = u_old[i + (width * j)];
            double ul = u_old[(i-1) + (width * j)];
            double ur = u_old[(i+1) + (width * j)];
            double utop = u_old[i + (width * (j+1))];
            double ubottom = u_old[i + (width * (j-1))];
            double test = 0;

            if (i > 0 && i < {{ POINT_SIZE_X }} - 1 && j > 0 && j < {{ POINT_SIZE_Y }} - 1){
                test = u1 + (r * (ul + ur + utop + ubottom - (4 * u1)));
                u_new[i + width * j] = test;
            }
        

        }""")

rendered_tpl = tpl.render(POINT_SIZE_X=XDir, POINT_SIZE_Y=YDir, POINT_WIDTH=YDir, DELTA=r)
mod = SourceModule(rendered_tpl)


heatCalculate = mod.get_function("heatCalculate")


GRID_X = int(nx/32)
GRID_Y = int(ny/8)


for i in range(nt + 1):
    u_old = u_new
    # print(i)
    heatCalculate(drv.InOut(u_new.ravel()), drv.In(u_old.ravel()), block=(32,8,1), grid=(GRID_X,GRID_Y))


u_new = u_new.reshape(XDir,YDir)
# # print(u_new)
# fig = pyplot.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, u_new[:], rstride=1, cstride=1, cmap=cm.viridis,
#     linewidth=0, antialiased=True)
# ax.set_xlim(0, 2)
# ax.set_ylim(0, 2)
# ax.set_zlim(1, 2.5)
# pyplot.show()