import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#TODO 少了PPT中的第三个方程！！！
# parameters
xy_limit = np.pi
xy_N = 80
xy_resolution = xy_limit / xy_N
time_limit = 20.
time_N = 400
time_resolution = time_limit / time_N
const_c = 0.1

# x, y, t
u = np.zeros([xy_N, xy_N, time_N])

# initial t = 0 & t = 1:
for kx in np.arange(xy_N):
    for ky in np.arange(xy_N):
        temp = xy_limit / xy_N
        u[kx, ky, 0] = 3 * np.sin(2 * kx * temp) * np.sin(ky * temp)
        u[kx, ky, 1] = 3 * np.sin(2 * kx * temp) * np.sin(ky * temp)

# calculate
coeff = (const_c * time_resolution / xy_resolution) ** 2
    
for time in np.arange(2, time_N, 1):
    u[1:-1, 1:-1, time] = \
        -u[1:-1, 1:-1, time - 2] \
        + (2 - 4 * coeff) *u[1:-1, 1:-1, time - 1] \
        + coeff * (u[0:-2, 1:-1, time - 1] + u[2:, 1:-1, time - 1] + u[1:-1, 0:-2, time - 1] + u[1:-1, 2:, time - 1])

# Plot function
def MyPlot(t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a, b = 0, 79
    temp1 = np.arange(a, b, 1)
    X, Y = np.meshgrid(temp1, temp1)
    ax.plot_wireframe(X, Y, u[a:b, a:b, t])
    ax.set_zlim([-3, 3])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Amplitude")
    ax.set_title("t = %f" % t)
    fig.savefig("2-D Oscillation t = %f.png" % t, dpi=72)
    plt.close(fig)
for t in np.arange(0, time_N, 2):
    MyPlot(t)
plt.show()
