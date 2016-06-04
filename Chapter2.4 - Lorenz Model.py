import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Initialize parameters
sigma = 10
b = 8 / 3
r = 25
step_length = 0.005
time_initial = 0.
time_final = 50.
x_initial = 5.
y_initial = -3.
z_initial = 60.

## My particular function
def Lorenz(t, x, y, z):
    kx = sigma * (y - x)
    ky = x * (r - z) - y
    kz = x * y - b * z
    return np.array([kx, ky, kz])
def fun1(x, y):
    return (-4. * x ** 3 * y ** 2)

## Numerical Integration Methods
def RK4Step(f, t, step, *args):
    args = np.array(args)
    k1 = f(t, *args)
    k2 = f(t + step / 2, *args + k1 * step / 2)
    k3 = f(t + step / 2, *args + k2 * step / 2)
    k4 = f(t + step, *args + step * k3)
    k = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (k * step)

## Do the integration
def MyInte(f, method, t0, t1, step, *args_initial):
    args_initial = np.array(args_initial)
    t_value = np.arange(t0, t1, step)
    dimension = args_initial.shape[0]
    N = t_value.shape[0]
    args_value = np.zeros((N, dimension))
    args_value[0, :] = args_initial.copy()
    for i in range(N - 1):
        args_value[i + 1, :] = args_value[i, :] + RK4Step(f, t_value[i], step, *args_value[i, :])
    return np.column_stack((t_value, args_value))

## Plot Function
def LorenzPlot(current_r):
    global r
    r = current_r
    value = MyInte(Lorenz, RK4Step, time_initial, time_final, step_length, x_initial, y_initial, z_initial)
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(value[:, 1], value[:, 2], value[:, 3], linewidth=1.8, color='c')
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")
    ax1.set_zlabel("Z Axis")
    ax1.set_title("Lorentz Attractor 3D r = %f" % r)
    ax2.plot(value[:, 1], value[:, 3], color='k')
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Z Axis")
    ax2.set_title("Projection on X-Z surface")
    # fig.savefig("Lorentz r = %f.png" % r, dpi=72)

r_value = np.array([0.1,
                    # 0.8, 1, 1.6, 5, 14, 15, 18, 22, 25, 30,
    # 1, 2, 14, 16, 25,
    28, 34, 99.96
                    ])
for j in r_value:
    LorenzPlot(j)
plt.show()
