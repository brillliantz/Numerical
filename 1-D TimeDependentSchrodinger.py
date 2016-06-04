import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from numba import jit
import time as tm

#################################################################
# parameters
sigma0 = 0.5
k0 = 5 * np.pi
mass = 1. / 2
hbar = 1.

x_limit = (0., 15.)
x_step = 0.01
x_array = np.arange(x_limit[0], x_limit[1], x_step)

time_limit = (0., 0.05)
time_step = x_step ** 2 / 400.
time_array = np.arange(time_limit[0], time_limit[1], time_step)

#################################################################
# Results
re_u = np.zeros([len(x_array), len(time_array)])
im_u = np.zeros([len(x_array), len(time_array)])

def init(x):
    '''
    return a tuple consiting Real part and Imaginary part.
    '''
    return (np.exp(-1. / 2. * ((x - 13.) / sigma0) ** 2) * np.cos(k0 * x),
            np.exp(-1. / 2. * ((x - 13.) / sigma0) ** 2) * np.sin(k0 * x))

#################################################################
# initial condition
re_u[:, 0], im_u[:, 0] = init(x_array)[0], init(x_array)[1]

#################################################################
# Bound Condition 
    # already be zero, no need to do anything.

#################################################################
# calculation
coeff = time_step / (2 * mass * (x_step) ** 2)

print tm.clock()
@jit
def calc(LEN, coeff, re_u, im_u):
    for t in np.arange(0, LEN - 2, 1):
        re_u[1:-1, t + 1] = re_u[1:-1, t] \
            - coeff * (im_u[2:, t] -2 * im_u[1:-1, t] + im_u[0:-2, t])
        im_u[1:-1, t + 1] = im_u[1:-1, t] \
                + coeff * (re_u[2:, t] -2 * re_u[1:-1, t] + re_u[0:-2, t])
        print ("%d / %d" % (t, LEN))

LEN = len(time_array)
calc(LEN, coeff, re_u, im_u)
print tm.clock()

im_u = im_u[:, 0:len(im_u[0,:]):80]
re_u = re_u[:, 0:len(re_u[0,:]):80]
abs_u = re_u ** 2 + im_u ** 2

#################################################################
# shouw animation or plot static results

m = 1
if (m == 1):
    #plt.plot(temp_x, Re_u[:, 0])
    #plt.plot(temp_x, Im_u[:, 0])
    fig = plt.figure()
    ax1 = fig.gca(projection='3d')
    #ax1.plot(x_array, abs_u[:,10], label='Re')
    #ax1.plot(x_array, abs_u[:,len(time_array)/2], label='Im')
    x3d, y3d = np.meshgrid(time_array, x_array)
    print "printing..."
    ax1.plot_surface(x3d, y3d, abs_u, 
                     cmap=cm.coolwarm
                     )
    # ax1.set_xlim([8,15])
    # ax1.set_ylim([0,1e-15])
    plt.show()
else:
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 15), ylim=(0, 1.))
    line, = ax.plot([], [], lw=2, c='r')
    
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        print "int!"
        return line,
    
    # animation function.  This is called sequentially
    def animate(i):
        print i
        x = x_array
        y = abs_u[:,i]
        line.set_data(x, y)
        return line,
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(time_array), interval=1, blit=True)
    
    #mywriter = animation.FFMpegWriter()
    """plt.rcParams['animation.ffmpeg_path'] ='E:\\Anaconda3\\envs\\env2\\Lib\\site-packages\\ffmpeg\\bin\\ffmpeg.exe'
    mywriter = animation.FFMpegWriter()
    anim.save('basic_animation.mp4', writer=mywriter, fps=30, extra_args=['-vcodec', 'libx264'])"""
    plt.show()




#################################################################
# save results to local drive and load them later
'''
im_u = im_u[:, 0:len(im_u[0,:]):10]
re_u = re_u[:, 0:len(re_u[0,:]):10]
np.save('comp.results/x_array', x_array)
np.save('comp.results/time_array', time_array)
#np.save('comp.results/re_u', re_u)
#np.save('comp.results/im_u', im_u)
abs_u = re_u**2 + im_u**2
np.save('comp.results/abs_u', abs_u)


x_array = np.load('comp.results/x_array.npy')
time_array = np.load('comp.results/time_array.npy')
re_u= np.load('comp.results/re_u.npy')
im_u= np.load('comp.results/im_u.npy')
abs_u= np.load('comp.results/abs_u.npy')



'''