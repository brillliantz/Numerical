import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

######################################################################
# Parameters
temprature = 300.
Kb = 0.1
# eta = 1E-6
# radius = 1E-8
# Gamma = 6 * np.pi * \eta * r
gamma = 100.
# Einstein relationship
D = Kb * temprature / gamma

t_period0 = 2
saw_period0 = 3.
saw_amplitude0 = 2.

######################################################################
# My functions
def SawTooth(x, amplitude = 2., saw_period = 1.):
  '''  
  Generate a sawtooth wave with changable period and amplitude.
  '''  
  fraction = 0.05
  re = np.zeros_like(x)
  wall = ((x % saw_period) - saw_period * fraction < 0)
  notwall = np.logical_not(wall)
  #if ((x % saw_period) - saw_period / 100 < 0):
  re[wall] = amplitude * (1 - (1. / fraction) * (x[wall] % saw_period))
  #else:
  re[notwall] = amplitude * ((x[notwall] % saw_period) - saw_period / 2) / (saw_period / 2)
  #re = amplitude * x * (10 - x) *(8 - x) * (5 - x)  * (3 - x) * (2 - x) /30
  return re

def SDEFunc1_1(y, t, pttl_func, pttl_func_args=(), t_period = t_period0):
  '''
  In Euler-Maruyama method, a initial value problem of 
  a first order stochastic ordinary equation can be expressed as:
    dy(t) = f(y, t)dt + g(y, t)dW_t,
  where dW_t is a wiener process. And can be expressed as: 
    z * sqrt(dt),
  where z is a random variable from standard normal distribution
  and dt is the time step length.
  
  Here SDEFunc1 represents f(y, t) and SDEFunc2 represents g(y, t).
  y:
    an array
  t:
    current time
  '''
  if ((t % t_period) < (t_period / 2)):
    dy = (1E-8) #TODO 这个函数还没向量化* np.ones(len(y))
    force = -(pttl_func(y + dy, *pttl_func_args) - pttl_func(y - dy, *pttl_func_args)) / (2 * dy)
    #print ("x = %.4f, t = %.4f, force = %.3f" % (y, t, force))
  else:
    force = 0.
    #print ("x = %.4f, t = %.4f, force = %.3f" % (y, t, force))  
  return force
def SDEFunc1_2(y, t, pttl_func, pttl_func_args=(), t_period = t_period0):
  '''
  In Euler-Maruyama method, a initial value problem of 
  a first order stochastic ordinary equation can be expressed as:
    dy(t) = f(y, t)dt + g(y, t)dW_t,
  where dW_t is a wiener process. And can be expressed as: 
    z * sqrt(dt),
  where z is a random variable from standard normal distribution
  and dt is the time step length.
  
  Here SDEFunc1 represents f(y, t) and SDEFunc2 represents g(y, t).
  y:
    an array
  t:
    current time
  '''
  dy = (1E-10) #TODO 这个函数还没向量化* np.ones(len(y))
  force = -(pttl_func(y + dy, *pttl_func_args) - pttl_func(y - dy, *pttl_func_args)) / (2 * dy)
  #print ("x = %.4f, t = %.4f, force = %.3f" % (y, t, force))  
  return force
def SDEFunc2(y, t):
  '''
  In Euler-Maruyama method, a initial value problem of 
  a first order stochastic ordinary equation can be expressed as:
      dy(t) = f(y, t)dt + g(y, t)dW_t,
  where dW_t is a wiener process. And can be expressed as: 
      z * sqrt(dt),
  where z is a random variable from standard normal distribution
  and dt is the time step length.
  
  Here SDEFunc1 represents f(y, t) and SDEFunc2 represents g(y, t).
  y:
    an array
  t:
    current time
  '''
  re = np.sqrt(D)
  return re

def SDEOneStep(y, t, dt, f, g, f_args = (), g_args = ()):
  '''
  Return: y(t + dt).
  y(t + dt) = y(t) + f(y, t)dt + g(y, t) * z * sqrt(dt)
  
  y: 
    an array
  t:
    current time
  dt:
    time step length
  f, g:
    the "slope functions" in a standard first order SDE.
  '''
  f1 = f(y, t, *f_args) * dt
  g1 = g(y, t, *g_args) * np.sqrt(dt) * np.random.randn()
  #print ("f1 = %.4f, g1 = %.4f\n" %(f1, g1))
  dy = f1 + g1
  return y + dy

######################################################################
# Calculations

ti, tf, t_step = 0., 50., 0.005
time = np.arange(ti, tf, t_step)
xi = 8.

N = len(time)
x_array = np.zeros(N)
x_array[0] = xi

delay = 40

#################################################
'''
Choose whether the potential function is time-dependent:
  time_dependent = 1 for time-dependent,
  time_dependent = 0 for time-independent.
'''
time_dependent = 1
if (time_dependent):
  SDEFunc1 = SDEFunc1_1
else:
  SDEFunc1 = SDEFunc1_2

#################################################
'''
Choose whether to plot the trojection or show the animation:
  calculate_all_and_plot = 1 to plot the projection of the particle,
  calculate_all_and_plot = 0 to show the animation of the motion of the particle
''' 
calculate_all_and_plot = 0
if (calculate_all_and_plot):
  for k in np.arange(N - 1):
    x_array[k + 1] = SDEOneStep(x_array[k], time[k], t_step, SDEFunc1, SDEFunc2, (SawTooth, (saw_amplitude0, saw_period0), t_period0), ())
  plt.plot(time, x_array, 'r')
  plt.xlabel("Time / s")
  plt.ylabel("Position of particle")
  plt.title("Trajection of motion of the particle")
  plt.show()
  exit()
else:
  pass
# Calculate the first several steps for animation delay
for k in np.arange(delay):
  x_array[k + 1] = SDEOneStep(x_array[k], time[k], t_step, SDEFunc1, SDEFunc2, (SawTooth, (saw_amplitude0, saw_period0), t_period0), ())



######################################################################
# Set up plot
fig = plt.figure(figsize=(16,10))

# plotting limits
xmin, xmax = 0., 10.
xran = np.arange(xmin, xmax, 0.01)


# top axes show the x-space data
ymin = -1
ymax = 1
ax1 = fig.add_subplot(211, xlim=(xmin, xmax),
                      ylim=(ymin - 0.1 * (ymax - ymin),
                            ymax + 0.1 * (ymax - ymin)))
particle_current, = ax1.plot([], [], 'ro', markersize=20, alpha=.7)
particle_previous, = ax1.plot([], [], 'ro', markersize=18, alpha=.05)

zero_line = ax1.plot([], []
                     #, c='y', label="zero line"
                     )
time_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes)
ax1.set_xlabel('X position of the particle')
ax1.set_ylabel('Y position (remains zero)')
ax1.set_title("Motion of the particle. (Transparent dots stand for the particle's previous position)")
#ax1.legend(loc="best")

# ax2 
ymin2 = SawTooth(xran, saw_amplitude0, saw_period0).min()
ymax2 = SawTooth(xran, saw_amplitude0, saw_period0).max()
ax2 = fig.add_subplot(212, xlim=(xmin, xmax),
                      ylim=(ymin2 - 0.1 * (ymax2 - ymin2),
                            ymax2 + 0.1 * (ymax2 - ymin2)))
pttl_line, = ax2.plot([], [], 'g', lw=3)
pttl_line_none, = ax2.plot([], [], 'g--', lw=3, alpha=.3)
ax2.set_title("The potential function (dash line means there is no potential field at that time)")
ax2.set_xlabel('X position of the particle')
ax2.set_ylabel(r'$E_p$ (potential energy)')
#ax2.legend(loc="best")


######################################################################
# Animate plot
def anim_init():
  """
  initialize animation
  """
  particle_previous.set_data([], [])
  particle_current.set_data([], [])
  pttl_line.set_data([], [])
  pttl_line_none.set_data([], [])
  #zero_line.set_data([], [])
  time_text.set_text("")
  return (
    particle_previous, 
    particle_current, 
    #zero_line, 
    pttl_line,
    pttl_line_none,
    time_text)

def anim_update(i):
  """
  perform animation step
  """
  global x_array, time, time_dependent
  x_array[i + 1] = SDEOneStep(x_array[i], time[i], t_step, SDEFunc1, SDEFunc2, (SawTooth, (saw_amplitude0, saw_period0), t_period0), ())
  
  particle_previous.set_data(x_array[i - delay: i + 1], np.zeros(len(x_array[i - delay: i + 1])))
  particle_current.set_data(x_array[i + 1], 0.)
  #zero_line.set_data(xran, np.zeros(len(xran)))
  #print time[i], x_array[i]
  if (time_dependent):
    if ((time[i] % t_period0) < (t_period0 / 2)):
        pttl_line.set_data(xran, SawTooth(xran, saw_amplitude0, saw_period0))
        pttl_line_none.set_data([], [])
    else:
        pttl_line.set_data([], [])
        pttl_line_none.set_data(xran, SawTooth(xran, saw_amplitude0, saw_period0))
  else:
    pttl_line.set_data(xran, SawTooth(xran, saw_amplitude0, saw_period0))
    pttl_line_none.set_data([], [])
  time_text.set_text("t = %.3f" % time[i] )
  return (
    particle_previous, 
    particle_current, 
    #zero_line, 
    pttl_line,
    pttl_line_none,
    time_text)

# call the animator.  blit=True means only re-draw the parts that have changed.

frames = np.int(N - delay - 1)
#frames = np.arange(delay, N, 1)
anim = animation.FuncAnimation(fig, anim_update, init_func=anim_init,
                               frames=frames, interval=5, blit=True, 
                               #fargs=(delay,)
                               )
#anim.save('SDE_5s.mp4', fps=30)
plt.show()
