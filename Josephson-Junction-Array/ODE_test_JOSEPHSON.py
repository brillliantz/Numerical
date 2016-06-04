# coding: utf-8
import scipy.integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time as tm

'''Parameters'''
# Capacity
C1 = 1.
# Resistor
R1 = 1.
# Total current
I1 = 5.
# Time to solve
time_step = 0.001
time_range = 50
time = np.arange(0, time_range, time_step)
# Range of I
I_range = 5.
I_step = 0.0005
I_array = np.arange(-I_range, I_range, I_step)

'''Initial Conditions'''
phi0 = 0.
u0 = 0.
y0 = (phi0, u0)

'''time it'''
print tm.clock()

'''Calculations'''
@jit
def func1(y, t, C = C1, R = R1, I = I1):
    phi = y[0]
    u = y[1]
    k_phi = u
    k_u = (I 
           + 2*np.sin(3*t) 
           - u / R - np.sin(phi)) / C
    return k_phi, k_u
@jit
def myplot(f, t, y0, arguments):
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.set_title('Initial: $\phi_0$ = %f, $U_0$ = %f, Time from 0 to %f. '
    % (u0, phi0, time_range))
    ax2.set_title('And parameters: C = %f, R = %f, I = %f '
    % (arguments[0], arguments[1], arguments[2]))    
    result = inte.odeint(func1, y0, time, arguments)
    ax1.plot(time, result[:, 0], label='$\phi$')
    ax2.plot(time, result[:, 1], label='$U$') 
    ax1.legend(loc="best")
    ax2.legend(loc="best")

argu3 = (C1, R1, I1)

not_draw_mean = 1
if (not_draw_mean == 1):
    myplot(func1, time, y0, argu3)
    plt.show()
else:
    '''Calculate the Mean'''
    u_mean_array = np.array([])
    for I1 in I_array:
        print I1
        result = inte.odeint(func1, y0, time, args=(C1, R1, I1))
        '''    ###'''
        #plt.plot(time, result[:, 1], label='$U$')
        ##plt.plot(time, result[:, 0], label='$\phi$')
        #plt.title('$\phi_0$ = %f, $U_0$ = %f, $I$ = %f, Time from 0 to %f. '
                          #% (u0, phi0, I1, time[len(time) - 1]))
        #plt.show()
        #plt.close()
        #start plot from t = t0.
        t0 = 800
        u_mean = np.mean(result[t0:, 1])
        u_mean_array = np.append(u_mean_array, u_mean)
    
    '''time it'''
    print tm.clock()
    
    plt.plot(I_array, u_mean_array)
    plt.title('$\phi_0$ = %f, $U_0$ = %f, Time from 0 to %f with step %f.  \n I in (-%f, %f) with step %f'
              % (u0, phi0, time_range, time_step, I_range, I_range, I_step))
    plt.show()    