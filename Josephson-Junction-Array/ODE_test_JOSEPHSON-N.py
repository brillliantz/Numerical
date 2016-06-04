# coding: utf-8
'''公式的手写版在寝室草稿纸'''
import scipy.integrate as inte
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit, float64
import time as tm


######################################
## Parameters 
# How many junctions
N = 10
# Normalized parameters
alpha0 = 0.1
gamma0 = 0.5
a0 = 1.0
# Time arange to solve in the calculation
time_step = 0.01
time_range = 600
time = np.arange(0, time_range, time_step)

######################################
## Initial Conditions 
y0 = np.zeros(2 * N)

######################################
## Calculations 
@jit
def phi_tt(y, t, a = 1., alpha = 0.1, gamma = 0.5):
    '''
    phi_t and phi_tt return the slope of phi and d(phi)/dt respectively.
    y is a vetor with length 6(=2 * 3).
    y[0]~y[6] is phi_(n-1), d(phi_(n-1))/dt, phi_(n), d(phi_(n))/dt, phi_(n+1), d(phi_(n+1))/dt, respectively.
    '''  
    pass
    re = -alpha * y[3] - np.sin(y[2]) - (gamma + np.sin(t)) + (y[0] - 2 * y[2] + y[4]) / a**2
    #print 'aaaaaaa'
    #print gamma
    #print 'aaaaaaa'
    return re


def every_six(y, t, a = 1., alpha = 0.1, gamma = 0.5):
    '''
    Each time, use 6 of the y array to calculate phi_tt.
    '''
    global N
    #LEN = len(y)
    #LEN2 = LEN / 2
    re = np.zeros(N)
    #TODO maybe useful: np.apply_along_axis(fun, axis=1, array)
    re[0] = phi_tt(np.hstack((y[-2:], y[:4])), t, a, alpha, gamma)
    for i in np.arange(1, N - 1):
        re[i] = phi_tt(y[2 * i - 2: 2 * i + 4], t, a, alpha, gamma)
    re[N - 1] = phi_tt(np.hstack((y[-4:], y[:2])), t, a, alpha, gamma)
    return re

@jit
def myfunc(y, t, a = 1., alpha = 0.1, gamma = 0.5):
    '''
    y is a vetor with length 2 * N. 
    y[0] ~ y[N] is phi1_t, phi1_tt, phi2_t, phi2_tt, ... , phiN_t, phiN_tt, respectively.
    '''
    LEN = len(y)
    re = np.zeros(LEN)
    index = np.arange(LEN)
    boolean_index_even = (index % 2 == 0)
    boolean_index_odd = (index % 2 == 1)

    re[boolean_index_even] = y[boolean_index_odd] #TODO 或者使用phi_t函数？画蛇添足吧
    re[boolean_index_odd] = every_six(y, t, a, alpha, gamma)
    return re
#@jit
def myfunc2(t, y, a = 1., alpha = 0.1, gamma = 0.5):
    '''
    the difference between 'myfunc2' and 'myfunc' is: the order of 't' and 'y'
    'myfunc2' is for inte.ode, 'myfunc' is for odeint.
    y is a vetor with length 2 * N. 
    y[0] ~ y[N] is phi1_t, phi1_tt, phi2_t, phi2_tt, ... , phiN_t, phiN_tt, respectively.
    '''
    LEN = len(y)
    re = np.zeros(LEN)
    index = np.arange(LEN)
    boolean_index_even = (index % 2 == 0)
    boolean_index_odd = (index % 2 == 1)

    re[boolean_index_even] = y[boolean_index_odd] #TODO 或者使用phi_t函数？画蛇添足吧
    re[boolean_index_odd] = every_six(y, t, a, alpha, gamma)
    return re

######################################
## Plot Functions
#@jit
def myplot(f, y0, t, arguments=(a0, alpha0, gamma0)):
    '''
    Plot U-t and phi-t relation in two sub-figures
    '''
    # creat two sub-figures
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    # set title and label
    ax1.set_title('Time from 0 to %f. $\gamma$ = %f, alpha = %f, a = %f'
    % (time[len(time) - 1], arguments[2], arguments[1], arguments[0]))
    ax1.set_xlabel('Time t / s')
    ax1.set_ylabel('Phase $\phi$ / rad')
    ax2.set_xlabel('Time t / s')
    ax2.set_ylabel('Votage $U$ / V')
    # calculate using ODEINT function
    print tm.clock()
    ##====================================================================== test
    r = inte.ode(myfunc2).set_integrator('dopri5')
    r.set_initial_value(y0, time[0]).set_f_params(*argu3)
    # 这里需要建立一个数组来装积分的结果
    result = np.zeros([time.shape[0]+1, 2*N])
    inte_process_counter = 0
    while r.successful() and r.t < time_range:
        result[inte_process_counter, :] = r.integrate(r.t+time_step)
        inte_process_counter += 1
        #print (r.t, r.integrate(r.t+time_step))
    # print inte_process_counter
    result = result[:time.shape[0], :]
    ##====================================================================== test    
    # result = inte.odeint(f, y0, t, args=arguments)
    print tm.clock()
    # plot a horizonal line at y = 0
    ax2.plot(time, np.zeros(len(time)))
    for i in np.arange(len(y0) / 2):
        ax1.plot(time, result[:, 2 * i], label='$\phi_{%d}$' % (i + 1)) 
        ax2.plot(time, result[:, 2 * i + 1], label='$U_{%d}$' % (i + 1))
    # location of the legends
    ax1.legend(loc="best")    
    ax2.legend(loc="best")
    plt.savefig('alpha = %f.png' % arguments[2], dpi=144, format='png')
    plt.close()
    #plt.show()

#@jit
#def U_plot2(f, y0, t, cut, arguments):
    #cut_index = np.int(cut / time_step)
    #fig = plt.figure(figsize=(10, 4))
    #ax1 = fig.add_subplot(111)
    #ax1.set_title('Time from %f to %f. $\gamma$ = %f, alpha = %f, a = %f'
    #% (cut, time[len(time) - 1], arguments[2], arguments[1], arguments[0]))
    #ax1.plot(time[cut_index:], np.zeros(len(time[cut_index:])))
    #result = inte.odeint(f, y0, t, args=arguments)
    #for i in np.arange(len(y0) / 2):
        #ax1.plot(time[cut_index:], result[cut_index:, 2 * i + 1], label='$U_{%d}$' % (i + 1))   
    #ax1.legend(loc="best")

judge = 1
if (judge):
    ################### Plot gamma-U curve ###################
    gamma_step = 0.001
    gamma_array = np.arange(0., 0.001, gamma_step)
    gamma_U_array = np.zeros([len(gamma_array), N + 1])
    counter = 0
    for mygamma in gamma_array:
        argu3 = (a0, alpha0, mygamma)
        # test
        print "\n--------------------\ncounter = %f, mygamma = %f" %(counter, mygamma)
        print tm.clock()
        ##====================================================================== test
        r = inte.ode(myfunc2).set_integrator('dopri5')
        r.set_initial_value(y0, time[0]).set_f_params(*argu3)
        # 这里需要建立一个数组来装积分的结果
        result = np.zeros([time.shape[0]+1, 2*N])
        inte_process_counter = 0
        while r.successful() and r.t < time_range:
            result[inte_process_counter, :] = r.integrate(r.t+time_step)
            inte_process_counter += 1
            #print (r.t, r.integrate(r.t+time_step))
        # print inte_process_counter
        result = result[:time.shape[0], :]
        ##====================================================================== test          
        print tm.clock()
        print "-----------\n"
        # test
        #result = inte.odeint(myfunc, y0, time, argu3)
        cut = 400.
        cut_index = np.int(cut / time_step)
        result_cut = result[cut_index: , :]
        index = np.arange(result_cut.shape[1])
        boolean_index_even = (index % 2 == 0)
        boolean_index_odd = (index % 2 == 1)
        result_cut_U = result_cut[:, boolean_index_odd]
        U_mean = np.mean(result_cut_U, axis=0)
        gamma_U_array[counter, 1:] = U_mean
    #     print U_mean
        counter += 1
    gamma_U_array[:, 0] = gamma_array
    np.save('comp.results/gamma_U_array', gamma_U_array)
    plt.figure(figsize=(12,8))
    plt.plot(gamma_U_array[:, 0], gamma_U_array[:, 1], lw=1.5)
    plt.xlabel("$\gamma$")
    plt.ylabel("$U$ (normalized)")
    plt.title(r'Time from %f to %f. a = %f, $\alpha$ = %f, $\gamma$ step = %f'
    % (cut, time_range, argu3[0], argu3[1], gamma_step))
    plt.show()
else:
    ################### Plot phi-t, U-t at different gamma value ###################
    # Time it
    print tm.clock()
    gamma_array = np.arange(0, 1, 0.5)
    for mygamma in gamma_array:
        print "ddd"
        argu3 = (a0, alpha0, mygamma)
        myplot(myfunc, y0, time, argu3)  
        print "mygamma = %f" % mygamma + "is done.\n"
    # Time it again
    print tm.clock()
    plt.show()
