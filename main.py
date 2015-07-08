##Collection of code to simulate CEST sequences

import numpy as np
from scipy.signal import argrelextrema
import scipy.optimize
from scipy.integrate import ode

def setCESTdefaults(dt = 1e-3, 
                    satDur = 4000, 
                    B1 = 1.0e-6, 
                    ti = 5, 
                    tacq = 10, 
                    tpresat = 50, 
                    tinterfreq = 1, 
                    accFactor = 1, 
                    hardTheta = np.pi/2, 
                    PElines = 1,
                    deltaPPM = 3.5, 
                    M0a = 1.0, 
                    relativeConc = 0.01, 
                    T1a = 2.0, 
                    T2a = 0.05, 
                    T1b = 1.5, 
                    T2b = 0.05, 
                    kb = 20.0):

    m = PElines/accFactor
   
    gamma = 2*np.pi*42.6e6 # rad/(s T)
    B0 = 7.0 #Tesla
    omega0 = gamma * B0
    omega1 = gamma * B1
    omegaWater = gamma * B0
    domegaSpecies = -6557.0 #rad/s
    omegaSpecies = omegaWater + domegaSpecies #chemical resonance frequency
    delta = deltaPPM*1e6/gamma


    #Pool a is water, b is the species
    M0b = M0a*relativeConc
    ka = M0b/M0a*kb #Transfer rate of pool a to pool b, s^-1. We want a-->b = b-->a

    sequenceParams = [satDur, ti, tacq, tpresat, accFactor, tinterfreq, hardTheta, m, dt, delta]
    physicsVariables = [B0, omega1, domegaSpecies,  M0a, M0b, T1a, T2a, T1b, T2b, ka, kb]
    Mstart = numpy.array([0,0,0,0,M0a,M0b,1.])
    
    print 'sequenceParams : [satDur = {0}, ti = {1}, tacq = {2}, tpresat = {3}, accFactor = {4}, tinterfreq = {5}, hardTheta = {6}, m = {7}, dt = {8}, delta = {9}]'.format(satDur, ti, tacq, tpresat, accFactor, tinterfreq, hardTheta, m, dt, delta)
    print 'physicsVariables : [B0 = {0}, omega1 = {1}, domegaSpecies = {2},  M0a = {3}, M0b = {4}, T1a = {5}, T2a = {6}, T1b = {7}, T2b = {8}, ka = {9}, kb = {10}]'.format(B0, omega1, domegaSpecies,  M0a, M0b, T1a, T2a, T1b, T2b, ka, kb)

    return Mstart, physicsVariables, sequenceParams


def xrot(phi):
    return np.array([[1,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,0,cos(phi),0,sin(phi),0,0],
                        [0,0,0,cos(phi),0,sin(phi),0],
                        [0,0,-sin(phi),0,cos(phi),0,0],
                        [0,0,0,-sin(phi),0,cos(phi),0],
                        [0,0,0,0,0,0,1],])

def yrot(phi):
    return np.array([[cos(phi),0,0,0,-sin(phi),0,0],
                        [0,cos(phi),0,0,0,-sin(phi),0],
                        [0,0,1,0,0,0,0],
                        [0,0,0,1,0,0,0],
                        [sin(phi),0,0,0,cos(phi),0,0],
                        [0,sin(phi),0,0,0,cos(phi),0],
                        [0,0,0,0,0,0,1]])

def zrot(phi):
    return np.array([[np.cos(phi), 0, np.sin(phi), 0, 0, 0, 0],
                        [0, np.cos(phi), 0, np.sin(phi), 0, 0, 0],
                        [-np.sin(phi), 0, np.cos(phi), 0, 0, 0, 0],
                        [0, -np.sin(phi), 0, np.cos(phi), 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],])


def ABtwoPool(dt, T1a=np.inf, T2a=np.inf, T1b = np.inf, T2b = np.inf, M0a=1000, M0b = 1.0, domega=0):
    ''' return the A matrix and B vector for the dM/dt magnetization evolution '''
    
    phi = domega*dt  # Resonant precession, radians.
    E1a = np.exp(-dt/T1a)
    E1b = np.exp(-dt/T1b)
    E2a = np.exp(-dt/T2a)
    E2b = np.exp(-dt/T2b)
    
    B = np.array([0, 0, 0, 0, M0a*(1. - E1a), M0b*(1. - E1b), 0])

    A = np.array([[E2a, 0, 0, 0, 0, 0, 0],
                     [0, E2b, 0, 0, 0, 0, 0],
                     [0, 0, E2a, 0, 0, 0, 0],
                     [0, 0, 0, E2b, 0, 0, 0],
                     [0, 0, 0, 0, E1a, 0, 0],
                     [0, 0, 0, 0, 0, E1b, 0],
                     [0, 0, 0, 0, 0, 0, 1 ]])
    return np.dot(A, sim.zrot(phi)),B

def freePrecessTwoPool(Mresult, 
                       t, 
                       A_fp, 
                       B_fp):
    if t > 0:
        Mresult_fp = np.empty((t+1,7))
        Mresult_fp[0,:] = np.array(Mresult)[-1,:]
        for i in range(1, t+1):
            Mresult_fp[i,:] = np.dot(A_fp, Mresult_fp[i-1,:]) + B_fp
        return np.concatenate((Mresult, Mresult_fp[1:-1]), 0)
    else:
        return Mresult



def cestSequenceEPI(Mstart, 
                    physicsVariables, 
                    sequenceParams):
    
    [satDur, ti, tacq, tpresat, accFactor, tinterfreq, hardTheta, m, dt, delta] = sequenceParams
    [B0, omega1, domegaSpecies,  M0a, M0b, T1a, T2a, T1b, T2b, ka, kb] = physicsVariables
    
    
    def dMdtTwoPool(t, M_vec, M0a = M0a, M0b = M0b, T1a = T1a, T2a = T2a, T1b = T1b, T2b = T2b,
            ka = ka, kb = kb, domegaa=-delta, domegab=-delta-domegaSpecies, omega1=omega1):

        A = np.array([[-1./T2a - ka, kb,  domegaa, 0, 0, 0, 0],
                         [ka, -1./T2b - kb, 0, domegab, 0, 0 , 0],
                         [-domegaa, 0, -1./T2a-ka, kb, omega1, 0, 0],
                         [0, -domegab, ka, -1./T2b-kb, 0, omega1, 0],
                         [0, 0, -omega1, 0, -1./T1a - ka, kb, M0a/T1a],
                         [0, 0, 0, -omega1, ka, -1./T1b - kb, M0b/T1b],
                         [0, 0, 0, 0, 0, 0,0]])
        return np.dot(A,M_vec)


    Mhistory = np.empty((1,7))
    Mhistory[0,:] = Mstart
    signals = []

    for m in range(m):
        ################    SATURATION PULSE    ##################################    

        Mresult = np.empty((int(satDur),7))
        Mresult[0,:] = Mhistory[-1,:]

        r = scipy.integrate.ode(dMdtTwoPool)
        r = r.set_integrator('dopri5')
        r = r.set_initial_value(Mresult[0,:], t=0)

        t = 0.0
        idx = 1
        while r.successful() and idx < satDur:
            Mresult[idx,:] = r.integrate(r.t + dt)
            t+= dt
            idx += 1

        if delta > 0.:
            Mresult[-1,0] = -Mresult[-1,0]
            Mresult[-1,2] = -Mresult[-1,2]

        ##################    END OF SATURATION PULSE  #####################################

        dt = 0.001
        A_fp, B_fp = sim.ABtwoPool(dt, T1a=T1a, T2a=T2a, T1b = T1b, T2b = T2b, M0a=M0a, M0b = M0b, domega=0)
        Mresult = sim.freePrecessTwoPool(Mresult, ti, A_fp, B_fp)## between sat pulse and aquisition pulse
        
        for i in range(accFactor):
            ##################   IMAGING SEQUENCE     ##########################################
            Mresult[-1][0:4] = [0,0,0,0] ## Spoiler Gradient
            Mresult = np.concatenate((Mresult, [np.dot(sim.yrot(hardTheta), Mresult[-1])]))
            signals.append(np.sqrt(Mresult[-1,0]**2 + Mresult[-1,2]**2))
            Mresult = sim.freePrecessTwoPool(Mresult, tacq, A_fp, B_fp)
            Mresult[-1][0:4] = [0,0,0,0] ## Spoiler Gradient
            #Mresult = freePrecessTwoPool(Mresult, tpresat, A_fp, B_fp)     ## tPresat - Does this exist? 

            #################     END OF IMAGING SEQUENCE     ####################################
        Mhistory = np.concatenate((Mhistory, Mresult), 0)
    
    Mhistory = sim.freePrecessTwoPool(Mhistory, tinterfreq, A_fp, B_fp)     ## after acquisition, before the next frequency offset
    
    try:
        return Mhistory, signals
    except:
        return Mhistory, signals, np.nan


def Zspectrum(freqs, 
              Mstart):

    signals = []
    Mresults = []
    for freq in freqs:
        print(freq)
        sequenceParams[-1] = freq
        Mresult, signal = sim.cestSequence(Mstart, physicsVariables, sequenceParams)
        Mstart = Mresult[-1,:]
        Mresults.append(Mresult)
        signals.append(signal)
    return Mresult, signals

def setAlterFreqs():
    freqs = []
    for i in range(-10000, 0, 300):
        freqs.append(i)
        freqs.append(-i)

    for i in range(len(freqs)):
        freqs[i] = float(freqs[i])
    return freqs
    
freqs = setAlterFreqs()
Mstart, physicsVariables, sequenceParams = setCESTdefaults(1e-6)

def zspectrum_N(params,
                freqs):
    
    arr = empty_like(freqs)*0
    shift =  params[0]
    
    for i in np.arange(0,len(params[1:]),3):
        
        A = params[i+1]
        w0 = params[i+2]
        lw =  params[i+3]
        tmp = np.divide(A,(1+4*((freqs-w0)/lw)**2))
        
        arr = arr+tmp
    return (arr+shift)

def h_residual_Zspectrum_N(params, 
                           y_data, 
                           w):
    
    return np.abs(y_data - zspectrum_N(params,w))

# first is amplitude
# then is the peak position in ppm
# line width (ppm)
# fourth parameter is the baseline shift, usually 1 if normalized

param_dict = {'water': [-.7, 0,0.6],
              'A' : [-0.05,3,.4]}

params_passed = [1.0]+\
                param_dict['water']+\
                param_dict['A']


