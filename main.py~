##Collection of code to simulate CEST sequences

import numpy as np
import scipy.optimize
from scipy.integrate import ode
import timeit

gamma = 2*np.pi*42.6e6 # rad/(s T)
B0 = 7.0 #Tesla
omega0 = gamma * B0

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
    Mstart = np.array([0,0,0,0,M0a,M0b,1.])
    
    print 'sequenceParams : [satDur = {0}, ti = {1}, tacq = {2}, tpresat = {3}, accFactor = {4}, tinterfreq = {5}, hardTheta = {6}, m = {7}, dt = {8}, delta = {9}]'.format(satDur, ti, tacq, tpresat, accFactor, tinterfreq, hardTheta, m, dt, delta)
    print 'physicsVariables : [B0 = {0}, omega1 = {1}, domegaSpecies = {2},  M0a = {3}, M0b = {4}, T1a = {5}, T2a = {6}, T1b = {7}, T2b = {8}, ka = {9}, kb = {10}]'.format(B0, omega1, domegaSpecies,  M0a, M0b, T1a, T2a, T1b, T2b, ka, kb)

    return Mstart, physicsVariables, sequenceParams


def xrot(phi):
    return np.array([[1,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,0,np.cos(phi),0,np.sin(phi),0,0],
                        [0,0,0,np.cos(phi),0,np.sin(phi),0],
                        [0,0,-np.sin(phi),0,np.cos(phi),0,0],
                        [0,0,0,-np.sin(phi),0,np.cos(phi),0],
                        [0,0,0,0,0,0,1],])

def yrot(phi):
    return np.array([[np.cos(phi),0,0,0,-np.sin(phi),0,0],
                        [0,np.cos(phi),0,0,0,-np.sin(phi),0],
                        [0,0,1,0,0,0,0],
                        [0,0,0,1,0,0,0],
                        [np.sin(phi),0,0,0,np.cos(phi),0,0],
                        [0,np.sin(phi),0,0,0,np.cos(phi),0],
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
    return np.dot(A, zrot(phi)),B

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



def cestSequence(Mstart, 
                    physicsVariables, 
                    sequenceParams):
    ## Takes a starting magnetization state and evolves it under the influence of a saturation pulse and imaging sequence

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
        A_fp, B_fp = ABtwoPool(dt, T1a=T1a, T2a=T2a, T1b = T1b, T2b = T2b, M0a=M0a, M0b = M0b, domega=0)
        Mresult = freePrecessTwoPool(Mresult, ti, A_fp, B_fp)## between sat pulse and aquisition pulse
        
        for i in range(accFactor):
            ##################   IMAGING SEQUENCE     ##########################################
            Mresult[-1][0:4] = [0,0,0,0] ## Spoiler Gradient
            Mresult = np.concatenate((Mresult, [np.dot(yrot(hardTheta), Mresult[-1])]))
            signals.append(np.sqrt(Mresult[-1,0]**2 + Mresult[-1,2]**2))
            Mresult = freePrecessTwoPool(Mresult, tacq, A_fp, B_fp)
            Mresult[-1][0:4] = [0,0,0,0] ## Spoiler Gradient
            #Mresult = freePrecessTwoPool(Mresult, tpresat, A_fp, B_fp)     ## tPresat - Does this exist? 

            #################     END OF IMAGING SEQUENCE     ####################################
        Mhistory = np.concatenate((Mhistory, Mresult), 0)
    
    Mhistory = freePrecessTwoPool(Mhistory, tinterfreq, A_fp, B_fp)     ## after acquisition, before the next frequency offset
    
    try:
        return Mhistory, signals
    except:
        return Mhistory, signals, np.nan


def Zspectrum(freqs, 
              Mstart, sequenceParams, physicsVariables):
    ## Iterates cestSequence over a list of frequency offsets
    ## Returns the magnetization history and the signal magnitudes for each offset
    signals = []
    Mresults = []
    for freq in freqs:
	percent_done = freqs.index(freq)/len(freqs)*100
	freqppm = freq/(2*np.pi*42.6e6*7.0)*1e6
        print('offset = {0} ppm, {1}% done'.format(freqppm, percent_done))
        sequenceParams[-1] = freq
        Mresult, signal = cestSequence(Mstart, physicsVariables, sequenceParams)
        Mstart = Mresult[-1,:]
        Mresults.append(Mresult)
        signals.append(signal)
    return Mresult, signals

def setAlterFreqs():
    ## Generates an alternating frequency offset list, to be iterated over using the cest sequence function
    freqs = []
    for i in range(-10000, 0, 300):
        freqs.append(i)
        freqs.append(-i)

    for i in range(len(freqs)):
        freqs[i] = float(freqs[i])

    gamma = 2*np.pi*42.6e6 # rad/(s T)
    B0 = 7.0 #Tesla
    omega0 = gamma * B0
    ppms = np.array(freqs)/omega0*1e6
    ppmsorted = np.sort(ppms)
    inds = [list(ppms).index(c) for c in ppmsorted]

    return freqs, ppmsorted, inds


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



###################################################################################################################
################################ Starting Multi-Pool/Pulsed-Saturation Code #######################################
###################################################################################################################

def setCESTdefaultsMP(dt = 1e-3,
                      ti = 5,
                      tacq = 10,
                      tpresat = 50,
                      accFactor = 1,
                      tinterfreq = 0,
                      hardTheta = np.pi/2,
                      PElines = 1,
                      delta = 0.0,
                      M0w = 1.0,
                      relaxationTimes = [2.0,0.05,  1.5,0.05,  1.5,0.05,  1.0,0.05,  1.0,0.05,  1.0,0.05,  1.0,0.05],
                      relativeConcentrations = [1.0, 0.01, 0.01, 0, 0, 0, 0],
                      exchangeRates = np.array([0, 200., 20., 0, 0, 0, 0]),
                      resonanceFrequencies = [0., 3.5, -3.5, 0, 0, 0, 0],
                      avePower = 1.0e-6,
                      dutyCycle = 0.5,
                      n = 100,
                      theta = 360
                      ):
    
    m = PElines/accFactor
    
    gamma = 2*np.pi*42.6e6 # rad/(s T)
    B0 = 7.0 #Tesla
    omega0 = gamma * B0

    reverseExchanges = [relativeConcentrations[i]*exchangeRates[i] for i in range(7)] #kwa = M0a/M0w*kaw
    
    
    Mstart = []
    for i in range(7):
        Mstart.append(0) #x
        Mstart.append(0) #y
        Mstart.append(M0w*relativeConcentrations[i])
        
    tr_pulse = find_tr(dt, dutyCycle, theta, avePower)/dt   # in seconds, divided by dt yields number of points in pulse
    satSequence = predefinedSatSequence(dt, tr_pulse, dutyCycle, n, theta, varianGaussian)
    satDur = len(satSequence)
    
    omega1 = gamma*satSequence
    
    sequenceParams = [satDur, ti, tacq, tpresat, accFactor, tinterfreq, hardTheta, m, dt, delta]
    physicsVariables = [B0, omega0, omega1, M0w, relaxationTimes, exchangeRates, relativeConcentrations, resonanceFrequencies, reverseExchanges]

    return Mstart, physicsVariables, sequenceParams, satSequence

def find_tr(dt, dutyCycle, theta, avePower):
    ## Returns the pulse duration time required to satisfy avePower and theta requirements
    ## Based on theory from http://onlinelibrary.wiley.com/doi/10.1002/mrm.22884/epdf
    p1 = 0.4279990617406086
    p2 = 0.3037023079606614
    tr = np.sqrt(p2/dutyCycle)*np.pi*theta/(180*gamma*p1*avePower)

    return tr

def predefinedSatSequence(dt, tr, dutyCycle, n, theta, pulseData):

    c1 = 0.147e-6*(1e-3/dt)
    gamma = 2*np.pi*42.6e6
    x = np.linspace((tr-dutyCycle*tr)/2, (tr+dutyCycle*tr)/2, tr*dutyCycle)
    B1max = c1*theta/(tr*dutyCycle)

    y = []
    for i in range(len(x)):
        ind = int(float(i)/len(x)*len(pulseData))
        y.append(pulseData[ind]/max(pulseData)*B1max)
    
    pulse = []
    for i in range(int((tr-dutyCycle*tr)/2)):
        pulse.append(0)
    for i in range(len(y)):
        pulse.append(y[i])
    for i in range(int((tr-dutyCycle*tr)/2)):
        pulse.append(0)
    
    satSequence = []
    for i in range(n+1):
        for j in range(len(pulse)):
            satSequence.append(pulse[j])
            
    return np.array(satSequence)


def xrotOneComponent(phi):
    return np.array([[1,0,0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])

def yrotOneComponent(phi):
    return np.array([[np.cos(phi), 0, np.sin(phi)],
                        [0,1,0],
                        [-np.sin(phi), 0, np.cos(phi)]])

def zrotOneComponent(phi):
    return np.array([[np.cos(phi), -np.sin(phi), 0],
                        [np.sin(phi), np.cos(phi), 0],
                        [0,0,1]])

def freePrecessAB(dt, T1=np.inf, T2=np.inf, M0=1.0, domega=0):
    ''' return the A matrix and B vector for the dM/dt magnetization evolution '''
    phi = domega*dt	 # Resonant precession, radians.
    E1 = np.exp(-dt/T1)
    E2 = np.exp(-dt/T2)

    B = np.array([0, 0, M0*(1 - E1)])

    A = np.array([[E2,0 , 0],
                     [0, E2, 0],
                     [0, 0 , E1]])
    return np.dot(A, zrotOneComponent(phi)),B

def freePrecess(McomponentHistory, t, A_fp, B_fp):
    
    if t > 0:
        Mresult_fp = np.empty((t+1,3))
        Mresult_fp[0,:] = np.array(McomponentHistory)[-1,:]
        for i in range(1, t+1):
            Mresult_fp[i,:] = np.dot(A_fp, Mresult_fp[i-1,:]) + B_fp
        return np.concatenate((McomponentHistory, Mresult_fp[1:-1]), 0)
    else:
        return McomponentHistory

def pulsedCEST(Mstart, physicsVariables, sequenceParams):
    starttimeunpack = timeit.default_timer()
    #Unpacking variables
    [satDur, ti, tacq, tpresat, accFactor, tinterfreq, hardTheta, m, dt, delta] = sequenceParams
    [B0, omega0, omega1, M0w, relaxationTimes, exchangeRates, relativeConcentrations, resonanceFrequencies, reverseExchanges] = physicsVariables
    [M0w, M0a, M0b, M0c, M0d, M0e, M0f] = relativeConcentrations
    [T1w, T2w, T1a, T2a, T1b, T2b, T1c, T2c, T1d, T2d, T1e, T2e, T1f, T2f] = relaxationTimes
    [kww, kaw, kbw, kcw, kdw, kew, kfw] = exchangeRates
    [kww, kwa, kwb, kwc, kwd, kwe, kwf] = reverseExchanges
    resonanceFrequencies = physicsVariables[-2]
    radianOffsets = [resonanceFrequencies[i]*omega0/1e6-delta for i in range(7)]
    [domegaw, domegaa, domegab, domegac, domegad, domegae, domegaf] = radianOffsets
    elapsedunpack = timeit.default_timer() - starttimeunpack

    
    def dMdt(t, M, kww = kww, kaw = kaw, kbw = kbw, kcw = kcw, kdw = kdw, kew = kew, kfw = kfw,
             kwa = kwa, kwb = kwb, kwc = kwc, kwd = kwd, kwe = kwe, kwf = kwf, 
             domegaw = domegaw, domegaa = domegaa, domegab = domegab, domegac = domegac,
             domegad = domegad, domegae = domegae, domegaf = domegaf,
             T1w = T1w, T2w = T2w, T1a = T1a, T2a = T2a, T1b = T1b, T2b = T2b, T1c = T1c, T2c = T2c,
             T1d = T1d, T2d = T2d, T1e = T1e, T2e = T2e, T1f = T1f, T2f = T2f, omega1=omega1):
        idx = int(t/dt)
        #if idx == 1000
        [wx, wy, wz, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, ex, ey, ez, fx, fy, fz] = M

        dwx = [-(kwa+kwb+kwc+kwd+kwe+kwf), domegaw, 0, kaw, 0, 0, kbw, 0, 0, kcw, 0, 0, kdw, 0, 0, kew, 0, 0, kfw, 0, 0]
        dwy = [-domegaw, -(kwa+kwb+kwc+kwd+kwe+kwf), omega1[idx], 0, kaw, 0, 0, kbw, 0, 0, kcw, 0, 0, kdw, 0, 0, kew, 0, 0, kfw, 0]
        dwz = [0, -omega1[idx], -(kwa+kwb+kwc+kwd+kwe+kwf), 0, 0, kaw, 0, 0, kbw, 0, 0, kcw, 0, 0, kdw, 0, 0, kew, 0, 0, kfw]
        dax = [kwa, 0, 0, -kaw, domegaa, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        day = [0, kwa, 0, -domegaa, -kaw, omega1[idx], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        daz = [0, 0, kwa, 0, -omega1[idx], -kaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dbx = [kwb, 0, 0, 0, 0, 0, -kbw, domegab, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dby = [0, kwb, 0, 0, 0, 0, -domegab, -kbw, omega1[idx], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dbz = [0, 0, kwb, 0, 0, 0, 0, -omega1[idx], -kbw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dcx = [kwc, 0, 0, 0, 0, 0, 0, 0, 0, -kcw, domegac, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dcy = [0, kwc, 0, 0, 0, 0, 0, 0, 0, -domegac, -kcw, omega1[idx], 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dcz = [0, 0, kwc, 0, 0, 0, 0, 0, 0, 0, -omega1[idx], -kcw, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ddx = [kwd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kdw, domegad, 0, 0, 0, 0, 0, 0, 0]
        ddy = [0, kwd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -domegad, -kdw, omega1[idx], 0, 0, 0, 0, 0, 0]
        ddz = [0, 0, kwd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -omega1[idx], -kdw, 0, 0, 0, 0, 0, 0]
        dex = [kwe, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kew, domegae, 0, 0, 0, 0]
        dey = [0, kwe, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -domegae, -kew, omega1[idx], 0, 0, 0]
        dez = [0, 0, kwe, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -omega1[idx], -kew, 0, 0, 0]
        dfx = [kwf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kfw, domegaf, 0]
        dfy = [0, kwf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -domegaf, -kfw, omega1[idx]]
        dfz = [0, 0, kwf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -omega1[idx], -kfw]

        B = array([wx/T2w, wy/T2w, -(M0w - wz)/T1w, ax/T2a, ay/T2a, -(M0a - az)/T1a,  bx/T2b, by/T2b, -(M0b - bz)/T1b,  
             cx/T2c, cy/T2c, -(M0c- cz)/T1c, dx/T2d, dy/T2d, -(M0d - dz)/T1d, ex/T2e, ey/T2e, -(M0e - ez)/T1e, fx/T2f, fy/T2f, -(M0f - fz)/T1f])

        dM = np.dot(np.array([dwx, dwy, dwz, dax, day, daz, dbx, dby, dbz, dcx, dcy, dcz, ddx, ddy, ddz, dex, dey, dez, dfx, dfy, dfz]),M) - B
        
        ## Crusher gradient
        #if tr*dt != 0.0:
        #    if round(t,5)%(round(tr*dt,5)) < 0.00001:
        #        dM[0],dM[1],dM[3],dM[4],dM[6],dM[7],dM[9],dM[10],dM[12],dM[13],dM[15],dM[16],dM[18],dM[19] = np.array([-wx, -wy, -ax, -ay, -bx,-by,-cx,-cy,-dx,-dy,-ex,-ey,-fx,-fy])/dt/dt
        
        return dM

    Mhistory = np.empty((1,21))
    Mhistory[0,:] = Mstart
    signals = []
    counter = 0
    starttimesat = timeit.default_timer()
    for m in range(m):
        ################    SATURATION PULSE    ##################################    
        Mresult = np.empty((int(satDur),21))
        Mresult[0,:] = Mhistory[-1,:]
        r = scipy.integrate.ode(dMdt)
        r = r.set_integrator('dopri5')
        r = r.set_initial_value(Mresult[0,:], t=0)

        t = 0.0
        indx = 1
        while r.successful() and indx < satDur:
            Mresult[indx,:] = r.integrate(r.t + dt)
            t+= dt
            indx += 1
            
        #if delta > 0.:
            #Mresult[-1,0] = -Mresult[-1,0]
            #Mresult[-1,2] = -Mresult[-1,2]

        ##################    END OF SATURATION PULSE  #####################################
        elapsedsat = timeit.default_timer() - starttimesat
        starttimeimaging = timeit.default_timer()
        dt = 0.001
        Mpools = []
        for i in range(7): #Evolve each pool separately
            M = Mresult[:, i*3:i*3+3]
            A_fp, B_fp = freePrecessAB(dt, T1=relaxationTimes[i*2], T2=relaxationTimes[i*2+1], M0 = relativeConcentrations[i], domega=0)
            M = freePrecess(M, ti, A_fp, B_fp)## between sat pulse and aquisition pulse

            for i in range(accFactor):
                ##################   IMAGING SEQUENCE     ##########################################
                M[-1][0:2] = [0,0] ## Spoiler Gradient
                M = np.concatenate((M, [np.dot(yrot(hardTheta), M[-1])]))
                signals.append(np.sqrt(M[-1,0]**2 + M[-1,1]**2))
                M = freePrecess(M, tacq, A_fp, B_fp)
                M[-1][0:2] = [0,0] ## Spoiler Gradient
                #Mresult = freePrecessTwoPool(Mresult, tpresat, A_fp, B_fp)     ## tPresat - Does this exist? 
                
                #################     END OF IMAGING SEQUENCE     ####################################
            M = freePrecess(M, tinterfreq, A_fp ,B_fp)
            Mpools.append(M)
            #Mhistory = np.concatenate((Mhistory, Mresult), 0)
    elapsedimaging = timeit.default_timer() - starttimeimaging
    Mhistory = Mpools     ## after acquisition, before the next frequency offset

    print('unpack = {0}, sat = {1}, imaging = {2}'.format(elapsedunpack, elapsedsat, elapsedimaging))
    return Mhistory, signals

def ZspectrumMP(freqs, Mstart, physicsVariables, sequenceParams):
    signals = []
    Mresults = []

    for freq in freqs:
        sequenceParams[-2] = freq
        n = sequenceParams[-1]
	percent_done = round(float(freqs.index(freq))/float(len(freqs)),3)*100
	freqppm = freq/(2*np.pi*42.6e6*7.0)*1e6
        print('offset = {0} ppm, {1}% done'.format(round(freqppm,2), percent_done))
        Mresult, signal = pulsedCEST(Mstart, physicsVariables, sequenceParams)
        
	Mstart = []
        for i in range(7):
            for j in range(3):
                Mstart.append(np.array(Mresult)[i,-1,j])
        Mresults.append(Mresult)
        signals.append(signal)

    return signals, Mresults


## This is the shape of the standard Varian gaussian pulse
varianGaussian = [13.999, 14.99, 16.042, 17.158, 18.343, 19.598, 20.927, 22.335, 23.824, 25.399, 27.062, 28.819, 30.673, 32.628, 34.689, 36.86, 39.145, 41.549, 44.077, 46.732, 49.521, 52.446, 55.515, 58.73, 62.098, 65.622, 69.309, 73.162, 77.187, 81.389, 85.773, 90.343, 95.104, 100.062, 105.22, 110.583, 116.155, 121.942, 127.946, 134.172, 140.624, 147.305, 154.219, 161.369, 168.757, 176.387, 184.261, 192.38, 200.747, 209.363, 218.228, 227.344, 236.711, 246.329, 256.196, 266.312, 276.676, 287.285, 298.137, 309.229, 320.558, 332.119, 343.909, 355.921, 368.15, 380.591, 393.236, 406.078, 419.109, 432.32, 445.703, 459.248, 472.945, 486.783, 500.751, 514.836, 529.028, 543.311, 557.674, 572.102, 586.582, 601.097, 615.633, 630.175, 644.706, 659.21, 673.67, 688.068, 702.389, 716.614, 730.726, 744.706, 758.537, 772.201, 785.679, 798.953, 812.006, 824.818, 837.372, 849.651, 861.636, 873.311, 884.658, 895.659, 906.3, 916.564, 926.434, 935.897, 944.937, 953.541, 961.694, 969.385, 976.6, 983.329, 989.561, 995.285, 1000.492, 1005.175, 1009.324, 1012.935, 1016.0, 1018.514, 1020.474, 1021.877, 1022.719, 1023.0, 1022.719, 1021.877, 1020.474, 1018.514, 1016.0, 1012.935, 1009.324, 1005.175, 1000.492, 995.285, 989.561, 983.329, 976.6, 969.385, 961.694, 953.541, 944.937, 935.897, 926.434, 916.564, 906.3, 895.659, 884.658, 873.311, 861.636, 849.651, 837.372, 824.818, 812.006, 798.953, 785.679, 772.201, 758.537, 744.706, 730.726, 716.614, 702.389, 688.068, 673.67, 659.21, 644.706, 630.175, 615.633, 601.097, 586.582, 572.102, 557.674, 543.311, 529.028, 514.836, 500.751, 486.783, 472.945, 459.248, 445.703, 432.32, 419.109, 406.078, 393.236, 380.591, 368.15, 355.921, 343.909, 332.119, 320.558, 309.229, 298.137, 287.285, 276.676, 266.312, 256.196, 246.329, 236.711, 227.344, 218.228, 209.363, 200.747, 192.38, 184.261, 176.387, 168.757, 161.369, 154.219, 147.305, 140.624, 134.172, 127.946, 121.942, 116.155, 110.583, 105.22, 100.062, 95.104, 90.343, 85.773, 81.389, 77.187, 73.162, 69.309, 65.622, 62.098, 58.73, 55.515, 52.446, 49.521, 46.732, 44.077, 41.549, 39.145, 36.86, 34.689, 32.628, 30.673, 28.819, 27.062, 25.399, 23.824, 22.335, 20.927, 19.598, 18.343, 17.158, 16.042]
