"""foline.py solves the approximate free-energy equation for the first-order line"""

import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt

pD = np.array([-0.669437,0.2063/(1/2)**2, 1.8/(1/2)**2])
pQ = np.array([-4.98606,0.2578/(3/2)**2, 0.14/(3/2)**2])

def Jc0(J1):
    return 1.5*J1/(pD[0]-pQ[0])
    

def Tcrit(Js, J1, J3):
    a, b, d = pQ-pD

    epsQ = 0.5 * J1 + 0.25 * J3
    epsD1 = - J1 + 0.25 * J3
    epsD0 = - 0.75 * J3

    gapQD = epsQ-min(epsD1,epsD0)
    gapD01 = abs(epsD0-epsD1)
    
    def rhs(T,J):
        return J*a-b/6*T**3/J**2 -d/20*T**5/J**4 + T*np.log(1+np.exp(-gapD01/(T+1e-6))) + gapQD

    Tcrit = [spo.root_scalar(rhs, bracket=[-0.1,2],args=(J),xtol=0.001).root for J in Js]
        
    return Tcrit

def Jcrit(Ts, J1, J3, enable_entropy = True):
    a, b, d = pQ-pD

    epsQ = 0.5 * J1 + 0.25 * J3
    epsD1 = - J1 + 0.25 * J3
    epsD0 = - 0.75 * J3

    gapQD = epsQ-min(epsD1,epsD0)
    gapD01 = abs(epsD0-epsD1)
    
    def rhs(J,T):
        return J*a-b/6*T**3/J**2 -d/20*T**5/J**4 + enable_entropy*T*np.log(1+np.exp(-gapD01/(T+1e-6))) + gapQD

    Jc = [spo.root_scalar(rhs, bracket=[0.34,0.6],args=(T)).root for T in Ts]
        
    return Jc

