''' Some helpers to detect the lines of maxima in e.g. the specific heat scans. '''

import numpy as np
import scipy.optimize as spo

def find_roots(bracket, f, divisions):
    xd = np.linspace(bracket[0] ,bracket[1], divisions)

    roots = []
    for i in range(divisions-1):
        x1 = xd[i]
        x2 = xd[i+1]

        if f(x1) > 0 and f(x2) < 0:
            sol = spo.root_scalar(f, bracket = (x1, x2))
            roots.append(sol.root)
    return roots
    
    

def deltaconv(x, xs, ys, alpha):
    ys = np.hstack([ys[0]+(ys[0]-ys[1])*np.arange(len(ys),0,-1),ys,ys[-1] + (ys[-1]-ys[-2])*np.arange(1,len(ys)+1)])
    xs = np.hstack([2*xs[0]-xs[1]-xs[-1]+xs, xs, 2*xs[-1]-xs[-2]+xs-xs[0]])
    xc = (x-xs)/alpha

    return np.trapz(-xc*np.exp(-xc**2/2)*ys/np.sqrt(2*np.pi*alpha**2), xs)

def find_maxima(x, y, alpha=None, divisions = 30):
    if alpha == None:
        alpha = (x[-1]-x[0])/60
    conv = [deltaconv(r, x, y, alpha) for r in x]

    roots = find_roots([x[0],x[-1]], lambda r: deltaconv(r, x,y, alpha), divisions=divisions)
    return roots
