import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

def fit_bootstrap(func, x, y, sigy, p0, samples=50):
    popt0, _ = spo.curve_fit(func, x, y, p0=p0, maxfev=20000)

    popts = []
    for i in range(samples):
        yr = y + np.random.normal(size=y.shape)*sigy
        popt, _ = spo.curve_fit(func, x, yr, p0=popt0, maxfev=20000)
        popts.append(popt)
        if np.std((popt-popt0)/popt0)> 10:
            print(popt-popt0)
    popts = np.array(popts)

    return popt0, np.std(popts,axis=0)

def fun(L, a,b):
    return a + b/L**3/2

# beta = 2L


Ls = np.array([8,12,16,24,32])
Es = np.array([-4.997951,-4.989471,-4.9874978,-4.9864774,-4.9862797])
sigEs = np.array([8.22e-05,0.000106,3.26e-05,3.78e-05,2.81e-05])

popt, perr = fit_bootstrap(fun, Ls,Es,sigEs, [-5,0])

print('E = {:.6g}±{:.6g}'.format(popt[0],perr[0]))
print('c = {:.6g}±{:.6g}'.format(popt[1],perr[1]))

plt.errorbar(1/Ls**3, Es,sigEs)
x = np.linspace(1e-9,0.002,10)
plt.plot(x, fun(1/x**(1/3),*popt),'--', zorder=20,color='black')
plt.text(0.95, 0.95, '$\\varepsilon^{S=3/2}_{\\mathrm{AFM}} = -4.98603(3)$',
    verticalalignment='top', horizontalalignment='right',
    transform=plt.gca().transAxes)
plt.xlim([0,None])
plt.xlabel('$1/L^3$')
plt.ylabel('$\\varepsilon^{S=3/2}_{\mathrm{AFM}}(L)$')
plt.tight_layout(pad=.1)
plt.savefig('../plots/spin32.pdf')
plt.show()
