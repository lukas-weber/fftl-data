"""
floline_check.py checks the asymptotic expressions against the exact solution of the free-energy argument used to approximate the first-order line.
"""

import foline
import numpy as np
import matplotlib.pyplot as plt


J1 = 1
J3 = 1

eAFM, a = (foline.pQ-foline.pD)[:2]

epsQ = 0.5 * J1 + 0.25 * J3
epsD1 = - J1 + 0.25 * J3
epsD0 = - 0.75 * J3

eQD = epsQ-epsD1

Jc0 = -eQD/eAFM

Jpre = eAFM/6/eQD**2 * a

Ts = np.linspace(0.05,0.5, 100)
Jcrit = foline.Jcrit(Ts, J1, J3)
Jcritapprox = Jc0 + Jpre * Ts**3

Jcritapprox = Jc0 - np.log(2)/eAFM * Ts 

plt.plot(Ts, Jcrit)
plt.plot(Ts, Jcritapprox)
plt.xlabel("$T/J_1$")
plt.ylabel("$J_c/J_1$")
plt.show()
plt.plot(Ts, np.abs(Jcrit-Jcritapprox))
plt.plot(Ts, Ts**3)
plt.plot(Ts, Ts**5)
plt.xlabel("$T/J_1$")
plt.ylabel("$(J_c-J_c^\\mathrm{approx})/J_1$")
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.show()


