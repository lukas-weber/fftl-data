import numpy as np
import scipy.optimize as spo
from collections import defaultdict
import mcextract as mce
import matplotlib.pyplot as plt

mc_1 = mce.MCArchive('../data/scaling5.json')
mc_075 = mce.MCArchive('../data/scaling_J3=0.75.json')
mc_025 = mce.MCArchive('../data/nd_scaling1.json')
mc_0 = mce.MCArchive('../data/scaling_J3=0.json')

mc_1_corrlen = mce.MCArchive('../data/scaling_cp_corrlen.json')


def plot_magQ():
    Ts = mc.get_parameter('T', unique=True)
    Js = mc.get_parameter('Jn', unique=True)
    for T in Ts:
        for J in Js:
            cond = {'T': T, 'Jn': J}
            Ls = mc.get_parameter('Lx', filter=cond)
            obsj = mc.get_observable('J', filter=cond)
 

            plt.errorbar(1/Ls, obsj.mean-1, obsj.error, label='$T = {:.3g}, J = {:.3f}$'.format(T,J))
        plt.legend()
        plt.show()

def plot(obsname,gamma, Tdim=0, log=True):
    Js = mc.get_parameter('Jn', unique=True)
    Ls = mc.get_parameter('Lx', unique=True)
    for L in Ls:
        for J in Js:
            cond = {'Lx': L, 'Jn':J}
            Ts = mc.get_parameter('T', filter=cond)
            idx = np.argsort(Ts)
            Ts = Ts[idx]
            obs = mc.get_observable(obsname, filter=cond)
            obsJ = mc.get_observable('J', filter=cond)
            obs.mean = obs.mean[idx]/ Ts**Tdim
            obs.error = obs.error[idx]/ Ts**Tdim
            obsJ.mean = obsJ.mean[idx]
            obsJ.error = obsJ.error[idx]

            plt.errorbar(Ts, L**gamma*obs.mean, L**gamma*obs.error, label='$L = {:.3g}, J = {:.3g}$'.format(L,J))
        plt.ylabel('{} $L^{{{:.2g}}}$'.format(obsname,gamma))
        plt.xlabel('$T$')
        plt.legend()
        plt.show()
def plot_T(obsname,gamma, Tdim=0, log=True):
    Ts = mc.get_parameter('T', unique=True)
    Ls = mc.get_parameter('Lx', unique=True)
    for L in Ls:
        for T in Ts[Ts<0.29]:
            cond = {'Lx': L, 'T':T}
            Js = mc.get_parameter('Jn', filter=cond)
            idx = np.argsort(Js)
            Js = Js[idx]
            obs = mc.get_observable(obsname, filter=cond)
            obsJ = mc.get_observable('J', filter=cond)
            obs.mean = obs.mean[idx]/ T**Tdim
            obs.error = obs.error[idx]/ T**Tdim
            obsJ.mean = obsJ.mean[idx]
            obsJ.error = obsJ.error[idx]


            plt.errorbar(Js, L**gamma*obs.mean, L**gamma*obs.error, label='$L = {:.3g}, T = {:.3g}$'.format(L,T))
        plt.ylabel('{} $L^{{{:.2g}}}$'.format(obsname,gamma))
        plt.xlabel('$J$')
        plt.legend()
        plt.show()

Lmax_degen = {
    0.20: 16,
    0.21: 16,
    0.22: 24,
    0.23: 24,
    0.24: 24,
    0.25: 32,
    0.26: 32,
    0.27: 48,
    0.28: 48,
}

Lmax_nodegen = {
    0.20: 16,
    0.21: 16,
    0.22: 24,
    0.23: 32,
    0.24: 32,
    0.25: 32,
    0.26: 32,
    0.27: 48,
    0.28: 48,
}

Lmax_corrlen = {
    0.20: 16,
    0.21: 16,
    0.22: 48,
    0.23: 48,
    0.24: 48,
    0.245: 48,
    0.25: 48,
    0.26: 32,
    0.27: 48,
    0.28: 48,
}

Lmax_075 = {
    0.20: 16,
    0.21: 16,
    0.22: 24,
    0.23: 24,
    0.24: 78,
    0.25: 78,
    0.26: 78,
    0.27: 48,
    0.28: 48,
}

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

def maxfun(J, J0, gamma, a, c):
    return a*(gamma**2)/np.abs((J-J0)**2+gamma**2)+c

def fit_maxs(mc, obsname, Lmax, Tmax, Tdim, Ldim, fitrange, fac=1):
    Ts = mc.get_parameter('T', unique=True)
    allmax = {}

    for T in Ts[Ts < Tmax]:
        Ls = mc.get_parameter('Lx', unique=True, filter=dict(T=T))
        maxs = []
        for L in Ls[Ls<=Lmax[T]]:
            cond = dict(Lx=L,T=T)
            Js = mc.get_parameter('Jn', filter=cond)
            idx = np.argsort(Js)
            obs = mc.get_observable(obsname, filter=cond)
            N = L**Ldim
            obs.mean = obs.mean[idx]*N*fac/T**Tdim
            obs.error = obs.error[idx]*N*fac/T**Tdim
            Js = Js[idx]
            imax = np.argmax(obs.mean)

            p0 = (Js[imax], 0.01/L, obs.mean[imax], 0)
            fr = fitrange/L**0.7
            if T > 0.26:
                fr=0.01/L**0.5
            fitr = [
                max(0,np.argmin(np.abs(Js-p0[0]+fr))-2),
                min(np.argmin(np.abs(Js-p0[0]-fr))+2,len(Js)),
            ]
            
            try:
                popt,perr = fit_bootstrap(maxfun, Js[fitr[0]:fitr[1]], obs.mean[fitr[0]:fitr[1]],obs.error[fitr[0]:fitr[1]], p0=p0)
                xx = np.linspace(Js[fitr[0]],Js[fitr[1]-1], 100)
                maxs.append((L,popt[2]+popt[3], (perr[2]**2+perr[3]**2)**0.5, popt[0],perr[0], (xx, maxfun(xx, *popt))))
            except (RuntimeError, ValueError) as e:
                print('(T={},L={}): {}'.format(T, L, e))
        allmax[T] = maxs
    return allmax


mcs = { 0: mc_0, 0.25: mc_025, 0.75: mc_075, 1.0: mc_1 }
critpoints = {
    1.0: (0.245, 0.392, 5, 1),
    0.0: (0.22, 0.353, 1, 1),
    0.75: (0.235, 0.369, 5, 1),
    0.25: (0.22, 0.354, 1, 1),
}

Lmaxs = {
    1.0: Lmax_degen,
    0.75: Lmax_075,
    0.25: Lmax_nodegen,
    0: Lmax_nodegen,
}


chi_maxs = {}
specheat_maxs = {}
corrlen_maxs = {}

chi_maxs[1.0] = fit_maxs(mc_1, 'JVar', Lmax=Lmaxs[1], Tmax=0.29, Tdim=1, Ldim=2, fitrange=0.009)
chi_maxs[0.75] = fit_maxs(mc_075, 'JVar', Lmax=Lmaxs[0.75], Tmax=0.29, Tdim=1, Ldim=2, fitrange=0.009)
chi_maxs[0.25] = fit_maxs(mc_025, 'JVar', Lmax=Lmaxs[0.25], Tmax=0.26, Tdim=1, Ldim=2, fitrange=0.012)
chi_maxs[0] = fit_maxs(mc_0, 'JVar', Lmax=Lmaxs[0], Tmax=0.26, Tdim=1, Ldim=2, fitrange=0.012)


specheat_maxs[1.0] = fit_maxs(mc_1, 'SpecificHeat', Lmax=Lmax_degen, Tmax=0.29, fac=3, Tdim=0, Ldim=0, fitrange=0.006)
specheat_maxs[0.75] = fit_maxs(mc_075, 'SpecificHeat', Lmax=Lmax_degen, Tmax=0.29, fac=3, Tdim=0, Ldim=0, fitrange=0.006)
specheat_maxs[0.25] = fit_maxs(mc_025, 'SpecificHeat', Lmax=Lmax_nodegen, Tmax=0.26, fac=3, Ldim=0, Tdim=0, fitrange=0.009)
specheat_maxs[0] = fit_maxs(mc_0, 'SpecificHeat', Lmax=Lmax_nodegen, Tmax=0.26, fac=3, Ldim=0, Tdim=0, fitrange=0.009)

corrlen_maxs[0] = fit_maxs(mc_0, 'JCorrLen', Lmax=Lmax_corrlen, Tmax=0.26, Ldim=0, Tdim=0, fitrange=0.014)
corrlen_maxs[1] = fit_maxs(mc_1_corrlen, 'JCorrLen', Lmax=Lmax_corrlen, Tmax=0.26, Ldim=0, Tdim=0, fitrange=0.016)


def plot_max(fig, axs, J2, mc, obsname, maxs, Lmax, Tdim=0, critexp=0, critpoint=(0,0), bshifts={}, crotation=-30, paneloffset=0):
    Ts = mc.get_parameter('T', unique=True)

    Tover = Ts[Ts>critpoint[0]].min()
    Tunder = Ts[Ts < Tover].max()
    
    Ls = mc.get_parameter('Lx', unique=True, filter=dict(T=Tover))
    for i, L in enumerate(Ls[Ls<=Lmax[Tover]]):
        cond = dict(Lx=L,T=Tover)
        Js = mc.get_parameter('Jn', filter=cond)
        idx = np.argsort(Js)
        obs = mc.get_observable(obsname, filter=cond)
        N = L**2
        obs.mean = obs.mean[idx]*N/ Tover**Tdim
        obs.error = obs.error[idx]*N/ Tover**Tdim
        Js = Js[idx]
        imax = np.argmax(obs.mean)

        axs[0].errorbar(Js,obs.mean,obs.error,ls='',label='$L={}$'.format(L))
        axs[0].plot(maxs[Tover][i][5][0], maxs[Tover][i][5][1], '-',color='black',markersize=0)

    axs[0].set_xlim([np.max([critpoint[1]-0.003,Js.min()]),np.min([critpoint[1]+0.004,Js.max()])])
    axs[0].set_ylabel('$J_1 \chi_Q$')
    axs[0].text(0.03,0.95,'$J_2/J_1\!=\!{}$'.format(J2),horizontalalignment='left', verticalalignment='top',transform=axs[0].transAxes)
    axs[0].text(0.03,0.83,'$T/J_1\!=\!{}$'.format(Tover),horizontalalignment='left', verticalalignment='top',transform=axs[0].transAxes)
    axs[0].text(0.02,0.15,'({})'.format(chr(paneloffset+ord('a'))),horizontalalignment='left', verticalalignment='bottom',transform=axs[0].transAxes)
    axs[0].legend(loc=1)

    axs[1].axhline(2.43/3,ls='--',color='black')
    Tmin = min(maxs.keys())
    shifts = defaultdict(lambda: 0)
    shifts.update(bshifts)
    for T, Tmaxs in maxs.items():
        Tmaxs = np.array(Tmaxs, dtype=object)
        Lss = Tmaxs[:,0]
        axs[1].errorbar(Lss,Tmaxs[:,1]*Lss**critexp,Tmaxs[:,2]*Lss**critexp, label='$T/J_1 = {:.3g}$'.format(T))
        axs[1].text(Tmaxs[-1,0]*1.15, Tmaxs[-1,1]*Tmaxs[-1,0]**critexp+shifts[T], '${}{}$'.format('T/J_1\\!=\\!'if T==Tmin else '', T),verticalalignment=('center' if T <= Tmin+0.01 else 'top'),fontsize=7)
    axs[1].set_yscale('log')
    axs[1].set_xscale('log',subs=[])
    axs[1].set_ylim([None,5])
    axs[1].set_xticks([10,20,40,80])
    axs[1].set_xticklabels(['10','20','40','80'])
    axs[1].set_xlim([10,85])
    axs[1].set_ylabel('$J_1 \chi_Q^{\\mathrm{max}} L^{-7/4}$')
    axs[1].text(0.05,0.95,'$T_c/J_1 = {}({})$'.format(critpoint[0],critpoint[2]),horizontalalignment='left', verticalalignment='top',transform=axs[1].transAxes)

    def extra(x, a, b, c):
        return a+b*x+c*x**2
    colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#c6761d']
    for i, T in enumerate([Tunder, Tover]):
        
        Tmaxs = np.array(maxs[T], dtype=object)
        axs[2].errorbar(1/Tmaxs[:,0], Tmaxs[:,3],Tmaxs[:,4],color=colors[1+i],marker=['v','>'][i], label='$T/J_1={}$'.format(T))
        popt, perr = fit_bootstrap(extra, 1/Tmaxs[:,0], Tmaxs[:,3], Tmaxs[:,4], p0=(Tmaxs[-1,3], 0, 0))
        axs[2].text(0.04, Tmaxs[1,3]-0.0002, '$T/J_1 = {}$'.format(T), rotation=crotation, fontsize=7)
    axs[2].set_ylabel('$J^{\mathrm{max}}/J_1$')
    axs[2].text(0.05,0.95,'$J_c/J_1 = {}({})$'.format(critpoint[1],critpoint[3]),horizontalalignment='left', verticalalignment='top',transform=axs[2].transAxes)
    axs[2].set_xlim([0,0.1])
    axs[2].set_ylim([None,Tmaxs[-1,3]+0.001])



def fig_critpoint():

    bshifts = defaultdict(lambda: {})
    bshifts.update({
        0.75: {0.21:0.4, 0.25:0.04},
        0.25: {0.2:0.4, 0.22: 0.2},
        0.: {0.2: 0.2, 0.21: 0.2},
    })

    crotations = defaultdict(lambda: -30)
    crotations.update({
        0.25: -35,
        0: -35,
        })

    fig, axs = plt.subplots(4,3,figsize=(5.9, 2*3.3), gridspec_kw=dict(width_ratios=(1.5,1,1)))

    axs[-1][0].set_xlabel('$J/J_1$')
    axs[-1][1].set_xlabel('$L$')
    axs[-1][2].set_xlabel('$1/L$')

    for i, (J2,row) in enumerate(zip(mcs.keys(), axs)):
        plot_max(fig, row, J2, mcs[J2],'JVar', maxs=chi_maxs[J2], Lmax=Lmaxs[J2], critpoint=critpoints[J2], Tdim=1, bshifts=bshifts[J2], critexp=-7/4, crotation=crotations[J2], paneloffset=i)
    axs[0][1].set_ylim(0.1, 4)
    axs[1][1].set_ylim(0.1, 4)
    axs[2][1].set_ylim(0.1, 4)
    plt.tight_layout(pad=0.15)
    plt.subplots_adjust(wspace=0.5)
    plt.savefig('../plots/critpoint.pdf')
    plt.show()

def plot_specheat(ax, mc, maxs, T, Lmax, obsname='SpecificHeat', xlim=None, ylim=None, chosen_Ls=[]):

    Ls = mc.get_parameter('Lx', unique=True, filter=dict(T=T))
    J2 = mc.get_parameter('J3', unique=True, filter=dict(T=T))[0]

    for i, L in enumerate(Ls[Ls<=Lmax[T]]):
        if len(chosen_Ls) > 0 and L not in chosen_Ls:
            continue
        cond = dict(Lx=L,T=T)
        Js = mc.get_parameter('Jn', filter=cond)
        idx = np.argsort(Js)
        obs = mc.get_observable(obsname, filter=cond)

        fac = 3 if obsname == 'SpecificHeat' else 1
        
        obs.mean = fac*obs.mean[idx]
        obs.error = fac*obs.error[idx]
        Js = Js[idx]
        
        ax.errorbar(Js,obs.mean,obs.error,ls='',label='$L={}$'.format(L))
        ax.plot(maxs[T][i][5][0], maxs[T][i][5][1], '-',color='black',markersize=0)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel('$C$')
    ax.set_xlabel('$J/J_1$')
    ax.text(0.05,0.95,'$J_2/J_1 = {},~T/J_1 = {}$'.format(J2, T), horizontalalignment='left', verticalalignment='top',transform=ax.transAxes)


    ax.legend(loc='center left')

def plot_compmax(ax, allmaxs):
    for name, T, maxs, kwargs in allmaxs:
        maxs = np.array(maxs[T], dtype=object)
        ax.errorbar(maxs[:,0], maxs[:,1]/maxs[0,1], maxs[:,2]/maxs[0,1], label=name, **kwargs)
    xx = np.linspace(12,32,100)
    ax.plot(xx, (xx/12)**(7/4), '--', markersize=0, color='black', label='$ (L/12)^{7/4}$')
    ax.set_xlabel('L')
    ax.set_ylabel('$\\chi_Q^\\mathrm{max}/\\chi_{Q,L=12}^\mathrm{max},~C^\\mathrm{max}/C^\mathrm{max}_{L=12}$')
    ax.set_ylim([None,8])
    ax.set_xlim([10,None])
    ax.legend(handlelength=2, borderpad=0)


def fig_specheat():
    fig, axs = plt.subplots(2,2,figsize=(5,4))
    axs = axs.flat
    plot_specheat(axs[0], mc_0, maxs=specheat_maxs[0], T=0.23, Lmax=Lmax_nodegen, xlim=[0.35,0.356], ylim=[None,10])
    plot_specheat(axs[1], mc_1, maxs=specheat_maxs[1], T=0.25, Lmax=Lmax_degen, xlim=[0.390,0.3952], ylim=[None,80])


    plot_compmax(axs[2], [
        ('$\chi_Q,~J_2/J_1=1$', 0.25, chi_maxs[1], {'color':'#e7298a', 'marker':'^', 'markerfacecolor':'white', 'ls':'-'}),
        ('$\chi_Q,~J_2/J_1=0$', 0.23, chi_maxs[0], {'color':'#e7298a', 'marker':'o', 'ls':'--'}),
        ('$C,~J_2/J_1 = 1$', 0.25, specheat_maxs[1], {'color':'#7570b3', 'marker':'^', 'markerfacecolor':'white', 'ls':'-'}),
        ('$C,~J_2/J_1 = 0$', 0.23, specheat_maxs[0], {'color':'#7570b3', 'marker':'o', 'ls':'--'}),
        ])
    axs[0].text(0.03,0.15,'(a)', transform=axs[0].transAxes)
    axs[1].text(0.03,0.15,'(b)', transform=axs[1].transAxes)
    axs[2].text(0.03,0.15,'(c)', transform=axs[2].transAxes)
    axs[3].text(0.03,0.15,'(d)', transform=axs[3].transAxes)

    from ising_critical import plot_ising8c
    plot_ising8c(axs[3])

    plt.tight_layout(pad=.1)
    plt.savefig('../plots/specheat.pdf')
    plt.show()

def fig_corrlen():
    fig, axs = plt.subplots(1,3,figsize=(5.9, 2))
    plot_specheat(axs[0], mc_0, maxs=corrlen_maxs[0], T=0.23, Lmax=Lmax_nodegen, obsname='JCorrLen', xlim=[0.35,0.356], ylim=[None,11.3])
    axs[0].set_ylabel('$\\xi_Q$')
    plot_specheat(axs[1], mc_1_corrlen, maxs=corrlen_maxs[1], T=0.25, Lmax=Lmax_nodegen, obsname='JCorrLen', xlim=[0.390,0.3952], ylim=[None,11.3], chosen_Ls = np.array([12,16,24,32]))
    axs[1].set_ylabel('$\\xi_Q$')

    maxs = np.array(corrlen_maxs[0][0.23], dtype=object)
    maxs2 = np.array(corrlen_maxs[1][0.25], dtype=object)
    axs[2].errorbar(maxs[:,0], maxs[:,1], maxs[:,2], label = '$J_2/J_1=0$')
    axs[2].errorbar(maxs2[::2,0], maxs2[::2,1], maxs2[::2,2], label = '$J_2/J_1=1$')
    axs[2].set_ylabel('$\\xi_Q^\mathrm{max}$')
    axs[2].set_xlabel('$L$')
    axs[2].legend()
    
    axs[0].text(0.03,0.17,'(a)', transform=axs[0].transAxes)
    axs[1].text(0.03,0.17,'(b)', transform=axs[1].transAxes)
    axs[2].text(0.03,0.17,'(c)', transform=axs[2].transAxes)

    plt.tight_layout(pad=.1)
    plt.savefig('../plots/corrlen_scaling.pdf')
    plt.show()

    

fig_corrlen()
fig_specheat()
fig_critpoint()
