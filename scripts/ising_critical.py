import numpy as np
import mcextract as mce
import matplotlib.pyplot as plt


def rot_specheat(phi, T, L, C, EM, chi):
    return (T**2*C*np.cos(phi)**2-2*EM*L**2*np.cos(phi)*np.sin(phi)+chi*T*np.sin(phi)**2)/T**2
def rot_chi(phi, T, L, C, EM, chi):
    return (T**2*C*np.sin(phi)**2+2*EM*L**2*np.cos(phi)*np.sin(phi)+chi*T*np.cos(phi)**2)/T

obs_specheat = dict(name=r'C_I', fun=rot_specheat)
obs_chi = dict(name=r'\chi_I', fun=rot_chi)

def bootstrop(fun, xs, sigxs, nsamples=50):
    y = fun(xs)
    ys = np.zeros([len(y), nsamples])
    for i in range(nsamples):
        xrs = []
        for x, sigx in zip(xs, sigxs):
            xrs.append(x + sigx * np.random.normal(size=len(x)))
        ys[:,i] = fun(xrs)
    return y, np.std(ys, axis=1)
        
def plot_rot(mc, ax, obs, phi, detune, plotargs={}, normalize=True):
    cond = dict(detune=detune)

    Ls = mc.get_parameter('Lx', filter=cond)
    T = mc.get_parameter('T', filter=cond)

    C = mc.get_observable('SpecHeat', filter=cond)
    chi = mc.get_observable('MagChi', filter=cond)
    EM = mc.get_observable('EMagCorr', filter=cond)

    rot_C, sig_rot_C = bootstrop(lambda xs: obs['fun'](phi, T, Ls, *xs), [C.mean, EM.mean, chi.mean], [C.error, EM.error, chi.error])

    philabel = '\\frac{{{:.0g}}}{{15}} \\pi'.format(phi*15/np.pi)
    if phi == 0:
        philabel = r'0 \vphantom{\frac{1}{15}}'
    norm = rot_C[0]
    if not normalize:
        norm = 1
    ax.errorbar(Ls, rot_C/norm, sig_rot_C/norm, label='${},~\\phi={}$'.format(obs['name'], philabel), **plotargs)

def plot_ising8c(ax):
    phi1 = np.pi/15
    phi2 = np.pi*3/15
    mc = mce.MCArchive('../data/ising_critical.json')
    plot_rot(mc, ax, obs_chi, phi2, True, plotargs={'ls':'-', 'markerfacecolor':'white', 'color':'#e7298a', 'marker':'^'})
    plot_rot(mc, ax, obs_chi, phi1, True, plotargs={'ls':'--', 'color':'#e7298a', 'marker':'o'})
    plot_rot(mc, ax, obs_specheat, phi2, True, plotargs={'ls':'-', 'markerfacecolor':'white', 'color':'#7570b3', 'marker':'^', 'zorder':10})
    plot_rot(mc, ax, obs_specheat, phi1, True, plotargs={'ls':'--', 'color':'#7570b3', 'marker':'o'})

    ax.set_ylabel('$\chi_I/\chi_{I,L=12},~C_I/C_{I,L=12}$')
    ax.set_xlim([10,37])
    ax.set_ylim([0.5,8])
    x = np.linspace(12,100,50)
    ax.plot(x, (x/12)**(7/4), '--', label='$(L/12)^{7/4}$', zorder=10, color='black')
    
    ax.legend(handlelength=2, borderpad=0)
    
    ax.set_xlabel('$L$')
    ax.text(0.99,0.03, '$(\\tilde{T},\\tilde{h})\!=\!(T_c+0.01,0)$',horizontalalignment='right', verticalalignment='bottom',transform=ax.transAxes)
    
def fig_rot(mc):
    f, ax = plt.subplots(1,1,figsize=(4,2.3))

    x = np.linspace(12,100,50)

    ax.text(0.97,0.03, '$(\\tilde{T}, \\tilde{h}) = (T_c, h)$',horizontalalignment='right', verticalalignment='bottom',transform=ax.transAxes)
    ax.plot(x, 10*(x/12)**(7/4), '--', label='$a L^{7/4}$', zorder=10, color='black')
    for i in range(4):
        plot_rot(mc, ax, obs_specheat, i*np.pi/15, False, normalize=False)
    ax.legend(handlelength=2, ncol=2, borderpad=0)
    ax.set_xlabel('$L$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('$C_I$')
    ax.set_xlim([8,110])
    ax.set_ylim([0.4,800])
    plt.tight_layout()
    plt.savefig('../plots/ising_critical.pdf')
    plt.show()

if __name__ == '__main__':
    mc = mce.MCArchive('../data/ising_critical.json')
    fig_rot(mc)
