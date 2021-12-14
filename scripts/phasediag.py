import numpy as np
import scipy.optimize as spo
import mcextract as mce
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import foline
import maxima_tools

mc_1 = mce.MCArchive('../data/kr_scan_degen2.json')
mc_075 = mce.MCArchive('../data/kr_scan_J3=0.75.json')
mc_025 = mce.MCArchive('../data/kr_scan_no_degen2.json')
mc_0 = mce.MCArchive('../data/kr_scan_J3=0.json')

mc_corr_0 = mce.MCArchive('../data/kr_corrlen_J3=0.json')
mc_corr_025 = mce.MCArchive('../data/kr_corrlen_J3=0.25.json')
mc_corr_075 = mce.MCArchive('../data/kr_corrlen_J3=0.75.json')
mc_corr_1 = mce.MCArchive('../data/kr_corrlen_J3=1.json')

mcs = {
    0: mc_0,
    0.25: mc_025,
    0.75: mc_075,
    1.0: mc_1,
}

mcs_corrlen = {
    0: mc_corr_0,
    0.25: mc_corr_025,
    0.75: mc_corr_075,
    1.0: mc_corr_1,
}


critpoints = {
    1.0: (0.245, 0.392, 5, 1),
    0.0: (0.22, 0.353, 1, 1),
    0.75: (0.235, 0.369, 5, 1),
    0.25: (0.22, 0.354, 1, 1),
}

def Jcs(J2, Tmax):
    Ts = np.linspace(0,Tmax, 100)
    Jc = np.array([foline.Jcrit(Ts, 1, J2), Ts]).T
    return Jc

def borders(x):
    res = 0.5*(x[1:]+x[:-1])
    res = np.hstack([[2*res[0]-res[1]],res,[2*res[-1]-res[-2]]])
    return res

def fit_maxs(mc, obsname, Tskip, alpha):
    Ts = mc.get_parameter('T', unique=True)
    Js = mc.get_parameter('Jn', unique=True)
    L = mc.get_parameter('Lx', unique=True)[-1]


    maxs = [] 
    for j, J in enumerate(Js):
        cond = {'Jn':J, 'Lx':L}

        obs = mc.get_observable(obsname, filter=cond)

        tskip = Tskip if J > 0.3525 else 0


        maxTs = maxima_tools.find_maxima(Ts[tskip:], obs.mean[tskip:], alpha)

        for maxT in maxTs:
            maxs.append([J, maxT])

    return np.array(maxs)


def isochore(mc, Tskip, alpha):
    obsname = 'J'
    
    Ts = mc.get_parameter('T', unique=True)
    Js = mc.get_parameter('Jn', unique=True)
    J2 = mc.get_parameter('J3', unique=True)[0]
    L = mc.get_parameter('Lx', unique=True)[-1]

    Ts = Ts[Tskip:]
    jcrit = 1.0

    isochore = []
    for j, J in enumerate(Js):
        cond = {'Jn':J, 'Lx':L}

        obs = mc.get_observable(obsname, filter=cond)

        inter = spi.interp1d(Ts, obs.mean[Tskip:])
        if (inter(0.2)-jcrit)*(inter(0.6)-jcrit) < 0:
            try:
                sol = spo.root_scalar(lambda T: inter(T)-jcrit, bracket=[0.2,0.6])
                isochore.append([J, sol.root])
            except Exception as e:
                print(e)

    return np.array(isochore)


def plot_color(fig, ax, mc,obsname, obslabel, Jc=None, critpoint=None, ylim=[0,0.6], Tskip=5, paneloffset=0, norm=None, cmap=None):
    Ts = mc.get_parameter('T', unique=True)
    Js = mc.get_parameter('Jn', unique=True)
    Ls = mc.get_parameter('Lx', unique=True)
    J2 = mc.get_parameter('J3',unique=True)[0]

    Js = Js[Js<0.55]
    meshT = borders(Ts)[Tskip:,None]
    meshJ = borders(Js)[None,:]
    meshO = np.zeros([len(Ts)-Tskip, len(Js)])

    for L in Ls:
        for j, J in enumerate(Js):
            cond = {'Jn':J, 'Lx':L}

            obs = mc.get_observable(obsname, filter=cond)
            meshO[:,j] = obs.mean[Tskip:]

    if obsname == 'SpecificHeat':        
        mesh = ax.pcolormesh(meshJ, meshT, 3*np.abs(meshO), norm=norm, cmap=cmap,rasterized=True)
    elif obsname == 'J':
        mesh = ax.pcolormesh(meshJ, meshT, meshO, norm=norm, cmap=cmap, rasterized=True)
    else:
        mesh = ax.pcolormesh(meshJ, meshT, np.abs(meshO), norm=norm, cmap=cmap, rasterized=True)
    if type(Jc) != type(None):
        ax.plot(Jc[:,0],Jc[:,1], '-',markersize=0, linewidth=1.5, color='red')
    if critpoint != None:
        cp = critpoint
        ax.plot([cp[1]], [cp[0]], 'o', color='red')
    ax.set_xlim([0.25,0.5])
    ax.set_ylim(ylim)
    ax.text(0.02,0.95,'({}) $J_2/J_1 = {:g}$'.format(chr(paneloffset+ord('a')), J2), color='white',horizontalalignment='left', verticalalignment='top',transform=ax.transAxes)


def Ctri(T):
    return (9*np.power(1/np.cosh(3/(4.*T)),2))/(16.*np.power(T,2))

def plot_cut(f, ax, mc, Jcut, obsname='SpecificHeat',  fac=3, color=None, marker=None, markersize=None):
    Js = mc.get_parameter('Jn')
    J2 = mc.get_parameter('J3',unique=True)[0]

    J0 = Js[np.argmin(np.abs(Js-Jcut))]
    cond = dict(Jn=J0)
    Ts = mc.get_parameter('T', filter=cond)
    C = mc.get_observable(obsname, filter=cond)

    name = '$J/J_1 = {:.2g}$'.format(J0)
    p = ax.errorbar(Ts, fac*C.mean, fac*C.error, label=name, color=color, marker=marker, markersize=markersize)


    return J0, p[0].get_color()

def plot_cuts(f,ax1,ax2,Jcd,Jcnd):
    Js = mc_1.get_parameter('Jn',unique=True)
    clrs = []
    artists = []

    J0ts = [0.35,0.40]
    for ax, J0t in zip([ax1,ax2],J0ts):
        J0 = Js[np.argmin(np.abs(Js-J0t))]
        Cs = []
        for name, mc1 in zip(['$J_2/J_1=1$','$J_2/J_1=0$'],[mc_1, mc_075]):
            cond = dict(Jn=J0)
            Ts = mc1.get_parameter('T', filter=cond)
            C = mc1.get_observable('SpecificHeat', filter=cond)
            p = ax.errorbar(Ts, 3*C.mean, 3*C.error, label=name)
            clrs.append(p[0].get_color())
            artists.append(p)
        ax.set_xlabel('$T/J_1$')
        ax.set_ylabel('$C$')

    def specheat_tri(T, J3):
        return (np.exp(1/(2.*T))*(9*np.exp(1/T) + np.exp(J3/T)*(2*np.exp(3/(2.*T))*np.power(-1 + J3,2) + np.power(1 + 2*J3,2))))/(2.*np.power(2 + np.exp(3/(2.*T)) + np.exp((1 + 2*J3)/(2.*T)),2)*np.power(T,2))

    Jcidxd = np.argmin(np.abs(Jcd[:,0]-J0ts[1]))
    Jcidxnd = np.argmin(np.abs(Jcnd[:,0]-J0ts[1]))

    leg1 = ax1.legend(loc = 'upper right')
    plot1, = ax1.plot(Ts, specheat_tri(Ts,1),'--',color=clrs[0], label='$J=0,~J_2/J_1=1$')
    plot2, = ax1.plot(Ts, specheat_tri(Ts,0.75),'--',dashes=[4,1,1,1],color=clrs[1],label='$ J=0,~J_2/J_1=0$')
    ax1.legend(handles=[plot1,plot2])
    ax1.add_artist(leg1)
    ax1.set_ylim([None,1.3])
    
    vline1 = ax2.axvline(Jcd[Jcidxd,1],color=clrs[0], ls='--',label='$F_Q=F_D$')
    vline2 = ax2.axvline(Jcnd[Jcidxnd,1],color=clrs[1], ls='--',label='$F_Q=F_{D0}$')
    ax2.legend(handles=[artists[2],artists[3],vline1,vline2],loc='upper right')
    
    ax1.text(0.02,0.95,'(c) $J/J_1={}$'.format(J0ts[0]), horizontalalignment='left', verticalalignment='top',transform=ax1.transAxes)
    ax2.text(0.02,0.95,'(d) $J/J_1={}$'.format(J0ts[1]), horizontalalignment='left', verticalalignment='top',transform=ax2.transAxes)

    f.align_ylabels([ax1,ax2])


def plot_folines(fig, ax):
    J3min = 0.25
    

    J3s = [0, 0.5,0.75,0.9,1.0]
    for J3 in J3s:
        Ts = np.linspace(0,0.24,100)
        Jc = np.array([foline.Jcrit(Ts, 1, J3), Ts]).T

        
        ax.plot(Jc[:,0], Jc[:,1], marker='', label='$J_2/J_1={}$'.format(J3))
        ax.text(Jc[-1,0], Jc[-1,1], '$J_2/J_1 = {}$'.format(J3), fontsize=7, rotation=22)
    ax.set_ylim([0,0.32])
    ax.set_xlim([0.345,0.405])
    
    dJdT = []
    J3s = np.linspace(J3min, 1, 100)
    for J3 in J3s:
        Ts = np.linspace(0,0.24,100)
        Jc = np.array([foline.Jcrit(Ts, 1, J3), Ts]).T

        d = Jc[-1,:]-Jc[-2,:]
        dJdT.append(d[0]/d[1])

    ax.set_xlabel('$J/J_1$')
    ax.set_ylabel('$T/J_1$')

def fig_corrlen():
    f, axs = plt.subplots(1,2,figsize=(5.90,2), sharey=True)

    norm = colors.Normalize(vmin=0, vmax=2)
    cmap = 'inferno'
    for i, (ax, J2) in enumerate(zip(axs, [0,1])):
        iso = isochore(mcs[J2], Tskip=2, alpha=0.1)
        plot_color(f, ax, mcs_corrlen[J2],'JCorrLen', '$\\xi_Q$', Tskip=2, ylim=[0.12,0.6], Jc=Jcs(J2, critpoints[J2][0]),paneloffset=i, norm=norm, cmap=cmap, critpoint=critpoints[J2])

        fline = Jcs(J2, 0.6)
        specheat_maxs = fit_maxs(mcs[J2], 'SpecificHeat', Tskip=6, alpha=0.02)
        ax.scatter(iso[:,0], iso[:,1], s=6, marker='^', color='#66ff66', linewidth=0, label='$\\langle l_\\Delta \\rangle = 1$')
        ax.scatter(specheat_maxs[:,0], specheat_maxs[:,1], s=4, linewidth=0, color='white', label='$C^\\mathrm{max}$')
        ax.legend(loc=4, borderpad=0, handletextpad=0, labelcolor='white')
    axs[0].set_xlim(0.25,0.4999)
    axs[0].set_ylabel('$T/J_1$')
    axs[0].set_xlabel('$J/J_1$')
    axs[1].set_xlabel('$J/J_1$')
    plt.tight_layout(pad=.2)

    f.subplots_adjust(right=0.88)
    bottom = axs[1].get_position().y0
    top = axs[1].get_position().y1
    cbarax = f.add_axes([0.90,bottom,0.02,top-bottom])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    f.colorbar(sm,label='$\\xi_Q$',cax=cbarax)
    plt.savefig('../plots/corrlen.pdf')
    plt.show()

    
def fig_phasediag():
    f, axs = plt.subplots(3,2, figsize=(5.9,2*3), sharey='row')
    axs[2][0].set_visible(False)
    axs[2][1].set_visible(False)
    axs = list(axs[:2,:].flatten())
    
    norm = colors.Normalize(vmin = 0.45, vmax = 1.5) 

    for i, (ax, J2) in enumerate(zip(axs, mcs.keys())):
        plot_color(f, ax, mcs[J2], 'J', '$\\langle{}l_\\Delta\\rangle$', Jc=Jcs(J2, critpoints[J2][0]), critpoint=critpoints[J2], paneloffset=i, norm=norm, cmap='inferno')
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    axs[0].set_xlim(0.25,0.4999)
    axs[2].set_xlim(0.25,0.4999)

    axs[0].set_ylabel('$T/J_1$')
    axs[2].set_ylabel('$T/J_1$')
    axs[2].set_xlabel('$J/J_1$')
    axs[3].set_xlabel('$J/J_1$')
    plt.tight_layout(pad=.2,rect=(0,0.1,1,1))
    axlines = f.add_axes([0.275,0.08, 0.45, 0.25])

    f.subplots_adjust(hspace=0.07, right=0.85)
    bottom = axs[3].get_position().y0
    top = axs[1].get_position().y1
    cbarax = f.add_axes([0.88,bottom,0.02,top-bottom])
    sm = cm.ScalarMappable(norm=norm, cmap='inferno')
    sm.set_clim(0.5,1.5)
    cb = f.colorbar(sm, label='$\\langle l_\\Delta \\rangle$', cax=cbarax)
    cb.set_ticks([0.5,0.75,1.0,1.25, 1.5])
    

    plot_folines(f, axlines)
    axlines.text(0.02,0.95,'(e)', horizontalalignment='left', verticalalignment='top',transform=axlines.transAxes)
    plt.savefig('../plots/phasediag.pdf')
    plt.show()

def fig_specheat_scan():
    Ts = np.linspace(0,0.6,100)

    Jc_d = np.array([foline.Jcrit(Ts, 1, 1), Ts]).T
    Jc_nd = np.array([foline.Jcrit(Ts, 1, 0), Ts]).T

    f, axs = plt.subplots(4,2,figsize=(5.9,7), sharex='col')
    #plot_cuts(f,axs[0][1],aylabel='$C$', xs[1][1], Jc_d, Jc_nd)

    norm=colors.LogNorm(vmin=0.4, vmax=10)
    cuts = {}
    
    for i, (row, J2) in enumerate(zip(axs, mcs.keys())):
        plot_color(f, row[0], mcs[J2], 'SpecificHeat', '$C$', Tskip=2, ylim=[0.12,0.6], paneloffset=i, Jc=Jcs(J2, critpoints[J2][0]), norm=norm, cmap='inferno', critpoint=critpoints[J2])
        specheat_maxs = fit_maxs(mcs[J2], 'SpecificHeat', Tskip=4+2*(J2==1), alpha=0.02)
        row[0].scatter(specheat_maxs[:,0], specheat_maxs[:,1], s=3, color='white', linewidth=0, label='$C^\\mathrm{max}$')

        cuts[J2] = [critpoints[J2][1]-0.03, critpoints[J2][1]+0.03]
        for Jcut in cuts[J2]:
            actual_Jcut, color = plot_cut(f, row[1], mcs[J2], Jcut)
            row[0].arrow(actual_Jcut, 0.16, 0, -0.03, color=color)

        row[1].text(0.02,0.95,'({}) $J_2/J_1 = {:g}$'.format(chr(len(mcs.keys())+i+ord('a')), J2), horizontalalignment='left', verticalalignment='top',transform=row[1].transAxes)
        row[1].legend(loc=2,bbox_to_anchor=(-0.03,0.84))
        row[1].set_ylabel('$C$')
        row[1].set_ylim([-0.05,5])
        row[1].set_xlim([0.1,0.6])
        row[0].set_ylabel('$T/J_1$')

    axs[-1,0].set_xlabel('$J/J_1$')
    axs[-1,-1].set_xlabel('$T/J_1$')
    plt.tight_layout(pad=.2)
    
    plt.subplots_adjust(wspace=0.17, top=0.9)
    left = axs[0,0].get_position().x0
    right = axs[0,0].get_position().x1
    cbarax = f.add_axes([left,0.92,right-left,0.02])
    
    sm = cm.ScalarMappable(norm=norm, cmap='inferno')
    cb = f.colorbar(sm,  orientation='horizontal',label='$C$', cax=cbarax)
    cbarax.xaxis.set_ticks_position('top')
    cbarax.xaxis.set_label_position('top')


    axins = inset_axes(axs[3,1], width=0.5, height=0.6, loc=1,
    bbox_to_anchor=(0.98,0.94), bbox_transform=axs[3,1].transAxes)

    actual_Jcut, color = plot_cut(f, axs[3,1], mcs[1], 0.26)
    actual_Jcut, _ = plot_cut(f, axins, mcs[1], actual_Jcut, color=color, marker='>', markersize=2)
    axs[3,0].arrow(actual_Jcut, 0.16, 0, -0.03, color=color)
    axins.set_ylim([0.4,0.55])
    axins.set_xlim([0.1,0.6])
    rects, connects = axs[3,1].indicate_inset_zoom(axins)
    
    for i, (row, J2) in enumerate(zip(axs, mcs.keys())):
        row[1].legend(loc=2,bbox_to_anchor=(-0.03,0.84))
    connects[0].set_visible(True)
    connects[1].set_visible(True)
    connects[2].set_visible(True)
    connects[3].set_visible(True)
    axins.tick_params(axis='both', which='major', labelsize=7)

    plt.savefig('../plots/specheat_scan.pdf')
    plt.show()

fig_corrlen()
fig_specheat_scan()
fig_phasediag()


