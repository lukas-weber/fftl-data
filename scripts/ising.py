import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import mcextract as mce
import scipy.interpolate as spi
import scipy.optimize as spo
import itertools

import maxima_tools

mc = mce.MCArchive('../data/ising_scan2.json')

norm = colors.Normalize(0,3)
cmap = 'inferno'


def rotated_interpolate(x, y, z, center, angle, xi):
    rxi = np.zeros_like(xi)
    rxi[:,0] = np.sin(angle)*(xi[:,1]-center[0]) + np.cos(angle)*(xi[:,0]-center[1]) + center[1]
    rxi[:,1] = np.cos(angle)*(xi[:,1]-center[0]) - np.sin(angle)*(xi[:,0]-center[1]) + center[0]
    
    return spi.interpn([y[:,0], x[0,:]], z, rxi, bounds_error=False, fill_value=0, method='linear')

def circ_slice(x, y, z, center, angle, radius, steps=100):
    phis = np.linspace(0, 2*np.pi, steps)
    xi = np.column_stack([center[1] + radius*np.sin(phis), center[0] + radius*np.cos(phis)])

    return phis, rotated_interpolate(x, y, z, center, angle, xi)


def rotate_interpolate(x, y, z, center, angle):
    xi = np.column_stack([y.flatten(),x.flatten()])
    vals = rotated_interpolate(x, y, z, center, angle, xi)

    return vals.reshape(x.shape)


def obs_scan(fig, ax, mc, phi):
    hs = mc.get_parameter('h',unique=True)
    Ts = mc.get_parameter('T', unique= True)
    Lx = mc.get_parameter('Lx',unique=True)[0]

    meshh,meshT = np.meshgrid(hs,Ts)

    C = mc.get_observable('SpecHeat')
    chi = mc.get_observable('MagChi')
    EM = mc.get_observable('EMagCorr')

    meshC = meshT**2*C.mean.reshape(meshh.shape)
    meshchi = meshT*chi.mean.reshape(meshh.shape)
    meshEM = EM.mean.reshape(meshh.shape)*Lx**2

    meshO = (meshC*np.cos(phi)**2-2*meshEM*np.cos(phi)*np.sin(phi)+meshchi*np.sin(phi)**2)/meshT**2

    Tc = 2.2691
    cp = (0, Tc)
    meshOr = rotate_interpolate(meshh,meshT,meshO, angle=phi, center=cp)


    rads = np.linspace(0.01, 0.1, 20)
    maxlines = []

    vert_maxlines = []
    hcuts = np.arange(len(hs))[::2]
    for hidx in hcuts:        
        maxTs = maxima_tools.find_maxima(Ts, meshOr[:,hidx],alpha=0.05)
        for maxT in maxTs:
            vert_maxlines.append([hs[hidx],maxT-Tc])
    vert_maxlines = np.array(vert_maxlines)
    Tidx = len(Ts[Ts-Tc < 0])

    maxs = Ts[Tidx+np.argmax(meshOr[Tidx:,:], axis=0)]-Tc
    
    plot = ax.pcolormesh(meshh,meshT-Tc,meshOr, norm=norm, cmap=cmap, shading='auto', rasterized=True)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1.5])

rows = 2
cols = 2
fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(5,4))

for i in range(rows):
    axs[i][0].set_ylabel('$T-T_c$')
for i in range(cols):
    axs[-1][i].set_xlabel('$h$')



phis = np.linspace(0, np.pi/2, 6)
for y, row in enumerate(axs):
    for x, ax in enumerate(row):
        idx = x*rows+y
        deno=15
        phi = idx/deno*np.pi
        obs_scan(fig, ax, mc, phi)

        philabel = '0' if phi == 0 else '\\frac{{{}}}{{{}}}\,\pi'.format(idx,deno)
        ax.text(0.02,0.95, '({}) $\\phi={}$'.format(chr(ord('a')+idx), philabel), color='white',horizontalalignment='left', verticalalignment='top',transform=ax.transAxes)

fig.tight_layout(pad=.2)
bottom = axs[-1][-1].get_position().y0
top = axs[0][-1].get_position().y1
cbarax = fig.add_axes([0.87,bottom,0.03,top-bottom])
fig.subplots_adjust(wspace=0.15,hspace=0.1, right=0.85,left=0.15)
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbarax, extend='max')
cb.set_label('$C_I$', labelpad=0)
plt.savefig('../plots/ising_rotation.pdf')
plt.show()
