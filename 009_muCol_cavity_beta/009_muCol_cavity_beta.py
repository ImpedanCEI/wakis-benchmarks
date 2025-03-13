import os, sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0

sys.path.append('../../wakis')

from wakis import SolverFIT3D
from wakis import GridFIT3D 
from wakis import WakeSolver

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 150
Ny = 150
Nz = 150

# Embedded boundaries
stl_walls = 'cavity_walls.stl'
stl_vacuum = 'cavity_vacuum.stl'
stl_windows = 'BE_windows.stl'

# Materials
stl_solids = {'walls': stl_walls,
              'vacuum' : stl_vacuum,
              'windows' : stl_windows
              }

stl_scale = [1e-3, 1e-3, 1e-3]

stl_materials = {'walls': [5.8e+07, 1.0, 5.8e+07],
                 'vacuum' : [1.0, 1.0, 0.0],
                 'windows' : [2.5e+07, 1.0, 2.5e+07]
                 }

background = 'pec' 

# Domain bounds
surf = pv.read(stl_walls)+pv.read(stl_vacuum)+pv.read(stl_windows)
surf = surf.scale(stl_scale)
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds

n_pml = 0
use_pml = True
if use_pml: #for pml
    n_pml = 10
    dz = (zmax-zmin)/Nz
    zmin -= n_pml*dz
    zmax += n_pml*dz
    Nz += 2*n_pml

# Set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials,
                stl_scale=stl_scale,
                tol=1e-3)
#grid.inspect()

# ------------ Beam source ----------------
# Beam parameters
beta =  0.88888109        # beam relativistic beta 
sigmaz = beta*26.302e-3   # [m] -> multiplied by beta to have f_max cte
q = 1e-9            # [C]
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
tinj = 8.54*sigmaz/(np.sqrt(beta)*beta*c)  # injection time offset [s] 

# Simualtion length
wakelength = 10*sigmaz  #[m]
add_space = 0   # no. cells
results_folder = f'results_lossy+windows_BCpml_nz150_addspace_tinj/'

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta, ti=tinj,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            add_space=add_space, results_folder=results_folder,
            Ez_file=results_folder+'Ez.h5', 
            save=True, logfile=False)

# ----------- Solver  ----------
# boundary conditions
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

if use_pml:
    bc_low=['pec', 'pec', 'pml']
    bc_high=['pec', 'pec', 'pml']

solver = SolverFIT3D(grid, wake, cfln=0.3,
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True, bg=background,
                     n_pml=n_pml)

# Add windows manually
for d in ['x', 'y', 'z']:
    solver.ieps[:, :, -3-n_pml:-n_pml, d] = 1/(stl_materials['windows'][0]*epsilon_0)
    solver.ieps[:, :, n_pml:3+n_pml, d] = 1/(stl_materials['windows'][0]*epsilon_0)
    solver.sigma[:, :, -3-n_pml:-n_pml, d] = stl_materials['windows'][2]
    solver.sigma[:, :, n_pml:3+n_pml, d] = stl_materials['windows'][2]

solver.update_tensors()

# -----------  Simulation ----------
# Plot settings
img_folder = results_folder+'img/'
if not os.path.exists(img_folder): os.mkdir(img_folder)
plotkw = {'title': img_folder+'Ez', 
            #'add_patch':'pipe', 'patch_alpha':0.3,
            'dpi':200,
            'figsize':[12,12],
            'vmin':-1e4, 'vmax':1e4,
            'cmap':'bwr',
            'plane': [int(Nx/2), slice(n_pml, Ny-n_pml), slice(n_pml, Nz-n_pml)],}

# Run wakefield time-domain simulation
solver.wakesolve(wakelength=wakelength, add_space=add_space+n_pml,
                plot=True, plot_from=800, plot_every=20, plot_until=7000,
                save_J=False,
                **plotkw)    


#-------------- Plots -------------

#--- Longitudinal wake and impedance ---
plot = True
if plot:
    #results_folder = f'results_beta{beta}_add0_inj/'
    #wake.load_results(results_folder)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e3, wake.WP, c='tab:red', lw=1.5, label='wakis')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(wake.f*1e-9, np.real(wake.Z), c='tab:red', ls='--', label='Re(Z) wakis')
    ax[1].plot(wake.f*1e-9, np.imag(wake.Z), c='tab:blue', ls='-.', label='Im(Z) wakis')
    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='tab:green', ls='-', alpha=0.5, label='Abs(Z) wakis')

    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig(f'{results_folder}ReImAbs.png')

    plt.show()

plot = False
if plot:
    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    folders = ['results_BCpec_nz150', 'results_lossy_BCpec_nz150', 'results_lossy+windows_BCpec_nz150', 'results_lossy+windows_BCpml_nz150_addspace']
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    legend = ['PEC cavity', 'Copper Cavity', 'Copper cavity + Be Windows', 'Copper cavity + Be Windows + PML']
    for i, folder in enumerate(folders):
        wake.load_results(folder)
        
        ax[0].plot(wake.s*1e3, wake.WP, c=colors[i], lw=1.5, alpha=0.8, label=f'WP {legend[i]}')
        ax[0].set_xlabel('s [mm]')
        ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
        ax[0].legend(fontsize=8)

        if i == 0:
            ax[1].plot(wake.f*1e-9, np.real(wake.Z), c=colors[i], alpha=0.8, ls='--', label=f'Re(Z) {legend[i]}')
            ax[1].plot(wake.f*1e-9, np.imag(wake.Z), c=colors[i], alpha=0.8, ls=':', label=f'Im(Z) {legend[i]}')
            ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c=colors[i], alpha=0.8, ls='-',  label=f'Abs(Z) {legend[i]}')
        else:
            ax[1].plot(wake.f*1e-9, np.real(wake.Z), c=colors[i], alpha=0.8, ls='--',)
            ax[1].plot(wake.f*1e-9, np.imag(wake.Z), c=colors[i],alpha=0.8, ls=':',)
            ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c=colors[i],alpha=0.8, ls='-',  label=f'Abs(Z) {legend[i]}')
        
        ax[1].set_xlabel('f [GHz]')
        ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')
        ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f'{results_folder}compare.png')

    plt.show()

#-------------- Compare with CST -------------

#--- Longitudinal wake and impedance w/ error ---
plot = False
if plot:
    #results_folder = f'results_beta{beta}_add0_inj/'
    #wake.load_results(results_folder)

    # CST wake
    cstWP = wake.read_txt(f'cst/WP_beta{beta}.txt')
    cstZ = wake.read_txt(f'cst/Z_beta{beta}.txt')
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e3, wake.WP, c='r', lw=1.5, label='wakis')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.2, label='CST')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    

    ax[1].plot(wake.f*1e-9, np.real(wake.Z), c='b', lw=1.5, label='Re(Z) wakis')
    ax[1].plot(wake.f*1e-9, np.imag(wake.Z), c='cyan', lw=1.3, label='Im(Z) wakis')
    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', lw=1.5, alpha=0.5, label='Abs(Z) wakis')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.2, label='Re(Z) CST')
    ax[1].plot(cstZ[0], cstZ[2], c='k', ls=':', lw=1.2, label='Im(Z) CST')
    ax[1].plot(cstZ[0], np.abs(cstZ[1]+1.j*cstZ[2]), c='k', ls='-', lw=1.2, alpha=0.5, label='Abs(Z) CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')

    ax[0].legend()
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig(f'{results_folder}benchmarkReImAbs.png')

    plt.show()


#--- 1d Ez field ---
plot = False
if plot:
    # E field
    d = wake.read_Ez('results_sigma5/Ez.h5',return_value=True)
    dd = wake.read_Ez('results_pec/Ez.h5',return_value=True)

    t, z = np.array(d['t']), np.array(d['z'])    
    dt = t[1]-t[0]
    steps = list(d.keys())

    # Beam J
    current = wake.read_Ez('Jz.h5',return_value=True)

    for n, step in enumerate(steps[:1740:20]):
        fig, ax = plt.subplots(1,1, figsize=[6,4], dpi=150)
        axx = ax.twinx()  

        ax.plot(z, d[step][1,1,:], c='g', lw=1.5, label=r'Ez(0,0,z) FIT | $\sigma$ = 5 S/m')
        ax.plot(z, dd[step][1,1,:], c='grey', lw=1.5, label='Ez(0,0,z) FIT | PEC')
        ax.set_xlabel('z [m]')
        ax.set_ylabel(r'$E_z$ field amplitude [V/m]', color='g')
        ax.set_ylim(-3e3, 3e3)
        ax.set_xlim(z.min(), z.max())
        
        # CST E field
        try:    
            cstfiles = sorted(os.listdir('cst/1d/'))
            cst = wake.read_txt('cst/1d/'+cstfiles[n])
            ax.plot(cst[0]*1e-2, cst[1], c='k', lw=1.5, ls='--', label=r'Ez(0,0,z) CST | $\sigma$ = 10 S/m')
        except:
            pass

        ax.legend(loc=1)

        # charge distribution
        axx.plot(z, current[step][1,1,:], c='r', lw=1.0, label='lambda λ(z)')
        axx.set_ylabel(r'$J_z$ beam current [C/m]', color='r')
        axx.set_ylim(-8e4, 8e4)

        fig.suptitle('timestep='+str(n*20))
        fig.tight_layout()
        fig.savefig('img/Ez1d_'+str(n*20).zfill(6)+'.png')

        plt.clf()
        plt.close(fig)

#-------------- Compare result files -------------

#--- Longitudinal wake and impedance ---

# compare beta
plot = False
if plot:

    fig, ax = plt.subplots(1,2, figsize=[12,4.5], dpi=150)

    # Read data
    beta = '0.4'
    keys = ['1', '10', '20', '30', '40', '50']
    colors = plt.cm.jet(np.linspace(0.1,0.9,len(keys)))
    res = {} 
    for i, k in enumerate(keys):
        # Wakis wake
        res[k] = wake.copy()
        res[k].load_results(f'results_beta{beta}_add{k}/')

        ax[0].plot(res[k].s, res[k].WP, c=colors[i], lw=1.5, alpha=0.8, label=r'$\beta=0.4$'+f' add={k}')
        ax[1].plot(res[k].f*1e-9, np.real(res[k].Z), c=colors[i], lw=1.5, alpha=0.8, label=r'Re: $\beta=0.4$'+f' add={k}')
        ax[1].plot(res[k].f*1e-9, np.imag(res[k].Z), c=colors[i], lw=1.5, ls=':', alpha=1.0, label=r'Im: $\beta=0.4$'+f' add={k}')
    
    # CST wake
    cstWP = wake.read_txt(f'cst/WP_beta{beta}.txt')
    cstZ = wake.read_txt(f'cst/Z_beta{beta}.txt')

    ax[0].plot(cstWP[0]*1e-3, cstWP[1], c='k', ls='--', lw=1.5, label=r'CST $\beta = 0.4$')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label=r'CST Re: $\beta = 0.4$')
    ax[1].plot(cstZ[0], cstZ[2], c='k', ls=':', lw=1.5, label=r'CST Im: $\beta = 0.4$')
    
    ax[0].set_xlabel('s [m]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='k')
    #ax[0].set_yscale('symlog')
    ax[0].set_ylim(-30, 30)
    ax[0].legend()
    ax[0].margins(x=0.01, tight=True)

    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Re/Im][$\Omega$]', color='k')
    #ax[1].set_yscale('symlog')
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].margins(x=0.01, tight=True)

    fig.suptitle('Benchmark with CST Wakefield Solver')
    #fig.tight_layout()
    
    fig.savefig(f'benchmark_addspace_beta{beta}.png')
    plt.show()

