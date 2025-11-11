import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0

import sys
sys.path.append('../../wakis')

from wakis import SolverFIT3D
from wakis import GridFIT3D 
from wakis import WakeSolver


# ---------- Domain setup ---------
# Embedded boundaries
stl_walls = 'cavity_walls.stl'
stl_vacuum = 'cavity_vacuum.stl'
stl_windows = 'BE_windows.stl'

stl_scale = [1e-3, 1e-3, 1e-3]
surf = pv.read(stl_walls)+pv.read(stl_vacuum)+pv.read(stl_windows)
surf = surf.scale(stl_scale)
# surf.plot()

stl_solids = {'walls': stl_walls,
              'vacuum' : stl_vacuum,
              'windows' : stl_windows
              }

stl_materials = {'walls': [5.8e+07, 1.0, 5.8e+07],
                 'vacuum' : [1.0, 1.0, 0.0],
                 'windows' : [2.5e+07, 1.0, 2.5e+07]
                 }

background = 'pec'

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
# Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

n_pml = 0
use_pml = True

# ------------ Beam source ----------------
# Beam parameters
beta = 0.88888109         # beam beta
sigmaz = beta*26.302e-3    #[m]
q = 1e-9            #[C]
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
tinj = 8.54*sigmaz/(np.sqrt(beta)*beta*c)  # injection time offset [s]

# Simualtion
wakelength = 10*sigmaz+tinj  #[m]
#wakelength = 0.2 #[m]
add_space = 0


# ----------- Solver & Simulation ----------
# boundary conditions
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']


scan_mesh = np.array([70])
for MeshNumber in scan_mesh:
    
    # Number of mesh cells
    Nx = MeshNumber
    Ny = MeshNumber
    Nz = MeshNumber
    if use_pml: #for pml
        n_pml = 10
        dz = (zmax-zmin)/Nz
        zmin -= n_pml*dz
        zmax += n_pml*dz
        Nz += 2*n_pml
        
        bc_low=['pec', 'pec', 'pml']
        bc_high=['pec', 'pec', 'pml']
    
    
    # set grid and geometry
    grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                    stl_solids=stl_solids, 
                    stl_materials=stl_materials,
                    stl_scale=stl_scale,
                    tol=1e-3)

    # grid.inspect()
    
    results_folder = f'All_tinj/results_WL{int(wakelength*1000)}mm_Mesh{MeshNumber}_npml{n_pml}/'
    # results_folder = f'OnlyVacuum/results_WL{int(wakelength*1000)}mm_Mesh{MeshNumber}/'
    wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta, add_space=add_space,
                xsource=xs, ysource=ys, xtest=xt, ytest=yt, ti=tinj,
                save=True, logfile=False, results_folder=results_folder,
                Ez_file=results_folder+'Ez.h5')
    
    # set Solver object
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

    # Plot settings
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('name', plt.cm.jet(np.linspace(0.1, 0.9))) # CST's colormap

    img_folder = results_folder+'img/'
    if not os.path.exists(img_folder): os.mkdir(img_folder)
    plotkw = {'title': img_folder+'Ez', 
                #'add_patch':'pipe', 'patch_alpha':0.3,
                'figsize':[12,12],
                'vmin':-1e4, 'vmax':1e4,
                'cmap':cmap,
                'plane': [int(Nx/2), slice(n_pml, Ny-n_pml), slice(add_space+n_pml, Nz-add_space-n_pml)]}


    solver.wakesolve(wakelength=wakelength, # Simulation wakelength in [m]
                    wake=wake,   # wake object of WakeSolver class
                    add_space=add_space,
                    # save_J=True,   # [OPT] Save source current Jz in HDF5 format
                    plot=True, plot_from=800, plot_every=20, plot_until=7000, # [OPT] Enable 2Dplot and plot frequency
                    **plotkw, # [OPT] plot arguments. ->See built-in plotting section for details 
                    )