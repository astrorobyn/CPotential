from CPotential.potential_utils import calculateCPotential
import numpy as np
import gizmo_analysis as ga
import h5py

nthreads=4 #28 for reg, 40 for skylake

# read snapshot
part = ga.io.Read.read_snapshots(simulation_directory='/Users/robyn/Documents/Research/Gizmo/data/m12i_res7100_md/', 
	properties=['position','mass'], assign_host_coordinates=False,)#particle_subsample_factor=1000)

all_pos = np.array([])
all_mass = np.array([])

for sp in ['dark','dark2','gas','star']:
    if sp=='star':
        star_pos = part[sp]['position']
    all_pos = np.append(all_pos,part[sp]['position'])
    all_mass = np.append(all_mass,part[sp]['mass'])

all_pos = np.reshape(all_pos,(-1,3))
print('position list shape:',all_pos.shape)
print('mass list shape:',all_mass.shape)
print('star particle positions:',star_pos.shape)

print('calculating potential at star particle positions')

potl = calculateCPotential(all_pos,all_mass,star_pos,print_flag = True, nthreads=nthreads)

potl = (potl-potl.max()) #set offset such that edge of box is 0


fname ='rom_rem_pot.hdf5'
print('saving to', fname)
 
with h5py.File(fname,'w') as pfile:
    pfile.create_dataset('potential',data = potl)
    pfile.create_dataset('position',data = star_pos)


