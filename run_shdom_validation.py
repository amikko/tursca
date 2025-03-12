import sys
postfix = sys.argv[1]
import os
scattering = [1,2]
coupling_coeff = 0.00003
scattering = [2]
for sca in scattering:
    for i_vza in range(7):
        #for i_vza in [5]:
        infilename = f'shdom_validation/shdom_view{i_vza}.nc'
        outfilename = f'shdom_validation/tursca{i_vza}_{sca}_{postfix}.dat'
        os.system(f'python tursca.py {sca} 1 {coupling_coeff} {infilename} {outfilename}')
