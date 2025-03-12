inputformat = './s4_validation/full_day/%s'
import time
import os
import sys
kant = int(sys.argv[1])
aerosols = [True, False]
aerosols = [True]
scatters = [1, 2]
#scatters = [1]
coupling_coeff = 0.0
for scatter in scatters:
    for aerosol in aerosols:
        for i in range(21):
            #infilename = inputformat % (f's4_2x2_aerosol_{i}.nc' if aerosol else f's4_2x2_clear_{i}.nc')
            #s4_2x2_fulldayaerosol_8.nc
            infilename = inputformat % (f's4_{kant}x{kant}_fulldayaerosol_{i}.nc' if aerosol else f's4_{kant}x{kant}_fulldayclear_{i}.nc')
            aerstr = 'aerosol' if aerosol else 'clear'
            scastr = 'ms' if scatter == 2 else 'ss'
            outfilename = inputformat % f'tursca_{kant}x{kant}_{aerstr}_{scastr}_{i}.dat'
            start_time = time.time()
            os.system(f'python tursca.py {scatter} 1 {coupling_coeff} {infilename} {outfilename}')
            end_time = time.time()
            print(end_time - start_time)
