import sys

multipixel = int(sys.argv[1])
longpixel = int(sys.argv[2])
longlongpixel = int(sys.argv[3])
print(multipixel,longpixel,longlongpixel)
if multipixel or longpixel or longlongpixel:
    scatter_modes = [2]
else:
    scatter_modes = [1,2]
    scatter_modes = [2]
#scatter_modes = [1]
aers = [True, False]
#aers = [True]
aers = [False]
polars = [True, False]
polars = [False]
vzas = [-60, -40, -20, 0, 20, 40, 60]
#vzas = [-60, -40, -20, 0, 20]
#vzas = [-60]
szas = [20, 60, 0, 40]
#szas = [20, 60]
coupling_coeff = 0.0000002
#multipixel = True
#multipixel = False
#longpixel = True
#longpixel = True
#longlongpixel = False
#longlongpixel = False
coupling_coeff = 0.0

import os
import time
for scatter in scatter_modes:
    for polar in polars:
        for aer in aers:
            for i_vza in range(len(vzas)):
                #for i_sza in range(len(szas)):
                for i_sza in [1]:
                    scatterstr = 'ss' if scatter == 1 else 'ms'
                    polarstr = 'polar' if polar else 'scalar'
                    polar_in = 4 if polar else 1
                    aerstr = 'aerosol' if aer else 'clear'
                    if longlongpixel:
                        infilename = f'siro_validation/input/vza{i_vza}_longlong_sza{i_sza}_{polarstr}_{aerstr}.nc'
                        tursca_res_folder = 'tursca_result_longlongpixel'
                    elif longpixel:
                        infilename = f'siro_validation/input/vza{i_vza}_long_sza{i_sza}_{polarstr}_{aerstr}.nc'
                        tursca_res_folder = 'tursca_result_longpixel'
                    elif multipixel:
                        infilename = f'siro_validation/input/vza{i_vza}_5x5_sza{i_sza}_{polarstr}_{aerstr}.nc'
                        tursca_res_folder = 'tursca_result_multipixel'
                    else:
                        infilename = f'siro_validation/input/vza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}.nc'
                        tursca_res_folder = 'tursca_result'
                    outfilename = f'siro_validation/{tursca_res_folder}/vza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}_{scatterstr}.dat'
                    start_time = time.time()
                    os.system(f'python tursca.py {scatter} {polar_in} {coupling_coeff} {infilename} {outfilename}')
                    end_time = time.time()
                    print('Total runtime: %f s' % (end_time - start_time))
#os.system('play ~/Downloads/compute-done.wav')
