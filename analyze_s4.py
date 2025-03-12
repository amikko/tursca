scatter_modes = [1,2]
aers = [True, False]
import os
import sys
aerosol = int(sys.argv[1])
kant = int(sys.argv[2])
import numpy as np
import matplotlib.pyplot as plt
import pickle
siroformat = 'siro_radiance_2x2_scalar_%s_fullday%d.pick'
inputformat = './s4_validation/full_day/%s'
siroformat = inputformat % siroformat
#aerosol = True
#scatter = 2

n_times = 21


elevation_max = 53.14 # in Helsinki at 21.06.2024

day_length = (18 + (56/60)) / 24 # day length on 21.06.2024: 18h56m
night_azi = 360 * (1 - day_length)
morning_azi = night_azi / 2
evening_azi = 360 - night_azi / 2
solar_azis = np.linspace(morning_azi,evening_azi,n_times)
solar_zens = np.zeros((n_times,))

tursca_rad = np.zeros((21,kant*kant,2))
siro_rad = np.zeros((21,4,2))
aerstr = 'aerosol' if aerosol else 'clear'
for scatter in [1,2]:
    scastr = 'ms' if scatter == 2 else 'ss'
    for i in range(21):
        outfilename = inputformat % f'tursca_{kant}x{kant}_{aerstr}_{scastr}_{i}.dat'
        tr = np.genfromtxt(outfilename)
        tursca_rad[i,:,scatter-1] = tr
        sirofname = siroformat % (aerstr, i)
        with open(sirofname,'rb') as pick:
            sr = pickle.load(pick)
        sr = sr[0,:,:]
        if scatter == 1:
            siro_rad[i,:,0] = np.sum(sr[:,:2],axis=1)
        else:
            siro_rad[i,:,1] = np.sum(sr[:,:],axis=1)

print(siro_rad)
print(tursca_rad)

for j in range(4):
    plt.plot(solar_azis,siro_rad[:,j,1],'r-',alpha=1.0/(j+1))
    plt.plot(solar_azis,siro_rad[:,j,0],'r:',alpha=1.0/(j+1))
for j in range(kant*kant):
    plt.plot(solar_azis,tursca_rad[:,j,1],'b-',alpha=1.0/(j+1))
    plt.plot(solar_azis,tursca_rad[:,j,0],'b:',alpha=1.0/(j+1))
plt.title(f'%s, {kant}x{kant}, 5700 couplings' % aerstr)
plt.xlabel('SAA')
plt.ylabel('transmittance')
plt.show()
