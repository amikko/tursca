import sys
postfix = sys.argv[1]
import numpy as np
import matplotlib.pyplot as plt

saving = True

folder = 'shdom_validation/'
vza = [0, 15, 30, 45, -15, -30, -45]
comp = 1.0
cloud_pos = [comp * 9,40]
#cloud_pos = [6,40] # for some reason the prp file for shdom does it like this
atm_height = 17 * comp
sza = 60
shadow_pos = [0,cloud_pos[1]-cloud_pos[0]/np.tan((90 - sza) * np.pi / 180)]
print(shadow_pos)
cloud_from_top = atm_height - cloud_pos[0]
cloud_view_x = []
shadow_view_x = []
for i in range(7):
    mult = np.tan((90 + vza[i]) * np.pi / 180.0)
    cloud_view_x.append(cloud_pos[1] - cloud_from_top/mult)
    shadow_view_x.append(shadow_pos[1] - atm_height/mult)
print(cloud_view_x)
print(shadow_view_x)

turscas = []
turscas_ss = []
shdomfile = '../../shdom/les2y21w16ar.out'
shdomfile2 = '../../shdom/les2y21w16br.out'
prpfile = '../shdom/les2y21w16.prp'
prp = np.genfromtxt(prpfile,skip_header=7)
maxext = np.max(prp[:,4])
print(maxext)
print('centre free mean path: %f' % (1/float(maxext)))
shdoms = np.genfromtxt(folder + shdomfile,comments='!')
shdoms2 = np.genfromtxt(folder + shdomfile2,comments='!')
print(shdoms.shape)
n = 64
if False:
    plt.plot(shdoms[:,0])
    plt.show()
    plt.plot(shdoms[:,1])
    plt.show()
    plt.plot(shdoms[:,2])
    plt.show()
rangs = [[0,1,2,3],[9,4,5,6]]
#for r in rangs:
allmode = True
for ir in range(2):
    r = rangs[0]
    for i in r:
        if allmode:
            alphai = 1.5 - i * 0.5
        else:
            alphai = 0
        i_ = rangs[ir][i]
        print(i_)
        try:
            print(folder + 'tursca%d_2_%s.dat' % (i_,postfix))
            turscas.append(np.genfromtxt(folder + 'tursca%d_2_%s.dat' % (i_,postfix)))
        except OSError:
            #print("err")
            continue
        turscas_ss.append(np.genfromtxt(folder + 'tursca%d_1_ss.dat' % i_))
        plt.plot(turscas[-1],'b-',alpha=1/(alphai+1))
        #plt.plot(turscas_ss[-1],'r',alpha=1/(alphai+1))
        #plt.plot([cloud_view_x[i],cloud_view_x[i]],[np.min(turscas[-1]),np.max(turscas[-1])])
        #plt.plot([shadow_view_x[i],shadow_view_x[i]],[np.min(turscas[-1]),np.max(turscas[-1])])
        #plt.plot(0.02*shdoms[(i*n):((i+1)*n),2]+0.006)
        if ir == 0:
            shdom_trans = 0.5 * shdoms[(i*n):((i+1)*n),2]
            plt.plot(shdom_trans,'k--',alpha=1/(alphai+1))
        else:
            shdomdata = shdoms2[(i*n):((i+1)*n),2]
            shdom_trans = 0.5 * np.roll(shdomdata[::-1],0)
            plt.plot(shdom_trans,'k--',alpha=1/(alphai+1))
        plt.title('vza: %d'% vza[i_])
        #if i_ == 0:
        plt.legend(('TURSCA','SHDOM'))
        plt.ylabel('transmittance')
        plt.xlabel('distance (km)')
        relerr = (turscas[-1] - shdom_trans)/shdom_trans
        #print(relerr)
        print('%2.2f' % (100 * np.mean(relerr)),'%2.2f' % (100 * np.std(relerr)))
        if not allmode:
            if saving:
                plt.savefig('shdom%d.pdf'% vza[i_])
            plt.show()
if allmode:
    plt.title('all VZAs')
    if saving:
        plt.savefig('shdom_all.pdf')
    plt.show()
