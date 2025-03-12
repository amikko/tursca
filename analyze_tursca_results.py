scatter_modes = [1,2]
aers = [True, False]
polars = [True, False]
vzas = [-60, -40, -20, 0, 20, 40, 60]
szas = [20, 60, 0, 40]
coupling_coeff = 0.00002
import os
import numpy as np
import matplotlib.pyplot as plt

tursca_res_folder = 'tursca_result'
#tursca_res_folder = 'tursca_result'
def load_case(scatter,polar,aer,i_vza,i_sza,multipixel=False):
    scatterstr = 'ss' if scatter == 1 else 'ms'
    polarstr = 'polar' if polar else 'scalar'
    polar_in = 4 if polar else 1
    aerstr = 'aerosol' if aer else 'clear'
    infilename = f'siro_validation/input/vza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}.nc'
    if multipixel and scatter != 1:
        outfilename = f'siro_validation/tursca_result_multipixel/vza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}_{scatterstr}.dat'
        tursca_trans = np.genfromtxt(outfilename)
        tursca_trans_mp = tursca_trans[12]
        outfilename = f'siro_validation/tursca_result_longpixel/vza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}_{scatterstr}.dat'
        tursca_trans = np.genfromtxt(outfilename)
        tursca_trans_lp = tursca_trans[tursca_trans.size // 2]
        outfilename = f'siro_validation/tursca_result_longlongpixel/vza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}_{scatterstr}.dat'
        tursca_trans = np.genfromtxt(outfilename)
        tursca_trans_llp = tursca_trans[tursca_trans.size // 2]
        try:
            outfilename = f'siro_validation/tursca_result_extralongpixel/vza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}_{scatterstr}.dat'
            tursca_trans = np.genfromtxt(outfilename)
            tursca_trans_elp = tursca_trans[tursca_trans.size // 2]
        except:
            tursca_trans_elp = np.nan
    else:
        tursca_trans_mp = np.nan
        tursca_trans_lp = np.nan
        tursca_trans_llp = np.nan
        tursca_trans_elp = np.nan
    outfilename = f'siro_validation/{tursca_res_folder}/vza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}_{scatterstr}.dat'
    tursca_trans = np.genfromtxt(outfilename)
    sirofilename = f'siro_validation/siro_rad/siro_radiance_vza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}.pick'
    try:
        sirofilename = f'siro_validation/siro_rad/radiancevza{i_vza}_sza{i_sza}_{polarstr}_{aerstr}.siro'
        siroin = np.genfromtxt(sirofilename)
    except:
        # whoops, I did misconfigured some runs, but this is a way to handle them.
        sirofilename = f'siro_validation/siro_rad/radiancevza{i_vza}_sza{i_sza}_polar_{aerstr}.siro'
        siroin = np.genfromtxt(sirofilename)
    old_mode = False
    if scatter == 1:
        siro_trans = np.sum(siroin[2:4])
    else:
        siro_trans = np.sum(siroin[2:])
    if old_mode:
        import pickle
        with open(sirofilename,'rb') as f:
            siro_trans = pickle.load(f)
        averaging = False
        if averaging:
            siro_trans = np.sum(siro_trans,axis=0) / siro_trans.shape[0]
        else:
            if polariz_mode > 1:
                siro_trans = siro_trans[0,:,:,:]
            else:
                siro_trans = siro_trans[0,:,:]
        if scatter == 1:
            if polar:
                siro_trans = np.sum(siro_trans[:,:2,:],axis=1)
            else:
                siro_trans = np.sum(siro_trans[:,:2],axis=1)
        else:
            if polar:
                siro_trans = np.sum(siro_trans[:,:,:],axis=1)
            else:
                siro_trans = np.sum(siro_trans[:,:],axis=1)
    return tursca_trans, siro_trans, tursca_trans_mp, tursca_trans_lp, tursca_trans_llp, tursca_trans_elp

def get_vzas(scatter,polar,aer,i_sza,multipixel=False):
    polardim = 4 if polar else 1
    tursca = np.zeros((len(vzas),polardim))
    turscamp = np.zeros((len(vzas),polardim))
    turscalp = np.zeros((len(vzas),polardim))
    turscallp = np.zeros((len(vzas),polardim))
    turscaelp = np.zeros((len(vzas),polardim))
    siro = np.zeros((len(vzas),polardim))
    for i_vza in range(len(vzas)):
        t,s,tmp,tlp,tllp,telp = load_case(scatter,polar,aer,i_vza,i_sza,multipixel)
        tursca[i_vza,:] = t
        siro[i_vza,:] = s
        turscamp[i_vza,:] = tmp
        turscalp[i_vza,:] = tlp
        turscallp[i_vza,:] = tllp
        turscaelp[i_vza,:] = telp
    return tursca,siro,turscamp,turscalp,turscallp,turscaelp

def plot_vzas(polar,aer,i_sza,saving=False,multipixel=False,plotting=True):
    tss,sss,tssmp,tsslp,_,_ = get_vzas(1,polar,aer,i_sza,multipixel)
    tms,sms,tmsmp,tmslp,tmsllp,tmselp = get_vzas(2,polar,aer,i_sza,multipixel)
    if not polar:
        plt.figure()
        plt.plot(vzas,tss,'b:',label='TURSCA (SS)')
        plt.plot(vzas,sss,'r:',label='Siro (SS)')
        plt.plot(vzas,tms,'b-',label='TURSCA 1x1 (MS)')
        plt.plot(vzas,tmslp,'b--',label='TURSCA 3x3 center (MS)')
        plt.plot(vzas,tmsmp,'b-.',label='TURSCA 5x5 center (MS)')
        plt.plot(vzas,tmsllp,'b-',label='TURSCA 7x7 center (MS)',alpha=0.5)
        #plt.plot(vzas,tmselp,'k--',label='TURSCA 21x21 center (MS)')
        plt.plot(vzas,sms,'r-',label='Siro (MS)')
        plt.ylabel('transmittance')
        plt.xlabel('viewing zenith angle (degrees)')
        plt.legend()
    else:
        fig,ax = plt.subplots(2,2,sharex=True)
        ax[0,0].plot(vzas,tss[:,0],'b:')
        ax[0,0].plot(vzas,sss[:,0],'r:')
        ax[0,0].set_title('Stokes I')
        ax[0,1].plot(vzas,tss[:,1],'b:')
        ax[0,1].plot(vzas,sss[:,1],'r:')
        ax[0,1].set_title('Stokes Q')
        ax[1,0].plot(vzas,tss[:,2],'b:')
        ax[1,0].plot(vzas,sss[:,2],'r:')
        ax[1,0].set_title('Stokes U')
        #ax[1,0].xlabel('VZA (degrees)')
        #ax[1,0].ylabel('transmittance')
        ax[1,1].plot(vzas,tss[:,3],'b:')
        ax[1,1].plot(vzas,sss[:,3],'r:')
        ax[1,1].set_title('Stokes V')
        #ax[1,1].xlabel('VZA (degrees)')

        ax[0,0].plot(vzas,tms[:,0],'b-')
        ax[0,0].plot(vzas,sms[:,0],'r-')
        ax[0,1].plot(vzas,tms[:,1],'b-')
        ax[0,1].plot(vzas,sms[:,1],'r-')
        ax[1,0].plot(vzas,tms[:,2],'b-')
        ax[1,0].plot(vzas,sms[:,2],'r-')
        ax[1,1].plot(vzas,tms[:,3],'b-')
        ax[1,1].plot(vzas,sms[:,3],'r-')
        fig.supylabel('transmittance')
        fig.supxlabel('viewing zenith angle (degrees)')
    aerstr = 'aerosol 0-2 km' if aer else 'clear sky'
    titlestr = f'{aerstr}, SZA: {szas[i_sza]}'
    if not polar:
        plt.title(titlestr)
    else:
        fig.suptitle(titlestr)

    if not saving:
        plt.show()
    else:
        polarstr = 'polar' if polar else 'scalar'
        aerstr = 'aerosol' if aer else 'clear'
        plt.savefig(f'{aerstr}_{polarstr}_sza{i_sza}.pdf')
    return ((sss,tss),(sms,tms,tmsmp,tmslp,tmsllp))
def savefigs():
    #for polar in [True,False]:
    for polar in [False]:
        for aer in [True,False]:
            for i_sza in [0,1]:
                plot_vzas(polar,aer,i_sza,True,multipixel=True)

def output_percentages(aer,i_sza):
    vals = plot_vzas(False,aer,i_sza,saving=False,multipixel=True,plotting=False)
    siro_ss = vals[0][0]
    siro_ms = vals[1][0]

    strs = ['1x1','3x3','5x5','7x7']

    tursca_ss = vals[0][1]
    res = ((tursca_ss - siro_ss)/siro_ss * 100)
    print(res,f'{np.mean(res):.2f} \\pm {np.std(res):.2f}')

    for i in range(len(strs)):
        print(strs[i])
        tursca_ms = vals[1][1+i]
        res = ((tursca_ms - siro_ms)/siro_ms * 100)
        print(res,f'{np.mean(res):.2f} \\pm {np.std(res):.2f}')
