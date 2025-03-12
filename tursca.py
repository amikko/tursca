# seuraavat tehtävät

# DONE!kink detection: tarkista että toimii! Se vähentää vähän nuita nodejen määrää.
# DONE!accurately set start and end nodes of each los
# POSTPONED!Selvitä miksei trace_path parallelisoi enää!!
#    se parallelisoi laskun kyllä! Mutta se kääntö on se missä kestää hirmuisen pitkään. Luulen että käännössä niitä funktioita on niin näppärästi hierarkisesti kutsuttu, että force inlinessä kestää hirmuisen pitkään. Tutki jos saisit inlineä vähennettyä ja lisäksi jos saisi jotain loop unrolling hässäkkää niin katoppa niitä.
# DONE!surface reflection
#    DONE!surface reflection sinne couplingiin!!
# scattering muller matrices store for use in each wl
# polarisaatio! + rotations!
# better solver for flux system, hand-crafted and parallelized?
# 99% scatter coupling weight!
# DONE!surf. reflection, single scattering and multiple scattering

# turbulent plume: graph of gaussian basis functions?

# raman scattering might be cool to have, 4x4xN_wl matrix for fluxes
# do not worry too much about errors
# wavelength is

"""
DONE!Tarkista path-transmittance laskut!!

DONE!Selvitä se debuggerin kaatama ongelma, jossa accessoidaan -1 indeksiä!

SEMIDONE!Selvitä mistä tulee tuo 50m??
    En tiedä, mutta nyt siellä on post-correction

DONE!Nopeempi systeeminsolveri!

Polarisaatio!
DONE!System solveriin tarvitaan templatoidut multiply() divide() -funktiot
 - Rotaatiot sopivasti

Antti note 9.8.2024: Since apparently it is very possible to create small
presized vecs and mats on the fly, then some matrix and vector operations could
use a bit refactoring. There is not, for example, need for the temp_mat_X and temp_vec_X variables in scattering.py

(Rakenna boundary-systeemi paremmaksi. Päätä automaatisesti millaiset normaalit tulee ja mihin suuntaan ne sojottaa.)

(Säilö kernelit!)


Validaatiotestit
Pluumitestit
    - tästä se coupling param tutkimus
    - tarkista sopiva aerosoli
    - black carbon?
Siro S4-simulaatiot, valitse sopiva S4 kaista
    - 2x2, 5x5, 10x10
    - yksi kaista 750nm
    - joku aerosoli myös!
    - scalar
    - polar
    - sulphate
#SHDOM polariz case
#    - 0.65µm
#    - tee sopivat aerosolit
#    - säädä sopiva ilmakehä
SHDOM scalar case
    - 1.65µm
    - tee sopivat aerosolit
    - säädä sopiva ilmakehä
    - vesipisara

Se 99% systeemikuvio

- Lisää varoitukset kun couplingit menee ylärajalle
- myös max_path_len varoitus
- BUGI: couplingeissa on 0,0,0,0 -couplauksia. Kenties ne on semmosia jotka
menee varjoon? tai semmoisia jotka ei linkitetä kun on liian kaukana?
- parallelisoi scatter coupling, nyt seriaali.
- optimoi flux_A_normalize

- jos on useempi havaintogeometria samassa skenessä, niin noodit voivat yhdistyä!!
- jos käytetään käyrän fittausta lopussa, niin pakota aina tietty määrä couplingeja
kullekin lossille!

- aallonpituuksien iterointi
- retrieval-mode, jossa patheja ei travi treissata uusiksi?

- konvergenssianalyysi jo A_coup-fluksien pohjalta?

- epäsymmetriset vaihefunktion integroitava logaritmisena??

- aallonpituuksien nopeutus: erottele vaihefunktioiden evaluaatio ja kaikki
aallonpituusriippuva!
"""

import numpy as np

# ongelma:
#   coupling matrixin luominen on tosi hidasta.
#   optimointeja:
#     - accessöi arrayhyn funktiolla siten että tarvii säilöä ja laskea vaan puolet
#     - yläraja nodeen kytkeytyviin patheihin? nyt kun vahvasti sirottava niin
#     silloin se linkittää ihan kaikkeen.
#     !!! Ratkaisu! mean free path on rajoittimena! !!!

# mean free path: (sigma * n)^-1
# test_case_1: 100x100
# test_case_2: 50x50
# test_case_3: 20x20
# test_case_4: 10x10

# animaatiot: tämä vaatii sen, että katsotaan polut kuntoon alusta ja lopusta!
# myös pintaheijastus!
# TODO: polkua treissatessa pitää tarkastella se split step -kuvio, eli jos
# aiempi steppi oli vain yhden min_stepin mittainen, niin voisiko virhettä minimoida
# se jos sen häivyttää ja nykyinen ja yksi steppi aiemmin voisi mergettää yhteen?

# source function at each node (and trace)
# scatter coupling ((xsec_sca1 + xsec_sca2) / dist^2) taikka mean free path?
# riittävä coupling coefficient thresholdi haetaan siten, että kutakin lossia
# kohden saadaan 2-stream, ja sitten lossien keskeiset sironnat on vaan bonusta?
# scattering flux (template)

# työläät osiot:
# source function at each node
# scatter coupling

# miten ratkaistaan yhtälö kun ei ole koko matriisia olemassa?

#10x10: 6650763 flux coup elements
#20x10: 589453027 flux coup elements

import yaml

import sys

if len(sys.argv) == 6:
    try: #original mode
        scatter_mode = int(sys.argv[1])
        polariz_mode = int(sys.argv[2])
        coupling_coeff = float(sys.argv[3])
        input_file = sys.argv[4]
        output_file = sys.argv[5]
        max_path_len = 150
        max_scatter_couplings = 22200
        n_fibo_dirs = 100
        source_basis_integral_steps = 20

        #voe ny hevov vidalis
        # säätyy tuo sironta kun minimum_steppiä
        if 'shdom' in input_file:
            minimum_step = 0.5
        else:
            minimum_step = 0.2
        if 'soot' in input_file or 'initial' in input_file:
            particle_muller_filename = 'soot_SCO2.dat'
        elif 'fullday' in input_file:
            particle_muller_filename = 'NIR_weak_abs.dat'
        elif 'shdom' in input_file:
            particle_muller_filename = 'shdom_validation/shdom_particle.dat'
        else:
            particle_muller_filename = 'siro_validation/fine_weak_mode.dat'
    except:
        pass
else:
    try:
        config_file = sys.argv[1]
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        with open(config_file) as fhandle:
            conf = yaml.safe_load(fhandle)
        scatter_mode = conf['scatter_mode']
        polariz_mode = conf['polariz_mode']
        coupling_coeff = conf['coupling_coeff']
        max_path_len = conf['max_path_len']
        max_scatter_couplings = conf['max_scatter_couplings']
        particle_muller_filename = conf['particle_muller_filename']
        minimum_step = conf['minimum_step']
        n_fibo_dirs = conf['n_fibo_dirs']
        source_basis_integral_steps = conf['source_basis_integral_steps']
    except:
        print("TURSCA")
        print("Usage:")
        print(f"python {sys.argv[0]} [config file] [input file] [output file]")
        print(f"python {sys.argv[0]} [scattering mode] [polarization mode] [scattering coupling coefficient] [input file] [output file]")
        exit()

import runpy
import data_io
solar_sel = 0
old_basis_integral_mode = False
uniform_mode = True # for weighting
no_weigthing_at_all = True # this is to disable all weighting
test_file = 'test_case_1.nc'
test_file = 's4_10x10_vza0.nc'
test_file = 's4_5x5_vza0.nc'
test_file = 's4_2x2_vza0.nc'
test_file = 'initial_2x2_vza0.nc'
#test_file = 's4_validation/s4_2x2_aerosol_high_morning.nc'
test_file = 's4_validation/full_day/s4_2x2_fulldayaerosol_%d.nc' % solar_sel
test_file = 'shdom_validation/shdom_view%d.nc' % solar_sel
#test_file = 's4_20x20_vza0.nc'
test_file = input_file
print(f"Using {test_file}")
siroradfile = 's4_validation/siro_radiance_2x2_scalar_aerosol_high_morning.pick'
siroradfile = 's4_validation/full_day/siro_radiance_2x2_scalar_aerosol_fullday%d.pick' % solar_sel
#
rayscaradfile = 's4_validation/raysca_radiance_2x2_polar_aerosol.pick'
#test_file = 's4_test.nc'
#test_file = 'initial_5x5_vza0.nc'
medium, instrument, source, boundary = data_io.read_input_nc(test_file)
n_pos = medium['position'].shape[0]

kant = 1

#print(medium['interpolation_parameter'])
#print(instrument['view_vector'][4999,:])
n_wl = source['input_wavelength'].shape[0]
n_los = instrument['position'].shape[0]

#n_los = 3
n_sca = medium['scatterer'].shape[1]

print(particle_muller_filename)
#particle_muller_filename = 'NIR-TWA.dat'
particle_muller_length = np.genfromtxt(particle_muller_filename).shape[0]
n_source = source['incident_direction'].shape[0]
#scatter_mode = 1 #0 = just surface refl., 1 = single scattering, 2 = multiple scattering
#polariz_mode = 1 #1 = scalar, 4 = full Stokes vector
#print(n_wl)
#print("scattering disabled!!")
#medium['scattering_cross_section'] = medium['scattering_cross_section'] * 0.0
#halt
n_refl_param = boundary['reflection_kernel_parameter'].shape[0]
n_boundary = boundary['reflection_kernel_parameter'].shape[1]


#coupling_coeff = 0.05 # 20x20
#max_scatter_len = 0.5
np.savetxt('med_pos.dat',medium['position'],'%f')
# oh? OH? is this a debug setup I see?! This surely would not cause trouble in
# model validation part, it would not, correct? NO! Away with you, return to the
# fires of Hell whence you came! Get falsed, idiot.
if False:
    temp_arr = np.zeros(medium['position'].shape)
    temp_arr[:51,0] = np.linspace(6371,6421,51) # this is the first basis
    temp_arr[51:,:] = medium['position'][51:,:]
    medium['position'] = temp_arr
    instrument['position'] = instrument['position'] * 1.01
    print(instrument['position'])
new_globals = runpy.run_module('trace_path',init_globals=globals())
newer_globals = runpy.run_module('polarization',init_globals=new_globals)
newerer_globals = runpy.run_module('phase_integrals',init_globals=newer_globals)
even_newer_globals = runpy.run_module('scattering',init_globals=newerer_globals)
for k in even_newer_globals:
    exec(f"{k} = even_newer_globals['{k}']")


if False:
    # set max path len = 1
    # set max scatter coupling = 1
    # set n_los = 3
    test_LU_decomp_solver()
    for i in range(3):
        print(flux_x[i])
    for i in range(3):
        print(flux_b[i])
    for i in range(3):
        l = []
        for j in range(3):
            l.append(flux_S_LU[i,j])
        print(l)
    halt
#print("stopped scattering!!")
#medium['scattering_cross_section'] = 0 * medium['scattering_cross_section']

populate_basis_definition(basis_pos,basis_lin,basis_param,basis_type,medium)
print("basis populated!")
populate_instr_definition(instr_pos,instr_view,instrument)
print("instr populated!")
populate_source(source)
print("source populated!")
populate_extinction_scattering(extinction, scattering, medium)
print("populated ext. sca.!")
populate_boundary_param(boundary)
print("populated boundary params!")
#print(basis_type)
set_up_muller_scattering()
normalize_table()

#for i in range(n_pos):
#    print(avg_scattering)
#halt
if False:
    print(basis_lin)
    test_basis()
    print(basis_weights)
    print(basis_pos)
if False:
    test_step()
    #globdict = argimport('trace_path',globdict)

    #for i in range(basis_weights.shape[0]):
    #    print(basis_weights[i] - basis_mid[i])
    print(basis_weights)
    print(basis_mid)
    print(basis_end)

if False:
    test_basis_contrib()
    print(basis_end)
if False:
    import time
    start_time = time.time()
    test_trace(final_trace=True,minimum_step=minimum_step)
    end_time = time.time()
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(n_los):
        #print(path_len[i])
        l = []
        for j in range(path_len[i]):
            #l.append(np.linalg.norm(path_steps[j,i]))
            pass
        #plt.plot(l[:-1],np.diff(l),'x')
        #print(l)
    #plt.show()
    #plt.hist(path_len,20)
    #plt.show()
    for i_los in [0,50]:
        print(i_los,i_los,i_los,i_los)
        for i in range(path_len[i_los]):
            pass
            #print(path_steps[i,i_los])
            #print(np.linalg.norm(path_steps[i,i_los]))

        #for i in range(path_len[i_los]):
        for i in range(0):
            #for i in range(3):
            l = []
            for j in range(n_pos):
                l.append((path_basis[i,j,i_los]))
            print(i,l)
    if False:
        arr = np.zeros((n_los,))
        for i_los in range(n_los):
            for j in range(path_len[i_los]):
                for k in range(n_pos):
                    if k > arr[i_los] and path_basis[j,k,i_los] > 0.0:
                        arr[i_los] = k
            #l.append(path_len[i_los])
        np.savetxt('arr3.txt',arr)
    print('trace_path elapsed:',end_time - start_time)
    start_time = time.time()
    test_source_basis()
    end_time = time.time()
    print('source_basis elapsed:',end_time - start_time)
    start_time = time.time()
    test_source_trans()
    end_time = time.time()
    print('source_trans elapsed:',end_time - start_time)
    start_time = time.time()
    test_path_trans()
    end_time = time.time()
    print('path_trans elapsed:',end_time - start_time)
    l = []
    i_los = 0
    i_source = 0
    #for i_node in range(path_len[i_los]):
        #for i_pos in range(n_pos):
        #    l.append(source_basis[path_len[i_los]-2, i_pos, i_los, i_source])
        #print(i_node,l)
        #l = []
    for i in range(n_los):
        l.append(transmittances[i,200])
    #print(l)
    np.savetxt('img.dat',np.array(l))
        #print(extinction[i,0])
    #for i in range(10):
    #print(basis_pos)
    #print(basis_param)
    i_los = 5050
    for i in range(10):
        pass
        #print(node_dist[i_los,i,i_los,i+1])

if False:
    populate_extinction_scattering(extinction, scattering, medium)
    for i in range(n_pos):
        l = []
        for j in range(10):
            l.append(extinction[i,j])
        #print(l)

if True:
    import time
    start_time = time.time()
    test_trace(minimum_step=minimum_step)

    end_time = time.time()
    #set_up_scattering()
    print('trace_path elapsed:',end_time - start_time)
    nodeamt = 0
    for i_los in range(n_los):
        for i_node in range(path_len[i_los]):
            nodeamt += 1
    if False:
        s = ""
        steps = ""
        steparr = np.zeros((n_los,))
        for i_los in range(n_los):
            nodeamt += path_len[i_los]
            s += (str(path_len[i_los]) + ' ')
            steps += (str(np.linalg.norm(path_steps[path_len[i_los]-1,i_los])) + ' ')
            steparr[i_los] = np.linalg.norm(path_steps[path_len[i_los]-1,i_los])
            if (i_los+1) % 5 == 0:
                s += "\n"
                steps += "\n"
        print(s)
        print(steps)
        import matplotlib.pyplot as plt
        plt.imshow(steparr.reshape((5,5)))
        plt.show()
    npsteps = path_steps.to_numpy()
    np.savetxt('steps_%d.dat' % path_len[0],np.sqrt(np.diag(npsteps[:,0,:] @ npsteps[:,0,:].T)))
    print("Node amt: ",nodeamt)
    start_time = time.time()
    test_source_basis(minimum_step)
    end_time = time.time()
    print('source_basis elapsed:',end_time - start_time)
    start_time = time.time()
    linkscaling()
    end_time = time.time()
    print('linkscaling elapsed:',end_time - start_time)
    start_time = time.time()
    test_coupling_matrix(coupling_coeff)
    end_time = time.time()
    print('coupling_matrix elapsed:',end_time - start_time)
    lt = link_table.to_numpy()
    print('link table:')
    print(lt)
    np.savetxt('linktable.dat',lt)
    sums = np.sum(lt,axis=1)
    try:
        print(sums.reshape(int(np.sqrt(sums.size)),int(np.sqrt(sums.size))))
    except:
        pass
    start_time = time.time()
    set_up_node_idx_inv()
    end_time = time.time()
    print('set up node idx inv: ', end_time - start_time)
    start_time = time.time()
    calc_scatter_basis()
    calc_scatter_basis_obs()
    end_time = time.time()
    print('scatter_basis elapsed:',end_time - start_time)
    #nodes = node_coupling.to_numpy().ravel()
    #import matplotlib.pyplot as plt
    #links = nodes[nodes>0]

    #print(i_los,i_node,j_los,j_node,node_basis_idx[i_los,i_node,j_los,j_node])
    print('scatter_coupling_amt:',scatter_coupling_amt)
    print('flux_A.shape:',flux_A.shape)
    print('flux_x.shape:',flux_x.shape)
    print('flux_bx.shape:',flux_bx.shape)
    print('flux_bc.shape:',flux_bc.shape)
    i_wl = 0
    start_time = time.time()
    set_up_coup_neighbours()
    #set_up_flux_matrix_coup(i_wl)
    end_time = time.time()
    print('set up coup neighbours: ', end_time - start_time)

    start_time = time.time()
    populate_phase_arrays(i_wl)
    end_time = time.time()
    print('populated phase arrays: ', end_time - start_time)

    start_time = time.time()
    set_up_dir_areas()
    end_time = time.time()
    print('set up dir areas: ', end_time - start_time)

    np.savetxt('dir_area.dat',dir_areas.to_numpy()[:,:,0])
    np.savetxt('area_amt.dat',area_amt.to_numpy())

    start_time = time.time()
    set_up_flux_matrix_obs(i_wl,i_source=0)
    end_time = time.time()
    print('set up flux matrix obs: ', end_time - start_time)
    start_time = time.time()
    flux_scattering_efficiency(i_wl)
    start_time2 = time.time()
    print('flux_scattering_efficiency: ',start_time2 - start_time)
    flux_A_SQ()
    start_time3 = time.time()
    print('flux_A_SQ: ',start_time3 - start_time2)
    #flux_A_normalize()
    #time.sleep(1)
    #print("midpoitn")
    flux_A_normalize()
    #print("no Aflux normalization!!")
    start_time4 = time.time()
    print('flux_A_normalize: ',start_time4 - start_time3)
    flux_A_trans(i_wl)
    end_time = time.time()
    print('flux_A_trans: ',end_time - start_time4)

    if False:
        start_time = time.time()
        set_up_flux_A_coup(i_wl)
        end_time = time.time()
        print('set up flux A coup: ', end_time - start_time)
        start_time = time.time()
        #test_weighting(0,48)
        #populate_phase_function_weight_table(i_wl)
        print('fluxAshape:',flux_A.shape)
        print('coup_idx_neigh_shape:',coup_idx_neighbour.shape)
        normalize_flux_A_obs()
        #test_lossy_weighting_obs(i_source=0)
        end_time = time.time()
        print('lossy weighting obs: ', end_time - start_time)
        start_time = time.time()
        normalize_flux_A_coup()
        #test_lossy_weighting_coup()
        end_time = time.time()
        print('lossy weighting coup: ', end_time - start_time)
        if False:
            if not no_weigthing_at_all:
                if scatter_mode > 1:
                    if not uniform_mode:
                        test_weighting()
                    else:
                        normalize_flux_A_coup()
                        normalize_flux_A_obs()


        #nac = norm_amt_Acoup.to_numpy()
        #nao = norm_amt_Aobs.to_numpy()
        #print(nac[:100])
        #print(nao[:100])
        #print('lossy weighting: ', end_time - start_time)
    start_time = time.time()
    #set_up_flux_b_coup(i_wl,i_source=0)
    i_source = 0
    set_up_flux_bc(i_wl,i_source)
    flux_b_normalize(i_source)
    #print("no bflux normalization!!")
    flux_b_trans(i_wl,i_source)
    end_time = time.time()
    print('set up flux b coup: ', end_time - start_time)
    out = flux_A.to_numpy()
    import netCDF4
    if False:
        with netCDF4.Dataset('flux_A_before.nc','w') as ds:
            ds.createDimension('dummy1',out.shape[0])
            ds.createDimension('dummy2',out.shape[1])
            if polariz_mode > 1:
                ds.createDimension('dummy3',out.shape[2])
                ds.createDimension('dummy4',out.shape[3])
                ds.createVariable('A','f4',('dummy1','dummy2','dummy3','dummy4'))
            else:
                ds.createVariable('A','f4',('dummy1','dummy2'))
            #ds.createVariable('S_y','f4',('dummy1','dummy4'))
            ds['A'][:] = out
    #add_diagonals()
    print("didnt crash diags")

    out = flux_A.to_numpy()
    #out2= flux_S_y.to_numpy()
    #print(out2)
    saving= False
    print(out.shape)
    #print(out2.shape)
    if False:
        import netCDF4
        with netCDF4.Dataset('flux_A.nc','w') as ds:
            ds.createDimension('dummy1',out.shape[0])
            ds.createDimension('dummy2',out.shape[1])
            if polariz_mode > 1:
                ds.createDimension('dummy3',out.shape[2])
                ds.createDimension('dummy4',out.shape[3])
                ds.createVariable('A','f4',('dummy1','dummy2','dummy3','dummy4'))
            else:
                ds.createVariable('A','f4',('dummy1','dummy2'))
            #ds.createVariable('S_y','f4',('dummy1','dummy4'))
            ds['A'][:] = out
            #ds['S_y'][:] = out2
        #for i in range(4):
            #for j in range(4):
                #np.savetxt(f'flux_A{i}{j}.dat',out[:,:,i,j])
    import netCDF4
    if False:
        out = flux_O.to_numpy()
        with netCDF4.Dataset('flux_O.nc','w') as ds:
            ds.createDimension('dummy1',out.shape[0])
            ds.createDimension('dummy2',out.shape[1])
            ds.createDimension('dummy3',out.shape[2])
            ds.createVariable('O','f4',('dummy1','dummy2','dummy3'))
            ds['O'][:] = out
    out = flux_bx.to_numpy()
    #print(out)
    print(out.shape)
    if polariz_mode > 1:
        out_ = out.reshape((out.shape[0]*out.shape[1],out.shape[2]))
    else:
        out_ = out.reshape((out.shape[0]*out.shape[1],))
    np.savetxt('flux_bx.dat',out_)

    print('Coup neigh amt:',coup_neigh_amt[None])
    shap = scatter_basis_obs.shape
    l = []
    for i in range(active_row_len[None]):
        l.append(active_rows[i])


    start_time = time.time()
    solver_preprocess(scatter_mode)
    if scatter_mode > 1:
        solve_fluxes_numpy()
        print('system solved.')
        solver_postprocess()
    sum_fluxes()
    end_time = time.time()
    print('solve flux system:',end_time-start_time)
    #out = flux_x.to_numpy()
    #for i in range(10):
    #    print(out[i])


        #print(path_len[i], np.sum(pbnp[:51,:,i],axis=1))
    #np.savetxt('flux_x.dat',out)
    #print(active_row_len[None])
    i_wl = 0
    cumulate_path_basis()
    pbnp = path_basis.to_numpy()
    psnp = path_steps.to_numpy()
    sbnp = source_basis.to_numpy()
    print(sbnp.shape)
    #for i in range(kant*kant):
    for i in range(1): #or n_los
        np.savetxt('path_basis_los%d.dat' % i,pbnp[:,:,i],fmt='%f')
        np.savetxt('path_steps_los%d.dat' % i,psnp[:,i],fmt='%f')
        np.savetxt('source_bais_los%d.dat' % i,sbnp[:,:,i,0],fmt='%f')

    #test_path_trans()

    sb = source_basis.to_numpy()
    pb = path_basis.to_numpy()
    i_los = 0
    for i in range(path_len[i_los]):
        pass
        #print(i,pb[i,:,i_los])
    ext = extinction.to_numpy()
    #print(ext[:,0])
    IIUV2IQUV = np.array([[1.0, 1.0, 0.0, 0.0],
                          [1.0,-1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

    nstv = node_scattering_table_value.to_numpy()
    np.savetxt('nstv.dat',nstv)
    nii = node_idx_inv.to_numpy()
    np.savetxt('nii.dat',nii,fmt='%d')
    IQUV2IIUV = np.linalg.inv(IIUV2IQUV)
    start_time = time.time()
    cumulate_path_result(i_wl)
    end_time = time.time()
    print('cumulate path result:',end_time - start_time)
    out_ = transmittances.to_numpy()
    if polariz_mode > 1:
        out = out_[:,i_wl,:] @ IIUV2IQUV
    else:
        out = out_[:,i_wl]
    print(out_.shape)
    #np.savetxt('trans2x2_%f_%d.dat' % (coupling_coeff,scatter_coupling_amt.to_numpy()),out)
    np.savetxt(output_file,out)
    print(out)
    if False:
        import pickle
        with open(siroradfile,'rb') as f:
            siro_trans = pickle.load(f)
        try:
            with open(rayscaradfile,'rb') as f:
                raysca_trans = pickle.load(f)
        except:
            print("No corresponding raysca file found")
        averaging = True
        if averaging:
            siro_trans = np.sum(siro_trans,axis=0) / siro_trans.shape[0]
        else:
            if polariz_mode > 1:
                siro_trans = siro_trans[0,:,:,:]
            else:
                siro_trans = siro_trans[0,:,:]
        print(siro_trans.shape)

        if polariz_mode > 1:
            if scatter_mode == 1:
                print(np.sum(siro_trans[:,:2,:],axis=1))
                print('^-- siro | raysca --v')
                print(raysca_trans[0,:,0,:])
            else:
                siroT = np.sum(siro_trans[:,:,:],axis=1)
                print(siroT)

        else:
            if scatter_mode == 1:
                #print('NO SURFACE FROM SIRO')
                siroT = np.sum(siro_trans[:,:2],axis=1)
                print(siroT)
                np.savetxt('s4_validation/full_day/siro%d.dat' % solar_sel,siroT)
                np.savetxt('s4_validation/full_day/tursca%d.dat' % solar_sel,out)
            elif scatter_mode == 0:
                print(np.sum(siro_trans[:,:1],axis=1))
            else:
                siroT = np.sum(siro_trans[:,:],axis=1)
                print(siroT)
                if 'shdom' in test_file:
                    np.savetxt('shdom_validation/tursca%d.dat' % solar_sel,out)
                else:
                    np.savetxt('s4_validation/full_day/siro%d.dat' % solar_sel,siroT)
                    np.savetxt('s4_validation/full_day/tursca%d.dat' % solar_sel,out)
                    print("proportionality: ")
                    print(out/siroT)
                    prefix = test_file.split('/')[-1].split('.')[0]
                    np.savetxt('%s_%f_%d_siro_prop.dat' % (prefix, coupling_coeff,scatter_coupling_amt[None]),out/siroT)
#print(medium['interpolation_parameter'])
