import taichi as ti
import taichi.math as tm
import numpy as np

zero_division_check = 1e-10

# tursca
# ScaGen

# NOTE: This file contains multiple definities of same functions for different
# polarization modes. If you can come up with a nice way to template those so
# that taichi function call signatures work nicely, please do make a pull request.
# It is nice to notice that Python itself can be used as a pre-processor of a kind
# this way :)

# stategia:
#  0. DONE!selvitä coupling matrix
#  1. DONE!ratkotaan scatter_coupling_basis
#  2. ratkaise tarvittavat sirontafunktiot
#  2.5. Säilö tulokset fluksimatriisiin
#  3. kun coupling on ratkaistu, niin lasketaan kaikki tarvittavat segmenttitransmittanssit
#  4. kun on segmenttitransmittanssit ja sirontafunktiot ratkottu, niin luodaan
#  lineaarinen systeemi jossa on muuttujia maksimissaan n_nodes ** 2. Tehdäänkö
#  2d array vai sitten 6d rakenne, jonka jotenkin voisi ratkaista? Pitäiskö olla
#  joku iteratiivinen hässäkkä? Semmoinen!
#  NOTE: näköjään pyörähtää ihan kätevästi tuo sparse-matrix! Sen voi sitten ratkaista!
#  5. lähteen kontribuutio kussakin nodessa
#  6. ratkaistaan lineaarinen systeemi.
#  7. kaikki nodet ovat vähintään yksittäissirottajia
#  8. kumuloidaan halutut fluksit takaisin havaitsimeen

# eri illumination sourcet ratkotaan erikseen! halutaan vaan transmittanssi tietystä
# sourcesta. Tosin systeemi pysyy samana, joten useemmalla sourcella voi
# no, mieti tätä!


n_nodes = n_los * max_path_len

active_rows = ti.field(ti.i32)
ar_block = ti.root.bitmasked(ti.i, (n_nodes + max_scatter_couplings))
ar_block.place(active_rows)
active_row_len = ti.field(ti.i32, shape=())

scatter_basis_obs = ti.field(ti.f32)
scbo_block = ti.root.bitmasked(ti.ij, (n_nodes,n_pos))
scbo_block.place(scatter_basis_obs)

scatter_basis = ti.field(ti.f32)
scb_block = ti.root.bitmasked(ti.ijk, (max_scatter_couplings,n_pos,3))
scb_block.place(scatter_basis)


node_basis_idx = ti.field(ti.i32) # this array basically tells that which
# (los_i, node_i, los_j, node_j) tuple corresponds to the scatter_basis[idx]
los1_block = ti.root.bitmasked(ti.i, (n_los,))
path1_block = los1_block.bitmasked(ti.j, (max_path_len,))
los2_block = path1_block.bitmasked(ti.k, (n_los,))
path2_block = los2_block.bitmasked(ti.l, (max_path_len,))
path2_block.place(node_basis_idx)

link_table = ti.field(ti.i32, shape=(n_los,n_los))
flux_scaeff = ti.field(ti.f32)

n_fluxblock = int(max_scatter_couplings / n_los)
if polariz_mode == 1:
    flux_A = ti.field(ti.f32)
    flux_O = ti.field(ti.f32)
    flux_x = ti.field(ti.f32)
    flux_c = ti.field(ti.f32)
    flux_bx = ti.field(ti.f32)
    flux_bc = ti.field(ti.f32)

elif polariz_mode == 4:
    flux_A = ti.Matrix.field(n=4, m=4, dtype=ti.f32)
    flux_O = ti.Matrix.field(n=4, m=4, dtype=ti.f32)
    flux_x = ti.Vector.field(n=4, dtype=ti.f32)
    flux_c = ti.Vector.field(n=4, dtype=ti.f32)
    flux_bx = ti.Vector.field(n=4, dtype=ti.f32)
    flux_bc = ti.Vector.field(n=4, dtype=ti.f32)

fluxA_block1 = ti.root.bitmasked(ti.i, (max_scatter_couplings,))
fluxA_block2 = fluxA_block1.bitmasked(ti.j, (max_scatter_couplings,))
fluxA_block2.place(flux_A)

fluxratio_block_ = ti.root.bitmasked(ti.i, (max_scatter_couplings,))
fluxratio_block = fluxratio_block_.dense(ti.j, (n_sca + 3,)) # the +3 stands for
# the surface, the total sum, and then the total sum for incident radiation
fluxratio_block.place(flux_scaeff)

#fluxA_block1 = ti.root.pointer(ti.i, (n_los,))
#fluxA_block2 = fluxA_block1.bitmasked(ti.i, (max_path_len + n_fluxblock,))
#fluxA_block3 = fluxA_block2.pointer(ti.j, (n_los,))
#fluxA_block4 = fluxA_block3.bitmasked(ti.j, (max_path_len + n_fluxblock,))
#fluxA_block4.place(flux_A)

fluxO_block1 = ti.root.dense(ti.i, (n_los,))
fluxO_block2 = fluxO_block1.bitmasked(ti.j, (max_path_len,))
fluxO_block3 = fluxO_block2.bitmasked(ti.k, (max_scatter_couplings,))
fluxO_block3.place(flux_O)

fluxx_block1 = ti.root.dense(ti.i, (n_los,))
fluxx_block2 = fluxx_block1.bitmasked(ti.j, (max_path_len,))
fluxx_block2.place(flux_x)

fluxc_block1 = ti.root.bitmasked(ti.i, (max_scatter_couplings,))
fluxc_block1.place(flux_c)

#fluxx_block1 = ti.root.pointer(ti.i, (n_los,))
#fluxx_block2 = fluxx_block1.bitmasked(ti.i, (max_path_len + n_fluxblock,))
#fluxx_block2.place(flux_x)

fluxbx_block1 = ti.root.dense(ti.i, (n_los,))
fluxbx_block2 = fluxbx_block1.bitmasked(ti.j, (max_path_len,))
fluxbx_block2.place(flux_bx)

fluxbc_block1 = ti.root.bitmasked(ti.i, (max_scatter_couplings,))
fluxbc_block1.place(flux_bc)

#fluxb_block1 = ti.root.pointer(ti.i, (n_los,))
#fluxb_block2 = fluxb_block1.bitmasked(ti.i, (max_path_len + n_fluxblock,))
#fluxb_block2.place(flux_b)

node_scattering_table = ti.Vector.field(n=4, dtype=ti.i32)
node_scattering_table_block = ti.root.pointer(ti.i, (max_scatter_couplings,))
node_scattering_table_block.place(node_scattering_table)

node_scattering_table_value = ti.field(dtype=ti.f32)
node_scattering_table_block_ = ti.root.pointer(ti.i, (max_scatter_couplings,))
node_scattering_table_block_.place(node_scattering_table_value)

#flux_S_LU = ti.field(ti.f32)
#flux_S_LU_block1 = ti.root.pointer(ti.i, (max_scatter_couplings,))
#flux_S_LU_block2 = flux_S_LU_block1.bitmasked(ti.j, (max_scatter_couplings,))
#flux_S_LU_block2.place(flux_S_LU)

# NOTE: this is actually a misnomer: the block matrix construction has the S matrix
# which would be the commented code above. As of 3.7.2024, Antti's brain was very
# lazy and did not figure out how to solve the block format of the system, so
# we just have to comply with solving the full system.

#flux_S_LU_block1 = ti.root.pointer(ti.i, (n_los,))
#flux_S_LU_block2 = flux_S_LU_block1.bitmasked(ti.i, (max_path_len + n_fluxblock,))
#flux_S_LU_block3 = flux_S_LU_block2.pointer(ti.j, (n_los,))
#flux_S_LU_block4 = flux_S_LU_block3.bitmasked(ti.j, (max_path_len + n_fluxblock,))
#flux_S_LU_block4.place(flux_S_LU)
flux_S_LU = ti.field(ti.f32)
flux_S_y = ti.field(ti.f32)

N_SLU = max_scatter_couplings * polariz_mode

flux_S_LU_block1 = ti.root.pointer(ti.i, (N_SLU,))
flux_S_LU_block2 = flux_S_LU_block1.bitmasked(ti.j, (N_SLU,))
flux_S_LU_block2.place(flux_S_LU)

flux_S_y_block1 = ti.root.pointer(ti.i, (N_SLU,))
flux_S_y_block1.place(flux_S_y)

node_coupling = ti.field(ti.f32)
los1_block_ = ti.root.pointer(ti.i, (n_los,))
path1_block_ = los1_block_.bitmasked(ti.j, (max_path_len,))
los2_block_ = path1_block_.pointer(ti.k, (n_los,))
path2_block_ = los2_block_.bitmasked(ti.l, (max_path_len,))
path2_block_.place(node_coupling)

norm_amt_Acoup = ti.field(ti.f32)
norm_block = ti.root.pointer(ti.i, (max_scatter_couplings,))
norm_block.place(norm_amt_Acoup)

norm_amt_Aobs = ti.field(ti.f32)
norm_block_ = ti.root.pointer(ti.i, (n_nodes,))
norm_block_.place(norm_amt_Aobs)

n_weighting = n_nodes if n_nodes < max_scatter_couplings else max_scatter_couplings

weight_V = ti.Vector.field(n=3, dtype=ti.f32)
weightv_block_ = ti.root.pointer(ti.ij, (n_weighting,n_los))
weightv_block_.place(weight_V)

weight_w = ti.field(ti.f32)
weightw_block_ = ti.root.pointer(ti.ij, (n_weighting,n_los))
weightw_block_.place(weight_w)

weight_coup = ti.field(ti.i32)
weighti_block_ = ti.root.pointer(ti.ij, (n_weighting,n_los))
weighti_block_.place(weight_coup)

n_phase_theta = 10
n_phase_phi = 10
n_lossy_threads = 8

n_indicator_table = n_lossy_threads if n_lossy_threads > n_los else n_los

phase_func_table = ti.field(dtype=ti.f32, shape=(n_pos,n_phase_theta,n_phase_phi))
phase_ang_table = ti.Vector.field(n=3, dtype=ti.f32, shape=(n_phase_theta,n_phase_phi))
indicator_table = ti.field(dtype=ti.f32, shape=(n_indicator_table,n_phase_theta,n_phase_phi))
#indicator_table = ti.field(dtype=ti.f32, shape=(n_phase_theta,n_phase_phi))

@ti.func
def calc_scasegment_tau(i_wl,i_los,i_node,j_los,j_node,mode_idx):
    #dist = node_dist[i_los,i_node,j_los,j_node]
    i_coup = node_basis_idx[i_los,i_node,j_los,j_node]
    tau = 0.0
    for i_pos in range(n_pos):
        tau += scatter_basis[i_coup,i_pos,mode_idx] * extinction[i_pos,i_wl]
        #for i_sca in range(n_sca):
        #     tau += scatter_basis[i_coup,i_pos,0] * (extinction[i_pos,i_wl] - scattering[i_sca,i_pos,i_wl])
    return tau

@ti.func
def calc_scasegment_transmittance(i_wl,i_los,i_node,j_los,j_node):
    tau = calc_scasegment_tau(i_wl,i_los,i_node,j_los,j_node,1)
    return tm.exp(-tau)

@ti.func
def calc_scasegment_trans_tail(i_wl,i_los,i_node,j_los,j_node):
    i_coup = node_basis_idx[i_los,i_node,j_los,j_node]
    trans = 1.0
    if scatter_basis[i_coup,0,2] < 0.0:
        #if False:
        # this is a signal from scatter_basis_contrib to indicate that no radiation
        # is lost toward ground
        trans = 1.0
        #pass
    else:
        tau = calc_scasegment_tau(i_wl,i_los,i_node,j_los,j_node,2)
        #print(tau)
        trans = 1-tm.exp(-tau)
    #print(trans)
    return trans

@ti.func
def calc_node_scattering(i_wl,i_sca,i_los,i_node,j_los,j_node,mode_idx):
    #dist = node_dist[i_los,i_node,j_los,j_node]
    Q = 0.0
    i_coup = node_basis_idx[i_los,i_node,j_los,j_node]
    for i_pos in range(n_pos):
        Q += scatter_basis[i_coup,i_pos,mode_idx] * scattering[i_sca,i_pos,i_wl]
    return Q

@ti.func
def calc_node_scattering_obs(i_wl,i_sca,i_los,i_node):
    #dist = node_dist[i_los,i_node,i_los,i_node-1]
    Q = 0.0
    i_flux = flux_idx(i_los,i_node)
    for i_pos in range(n_pos):
        Q += scatter_basis_obs[i_flux,i_pos] * scattering[i_sca,i_pos,i_wl]
        #print(i_sca,i_los,i_pos,scatter_basis_obs[i_flux,i_pos],scattering[i_sca,i_pos,i_wl])
    return Q

@ti.func
def calc_node_extinction(i_wl,i_los,i_node,j_los,j_node):
    #dist = node_dist[i_los,i_node,j_los,j_node]
    Q_ex = 0.0
    i_coup = node_basis_idx[i_los,i_node,j_los,j_node]
    for i_pos in range(n_pos):
        Q_ex += scatter_basis[i_coup,i_pos,0] * extinction[i_pos,i_wl]
    return Q_ex

@ti.func
def calc_node_extinction_obs(i_wl,i_los,i_node):
    #dist = node_dist[i_los,i_node,i_los,i_node-1]
    Q_ex = 0.0
    i_flux = flux_idx(i_los,i_node)
    for i_pos in range(n_pos):
        Q_ex += scatter_basis_obs[i_flux,i_pos] * extinction[i_pos,i_wl]
        #print(i_sca,i_los,i_pos,scatter_basis_obs[i_flux,i_pos],scattering[i_sca,i_pos,i_wl])
    return Q_ex

def linkscaling():
    ps = path_steps.to_numpy()
    psl = np.zeros_like(ps)
    pl = path_len.to_numpy()
    n_point = np.sum(pl)
    p = np.zeros((n_point,3))
    idx = 0
    for i_los in range(n_los):
        for i_node in range(pl[i_los]):
            p[idx] = ps[i_node,i_los]
            idx += 1
    pm = np.mean(p,axis=0)
    pstd = np.std(p,axis=0)
    pmm = p - pm
    for i in range(3):
        if pstd[i] < zero_division_check:
            # this is likely to happen when one dimension is 0 entirely.
            pstd[i] = 1
    p_scaled = pmm / pstd
    p_cov = np.cov(p_scaled.T)
    #print(p_cov)
    s,V = np.linalg.eig(p_cov)
    for i in range(3):
        if s[i] < zero_division_check:
            # this is likely to happen when one dimension is 0 entirely.
            s[i] = 1
    #print(s,V)
    ds = np.diag(1/np.sqrt(s))
    projected_ps = np.dot(p_scaled, V) @ ds
    idx = 0
    for i_los in range(n_los):
        for i_node in range(pl[i_los]):
            psl[i_node,i_los,:] = projected_ps[idx]
            idx += 1
    proj_mat = projected_ps.T@projected_ps
    print(proj_mat)
    for i_los in range(n_los):
        for i_node in range(pl[i_los]):
            path_steps_linkscale[i_node,i_los] = psl[i_node,i_los,:]
    #path_steps_linkscale.from_numpy(psl[:,:,:])
    #print(psl)
    np.savetxt('psl.dat',psl.reshape((psl.shape[0]*psl.shape[1],3)))

@ti.func
def scatter_coupling(i_node,i_los,j_node,j_los):
    coupling = 0.0
    # this is the case with the same exact same node on the exactly same los!
    # however, this should not happen at this point anyway, so this check might
    # be moot.
    if i_los != j_los or i_node != j_node:
        #coupling = avg_scatter_mat[i_node,i_los] + avg_scatter_mat[j_node,j_los]
        #coupling = avg_scatter_mat[i_node,i_los] + avg_scatter_mat[j_node,j_los]
        coupling = tm.sqrt(avg_scatter_mat[i_node,i_los] * avg_scatter_mat[j_node,j_los])
        #coupling = ti.max(avg_scatter_mat[i_node,i_los],avg_scatter_mat[j_node,j_los])
    #for i_pos in range(n_pos):
    #if i_node == 50:
    #    print(i_los,i_node,j_los,j_node,coupling,avg_scatter_mat[i_node,i_los],avg_scatter_mat[j_node,j_los])
    # the surface contribution
    if i_node == (path_len[i_los]-1):
        coupling += scattering_atmos_max[None] * boundary_refl_param[0,0]
    #    pbi = path_basis[i_node+1,i_pos,i_los] - path_basis[i_node,i_pos,i_los]
    #    pbj = path_basis[j_node+1,i_pos,j_los] - path_basis[j_node,i_pos,j_los]
    #    if pbi > 0.0 or pbj > 0.0:
    #        coupling += avg_scattering[i_pos] * pbi + avg_scattering[i_pos] * pbj
    dist_real = tm.length(path_steps[j_node,j_los] - path_steps[i_node,i_los])
    dist_scale = tm.length(path_steps_linkscale[j_node,j_los] - path_steps_linkscale[i_node,i_los])
    # mean free path: (sigma * n)^-1
    #print(1.0/(0.5 * coupling))
    if dist_real > 1.0/(0.5 * coupling):
        pass
        #coupling = 0.0

    if dist_scale < zero_division_check:
        #print(f'dist_scale = 0! i_los,i_node,j_los,j_node = ({i_los},{i_node},{j_los},{j_node}). Setting to 1.')
        dist_scale = 1
    #return coupling / dist, dist
    #return coupling, dist
    #print(dist_scale,path_steps_linkscale[j_node,j_los],path_steps_linkscale[i_node,i_los])
    #return coupling / (dist_scale ** 2), dist_real
    return coupling / (dist_scale), dist_real

# memory reduction hack:
# assume symmetric matrix -> do not access the element through a normal array
# but a function
# - when coupling is symmetric, then only half of the points need to be computed

from scipy.spatial import ConvexHull
def weighting_(i_los,i_node):
    """
    All the fluxes from i_node on i_los are weighted depending on their distribution
    across the sphere.

    NOTE: This is a regular Python function because of the scipy.spatial.ConvexHull
    """
    V = []
    for j_los in range(n_los):
        for j_node in range(path_len[j_los]):
            if node_basis_idx[i_los,i_node,j_los,j_node] > 0:
                V.append(path_steps[j_node,j_los] - path_steps[i_node,i_los])
    for idx_v,v in enumerate(V):
        V[idx_v] = v / np.linalg.norm(v)

    # check the overlapping directions
    overlap_epsilon = 1e-3 #TODO: derive this explicitly from an angular limit
    overlap_ma = -1 * np.ones(len(V))
    # overlap_ma is -1 for each vector which is computed, all the other values
    # denote that directions are the same
    for idx_v in range(len(V)):
        for idx_u in range(idx_v):
            if overlap_ma[idx_u] < 0 and np.linalg.norm(V[idx_v] - V[idx_u]) < overlap_epsilon:
                overlap_ma[idx_v] = idx_u
                break

    n_uniq_v = np.sum(overlap_ma < 0)
    if n_uniq_v == 2:
        # in this case we just split the weight equally to the both unique
        # vectors, and then
        pass

    P = np.zeros((n_uniq_v,3))

    print(overlap_ma)
    i = 0

    P = np.zeros()
    for idx_v in range(len(V)):
        if overlap_ma[idx_v] == -1:
            P[i,:] = V[idx_v]
            i += 1
    print(V)
    print(P)
    #well, you've made your bed. what now, smart guy
    simp = ConvexHull(P).simplices
    print(simp)

    """
    1. Yhdistä epsilonin päässä olevat pisteet toisiinsa ja kirjaa ne ylös.
    2. Laske pistejoukon convex hull
    3. Kunkin pisteen edgejen pituuksien keskiarvo otetaan ympyrän säteeksi (kertaa jokin pallopintaskaalatekijä??)
    3.5. Vai tehäänkö line/dual graafi?
    4. Tämän pisteen voronoi-pinta-ala on pallopinnan ympyrän pinta-ala tuolla säteellä
    5. Kun kaikille pisteille on tämä laskettu, niin skaalataan ne siten, että alojen summa vastaa pallopintaa
    6. Yhdistettyjen pisteiden pinta-ala jaetaan tasaisesti niiden kesken.
    7. Tällöin kaikilla pallopinnan pisteillä on pinta-ala.

    komplementtigraafi johonkin kohtaan!
    concave hull (Lindqvist & Muinonen)
    """
    pass
    """
    Plano nuevo
    1. Laske kaikkien pistiden great circle distance toisiinsa!
    2. Yhdistä pisteet jotka ovat epsilonin päässä toisistaan (luonnollinen kulmaeps.)
    3. Kullekin pisteelle laske keskim. etäisyys naapureihinsa. (density of some kind)
    4. Painota kutakin pistettä normalisoidun keskim. etäysyyden avulla.
    5. Halutessasi voit myös tehdä sen 1/8 semisfääri-korjauksen, jossa energiaa
    hukkaantuu jos jossain semisfäärissä ei ole yhtään vektoria
    """


def test_weighting_(i_los,i_node):
    print("weighting the los %d, node %d" % (i_los,i_node))
    weighting(i_los,i_node)

@ti.func
def check_shadow(i_node,i_los,j_node,j_los):
    """
    Here we check if the boundaries block the coupling between
    these two points.
    """
    N_check_steps = 10
    in_shadow = False
    r_start = path_steps[i_node,i_los]
    r_end = path_steps[j_node,j_los]
    d = r_end - r_start
    for i in range(1,N_check_steps):
        r_check = r_start + i/N_check_steps * d
        if not is_in_medium(r_check,stopper_inner,stopper_outer):
            in_shadow = True
            break
    return in_shadow

@ti.func
def create_coupling_matrix(coupling_coeff,total_coupling):
    coupled_weight = 0.0
    ti.loop_config(serialize=True)
    for i_coup in range(max_scatter_couplings):
        i_node,i_los,j_node,j_los = node_scattering_table[i_coup]
        if i_node == 0 and i_los == 0 and j_node == 0 and j_los == 0:
            continue
        coupling, dist = scatter_coupling(i_node,i_los,j_node,j_los)
        #print(i_los,i_node,j_los,j_node,coupling)
        if coupling > coupling_coeff:
            node_basis_idx[i_los,i_node,j_los,j_node] = scatter_coupling_amt[None]
            scatter_coupling_amt[None] += 1
            node_dist[i_los,i_node,j_los,j_node] = dist
            node_coupling[i_los,i_node,j_los,j_node] = coupling
            link_table[i_los,j_los] += 1
            coupled_weight += coupling

    print(f"Coupled: {100*coupled_weight / (total_coupling):%1.2f} % of total coupling weight.")
    print(scatter_coupling_amt[None],max_scatter_couplings)

@ti.func
def scatter_basis_contrib(i_coup):
    i_los,i_node,j_los,j_node = node_idx_inv[i_coup]
    happening = True
    #N_int_steps = 10
    N_int_steps = n_pos
    int_coeff = 1 / (N_int_steps + 1)
    path_dir = path_steps[j_node,j_los] - path_steps[i_node,i_los]
    #print(i_coup,tm.length(path_steps[j_node,j_los]),tm.length(path_steps[i_node,i_los]))
    if i_los == 0 and i_node == 0 and j_los == 0 and j_node == 0:
        #this indicates uninitialized coupling
        happening = not happening
    if happening:
        for i_pos in range(n_pos):
            #scatter_basis[i_coup,i_pos,0] = path_basis[i_node,i_pos,i_los]
            scatter_basis[i_coup,i_pos,0] = eval_basis(path_steps[j_node,j_los],i_pos)
            scatter_basis[i_coup,i_pos,2] = eval_basis(path_steps[i_node,i_los],i_pos)
        for i_int in range(N_int_steps+1):
            r = path_steps[i_node,i_los] + (i_int / N_int_steps) * path_dir
            for i_pos in range(n_pos):
                #scatter_basis[i_coup,i_pos] = node_dist[i_los,i_node,j_los,j_node] * 0.5 * (path_basis[i_node,i_pos,i_los] + path_basis[j_node,i_pos,j_los])
                # NOTE: the scattering happens at point j_los, j_node. Even if radiation comes from i_los, i_node, it should not have an effect how to scatter radiation at point j.
                # Therefore the path basis at point j is used. Why then have the full node_dist distance between points i and j here and not just 0.5*dist?
                # Good question. The idea is to _extend_ the scattering length onwards from the point j, so that the point j is adequately presented here.

                # However, the scattering path tau needs the average between the nodes, so it needs another variable. Therefore
                # the scatter_basis is now extended onto 3D array, where the first one is used for scattering and another one for extinction calculations

                #scatter_basis[i_coup,i_pos,0] = node_dist[i_los,i_node,j_los,j_node] * path_basis[i_node,i_pos,i_los]

                #scatter_basis[i_coup,i_pos,0] = dist * path_basis[i_node,i_pos,i_los]

                #scatter_basis[i_coup,i_pos,0] = 1.0 * path_basis[j_node,i_pos,j_los]
                # the above is a bit weird, so let's fallback to below
                #scatter_basis[i_coup,i_pos,0] = node_dist[i_los,i_node,j_los,j_node] * 0.5 * (path_basis[i_node,i_pos,i_los] + path_basis[j_node,i_pos,j_los])

                sca_basis = eval_basis(r,i_pos) * int_coeff * node_dist[i_los,i_node,j_los,j_node]

                #if i_int < 4:
                    # there might be a justified reason to put a 2x multiplier into this...
                #scatter_basis[i_coup,i_pos,0] += sca_basis
                    #print(i_int,i_pos,'scabas0')

                scatter_basis[i_coup,i_pos,1] += sca_basis
                    #print(i_int,i_pos,'scabas1')
                #scatter_basis[i_coup,i_pos,1] = node_dist[i_los,i_node,j_los,j_node] * 0.5 * (path_basis[i_node,i_pos,i_los] + path_basis[j_node,i_pos,j_los])

        if False:
            path_dir /= tm.length(path_dir)
            end_point = find_next_boundary_crossing(path_steps[j_node,j_los], path_dir)
            #print(path_steps[j_node,j_los], path_dir, end_point)
            closest_boundary = find_closest_boundary(end_point)
            if closest_boundary == 0:
                # 0 inner, 1 outer
                # the main idea behind this scheme is to inform the transmittance calculations that
                # no radiation is lost through this boundary.
                scatter_basis[i_coup,0,2] = -1.0
            else:
                end_dir = end_point - path_steps[j_node,j_los]
                end_dist = tm.length(end_dir)
                for i_int in range(N_int_steps+1):
                    r = path_steps[j_node,j_los] + (i_int / N_int_steps) * end_dir
                    for i_pos in range(n_pos):
                        sca_basis = eval_basis(r,i_pos) * int_coeff * end_dist
                        scatter_basis[i_coup,i_pos,2] += sca_basis

        #for i_pos in range(n_pos):
        #    print(i_pos,scatter_basis[i_coup,i_pos,0],scatter_basis[i_coup,i_pos,1],scatter_basis[i_coup,i_pos,2])

@ti.func
def scatter_basis_contrib_obs(i_los):
    for i_node in range(path_len[i_los]):
        i_flux = flux_idx(i_los,i_node)
        #l = 0.0
        #if 0 < i_node and i_node < (path_len[i_los]-1):
        #    l = 0.5 * (node_dist[i_los,i_node-1,i_los,i_node] + node_dist[i_los,i_node,i_los,i_node + 1])
        #    #l = node_dist[i_los,i_node-1,i_los,i_node]# + node_dist[i_los,i_node,i_los,i_node + 1])
        #elif i_node == 0:
        #    l = node_dist[i_los,i_node,i_los,i_node+1]
        #elif i_node == (path_len[i_los]-1):
        #    l = node_dist[i_los,i_node-1,i_los,i_node]
        #l = 1.0
        for i_pos in range(n_pos):
            scatter_basis_obs[i_flux,i_pos] = path_basis[i_node,i_pos,i_los]# * l

@ti.kernel
def calc_scatter_basis():
    #ti.loop_config(serialize=True)
    for i_coup in range(scatter_coupling_amt[None]):
        scatter_basis_contrib(i_coup)

@ti.kernel
def calc_scatter_basis_obs():
    for i_los in range(n_los):
        scatter_basis_contrib_obs(i_los)

@ti.func
def refl_kernel_lambertian(d_in,r):
    closest = find_closest_boundary(r)
    surf_norm = stopper_inner.up(r)
    if closest == 1:
        surf_norm = stopper_outer.up(r)
    costerm = tm.dot(d_in,surf_norm)
    #d_norm = (r1 - r0) / tm.length(r1 - r0) # could be node_dist
    surfmult = 1.0
    if closest == 1:
        surfmult = 0.0
    ph = surfmult * boundary_refl_param[0,0] * costerm / tm.pi
    return ph#, boundary_refl_param[0,0]


@ti.func
def get_integrated_reflfun(i_coup,dir_out,dir_in):
    color = get_color_in_dir(i_coup,dir_out)
    origin = tm.vec3(0.0, 1.0, 0.0)
    #dir_amt = area_amt[color-1,i_coup]
    R = rotmat(origin,dir_in)
    _, _, j_los, j_node = node_idx_inv[i_coup]
    r = path_steps[j_node,j_los]
    closest = find_closest_boundary(r)
    surf_norm = stopper_inner.up(r)
    if closest == 1:
        surf_norm = stopper_outer.up(r)
    refl = 0.0
    alb = 0.0
    for i_fibo in range(n_fibo_dirs):
        fibo_rot = matrix_vec_mult_3(R,phase_dir_w[i_fibo])
        towards_medium = 1.0
        if tm.dot(fibo_rot,surf_norm) < 0.0:
            towards_medium = 0.0
        if dir_areas[i_fibo,i_coup,1] == color:
            refl_ = refl_kernel_lambertian(-dir_in,r) * phase_area_w[i_fibo] * towards_medium
            refl += refl_
    #print(dir_amt)
    #return refl/2# * (1/(2*tm.pi))# / dir_amt
    return refl

@ti.func
def get_integrated_reflfun_b(i_coup,i_source):
    color = get_color_in_dir_b(i_coup,i_source)
    origin = tm.vec3(0.0, 1.0, 0.0)
    dir_in = source_dir[i_source]
    R = rotmat(origin,dir_in)
    i_los, i_node, _, _ = node_idx_inv[i_coup]
    r = path_steps[i_node,i_los]
    closest = find_closest_boundary(r)
    surf_norm = stopper_inner.up(r)
    if closest == 1:
        surf_norm = stopper_outer.up(r)
    refl = 0.0
    for i_fibo in range(n_fibo_dirs):
        fibo_rot = matrix_vec_mult_3(R,phase_dir_w[i_fibo])
        towards_medium = 1.0
        if tm.dot(fibo_rot,surf_norm) < 0.0:
            towards_medium = 0.0
        if dir_areas_b[i_fibo,i_los,i_node,1] == color:
            refl += refl_kernel_lambertian(-dir_in,r) * phase_area_w[i_fibo] * towards_medium
    return refl

@ti.func
def get_integrated_reflmat(i_coup,dir_out,dir_in):
    ph_ = get_integrated_reflfun(i_coup,dir_out,dir_in)
    ph = tm.mat4(0)
    ph[0,0] = 0.5 * ph_
    ph[1,0] = 0.5 * ph_
    ph[0,1] = 0.5 * ph_
    ph[1,1] = 0.5 * ph_
    return ph


if polariz_mode == 1:
    @ti.func
    def scatter_kernel_source(i_sca,i_source,j_los,j_node,k_los,k_node):
        """
        The scattering function (or the Mueller matrix) at (j_los,j_node) from the
        direction of i_source to (j_los,j_node) and scattered from (j_los,j_node)
        to the direction of (k_los,k_node).
        """
        ph = 0.0
        if k_node != -1:
            incident_dir = source_dir[i_source]
            # combined weigths of scattering functions
            # its scattering
            #ph = 1/(4*tm.pi) #isotropic
            r0 = path_steps[j_node,j_los]
            r1 = path_steps[k_node,k_los]
            d = r1 - r0
            d = d/tm.length(d)
            ph = phase_function(i_sca,incident_dir,d)
            #print(i_sca,j_los,j_node,k_los,k_node,tm.dot(incident_dir,d),ph)
        return ph

    @ti.func
    def refl_kernel(i_los,i_node,j_los,j_node,k_los,k_node):
        r_i = path_steps[i_node,i_los]
        r_j = path_steps[j_node,j_los]
        #r_k = path_steps[k_node,k_los]
        d_in = r_i - r_j
        d_in = d_in/tm.length(d_in)
        ph = refl_kernel_lambertian(d_in,r_j)
        if ph < 0.0:
            #print(ph,i_los,i_node,j_los,j_node,k_los,k_node,d_in)
            #print(tm.length(r_i),tm.length(r_j),tm.length(r_k))
            pass
        ph = 0.0
        return ph

    @ti.func
    def refl_kernel_source(i_source,j_los,j_node,k_los,k_node):
        sour_dir = -source_dir[i_source]
        r0 = path_steps[j_node,j_los]
        #r1 = path_steps[k_node,k_los]
        ph = refl_kernel_lambertian(sour_dir,r0)
        if ph < 0.0:
            print("aaargh, reflecting node is in shadow")
        return ph

    @ti.func
    def scatter_kernel(i_sca,i_los,i_node,j_los,j_node,k_los,k_node) -> ti.f32:
        """
        The scattering function (or the Mueller matrix) at (j_los,j_node) from the
        direction of (i_los,i_node) to (j_los,j_node) and scattered from (j_los,j_node)
        to the direction of (k_los,k_node).
        """
        r_i = path_steps[i_node,i_los]
        r_j = path_steps[j_node,j_los]
        ph = 0.0
        d_in = r_i - r_j
        d_in = d_in/tm.length(d_in)
        d_out = tm.vec3(0.0)
        if k_node != -1:
            r_k = path_steps[k_node,k_los]
            d_out = r_k - r_j
            d_out = d_out/tm.length(d_out)
        d_out = -instr_view[j_los]
        ph = phase_function(i_sca,d_in,d_out)
        return ph

    @ti.func
    def set_up_flux_A_obs_SQ():
        for i_coup in range(scatter_coupling_amt[None]):
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]

            R = 0.0
            d_out = tm.vec3(0.0)
            if j_node == 0:
                d_out = -instr_view[j_los]
            else:
                d_out = path_steps[j_node-1,j_los] - path_steps[j_node,j_los]
                d_out /= tm.length(d_out)
            if (path_len[j_los]-1) == j_node:
                d_in = path_steps[j_node,j_los] - path_steps[i_node,i_los]
                d_in /= tm.length(d_in)
                R = get_integrated_reflfun(i_coup,d_out,d_in)
            phasevec = get_integrated_phasefun(i_coup,d_out)
            S = flux_scaeff[i_coup,0] * phasevec[0]
            if n_sca > 1:
                S += flux_scaeff[i_coup,1] * phasevec[1]
                if n_sca > 2:
                    S += flux_scaeff[i_coup,2] * phasevec[2]
                    if n_sca > 3:
                        S += flux_scaeff[i_coup,3] * phasevec[3]
            flux_O[j_los, j_node, i_coup] = R + S

    """
    Lambertian pinnan coupl strength: max atmos coupl * alb

    Ei mitään ratioo vaan suoraan scattering! Miten painotetaan vaihefunktiot?
    Se columnin summa pitäisi vastata sirontaatehokkuutta. Eli kukin sirottaja skaalataan
    oman sirontatehokkuutensa kautta ja lopuksi summataan?

    ja laitettava se 4*pi sinne lambertiin myös??

    Pisteittäinen scattering tarvitaan! on liikaa jos skaalataan path stepin pituudella sitä!!
    flux_O ja flux_A laitettava vec4?? ja sitten se viimeinen vec-elementti oisi se painotettu summa?

    ja sitten ne elementit kerrotaan niillä transmittansseilla

    Polarisaatio: ratkaise skalaarina fluksit ja sitten vaan kerro mullerillä??
    """
    @ti.func
    def set_up_flux_scattering_efficiency(i_wl):
        for i_coup in range(scatter_coupling_amt[None]):
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            Q_tot = 0.0
            Q_tot_b = 0.0
            if (path_len[j_los]-1) == j_node:
                Q = boundary_refl_param[0,0]
                flux_scaeff[i_coup,n_sca] = Q
                Q_tot += Q
            if (path_len[i_los]-1) == i_node:
                Q = boundary_refl_param[0,0]
                Q_tot_b += Q
            for i_sca in range(n_sca):
                Q = calc_node_scattering(i_wl,i_sca,i_los,i_node,j_los,j_node,0)
                flux_scaeff[i_coup,i_sca] = Q
                Q_tot += Q
                Q_b = calc_node_scattering(i_wl,i_sca,i_los,i_node,j_los,j_node,2)
                Q_tot_b += Q_b
            #print(i_los,i_node,j_los,j_node,Q_tot,Q_tot_b)
            flux_scaeff[i_coup,n_sca+1] = Q_tot
            flux_scaeff[i_coup,n_sca+2] = Q_tot_b

            #print(i_coup,flux_scaeff[i_coup,0],flux_scaeff[i_coup,1],flux_scaeff[i_coup,2],flux_scaeff[i_coup,3])

    @ti.func
    def set_up_flux_A_coup_SQ():
        for i_cc in range(coup_neigh_amt[None]):
            i_coup, j_coup = coup_idx_neighbour[i_cc]
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            k_los, k_node = 0, 0
            if i_coup != j_coup:
                _    , _     , k_los, k_node = node_idx_inv[j_coup]
            else:
                k_los, k_node, _    , _      = node_idx_inv[j_coup]
            R = 0.0
            d_out = path_steps[k_node,k_los] - path_steps[j_node,j_los]
            d_out /= tm.length(d_out)
            if (path_len[j_los]-1) == j_node:
                #SQ = refl_kernel(i_los,i_node,j_los,j_node,k_los,k_node)
                d_in = path_steps[j_node,j_los] - path_steps[i_node,i_los]
                d_in /= tm.length(d_in)
                R = get_integrated_reflfun(i_coup,d_out,d_in)
            phasevec = get_integrated_phasefun(i_coup,d_out)
            S = flux_scaeff[i_coup,0] * phasevec[0]
            if n_sca > 1:
                S += flux_scaeff[i_coup,1] * phasevec[1]
                if n_sca > 2:
                    S += flux_scaeff[i_coup,2] * phasevec[2]
                    if n_sca > 3:
                        S += flux_scaeff[i_coup,3] * phasevec[3]
            flux_A[i_coup,j_coup] = -(R + S)

    @ti.func
    def set_up_flux_A_normalize():
        #ti.loop_config(serialize=True)
        for j_coup in range(scatter_coupling_amt[None]):
            tot_S = 0.0
            #for i_los in range(n_los):
            #    for i_node in range(path_len[i_los]):
            #        tot_S += flux_O[i_los, i_node, j_coup]
            for i_coup in range(scatter_coupling_amt[None]):
                tot_S += -flux_A[i_coup, j_coup]
            if tot_S > 0.0:
                normalizer = flux_scaeff[j_coup,n_sca+1] / tot_S
                #normalizer = 1/tot_S
                #print(j_coup,tot_S,normalizer)
                #for i_los in range(n_los):
                #    for i_node in range(path_len[i_los]):
                #        flux_O[i_los, i_node, j_coup] *= normalizer
                for i_coup in range(scatter_coupling_amt[None]):
                    flux_A[i_coup, j_coup] *= normalizer

    @ti.func
    def set_up_flux_A_obs_trans(i_wl):
        for i_coup in range(scatter_coupling_amt[None]):
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            trans = calc_scasegment_transmittance(i_wl,i_los,i_node,j_los,j_node)
            #trans_tail = calc_scasegment_trans_tail(i_wl,i_los,i_node,j_los,j_node)
            flux_O[j_los, j_node, i_coup] *= trans# * trans_tail

    @ti.func
    def set_up_flux_A_coup_trans(i_wl):
        for i_cc in range(coup_neigh_amt[None]):
            i_coup, j_coup = coup_idx_neighbour[i_cc]
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            trans = calc_scasegment_transmittance(i_wl,i_los,i_node,j_los,j_node)
            #trans_tail = calc_scasegment_trans_tail(i_wl,i_los,i_node,j_los,j_node)
            #print(trans,trans_tail)
            flux_A[i_coup, j_coup] *= trans# * trans_tail

    @ti.kernel
    def set_up_flux_bc(i_wl: int, i_source: int):
        for i_coup in range(scatter_coupling_amt[None]):
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            if j_los == 0 and j_node == 0 and i_los == 0 and i_node == 0:
                continue
            d_out = path_steps[j_node,j_los] - path_steps[i_node,i_los]
            d_out /= tm.length(d_out)
            R = 0.0
            if (path_len[j_los]-1) == j_node:
                R = get_integrated_reflfun_b(i_coup,i_source)
            phasevec = get_integrated_phasefun_b(i_coup,i_source)
            Q = calc_node_scattering(i_wl,0,i_los,i_node,j_los,j_node,2)
            S = Q * phasevec[0]
            if n_sca > 1:
                Q = calc_node_scattering(i_wl,1,i_los,i_node,j_los,j_node,2)
                S += Q * phasevec[1]
                if n_sca > 2:
                    Q = calc_node_scattering(i_wl,2,i_los,i_node,j_los,j_node,2)
                    S += Q * phasevec[2]
                    if n_sca > 3:
                        Q = calc_node_scattering(i_wl,3,i_los,i_node,j_los,j_node,2)
                        S += Q * phasevec[3]
            flux_bc[i_coup] = R + S
            #print(i_coup,i_los,i_node,j_los,j_node,flux_bc[i_coup])
            #splittaa transmittanssi ja tämä lasku, pitää painottaa taas!!!

    @ti.kernel
    def flux_b_normalize(i_source : int):
        for i_los in range(n_los):
            for i_node in range(path_len[i_los]):
                tot_S = 0.0
                flux_Q = 0.0
                for i_coup in range(scatter_coupling_amt[None]):
                    _i_los, _i_node, _j_los, _j_node = node_idx_inv[i_coup]
                    if _i_los == i_los and _i_node == i_node:
                        flux_Q = flux_scaeff[i_coup,n_sca+2]
                        tot_S += flux_bc[i_coup]
                        #print(i_node,i_coup,flux_Q,tot_S)
                if tot_S > 0.0:
                    normalizer = flux_Q / tot_S * 4*tm.pi
                    #print(flux_Q,tot_S)
                    for i_coup in range(scatter_coupling_amt[None]):
                        _i_los, _i_node, _, _ = node_idx_inv[i_coup]
                        if _i_los == i_los and _i_node == i_node:
                            flux_bc[i_coup] *= normalizer

    @ti.kernel
    def flux_b_trans(i_wl : int,i_source : int):
        # tarvii kaksi transsia: incidentti sekä i_coup!
        for i_coup in range(scatter_coupling_amt[None]):
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            if j_los == 0 and j_node == 0 and i_los == 0 and i_node == 0:
                continue
            trans_source = calc_source_transmittance(i_node,i_wl,i_los,i_source)
            trans_coup = calc_scasegment_transmittance(i_wl,i_los,i_node,j_los,j_node)
            flux_bc[i_coup] *= trans_source * trans_coup

    @ti.kernel
    def flux_A_trans(i_wl : int):
        #set_up_flux_A_obs_trans(i_wl)
        set_up_flux_A_coup_trans(i_wl)

    @ti.kernel
    def flux_A_normalize():
        set_up_flux_A_normalize()

    @ti.kernel
    def flux_A_SQ():
        #set_up_flux_A_obs_SQ()
        set_up_flux_A_coup_SQ()

    @ti.kernel
    def flux_scattering_efficiency(i_wl : int):
        set_up_flux_scattering_efficiency(i_wl)

    @ti.func
    def set_up_flux_A_obs(i_wl):
        for i_coup in range(scatter_coupling_amt[None]):
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            trans = calc_scasegment_transmittance(i_wl,i_los,i_node,j_los,j_node)
            SQ = 0.0
            if (path_len[j_los]-1) == j_node:
                SQ = refl_kernel(i_los,i_node,j_los,j_node,j_los,j_node-1)
            for i_sca in range(n_sca):
                S = scatter_kernel(i_sca,i_los,i_node,j_los,j_node,j_los,j_node-1)
                Q = calc_node_scattering_obs(i_wl,i_sca,j_los,j_node)
                SQ += S * Q
            flux_O[j_los, j_node, i_coup] = trans * SQ

    @ti.func
    def set_up_flux_b_obs(i_wl,i_los,i_source):
        for i_node in range(path_len[i_los]):
            trans = calc_source_transmittance(i_node,i_wl,i_los,i_source)

            SQ = 0.0
            if (path_len[i_los]-1) == i_node:
                SQ = refl_kernel_source(i_source,i_los,i_node,i_los,i_node-1)
            #else:
            #print("")
            #print(SQ)
            for i_sca in range(n_sca):
                S = scatter_kernel_source(i_sca, i_source,i_los,i_node,i_los,i_node-1)
                Q = calc_node_scattering_obs(i_wl,i_sca,i_los,i_node)
                #print(i_sca,S,Q)
                SQ += S * Q
                #print(SQ)
            #print("flux b sca SQ, trans:",SQ,trans)
            i = flux_idx(i_los,i_node)
            #flux_b[i] = trans * SQ
            #print(i_los,i_node,trans,SQ)
            flux_bx[i_los,i_node] = trans * SQ

    @ti.kernel
    def set_up_flux_A_coup(i_wl: int):
        for i_cc in range(coup_neigh_amt[None]):
            i_coup, j_coup = coup_idx_neighbour[i_cc]
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            k_los, k_node = 0, 0
            if i_coup != j_coup:
                _    , _     , k_los, k_node = node_idx_inv[j_coup]
            else:
                k_los, k_node, _    , _      = node_idx_inv[j_coup]
            trans = calc_scasegment_transmittance(i_wl,i_los,i_node,j_los,j_node)
            trans_tail = calc_scasegment_trans_tail(i_wl,i_los,i_node,j_los,j_node)
            SQ = 0.0
            Q_tot = 0.0
            d_out = path_steps[k_node,k_los] - path_steps[j_node,j_los]
            d_out /= tm.length(d_out)
            if (path_len[j_los]-1) == j_node:
                #SQ = refl_kernel(i_los,i_node,j_los,j_node,k_los,k_node)
                d_in = path_steps[j_node,j_los] - path_steps[i_node,i_los]
                d_in /= tm.length(d_in)
                SQ = get_integrated_reflfun(i_coup,d_out,d_in)
                #print(get_integrated_reflfun(i_coup,d_out,d_in))
            SQ += get_integrated_phasefun(i_coup,d_out)
            for i_sca in range(n_sca):
                #S = scatter_kernel(i_sca, i_los,i_node,j_los,j_node,k_los,k_node)
                Q = calc_node_scattering(i_wl,i_sca,j_los,j_node,k_los,k_node)
                #SQ += S * Q
                Q_tot += Q
            Q_tot /= calc_node_extinction(i_wl,j_los,j_node,k_los,k_node)
            SQ *= Q_tot
            #print(SQ,Q_tot,(1-trans))
            #print(trans,SQ,trans_tail)
            if tm.isnan( -trans * SQ):
                print("AAARGH in ",i_los,i_node,j_los,j_node,k_los,k_node,'coups',i_coup,j_coup)
            #flux_A[n_nodes + i_coup - 1,n_nodes + j_coup - 1] = -(1-trans) * SQ# * (trans_tail)
            #flux_A[i_coup,j_coup] = -(1-trans) * SQ# * (2*tm.pi) # * (trans_tail)
            flux_A[i_coup,j_coup] = -(trans) * SQ# * (trans_tail)# * (2*tm.pi) # * (trans_tail)
            flux_A[i_coup,j_coup] = SQ# * (trans_tail)# * (2*tm.pi) # * (trans_tail)
            if -(trans) * SQ > 0:
                print(-(trans) * SQ,i_los,i_node,j_los,j_node,k_los,k_node)
            norm_amt_Acoup[i_coup] += 1

    @ti.kernel
    def set_up_flux_b_coup(i_wl: int, i_source: int):
        for i_coup in range(scatter_coupling_amt[None]):
            j_los, j_node, k_los, k_node = node_idx_inv[i_coup]
            if j_los == 0 and j_node == 0 and k_los == 0 and k_node == 0:
                continue
            trans = calc_source_transmittance(j_node,i_wl,j_los,i_source)
            SQ = 0.0
            if (path_len[j_los]-1) == j_node:
                SQ = refl_kernel_source(i_source,j_los,j_node,k_los,k_node)
            #else:
            for i_sca in range(n_sca):
                S = scatter_kernel_source(i_sca,i_source,j_los,j_node,k_los,k_node)
                Q = calc_node_scattering(i_wl,i_sca,j_los,j_node,k_los,k_node,0)
                SQ += S * Q
            #flux_b[n_nodes + i_coup - 1] = trans * SQ
            flux_bc[i_coup] = trans * SQ
            #print(n_nodes + i_coup - 1,trans,SQ)

elif polariz_mode == 4:
    @ti.func
    def scatter_kernel_source(i_sca,i_source,j_los,j_node,k_los,k_node):
        """
        The scattering function (or the Mueller matrix) at (j_los,j_node) from the
        direction of i_source to (j_los,j_node) and scattered from (j_los,j_node)
        to the direction of (k_los,k_node).
        """
        ph = tm.mat4(0)
        if k_node != -1:
            sour_dir = source_dir[i_source]
            # combined weigths of scattering functions
            # its scattering
            #ph = 1/(4*tm.pi) #isotropic
            r0 = path_steps[j_node,j_los]
            r1 = path_steps[k_node,k_los]
            d = r1 - r0
            d = d/tm.length(d)
            ph = phase_matrix(i_sca,sour_dir,d)
        return ph

    @ti.func
    def refl_kernel(i_los,i_node,j_los,j_node,k_los,k_node):
        r_i = path_steps[i_node,i_los]
        r_j = path_steps[j_node,j_los]
        r_k = path_steps[k_node,k_los]
        d_in = r_i - r_j
        d_in = d_in/tm.length(d_in)
        ph_ = refl_kernel_lambertian(d_in,r_j)
        ph = tm.mat4(0)

        """
        Note from Antti: Depolarizing lambertian kernel is not invertible.
        i.e. [a a 0 0
              a a 0 0
              0 0 0 0
              0 0 0 0]
        switching to polarization-retaining lambertian kernel:
             [a 0 0 0
              0 a 0 0
              0 0 a 0
              0 0 0 a]
        Another note: No matrix is divided by this, so there shouldn't be a problem!
        """
        ph[0,0] = 0.5 * ph_
        ph[1,0] = 0.5 * ph_
        ph[0,1] = 0.5 * ph_
        ph[1,1] = 0.5 * ph_
        return ph

    @ti.func
    def refl_kernel_source(i_source,j_los,j_node,k_los,k_node):
        sour_dir = -source_dir[i_source]
        r0 = path_steps[j_node,j_los]
        ph = tm.mat4(0)
        ph_ = refl_kernel_lambertian(sour_dir,r0)
        ph[0,0] = 0.5 * ph_
        ph[1,0] = 0.5 * ph_
        ph[0,1] = 0.5 * ph_
        ph[1,1] = 0.5 * ph_
        return ph

    @ti.func
    def scatter_kernel(i_sca,i_los,i_node,j_los,j_node,k_los,k_node) -> tm.mat4:
        """
        The scattering function (or the Mueller matrix) at (j_los,j_node) from the
        direction of (i_los,i_node) to (j_los,j_node) and scattered from (j_los,j_node)
        to the direction of (k_los,k_node).
        """
        r_i = path_steps[i_node,i_los]
        r_j = path_steps[j_node,j_los]
        ph = tm.mat4(0)
        if k_node != -1:
            r_k = path_steps[k_node,k_los]
            d_in = r_i - r_j
            d_in = d_in/tm.length(d_in)
            d_out = r_k - r_j
            d_out = d_out/tm.length(d_out)
            ph = phase_matrix(i_sca,d_in,d_out)
        return ph

    @ti.func
    def set_up_flux_A_obs(i_wl,i_los):
        for i_coup in range(scatter_coupling_amt[None]):
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]

            trans = calc_scasegment_transmittance(i_wl,i_los,i_node,j_los,j_node)
            SQ = tm.mat4(0)
            Q_tot = 0.0
            dir_in = path_steps[j_node,j_los] - path_steps[i_node,i_los]
            if tm.length(dir_in) < zero_division_check:
                continue
            dir_in = dir_in / tm.length(dir_in)
            dir_out = tm.vec3(0.0)
            if j_node == 0:
                dir_out = -instr_view[j_los]
            else:
                dir_out = path_steps[j_node-1,j_los] - path_steps[j_node,j_los]
            dir_out = dir_out / tm.length(dir_out)
            #if tm.length(dir_out) < zero_division_check:
            #    continue # this is just error-hiding: let's crash if that's the case!!
            if (path_len[j_los]-1) == j_node:
                #SQ = refl_kernel(i_los,i_node,j_los,j_node,j_los,j_node-1)
                SQ = get_integrated_reflmat(i_coup,dir_out,dir_in)

            SQ += get_integrated_phasemat(i_coup,dir_out)
            for i_sca in range(n_sca):
                #S = scatter_kernel(i_sca, i_los,i_node,j_los,j_node,j_los,j_node-1)

                Q = calc_node_scattering_obs(i_wl,i_sca,j_los,j_node)
                #ph = 1.0 / phase_function(i_sca,dir_in,dir_out)
                #SQ += S * Q * ph
                Q_tot += Q

            i = flux_idx(i_los,i_node)
            Q_tot /= calc_node_extinction_obs(i_wl,j_los,j_node)
            SQ *= (1/Q_tot)
            rotated_SQ = rotate_polarization(dir_in,dir_out,SQ)
            #print(dir_in,dir_out,rotated_SQ)
            #flux_A[i, n_nodes + i_coup - 1] = (1 - trans) * rotated_SQ
            #flux_O[j_los, j_node, i_coup] = (1 - trans) * rotated_SQ# * (2*tm.pi)
            flux_O[j_los, j_node, i_coup] = (trans) * rotated_SQ# * (2*tm.pi)
            # the -1 in i_coup index is to fill the matrix appropriately
            # the minus sign in front of the element is to fill the element
            # accordingly.
            norm_amt_Aobs[i] += 1

    @ti.func
    def set_up_flux_b_obs(i_wl,i_los,i_source):
        inat = incident_stokes(i_wl)
        for i_node in range(path_len[i_los]):
            trans = calc_source_transmittance(i_node,i_wl,i_los,i_source)
            transvec = tm.vec4(trans)
            SQ = tm.vec4(0.0)
            if (path_len[i_los]-1) == i_node:
                S = refl_kernel_source(i_source,i_los,i_node,i_los,i_node-1)
                SQ = matrix_vec_mult_4(S,inat)
            #else:
            for i_sca in range(n_sca):
                S = scatter_kernel_source(i_sca, i_source,i_los,i_node,i_los,i_node-1)
                Q = calc_node_scattering_obs(i_wl,i_sca,i_los,i_node)
                SQ += matrix_vec_mult_4(S * Q,inat)
            i = flux_idx(i_los,i_node)
            dir_in = source_dir[i_source]
            dir_out = tm.vec3(0.0)
            if i_node == 0:
                dir_out = -instr_view[i_los]
            else:
                dir_out = path_steps[i_node-1,i_los] - path_steps[i_node,i_los]
            dir_out = dir_out / tm.length(dir_out)
            forward_rota = forward_rotation(dir_out, dir_in)
            SQ_transvec = pointwise_product_4(SQ,transvec)
            #flux_b[i] = matrix_vec_mult_4(forward_rota,SQ_transvec)
            flux_bx[i_los,i_node] = matrix_vec_mult_4(forward_rota,SQ_transvec)

    @ti.kernel
    def set_up_flux_A_coup(i_wl: int):
        for i_cc in range(coup_neigh_amt[None]):
            i_coup, j_coup = coup_idx_neighbour[i_cc]
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            k_los, k_node = 0, 0
            if i_coup != j_coup:
                _    , _     , k_los, k_node = node_idx_inv[j_coup]
            else:
                k_los, k_node, _    , _      = node_idx_inv[j_coup]
            trans = calc_scasegment_transmittance(i_wl,i_los,i_node,j_los,j_node)
            SQ = tm.mat4(0)
            Q_tot = 0.0
            dir_in = path_steps[j_node,j_los] - path_steps[i_node,i_los]
            dir_in = dir_in / tm.length(dir_in)
            if tm.length(dir_in) < zero_division_check:
                continue
            dir_out = path_steps[k_node,k_los] - path_steps[j_node,j_los]
            dir_out = dir_out / tm.length(dir_out)
            #if tm.length(dir_out) < zero_division_check:
            #    continue # this is just error-hiding: let's crash if that's the case!!
            if (path_len[j_los]-1) == j_node:
                #SQ = refl_kernel(i_los,i_node,j_los,j_node,k_los,k_node)
                SQ = get_integrated_reflmat(i_coup,dir_out,dir_in)
            SQ += get_integrated_phasemat(i_coup,dir_out)
            for i_sca in range(n_sca):
                #S = scatter_kernel(i_sca, i_los,i_node,j_los,j_node,k_los,k_node)
                Q = calc_node_scattering(i_wl,i_sca,j_los,j_node,k_los,k_node)
                #ph = 1.0 / phase_function(i_sca,dir_in,dir_out)
                #ph = 1.0 # wtf is this Antti of the past?
                #SQ += S * Q * ph
                Q_tot += Q
            if tm.isnan(SQ[0,0]):
                print("AAARGH in A_coup creation: ",i_los,i_node,j_los,j_node,k_los,k_node,phase_function(1,dir_in,dir_out),tm.length(dir_in + dir_out),tm.acos(tm.dot(dir_in,dir_out)))
            Q_tot /= calc_node_extinction(i_wl,j_los,j_node,k_los,k_node)
            SQ *= (1/Q_tot)
            rotated_SQ = rotate_polarization(dir_in,dir_out,SQ)
            #flux_A[n_nodes + i_coup - 1,n_nodes + j_coup - 1] = -(1-trans) * rotated_SQ
            #flux_A[i_coup,j_coup] = -(1-trans) * rotated_SQ# * (2*tm.pi)
            flux_A[i_coup,j_coup] = -(trans) * rotated_SQ# * (2*tm.pi)
            norm_amt_Acoup[i_coup] += 1

    @ti.kernel
    def set_up_flux_b_coup(i_wl: int, i_source: int):
        inat = incident_stokes(i_wl)
        for i_coup in range(scatter_coupling_amt[None]):
            j_los, j_node, k_los, k_node = node_idx_inv[i_coup]
            if j_los == 0 and j_node == 0 and k_los == 0 and k_node == 0:
                continue
            trans = calc_source_transmittance(j_node,i_wl,j_los,i_source)
            transvec = tm.vec4(trans)
            SQ = tm.vec4(0.0)
            if (path_len[j_los]-1) == j_node:
                S = refl_kernel_source(i_source,j_los,j_node,k_los,k_node)
                SQ = matrix_vec_mult_4(S,inat)
            #else:
            for i_sca in range(n_sca):
                S = scatter_kernel_source(i_sca,i_source,j_los,j_node,k_los,k_node)
                Q = calc_node_scattering(i_wl,i_sca,j_los,j_node,k_los,k_node)
                SQ += matrix_vec_mult_4(S * Q,inat)
            dir_in = source_dir[i_source]
            dir_out = path_steps[k_node,k_los] - path_steps[j_node,j_los]
            dir_out = dir_out / tm.length(dir_out)

            forward_rota = forward_rotation(dir_out, dir_in)
            SQ_transvec = pointwise_product_4(SQ,transvec)
            flux_bc[i_coup] = matrix_vec_mult_4(forward_rota,SQ_transvec)
            #flux_b[n_nodes + i_coup - 1] = matrix_vec_mult_4(forward_rota,SQ_transvec)
            #print(n_nodes + i_coup - 1,trans,S,Q)

@ti.kernel
def normalize_flux_A_obs():
    for i_coup in range(scatter_coupling_amt[None]):
        i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
        i = flux_idx(i_los,i_node)
        dist = tm.length(path_steps[j_node,j_los] - path_steps[i_node,i_los])
        not_needed = 0.0
        A = coup_distance_to_radius(dist,not_needed) * 4
        #A = 2.0
        #flux_A[i,n_nodes + i_coup - 1] /= norm_amt_Aobs[i]
        #flux_A[i,n_nodes + i_coup - 1] *= A / (4*tm.pi)
        #flux_A[i,n_nodes + i_coup - 1] *= (2*tm.pi)
        #print('A_obs weight:',A / (2*tm.pi))


        #print(i,trans,S,Q)
    # the obs. fluxes
    # calculate the source contribution at each node
    # accumulate the emission term
    # scattering toward the instrument
    # the coupling fluxes
    # scattering toward the coupled node from the source
    # TODO: löydä linkitykset ekana, ja tallenna ne sitten johonkin tableen.
    # pitää tehdä mahollisimman paljon esilaskentaa kun pitää joka aallonpituudelle
    # rakentaa matriisi erikseen.

#@ti.func
#def normalize_node_fluxes(i_los):
    #for i_los in range(n_los):
        #for i_node in range(path_len[i_los]):


@ti.kernel
def set_up_node_idx_inv():
    ti.loop_config(serialize=True)
    for i_los in range(n_los):
        for i_node in range(path_len[i_los]):
            for j_los in range(n_los):
                for j_node in range(path_len[i_los]):
                    if i_node == j_node and i_los == j_los:
                        continue
                    else:
                        i_coup = node_basis_idx[i_los,i_node,j_los,j_node]
                        node_idx_inv[i_coup] = tm.ivec4(i_los,i_node,j_los,j_node)

@ti.kernel
def set_up_coup_neighbours():
    # now non-parallelized, because this causes problems
    # later on if not done seqeuentially
    ti.loop_config(serialize=True)
    for i_coup in range(scatter_coupling_amt[None]):
        v_i = node_idx_inv[i_coup]
        if v_i[0] == 0 and v_i[1] == 0 and v_i[2] == 0 and v_i[3] == 0:
            #uninitialized
            continue
        i_los, i_node, j_los, j_node = v_i[0],v_i[1],v_i[2],v_i[3]
        for j_coup in range(scatter_coupling_amt[None]):
            v_j = node_idx_inv[j_coup]
            if v_j[0] == 0 and v_j[1] == 0 and v_j[2] == 0 and v_j[3] == 0:
                continue
            if j_los == v_j[0] and j_node == v_j[1] and not (i_coup == 0 and j_coup == 0):
                coup_idx_neighbour[coup_neigh_amt[None]] = tm.ivec2(i_coup,j_coup)
                coup_neigh_amt[None] += 1



@ti.kernel
def normalize_flux_A_coup():
    for i_cc in range(coup_neigh_amt[None]):
        i_coup, j_coup = coup_idx_neighbour[i_cc]
        j_los,j_node,k_los,k_node = node_idx_inv[j_coup]
        dist = tm.length(path_steps[k_node,k_los] - path_steps[j_node,j_los])
        not_needed = 0.0
        A = coup_distance_to_radius(dist,not_needed) * 4
        #A = 2.0
        #flux_A[n_nodes + i_coup - 1,n_nodes + j_coup - 1] /= norm_amt_Acoup[i_coup]
        #flux_A[n_nodes + i_coup - 1,n_nodes + j_coup - 1] /= (norm_amt_Acoup[i_coup]/(4*tm.pi)/1.4142)
        #flux_A[n_nodes + i_coup - 1,n_nodes + j_coup - 1] *= A / (4*tm.pi)
        #flux_A[n_nodes + i_coup - 1,n_nodes + j_coup - 1] *= (2*tm.pi)


@ti.func
def sphere_metric(v1,v2):
    # this returns the great circle distance between v1 and v2, assuming normalzied
    dot = tm.dot(v1,v2)
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    #print(v1,v2,tm.dot(v1,v2),tm.acos(dot))
    return tm.acos(dot)

@ti.func
def weighting(i_los,i_node):
    #weight_V.fill(tm.vec3(0.0)) # the normalized direction vectors
    #weight_w.fill(0.0) # the average distance to other points
    #weight_coup.fill(-1) # if positive, the index of which is this vector merged to. if negative, the amount of how many indices are merged into this

    # yhdistetään flux_A_coup ja flux_A_obs!
    # painot ovat käytännössä samat kummassakin!
    # HUOM! ei ole! flux_A_obs on aina vakiota ja se tiedetään!
    # flux_A_coup
    V_idx = 0
    #note: tän luupin voisi käydä nopeemminkin jotenkin
    for j_los in range(n_los):
        for j_node in range(path_len[j_los]):
            if node_basis_idx[i_los,i_node,j_los,j_node] > 0:
                weight_coup[V_idx,i_los] = node_basis_idx[i_los,i_node,j_los,j_node]
                weight_V[V_idx,i_los] = path_steps[j_node,j_los] - path_steps[i_node,i_los]
                weight_V[V_idx,i_los] /= tm.length(weight_V[V_idx,i_los])
                V_idx += 1
    # if vectors are within this many radians to each other, then
    # they are agglomorated.
    #overlap_epsilon = 0.1 * tm.pi / 180.0

    for idx_v in range(V_idx):
        for idx_u in range(idx_v):
            sphere_dist = sphere_metric(weight_V[idx_v,i_los],weight_V[idx_u,i_los])
            weight_w[idx_v,i_los] += sphere_dist
            weight_w[idx_u,i_los] += sphere_dist

    if V_idx > 0:
        #compute the mean distances
        for idx_v in range(V_idx):
            weight_w[idx_v,i_los] /= V_idx
        # square the means: the idea is to represent an area of some sorts
        for idx_v in range(V_idx):
            weight_w[idx_v,i_los] *= weight_w[idx_v,i_los]

    # the mean distance of pi means that a particular point is directly on the other side of the sphere. Let's grant them the whole half hemisphere, 0.5
    # the mean distance of zero means that all the couplings are toward the same direction. The radiation in there should be divided equally.
    # however, let's give them some minimum weight and scale everything between the zero and pi accordingly.

    minimum_weight = 0.001 # of the total solid angle of the sphere.
    total_weight = 0.0
    k = (0.5 - minimum_weight) / tm.pi

    for idx_v in range(V_idx):
        weight_w[idx_v,i_los] = k * weight_w[idx_v,i_los] + minimum_weight
        total_weight += weight_w[idx_v,i_los]
    for idx_v in range(V_idx):
        weight_w[idx_v,i_los] /= total_weight
        #below: whoops, the 4pi is taken into account in the phase
        #function already
        #weight_w[idx_v,i_los] *= (4*tm.pi)

    i = flux_idx(i_los,i_node)
    for idx_v in range(V_idx):
        i_coup = weight_coup[idx_v,i_los]
        # normalize_flux_A_obs
        flux_A[i, n_nodes + i_coup - 1] *= weight_w[idx_v,i_los]
        # normalize_flux_A_coup
        for i_cc in range(coup_neigh_amt[None]):
            i_coup_, j_coup = coup_idx_neighbour[i_cc]
            if i_coup == i_coup_:
                flux_A[n_nodes + i_coup - 1, n_nodes + j_coup - 1] *= weight_w[idx_v,i_los]

    for idx_v in range(V_idx):
        weight_V[idx_v,i_los] = tm.vec3(0.0)
        weight_w[idx_v,i_los] = 0.0
        weight_coup[idx_v,i_los] = -1




@ti.func
def coup_distance_to_radius(dist,dist_to_minimum):
    #area_at_minimum = 4*tm.pi
    area_at_minimum = 2*tm.pi
    #area_at_minimum = 3*tm.pi/4
    #area_at_minimum = tm.pi/2
    #area_at_minimum = tm.pi/4
    #d = tm.max(dist - dist_to_minimum,)
    # not sure if the above is reasonable.
    # v---this has been found to keep the results somewhat stable
    d = dist
    #print(d)
    #A = area_at_minimum / (d + 1) ** 2
    A = area_at_minimum / (d + 1) ** 2
    #A = area_at_minimum
    #A = area_at_minimum / (d + 1)
    #A = area_at_minimum / (d) ** 2
    radius = A / 4
    # curious, but unproven: with the regular sphere area A, we can obtain the
    # circle radius in sphere_metric by dividing with 4.
    return radius



@ti.func
def populate_indicator(R,coupled_line,minimum_dist,i_thread_or_i_los):
    coup_dist = tm.length(coupled_line)
    coup_radius = coup_distance_to_radius(coup_dist,minimum_dist)
    coupled_line /= coup_dist
    coupled_line_rot = matrix_vec_mult_3(R,coupled_line)
    for i_theta in range(n_phase_theta):
        for i_phi in range(n_phase_phi):
            v = phase_ang_table[i_theta,i_phi]
            sphere_dist = sphere_metric(coupled_line_rot,v)
            if sphere_dist < coup_radius:
                indicator_table[i_thread_or_i_los,i_theta,i_phi] = 1.0

@ti.func
def weight_Acoup(R,minimum_dist,i_coup,j_node,j_los,i_thread):
    for i_theta in range(n_phase_theta):
        for i_phi in range(n_phase_phi):
            indicator_table[i_thread,i_theta,i_phi] = 0.0
    for i_cc in range(coup_neigh_amt[None]):
        _i_coup, j_coup = coup_idx_neighbour[i_cc]
        if _i_coup == i_coup:
            # i_coup and j_coup are now neighbours
            _, _, k_los, k_node = node_idx_inv[j_coup]
            coupled_line = path_steps[k_node,k_los] - path_steps[j_node,j_los]
            populate_indicator(R,coupled_line,minimum_dist,i_thread)
    return calculate_weight(j_node,j_los,i_thread)

@ti.func
def weight_Aobs(R,minimum_dist,i_node,i_los):
    for i_theta in range(n_phase_theta):
        for i_phi in range(n_phase_phi):
            indicator_table[i_los,i_theta,i_phi] = 0.0
    for i_coup in range(scatter_coupling_amt[None]):
        _i_los, _i_node, j_los, j_node = node_idx_inv[i_coup]
        if _i_los == i_los and _i_node == i_node:
            coupled_line = path_steps[j_node,j_los] - path_steps[i_node,i_los]
            populate_indicator(R,coupled_line,minimum_dist,i_los)
    return calculate_weight(i_node,i_los,i_los)

@ti.func
def calculate_weight(j_node,j_los,i_thread_or_i_los):
    total_phase = 0.0
    coupled_phase = 0.0
    for i_pos in range(n_pos):
        basis_contrib = path_basis[j_node,i_pos,j_los]
        if basis_contrib == 0.0:
            continue
        for i_theta in range(n_phase_theta):
            # uncomment this for abstract art
            #print(int(indicator_table[i_theta,0]),int(indicator_table[i_theta,1]),int(indicator_table[i_theta,2]),int(indicator_table[i_theta,3]), int(indicator_table[i_theta,4]), int(indicator_table[i_theta,5]), int(indicator_table[i_theta,6]), int(indicator_table[i_theta,7]), int(indicator_table[i_theta,8]), int(indicator_table[i_theta,9]))
            for i_phi in range(n_phase_phi):
                #print(indicator_table[i_theta,i_phi])
                phase_func_table[i_pos,i_theta,i_phi]
                total_phase += basis_contrib * phase_func_table[i_pos,i_theta,i_phi]
                coupled_phase += indicator_table[i_thread_or_i_los,i_theta,i_phi] * phase_func_table[i_pos,i_theta,i_phi] * basis_contrib
    weight = 0.0
    if total_phase > 0.0:
        weight = coupled_phase / total_phase
    return weight

@ti.func
def lossy_weighting_coup(i_coup,i_thread):
    """
    0. Esilaske keskim. phase function (theta,phi) -gridissä kullekin basis-funktiolle.
    1. Täräyttele pallopinnalle i_coupin jälkimmäisen noden couplaukset.
    2. Määrää kullekin niille pinta-ala sen mukaan kuinka lähellä kyseinen linkitetty noodi on.
        Huom. se max.couplaus voisi olla vaikka 2pi, eli puolipallo kun ollaan täysin kiinni toisessa noodissa
        tai sitten se voisi olla jokin vakio lähimmälle noodille ja siitä kauemmaksi mentäessä pienentyy!
    3. Mäppää pallopinnan pinta-alat samaan gridiin vaihefunktion kanssa indikaattorifunktion muotoon.
    4. Laske indikaattorifunktion ja vaihefunktion tulo.
    5. Tästä on saatu skaalauskerroin jolla i_couppia täytyy skaalata.

    Kysymykset: miten diskretisoidaan gridi?
                miten skaalataan linkin pinta-ala? Oisko lähin se pi?
    """


    i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
    origin = path_steps[j_node,j_los] - path_steps[i_node,i_los]
    origin /= tm.length(origin)
    phase_table_origin = tm.vec3(1.0, 0.0, 0.0)
    R = rotmat(origin,phase_table_origin)
    minimum_dist = tm.inf
    for i_cc in range(coup_neigh_amt[None]):
        # katso tässä luupissa kuosiin määräpainotus?
        # Hei, tämähän käsittely pitää tehdä myös A_obsille!!
        _i_coup, j_coup = coup_idx_neighbour[i_cc]
        if _i_coup == i_coup:
            _, _, k_los, k_node = node_idx_inv[j_coup]
            coupled_line = path_steps[k_node,k_los] - path_steps[j_node,j_los]
            coup_dist = tm.length(coupled_line)
            if coup_dist < minimum_dist:
                minimum_dist = coup_dist

    weight = weight_Acoup(R,minimum_dist,i_coup,j_node,j_los,i_thread)

    #print(i_coup,weight)
    if weight == 0.0:
        weight = 1.0
    for i_cc in range(coup_neigh_amt[None]):
        _i_coup, j_coup = coup_idx_neighbour[i_cc]
        if _i_coup == i_coup:
            flux_A[n_nodes + i_coup - 1,n_nodes + j_coup - 1] *= weight

@ti.func
def lossy_weighting_obs(i_node,i_los,i_source):
    origin = source_dir[i_source]
    origin /= tm.length(origin)
    phase_table_origin = tm.vec3(1.0, 0.0, 0.0)
    R = rotmat(origin,phase_table_origin)
    minimum_dist = tm.inf
    for i_coup in range(scatter_coupling_amt[None]):
        _i_los, _i_node, j_los, j_node = node_idx_inv[i_coup]
        if _i_los == i_los and _i_node == i_node:
            coupled_line = path_steps[j_node,j_los] - path_steps[i_node,i_los]
            coup_dist = tm.length(coupled_line)
            if coup_dist < minimum_dist:
                minimum_dist = coup_dist

    weight = weight_Aobs(R,minimum_dist,i_node,i_los)
    if weight == 0.0:
        weight = 1.0

    i_flux = flux_idx(i_los,i_node)
    for i_coup in range(scatter_coupling_amt[None]):
        _i_los, _i_node, j_los, j_node = node_idx_inv[i_coup]
        if _i_los == i_los and _i_node == i_node:
            flux_A[i_flux, n_nodes + i_coup - 1] *= weight

@ti.kernel
def test_lossy_weighting_coup():
    ti.loop_config(serialize=True)
    for i_thread in range(n_lossy_threads):
        start_idx = i_thread * scatter_coupling_amt[None] // n_lossy_threads
        end_idx = (i_thread+1) * scatter_coupling_amt[None] // n_lossy_threads
        for i_coup in range(start_idx,end_idx):
            lossy_weighting_coup(i_coup,i_thread)

@ti.kernel
def test_lossy_weighting_obs(i_source:int):
    ti.loop_config(serialize=True)
    for i_los in range(n_los):
        for i_node in range(path_len[i_los]):
            lossy_weighting_obs(i_node,i_los,i_source)

@ti.kernel
def test_weighting():
    for i_los in range(n_los):
        for i_node in range(path_len[i_los]):
            weighting(i_los,i_node)

@ti.kernel
def set_up_flux_matrix_obs(i_wl: int,i_source: int):
    set_up_flux_A_obs(i_wl)
    for i_los in range(n_los):
        set_up_flux_b_obs(i_wl,i_los,i_source)

@ti.func
def populate_LU():
    for i_cc in range(coup_neigh_amt[None]):
        i, j = coup_idx_neighbour[i_cc]
        flux_S_LU[n_nodes + i - 1,n_nodes + j - 1] = flux_A[n_nodes + i - 1,n_nodes + j - 1]

@ti.func
def LU_decomp():
    """
    With kinds regards to Arttu Väisänen

    Inspiration from https://en.wikipedia.org/wiki/LU_decomposition MATLAB section
    function LU = LUDecompDoolittle(A)
        n = length(A);
        LU = A;
        % decomposition of matrix, Doolittle's Method
        for i = 1:1:n
            for j = 1:(i - 1)
                LU(i,j) = (LU(i,j) - LU(i,1:(j - 1))*LU(1:(j - 1),j)) / LU(j,j);
            end
            j = i:n;
            LU(i,j) = LU(i,j) - LU(i,1:(i - 1))*LU(1:(i - 1),j);
        end
        %LU = L+U-I
    end
    """
    N = scatter_coupling_amt[None]
    for i in range(N):
        for j in range(N):
            flux_S_LU[i,j] = flux_A[n_nodes + i - 1,n_nodes + j - 1]
        flux_S_y[i] = flux_b[n_nodes + i - 1]
        #print(i,flux_S_LU[i,i])
    for i in range(N):
        for j in range(i):
            dotprod = 0.0
            #    for idx in range(4):
            #        for jdx in range(4):
            #            temp_mat_a[None][idx,jdx] = 0.0
            for k in range(j):
                dotprod += flux_S_LU[i,k] * flux_S_LU[k,j]
                #    matrix_mult(temp_mat_b[None],flux_A[i,k],flux_A[k,j])
                #    temp_mat_a += temp_mat_b[None]
            flux_S_LU[i,j] = (flux_S_LU[i,j] - dotprod) / flux_S_LU[j,j];
            #    matrix_inverse(temp_mat_b[None],flux_A[j,j])
            #    matrix_mult(flux_A[i,j],(flux_A[i,j] - temp_mat_a[None]),temp_mat_b[None])
        for j in range(i,N):
            dotprod = 0.0
            for k in range(i):
                dotprod += flux_S_LU[i,k] * flux_S_LU[k,j]
            flux_S_LU[i,j] = flux_S_LU[i,j] - dotprod
            # now the matrix LU should be L+U-I
        #print(i,flux_S_LU[i,i])
if polariz_mode == 1:
    @ti.func
    def solve_flux_system():
        N = active_row_len[None]

        """

        TODO:
            The matrix in question is of form
            A = [P Q
                 R S]
            where P = 1 and R = 0, which has an inverse of form
            inv(A) = [1, -Q*inv(S)
                      0,    inv(S)]
            so we only need to compute inv(S) and obtain the solution from there


        function x = SolveLinearSystem(LU, B)
            n = length(LU);
            y = zeros(size(B));
            % find solution of Ly = B
            for i = 1:n
                y(i,:) = B(i,:) - LU(i,1:i)*y(1:i,:);
            end
            % find solution of Ux = y
            x = zeros(size(B));
            for i = n:(-1):1
                x(i,:) = ( y(i,:) - LU(i,(i + 1):n)*x((i + 1):n,:))/LU(i, i);
            end
        end
        """
        LU_decompostion_solver = True
        N = scatter_coupling_amt[None]
        if LU_decompostion_solver:
            for i in range(N):
                dotprod = 0.0
                for j in range(i):
                    dotprod += flux_S_LU[i,j] * flux_S_y[j]
                flux_S_y[i] = flux_b[n_nodes + i - 1] - dotprod
            for _i in range(-N+1,1):
                i = -_i
                dotprod = 0.0

                for j in range(i,N):
                    dotprod += flux_S_LU[i,j] * flux_x[n_nodes + j - 1]
                flux_x[n_nodes + i - 1] = (flux_S_y[i] - dotprod) / flux_S_LU[i,i]
        else:
            # This is an old Gaussian elimination with pivoting solution, very slow
            for _i_pivot in range(N):
                i_pivot = active_rows[_i_pivot]
                for _i in range(i_pivot+1,N):
                    i = active_rows[_i]
                    scaling_factor = flux_A[i,i_pivot] / flux_A[i_pivot,i_pivot]
                    for _j in range(i_pivot,N):
                        j = active_rows[_j]
                        flux_A[i,j] -= scaling_factor * flux_A[i_pivot,j]
                    flux_b[i] -= scaling_factor * flux_b[i_pivot]
            for _i in range(-N+1,1):
                i = active_rows[-_i] # taichi loops do not handle the three argument range, this is a workaround for range(N-1,-1,-1)
                flux_x[i] = flux_b[i]
                for _j in range(i + 1, N):
                    j = active_rows[_j]
                    flux_x[i] -= flux_A[i,j] * flux_x[j]
                flux_x[i] /= flux_A[i,i]

elif polariz_mode == 4:
    @ti.func
    def solve_flux_system(scattering_mode):
        N = active_row_len[None]
        if scattering_mode == 1:
            # single-scattering
            for _i in range(N):
                i = active_rows[_i]
                flux_x[i] = flux_b[i]
        else:
            for _i in range(N):
                i = active_rows[_i]
                cumvec = tm.vec4(0)
                for _j in range(i):
                    j = active_rows[_j]
                    cumvec += matrix_vec_mult_4(flux_A[i,j], flux_S_y[j])
                flux_S_y[i] = flux_b[i] - cumvec
            for _i in range(-N+1,1):
                i = active_rows[-_i]
                cumvec = tm.vec4(0)
                for _j in range(i,N):
                    j = active_rows[_j]
                    cumvec += matrix_vec_mult_4(flux_A[i,j], flux_x[j])

                #inv_A_ii = matrix_inverse(flux_A[i,i])
                inv_A_ii = flux_A[i,i]
                #print(inv_A_ii)
                #print(inv_A_ii, (flux_S_y[i] - cumvec))
                temp_vec = flux_S_y[i] - cumvec
                #print(flux_A[i,i])
                #print(temp_vec)
                flux_x[i] = matrix_vec_mult_4(inv_A_ii, temp_vec)

def solve_fluxes_numpy():
    A_coup = flux_S_LU.to_numpy()
    N = scatter_coupling_amt[None] * polariz_mode
    if False:
        import netCDF4
        with netCDF4.Dataset('A_coup.nc','w') as ds:
            ds.createDimension('dummy1',A_coup[:N,:N].shape[0])
            ds.createDimension('dummy2',A_coup[:N,:N].shape[1])
            ds.createVariable('A_coup','f4',('dummy1','dummy2'))
            ds['A_coup'][:] = A_coup[:N,:N]
    b_coup = flux_S_y.to_numpy()
    np.savetxt('b_coup.dat',b_coup)
    x_coup = np.zeros_like(b_coup)
    x_coup[:N] = np.linalg.solve(A_coup[:N,:N]+np.eye(N),b_coup[:N])
    #x_coup[:N] = np.linalg.solve(A_coup[:N,:N],b_coup[:N])
    #negative_fluxes = np.sum(x_coup < 0)
    #print(x_coup)
    #if negative_fluxes > 0:
    #    print("Warning: negative fluxes: %d" % negative_fluxes)
    flux_S_y.from_numpy(x_coup)

@ti.kernel
def test_LU_decomp_solver():
    # test_b = [1,2,5]
    # test_A = [[4,3,2],
    #           [8,7,9],
    #           [1,0,-1]]
    # result should be:
    # x = [7.111111, -10.555555, 2.111111]
    flux_b[0] = 1
    flux_b[1] = 2
    flux_b[2] = 5

    flux_S_LU[0,0] = 4
    flux_S_LU[0,1] = 3
    flux_S_LU[0,2] = 2
    flux_S_LU[1,0] = 8
    flux_S_LU[1,1] = 7
    flux_S_LU[1,2] = 9
    flux_S_LU[2,0] = 1
    flux_S_LU[2,1] = 0
    flux_S_LU[2,2] = -1

    active_rows[0] = 0
    active_rows[1] = 1
    active_rows[2] = 2

    active_row_len[None] = 3

    LU_decomp()
    solve_flux_system(2)


@ti.kernel
def set_up_scattering():
    scatter_coupling_amt[None] = 0 # 0.03333081

if polariz_mode == 1:
    @ti.kernel
    def add_diagonals():
        for _ in range(1):
            for i in range(flux_b.shape[0]):
                if flux_b[i] != 0.0:
                    active_rows[active_row_len[None]] = i
                    active_row_len[None] += 1
            for _i in range(active_row_len[None]):
                i = active_rows[_i]
                flux_A[i,i] = 1.0
        #N = scatter_coupling_amt[None]
        #for _i in range(N):
        #    i = n_nodes + _i - 1
        #    flux_A[i,i] = 1.0

elif polariz_mode == 4:
    @ti.kernel
    def add_diagonals():
        for _ in range(1):
            for i in range(flux_b.shape[0]):
                if flux_b[i][0] != 0.0 or flux_b[i][1] != 0.0 or flux_b[i][2] != 0.0 or flux_b[i][3] != 0.0:
                    active_rows[active_row_len[None]] = i
                    active_row_len[None] += 1
        for i in range(scatter_coupling_amt[None]):
            eye = tm.mat4(0)
            eye[0,0] = 1.0
            eye[1,1] = 1.0
            eye[2,2] = 1.0
            eye[3,3] = 1.0
            #flux_A[i + n_nodes - 1,i + n_nodes - 1] = eye
            flux_A[i + n_nodes - 1,i + n_nodes - 1][0,0] = 1.0
            flux_A[i + n_nodes - 1,i + n_nodes - 1][1,1] = 1.0
            flux_A[i + n_nodes - 1,i + n_nodes - 1][2,2] = 1.0
            flux_A[i + n_nodes - 1,i + n_nodes - 1][3,3] = 1.0
            #print(i)
"""
#TODO: Taichi bug report. For some reason this function populates off-diagonals too
# taichi310 environment stack for this
elif polariz_mode == 4:
    @ti.kernel
    def add_diagonals():
        for _ in range(1):
            for i in range(flux_b.shape[0]):
                if flux_b[i][0] != 0.0 or flux_b[i][1] != 0.0 or flux_b[i][2] != 0.0 or flux_b[i][3] != 0.0:
                    active_rows[active_row_len[None]] = i
                    active_row_len[None] += 1
            for _i in range(active_row_len[None]):
                i = active_rows[_i]
                eye = tm.mat4(0)
                eye[0,0] = 1.0
                eye[1,1] = 1.0
                eye[2,2] = 1.0
                eye[3,3] = 1.0
                flux_A[i,i] = eye
                #print(i)
"""
@ti.func
def populate_x_obs():
    N = active_row_len[None]
    # single-scattering
    for i_los in range(n_los):
        for i_node in range(path_len[i_los]):
            flux_x[i_los,i_node] = flux_bx[i_los,i_node]
        #print(i,flux_x[i])

if polariz_mode == 1:
    @ti.func
    def sum_ss_ms_flux():
        M = scatter_coupling_amt[None]
        for i_los in range(n_los):
            for i_node in range(path_len[i_los]):
                #i = flux_idx(i_los,i_node)
                sumflux = 0.0
                for i_coup in range(M):
                    sumflux += flux_O[i_los,i_node,i_coup] * flux_c[i_coup]
                flux_x[i_los,i_node] += sumflux

elif polariz_mode == 4:
    @ti.func
    def sum_ss_ms_flux():
        M = scatter_coupling_amt[None]
        for i_los in range(n_los):
            for i_node in range(path_len[i_los]):
                #i = flux_idx(i_los,i_node)
                sumflux = tm.vec4(0.0)
                for i_coup in range(M):
                    sumflux += matrix_vec_mult_4(flux_O[i_los,i_node,i_coup],flux_c[i_coup])
                flux_x[i_los,i_node] += sumflux

if polariz_mode == 1:
    @ti.kernel
    def solver_preprocess(scattering_mode: int):
        populate_x_obs()
        if scattering_mode > 1:
            N = scatter_coupling_amt[None]
            for i_coup in range(N):
                for j_coup in range(N):
                    flux_S_LU[i_coup,j_coup] = flux_A[i_coup,j_coup]
                flux_S_y[i_coup] = flux_bc[i_coup]

elif polariz_mode == 4:
    @ti.kernel
    def solver_preprocess(scattering_mode: int):
        populate_x_obs()
        if scattering_mode > 1:
            N = scatter_coupling_amt[None]
            for _i in range(N):
                i = 4 * _i
                for _j in range(N):
                    j = 4 * _j
                    flux_S_LU[i+0,j+0] = flux_A[_i,_j][0,0]
                    flux_S_LU[i+1,j+0] = flux_A[_i,_j][1,0]
                    flux_S_LU[i+2,j+0] = flux_A[_i,_j][2,0]
                    flux_S_LU[i+3,j+0] = flux_A[_i,_j][3,0]
                    flux_S_LU[i+0,j+1] = flux_A[_i,_j][0,1]
                    flux_S_LU[i+1,j+1] = flux_A[_i,_j][1,1]
                    flux_S_LU[i+2,j+1] = flux_A[_i,_j][2,1]
                    flux_S_LU[i+3,j+1] = flux_A[_i,_j][3,1]
                    flux_S_LU[i+0,j+2] = flux_A[_i,_j][0,2]
                    flux_S_LU[i+1,j+2] = flux_A[_i,_j][1,2]
                    flux_S_LU[i+2,j+2] = flux_A[_i,_j][2,2]
                    flux_S_LU[i+3,j+2] = flux_A[_i,_j][3,2]
                    flux_S_LU[i+0,j+3] = flux_A[_i,_j][0,3]
                    flux_S_LU[i+1,j+3] = flux_A[_i,_j][1,3]
                    flux_S_LU[i+2,j+3] = flux_A[_i,_j][2,3]
                    flux_S_LU[i+3,j+3] = flux_A[_i,_j][3,3]
                    #if flux_A[n_nodes + _i - 1,n_nodes + _j - 1][0,0] == 1.0:
                    #    print(i,j,flux_A[n_nodes + _i - 1,n_nodes + _j - 1])
                flux_S_y[i+0] = flux_bc[_i][0]
                flux_S_y[i+1] = flux_bc[_i][1]
                flux_S_y[i+2] = flux_bc[_i][2]
                flux_S_y[i+3] = flux_bc[_i][3]

if polariz_mode == 1:
    @ti.kernel
    def solver_postprocess():
        N = scatter_coupling_amt[None]
        for i_coup in range(N):
            flux_c[i_coup] = flux_S_y[i_coup]

elif polariz_mode == 4:
    @ti.kernel
    def solver_postprocess():
        N = scatter_coupling_amt[None]
        for _i in range(N):
            i = 4 * _i
            flux_c[_i][0] = flux_S_y[i+0]
            flux_c[_i][1] = flux_S_y[i+1]
            flux_c[_i][2] = flux_S_y[i+2]
            flux_c[_i][3] = flux_S_y[i+3]

@ti.kernel
def sum_fluxes():
    # flux_x_obs = flux_b_obs + S_obs @ inv(I - S_coup) @ flux_b_coup

    #populate_LU()

    #there's no need to compute the decomposition beforehand, but it is done
    #in conjunction of the system solving
    #ti.loop_config(serialize=True)
    #print('scattering_mode:',scattering_mode)
    #ti.loop_config(serialize=True)
    #LU_decomp()

    #solve_flux_system()

    sum_ss_ms_flux()

@ti.func
def find_arr_min_idx(arr,arrlen):
    i_min = 0
    minval = tm.inf
    for i in range(arrlen):
        if arr[i] < minval:
            i_min = i
            minval = arr[i]
        elif arr[i] == 0.0:
            i_min = i
            minval = arr[i]
            break
    return i_min, minval

@ti.func
def find_max_couplings():
    current_min_value = 0.0
    current_min_idx = 0
    total_coupling = 0.0
    ti.loop_config(serialize=True)
    for i_los in range(n_los):
        for i_node in range(path_len[i_los]):
            for j_los in range(n_los):
                for j_node in range(path_len[j_los]):
                    if i_los == j_los and i_node == j_node:
                        continue
                    if check_shadow(i_node,i_los,j_node,j_los):
                        #print(i_los,i_node,' in shadow of ',j_los,j_node)
                        #coupling = -coupling
                        continue
                    coupling, _ = scatter_coupling(i_node,i_los,j_node,j_los)
                    #print(i_los,i_node,j_los,j_node,coupling)
                    total_coupling += coupling
                    if coupling > current_min_value:
                        node_scattering_table[current_min_idx] = tm.ivec4(i_node,i_los,j_node,j_los)
                        node_scattering_table_value[current_min_idx] = coupling
                        current_min_idx, current_min_value = find_arr_min_idx(node_scattering_table_value, max_scatter_couplings)
    return total_coupling, current_min_value

@ti.kernel
def test_coupling_matrix(coupling_coeff: float):
    # non-parallelized because some race conditions
    total_coupling, min_coupling = find_max_couplings()
    create_coupling_matrix(coupling_coeff,total_coupling)

@ti.func
def flux_idx(i_los,i_node):
    return i_los * max_path_len + i_node

if polariz_mode == 1:
    @ti.kernel
    def cumulate_path_result(i_wl: int):
        for i_los in range(n_los):
            for i_node in range(1,path_len[i_los]):
                transmittances[i_los,i_wl] += calc_path_transmittance(i_node,i_wl,i_los) * flux_x[i_los,i_node]

elif polariz_mode == 4:
    @ti.kernel
    def cumulate_path_result(i_wl: int):
        for i_los in range(n_los):
            for i_node in range(1,path_len[i_los]):
                transmittances[i_los,i_wl] += calc_path_transmittance(i_node,i_wl,i_los) * flux_x[i_los,i_node]

@ti.kernel
def test_scatter_basis():
    for i_los in instr_pos:
        scatter_basis_contrib(i_los)
