import taichi as ti
import taichi.math as tm
import numpy as np

#ti.init(debug=True)
#ti.init(arch=ti.cpu,debug=False)
ti.init(arch=ti.cpu,debug=True)
#ti.init(arch=ti.cpu,offline_cache=False,debug=True)

#ti.init(arch=ti.amdgpu)#,debug=True)

"""
TODO:
Stoppers
 - Tänne tarvitaan semmoinen vastaavanlainen rakenne kuten medium basis -hässäkässä
 eli otetaan n_stoppers ja sitten parametrit yms.
Parallelisaatio
 - parallelisaatiossa apufieldit nollataan, pitäisi olla joku parempi ratkaisu tähän
 - tarkista että stopperit ja mediumit ovat yhtäpitävät!

Mysteereitä
 miten toteuttaa sopivat välimuuttujat?

 Jos tehdään sirontagraafi, niin sen voisi sitten jotenkin analysoida ja hajauttaa
 erilliset osat! Tällä tavoin voisi laskennan toteuttaa useammassa vaiheessa eri
 matriisin osa-alueille.
"""
@ti.dataclass
class Stopper_flat:
    center: tm.vec3
    up_vector_flat: tm.vec3
    size_parameter: ti.f32

    @ti.func
    def up(self, r):
        return self.up_vector_flat

    @ti.func
    def value(self,r):
        return tm.dot((r - self.center),self.up_vector_flat)

    @ti.func
    def param(self,r):
        pass


@ti.dataclass
class Stopper_sphr_out:
    center: tm.vec3
    size_parameter: ti.f32

    @ti.func
    def up(self, r):
        return tm.normalize(r)

    @ti.func
    def value(self,r):
        return tm.length(r) - self.size_parameter

    @ti.func
    def param(self,r):
        pass

@ti.dataclass
class Stopper_sphr_in:
    center: tm.vec3
    size_parameter: ti.f32

    @ti.func
    def up(self, r):
        return -tm.normalize(r)

    @ti.func
    def value(self,r):
        return self.size_parameter - tm.length(r)

    @ti.func
    def param(self,r):
        pass

def create_stopper(type, center, param_or_vec):
    if type == 0:
        return Stopper_flat(center, param_or_vec, center[2])
    elif type == 1:
        return Stopper_sphr_out(center, param_or_vec)
    elif type == 2:
        return Stopper_sphr_in(center, param_or_vec)



# well, well, well. If it aint the consequences of my actions?
# Curse you, hardcoded boundaries!!!
if False:
    stopper_inner = create_stopper(1, tm.vec3(0,0,0), 6371.0 ** 2)
    stopper_outer = create_stopper(2, tm.vec3(0,0,0), 6421.0 ** 2)

if boundary['shape'][0] == 0:
    stopper_inner = create_stopper(0, tm.vec3(0,0,0), tm.vec3(0,0,1))
    stopper_outer = create_stopper(0, tm.vec3(0,0,boundary['parameter'][1]), tm.vec3(0,0,-1))
    spherical = False
elif boundary['shape'][1] == 1:
    stopper_inner = create_stopper(1, tm.vec3(0,0,0), boundary['parameter'][0])
    stopper_outer = create_stopper(2, tm.vec3(0,0,0), boundary['parameter'][1])
    spherical = True
print("boundary_vals",boundary['parameter'][0],boundary['parameter'][1])

stop = create_stopper(1, tm.vec3(0,0,0), 1.0)
@ti.kernel
def test_stopper():
    print(stop.value(tm.vec3(1,10,0)))
#test_stopper()

# In the end we get bunch of stoppers and they all should be greater
# than zero for us to be in the medium.

# Fieldit on niitä sparseja!!

# kirjoita ti.funktio, joka ottaa R3-pisteen, N-arrayn R3-pisteitä (kantafunktioiden lokaatiot)
# N-arrayn inttejä, jotka kertoo että millainen kantafunktio on kyseessä,
# N-array floatteja, jotka on kantafunktioiden parametrit
# ulos tulee: N-array floatteja, jotka meinaa että kuinka suuri kontribuutio kutakin
# kantafunktiota tulee kuhunkin pisteeseen.

"""
# vec3 is a built-in vector type suppied in the `taichi.math` module
vec3 = ti.math.vec3
n = 10
# Declares a struct comprising three vectors and one floating-point number
particle = ti.types.struct(
  pos=vec3, vel=vec3, acc=vec3, mass=float,
)
# Declares a 1D field of the struct particle by calling field()
particle_field = particle.field(shape=(n,))
"""

@ti.func
def find_nearest_idx_floor(lin_r,arr,start_idx,end_idx):
    found = False
    in_domain = True
    if lin_r < arr[start_idx] or lin_r > arr[end_idx]:
        in_domain = False
    lower_limit = 0
    if in_domain:
        search_len = end_idx - start_idx
        upper_limit = search_len
        mean_idx = search_len // 2
        while search_len > 1:
            mean_idx = (upper_limit + lower_limit) // 2
            val = arr[mean_idx + start_idx]
            if val > lin_r:
                upper_limit = mean_idx
            else:
                lower_limit = mean_idx
            search_len = upper_limit - lower_limit
    else:
        # the point is that the find_nearest_idx_floor returns -1
        # i.e. the point is outside the array
        # this weird set up is because taichi functions allow at most one return statement :)
        lower_limit = -start_idx - 1
    return lower_limit + start_idx

instr_pos = ti.Vector.field(n=3, dtype=ti.f32,shape=(n_los,))
instr_view = ti.Vector.field(n=3, dtype=ti.f32,shape=(n_los,))

basis_pos = ti.Vector.field(n=3, dtype=ti.f32, shape=(n_pos,))
basis_lin = ti.field(dtype=ti.f32, shape=(n_pos,))
basis_param = ti.field(dtype=ti.f32, shape=(n_pos,))
basis_weights = ti.field(dtype=ti.f32, shape=(n_pos,n_los))
basis_mid = ti.field(dtype=ti.f32, shape=(n_pos,n_los))
basis_kink = ti.field(dtype=ti.f32, shape=(n_pos,n_los))
basis_end = ti.field(dtype=ti.f32, shape=(n_pos,n_los))

basis_type_index = ti.field(dtype=ti.i32, shape=(n_pos,))
basis_type = ti.field(dtype=ti.i32, shape=(3,2))
# for each of the type functions (the first dimension)
# the start and end indices of corresponding types
source_dir = ti.Vector.field(n=3, dtype=ti.f32,shape=(n_source,))
source_pos = ti.Vector.field(n=3, dtype=ti.f32,shape=(n_source,))


path_steps = ti.Vector.field(n=3, dtype=ti.f32)
ps_block_ = ti.root.bitmasked(ti.i, (max_path_len))
ps_block = ps_block_.pointer(ti.j, (n_los,))
ps_block.place(path_steps)

path_steps_linkscale = ti.Vector.field(n=3, dtype=ti.f32)
psl_block_ = ti.root.bitmasked(ti.i, (max_path_len))
psl_block = psl_block_.pointer(ti.j, (n_los,))
psl_block.place(path_steps_linkscale)


path_len = ti.field(dtype=ti.i32, shape=(n_los,))

#x = ti.Vector.field(n=3,dtype=ti.f32)
#block = ti.root.pointer(ti.ij, (4,4))
#block.place(x)

path_basis = ti.field(ti.f32)
pb_block = ti.root.bitmasked(ti.ijk, (max_path_len,n_pos,n_los))
pb_block.place(path_basis)

source_basis = ti.field(ti.f32)
sb_block = ti.root.bitmasked(ti.ijkl, (max_path_len,n_pos,n_los,n_source))
sb_block.place(source_basis)

node_dist = ti.field(ti.f32)
los1_block = ti.root.pointer(ti.i, (n_los,))
path1_block = los1_block.bitmasked(ti.j, (max_path_len,))
los2_block = path1_block.pointer(ti.k, (n_los,))
path2_block = los2_block.bitmasked(ti.l, (max_path_len,))
path2_block.place(node_dist)

#tr_block = ti.root.bitmasked(ti.ijk, (n_los,n_los,10))
#tr_block.place(transmittance)
#spec_block = path2_block.dense(ti.i,(n_wl,))
#spec_block.place(transmittance)
if polariz_mode == 1:
    transmittances = ti.field(dtype=ti.f32, shape=(n_los,n_wl))
elif polariz_mode == 4:
    transmittances = ti.Vector.field(n=4, dtype=ti.f32, shape=(n_los,n_wl))

#node_partial_trans = ti.field(ti.f32, shape=(n_los,n_wl))
#ti.root.bitmasked(ti)

extinction = ti.field(dtype=ti.f32, shape=(n_pos,n_wl))

scattering = ti.field(dtype=ti.f32, shape=(n_sca,n_pos,n_wl))
avg_scattering = ti.field(dtype=ti.f32, shape=(n_pos,))
boundary_refl_param = ti.field(dtype=ti.f32, shape=(n_refl_param,n_boundary))
scattering_atmos_max = ti.field(ti.f32, shape=())

avg_scatter_mat = ti.field(ti.f32)
asm_block_ = ti.root.bitmasked(ti.i, (max_path_len,))
asm_block = asm_block_.pointer(ti.j, (n_los,))
asm_block.place(avg_scatter_mat)

@ti.func
def append_path(basis,new_basis,steps,new_step,len_path,i_los,first):
    #for i in range(n_pos):
        #basis[len_path[i_los],i,i_los] = new_basis[i,i_los]
    steps[len_path[i_los],i_los] = new_step
    len_path[i_los] += 1
    if not first:
        dist = tm.length(steps[len_path[i_los] - 2,i_los] - steps[len_path[i_los] - 1,i_los])
        node_dist[i_los,len_path[i_los] - 2,i_los,len_path[i_los] - 1] = dist

@ti.func
def clear_arr(arr,i_los):
    for i in range(arr.shape[0]):
        arr[i,i_los] = 0.0

def populate_boundary_param(boundary):
    refl_params = boundary['reflection_kernel_parameter']
    for i in range(n_refl_param):
        for j in range(n_boundary):
            boundary_refl_param[i,j] = refl_params[i,j]
            #print(boundary_refl_param[i,j])

def populate_instr_definition(instr_pos,instr_view,instrument):
    ip = instrument['position']
    iv = instrument['view_vector']
    for i in range(instrument['position'].shape[0]):
        instr_pos[i] = tm.vec3(ip[i,0],ip[i,1],ip[i,2])
        instr_view[i] = tm.vec3(iv[i,0],iv[i,1],iv[i,2])

def populate_basis_definition(basis_pos,basis_lin,basis_param,basis_type,medium):
    mp = medium['position']
    for i in range(mp.shape[0]):
        basis_pos[i] = tm.vec3(mp[i,0],mp[i,1],mp[i,2])
        basis_param[i] = medium['interpolation_parameter'][i]
    populate_basis_type(basis_type,medium)
    for j in range(basis_type.shape[0]):
        #print(basis_type[j,0],basis_type[j,1])
        for i in range(basis_type[j,0],basis_type[j,1]):
            if j == 0:
                basis_lin[i] = basis_pos[i][2]
            elif j == 1:
                basis_lin[i] = np.linalg.norm(basis_pos[i])
                #print(i,basis_lin[i])
    print(basis_lin)

def populate_basis_type(basis_type,medium):
    fun_types = medium['interpolation_function']
    for i in range(n_pos):
        basis_type_index[i] = medium['interpolation_function'][i]
    found = np.zeros((basis_type.shape[0],))
    for i in range(fun_types.size):
        for j in range(basis_type.shape[0]):
            if fun_types[i] == j and not found[j]:
                found[j] = 1
                basis_type[fun_types[i],0] = i
            basis_type[fun_types[i],1] = i+1
    print(basis_type)

def populate_extinction_scattering(extinction, scattering, medium):
    cm_in_km = 1e5
    n_abs = medium['absorber'].shape[1]
    tau_abs = np.zeros((n_pos,n_wl),dtype=np.single)
    tmp_sca = np.zeros((n_pos,n_wl),dtype=np.single)
    tau_sca = np.zeros((n_sca,n_pos,n_wl),dtype=np.single)
    tau_ext = np.zeros((n_pos,n_wl),dtype=np.single)

    for idx_pos in range(n_pos):
        for idx_abs in range(n_abs):
            tau_abs[idx_pos,:] += cm_in_km * medium['absorber'][idx_pos,idx_abs] * medium['absorbing_cross_section'][idx_pos,:,idx_abs]
        for idx_sca in range(n_sca):
            pos_sca_tau = cm_in_km * medium['scatterer'][idx_pos,idx_sca] * medium['scattering_cross_section'][idx_pos,:,idx_sca]
            tmp_sca[idx_pos,:] += pos_sca_tau
            avg_scattering[idx_pos] += np.mean(pos_sca_tau)
            tau_sca[idx_sca,idx_pos,:] = pos_sca_tau
        if avg_scattering[idx_pos] > scattering_atmos_max[None]:
            scattering_atmos_max[None] = avg_scattering[idx_pos]
        #print(idx_pos,avg_scattering[idx_pos])
        tau_ext[idx_pos,:] = tmp_sca[idx_pos,:] + tau_abs[idx_pos,:]
    scattering.from_numpy(tau_sca)
    sca = scattering.to_numpy()
    #if sca.shape[0] > 1:
    #    np.savetxt('scatter.dat',sca[1,:,:])
    extinction.from_numpy(tau_ext)

def populate_source(source):
    sp = source['position']
    sid = source['incident_direction']
    for i in range(n_source):
        source_pos[i] = tm.vec3(sp[i,0],sp[i,1],sp[i,2])
        source_dir[i] = tm.vec3(sid[i,0],sid[i,1],sid[i,2])

@ti.func
def interpolate_layer_basis(r,weights,pos,lin,start_idx,end_idx,i_los):
    # assume z-axis is the upward direction
    lin_r = r[2]
    idx = find_nearest_idx_floor(lin_r,lin,start_idx,end_idx-1)

    if idx != -1:
        gap_length = lin[idx+1] - lin[idx]
        weights[idx,i_los] = (lin[idx+1] - lin_r) / gap_length
        weights[idx + 1,i_los] = (lin_r - lin[idx]) / gap_length
        #print(lin_r,idx,start_idx,end_idx,weights[idx,i_los],weights[idx+1,i_los])

@ti.func
def interpolate_sphr_basis(r,weights,pos,lin,start_idx,end_idx,i_los):
    # the
    center = tm.vec3(0.0, 0.0, 0.0)
    lin_r = tm.length(r-center)
    idx = find_nearest_idx_floor(lin_r,lin,start_idx,end_idx-1)
    # at this point, lin_r should be found from between idx and idx+1 in self.lin
    # ASSUMING LINEARITY BETWEEN THE POINTS
    if idx != -1:
        gap_length = lin[idx+1] - lin[idx]
        weights[idx,i_los] = (lin[idx+1] - lin_r) / gap_length
        weights[idx + 1,i_los] = (lin_r - lin[idx]) / gap_length
    #print((lin[idx+1] - lin_r) / gap_length, (lin_r - lin[idx]) / gap_length)

@ti.func
def interpolate_gauss_basis(r,weights,pos,std,start_idx,end_idx,i_los):
    max_dist = 2.0 # multiplier of standard deviation; how far can we obtain
    # results
    for fun_idx in range(start_idx,end_idx):
        fun_dist = tm.length(r - pos[fun_idx])
        if fun_dist < (max_dist * std[fun_idx]):
            var = std[fun_idx] ** 2
            exp = -fun_dist ** 2 / var
            div = ti.sqrt((2 * tm.pi) ** 3 * var)
            val = ti.exp(exp) / div
            weights[fun_idx,i_los] = val
        else:
            weights[fun_idx,i_los] = 0.0

@ti.func
def basis_contrib(r,weights,basis_type,pos,lin,param,i_los):
    for b_idx in range(basis_type.shape[0]):
        start_idx = basis_type[b_idx,0]
        end_idx = basis_type[b_idx,1]
        if end_idx - start_idx == 0:
            continue
        if b_idx == 0:
            interpolate_layer_basis(r,weights,pos,lin,start_idx,end_idx,i_los)
        elif b_idx == 1:
            interpolate_sphr_basis(r,weights,pos,lin,start_idx,end_idx,i_los)
        elif b_idx == 2:
            interpolate_gauss_basis(r,weights,pos,param,start_idx,end_idx,i_los)

@ti.func
def eval_layer_basis(r,pos,start_idx,end_idx):
    lin_r = r[2]
    ev = 0.0
    idx_low = pos-1
    idx_high = pos+1
    if pos == start_idx:
        idx_low = pos
    elif pos == (end_idx-1):
        idx_high = pos
    if basis_lin[pos] <= lin_r and lin_r < basis_lin[idx_high]:
        gap_length = basis_lin[idx_high] - basis_lin[pos]
        ev = (basis_lin[idx_high] - lin_r) / gap_length
    elif basis_lin[idx_low] <= lin_r and lin_r < basis_lin[pos]:
        gap_length = basis_lin[pos] - basis_lin[idx_low]
        ev = (lin_r - basis_lin[idx_low]) / gap_length
    return ev


@ti.func
def eval_sphr_basis(r,pos,start_idx,end_idx):
    center = tm.vec3(0.0, 0.0, 0.0)
    lin_r = tm.length(r-center)
    ev = 0.0
    idx_low = pos-1
    idx_high = pos+1
    if pos == start_idx:
        idx_low = pos
    elif pos == (end_idx-1):
        idx_high = pos
    if basis_lin[pos] <= lin_r and lin_r < basis_lin[idx_high]:
        gap_length = basis_lin[idx_high] - basis_lin[pos]
        ev = (basis_lin[idx_high] - lin_r) / gap_length
    elif basis_lin[idx_low] <= lin_r and lin_r < basis_lin[pos]:
        gap_length = basis_lin[pos] - basis_lin[idx_low]
        ev = (lin_r - basis_lin[idx_low]) / gap_length

    return ev

@ti.func
def eval_gauss_basis(r,pos,start_idx,end_idx):
    max_dist = 2.0 # multiplier of standard deviation; how far can we obtain
    # results
    std = basis_param[pos]
    fun_dist = tm.length(r - basis_pos[pos])
    ev = 0.0
    if fun_dist < (max_dist * std):
        var = std ** 2
        exp = -fun_dist ** 2 / var
        div = ti.sqrt((2 * tm.pi) ** 3 * var)
        ev = ti.exp(exp) / div
    return ev


@ti.func
def eval_basis(r,i_pos):
    ev = 0.0
    if (basis_type[0,1] - basis_type[0,0]) != 0 and basis_type[0,0] <= i_pos and i_pos < basis_type[0,1]:
        ev = eval_layer_basis(r,i_pos,basis_type[0,0],basis_type[0,1])
    elif (basis_type[1,1] - basis_type[1,0]) != 0 and basis_type[1,0] <= i_pos and i_pos < basis_type[1,1]:
        ev = eval_sphr_basis(r,i_pos,basis_type[1,0],basis_type[1,1])
    elif (basis_type[2,1] - basis_type[2,0]) != 0 and basis_type[2,0] <= i_pos and i_pos < basis_type[2,1]:
        ev = eval_gauss_basis(r,i_pos,basis_type[2,0],basis_type[2,1])
    return ev

@ti.kernel
def test_basis():
    r = tm.vec3(6370.95,-0.00,-0.00)
    basis_contrib(r,basis_weights,basis_type,basis_pos,basis_lin,basis_param)


@ti.func
def calc_interp_err(r_1,r_2,r_3,w_1,w_2,w_3,i_los):
    # the linear interpolation error
    # which norm? inf-norm and 2-norm
    # now there's 2-norm
    d1 = tm.length(r_2 - r_1)
    d2 = tm.length(r_3 - r_2)
    d_tot = d1 + d2
    err = 0.0
    for i in range(w_1.shape[0]):
        err += ti.abs(w_2[i,i_los] - ((d_tot - d1) * w_1[i,i_los] + (d_tot - d2) * w_3[i,i_los]) / d_tot)
        #print('err:',w_1[i],w_2[i],w_3[i],w_2[i] - ((d_tot - d1) * w_1[i] + (d_tot - d2) * w_3[i]) / d_tot)
    return err

@ti.func
def sphere_line_intersect(line_o,line_d,sphere_c,sphere_r):
    # this function will solve the next intersection from line_o along line_d
    # with sphere centered at sphere_c with radius sphere_r
    # The reasoanble values for this are [eps,infty). If there are no solutions,
    # then returns -1.0.
    # Formula copied from https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection (20.9.2024)
    d = 0.0
    within_sphere = False
    losc = line_o - sphere_c
    b = tm.dot(line_d, losc)
    T = b ** 2 - (tm.dot(losc,losc) - sphere_r ** 2)
    if T < 0.0:
        d = -1.0
    elif T == 0.0:
        d = -b
    else:
        # two solutions
        # the line_o has three choices: before both solutions, between the solutions
        # or past the both solutions.
        # if line_o is before both, then d1 is the smaller one of them and that's
        # the next one. if line_o is in between, then d2 is the next one. if past,
        # then d2 is still the closest, but we can return just something minus-signed
        sqrT = tm.sqrt(T)
        d1 = -b - sqrT
        if 0.0 < d1:
            d = d1
        else:
            d2 = -b + sqrT
            d = d2
            if 0.0 < d2:
                within_sphere = True
    return (d, within_sphere)

@ti.func
def plane_line_intersect(line_o,line_d,plane_o,plane_norm):
    # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    ln = tm.dot(line_d,plane_norm)
    num = tm.dot(plane_o - line_o,plane_norm)
    d = tm.inf
    if ln > 0.0:
        d = num / ln
    return d

@ti.func
def interp_crossing(step_dir,step_start):
    # this is now tuned without atmospheric refraction
    # here you can figure that out
    t = tm.inf
    override = False
    new_step = tm.vec3(0.0)
    for idx_pos in range(n_pos):
        t_ = -1.0
        if basis_type_index[idx_pos] == 0:
            # find cutting point with layer
            plane_o = tm.vec3(0.0,0.0,basis_pos[idx_pos][2])
            plane_norm = tm.vec3(0.0,0.0,1.0)
            if step_start[2] > plane_o[2]:
                plane_norm = tm.vec3(0.0,0.0,-1.0)
            t_ = plane_line_intersect(step_start,step_dir,plane_o,plane_norm)
        elif basis_type_index[idx_pos] == 1:
            # sphere
            sphere_c = tm.vec3(0.0, 0.0, 0.0)
            t_, _ = sphere_line_intersect(step_start, step_dir, sphere_c, basis_lin[idx_pos])
        elif basis_type_index[idx_pos] == 2:
            # gaussian
            sphere_c = basis_pos[idx_pos]
            sphere_r = basis_param[idx_pos]
            t_, override = sphere_line_intersect(step_start, step_dir, sphere_c, 2.0 * sphere_r)
            #print('gauss',idx_pos,t_)
        if override:
            # in this case, we take the minimum_step steps
            t = 0.0
            break
        if t_ < 0.0:
            continue
        elif t_ < t:
            t = t_
    if not tm.isinf(t):
        new_step = step_start + t * step_dir
    #print(new_step,step_start,t,step_dir)
    return new_step

@ti.func
def find_step_alternative(r_start,weights_start,weights_mid,weights_end,basis_type,step_dir,minimum_step,interp_error,i_los,halver):
    # refraction here too?
    r_new = r_start + minimum_step * step_dir
    if is_in_medium(r_new,stopper_inner,stopper_outer):
        r_new = interp_crossing(step_dir,r_new)
    if halver:
        r_new = r_start + (r_new - r_start) * 0.5
    return r_new

@ti.func
def find_step(r_start,weights_start,weights_mid,weights_end,basis_type,step_dir,minimum_step,interp_error,i_los):
    """
    Will find an adequate next step for the path
    Inputs:
        r_start, 3-vector
        weights_start, ndarray of floats, shape (n_function_pos,)
        basis_funs, list of Basis_xxx dataclasses
        step_dir, 3-vector
        minimum_step, float
        interp_error, float
    Outputs:
        r_end, 3-vector
        weights_end, ndarray of floats, shape (n_function_pos,)

    TODO: pitääkö olla max_step? Jotta sironta saadaan riittävän näppärästi näkyviin?
    """
    r_mid = 0.5 * minimum_step * step_dir + r_start
    basis_contrib(r_mid,weights_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
    #print(weights_mid)
    r_end = minimum_step * step_dir + r_start
    basis_contrib(r_end,weights_end,basis_type,basis_pos,basis_lin,basis_param,i_los)
    #print(weights_end)
    step_err = calc_interp_err(r_start,r_mid,r_end,weights_start,weights_mid,weights_end,i_los)
    step_taken = False
    while ti.abs(step_err) < interp_error:
        # plan: one step at a time, check interp error to the second to last spot
        # if error too large -> the step is from start to second to last spot
        # if error not too large -> take another minimum_step
        r_end = minimum_step * step_dir + r_end
        r_mid = r_start + 0.5 * (r_end - r_start)
        clear_arr(weights_mid,i_los)
        clear_arr(weights_end,i_los)
        basis_contrib(r_mid,weights_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
        basis_contrib(r_end,weights_end,basis_type,basis_pos,basis_lin,basis_param,i_los)
        step_err = calc_interp_err(r_start,r_mid,r_end,weights_start,weights_mid,weights_end,i_los)
        #print(r_mid,step_err,interp_error)
        step_taken = True
    if step_taken:
        # we try to stay within the error bounds, but progress nonetheless
        # if the error is too large on the smallest step too, we take it
        # otherwise we try to stay within the error and take a step back
        r_end = r_end - minimum_step * step_dir
    clear_arr(weights_end,i_los)
    basis_contrib(r_end,weights_end,basis_type,basis_pos,basis_lin,basis_param,i_los)
    return r_end

@ti.func
def reassessment_error(r_1,r_2,r_3,weights_start,weights_mid,weights_end,i_los):
    clear_arr(weights_start,i_los)
    clear_arr(weights_mid,i_los)
    clear_arr(weights_end,i_los)
    basis_contrib(r_1,weights_start,basis_type,basis_pos,basis_lin,basis_param,i_los)
    basis_contrib(r_2,weights_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
    basis_contrib(r_3,weights_end,basis_type,basis_pos,basis_lin,basis_param,i_los)
    return calc_interp_err(r_1,r_2,r_3,weights_start,weights_mid,weights_end,i_los)

@ti.func
def calc_orig_err(r_start,r_kink_start,r_kink_end,r_end,i_los):
    clear_arr(weights_start,i_los)
    clear_arr(weights_mid,i_los)
    clear_arr(weights_kink,i_los)
    clear_arr(weights_end,i_los)
    final_mean = kink_end + 0.5 * (final_step - kink_end)
    kink_mean = kink_start + 0.5 * (kink_end - kink_start)
    start_mean = start_step + 0.5 * (kink_start - start_step)
    basis_contrib(r_start,weights_start,basis_type,basis_pos,basis_lin,basis_param,i_los)
    basis_contrib(r_kink_start,weights_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
    basis_contrib(r_kink_end,weights_kink,basis_type,basis_pos,basis_lin,basis_param,i_los)
    basis_contrib(r_end,weights_end,basis_type,basis_pos,basis_lin,basis_param,i_los)

@ti.func
def calc_reassess_err():
    pass

@ti.func
def reassess_final(path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los,weights_start,weights_mid,weights_end,minimum_step):
    """
    The final step is reassessed so that a lingering tiny step is cleared.
    OR
    The final step is tuned to the boudnary
    """
    curr_path_len = path_len[i_los]
    final_step_len = node_dist[i_los,curr_path_len - 2,i_los,curr_path_len - 1]
    final_step = path_steps[curr_path_len-1,i_los]

    if final_step_len < minimum_step:
        #the last step is extremely small
        prev_step = path_steps[curr_path_len-3,i_los]
        node_dist[i_los,path_len[i_los] - 2,i_los,path_len[i_los] - 1] = 0.0
        node_dist[i_los,path_len[i_los] - 3,i_los,path_len[i_los] - 2] = tm.length(final_step - prev_step)
        path_steps[curr_path_len-2,i_los] = final_step
        for i_pos in range(n_pos):
            path_basis[curr_path_len-2,i_pos,i_los] = path_basis[curr_path_len-1,i_pos,i_los]
            path_basis[curr_path_len-1,i_pos,i_los] = 0.0


        path_len[i_los] -= 1
    else:
        n_step_split = 40
        prev_step = path_steps[curr_path_len-2,i_los]
        step_dir = (final_step - prev_step) / tm.length(final_step - prev_step)
        step_tuning_scale = 4 * minimum_step if minimum_step * 4 > 0.2 else 0.2
        for i in range(n_step_split):
            new_step = final_step + (i / n_step_split) * step_dir * step_tuning_scale
            in_medium = is_in_medium(new_step,stopper_inner,stopper_outer)
            if not in_medium:
                #the recalibrated step is the one just before that.
                new_final_step = final_step + ((i-1) / n_step_split) * step_dir * step_tuning_scale
                path_steps[curr_path_len-1,i_los] = new_final_step
                #path_basis[curr_path_len-1,i_pos,i_los]
                basis_contrib(path_steps[curr_path_len-1,i_los],weights_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
                for i_pos in range(n_pos):
                    path_basis[curr_path_len-1,i_pos,i_los] = weights_mid[i_pos,i_los]
                node_dist[i_los,curr_path_len-2,i_los,curr_path_len-1] = tm.length(new_final_step - prev_step)
                break
        #the last step is not exactly on the boundary
        #TODO: set the step onto the boundary!!!!
        # recompute basis!



@ti.func
def reassess_steps(path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los,weights_start,weights_mid,weights_end):
    """
    The past few steps are to be reassessed if there is a "kink" in the path and
    the step boundaries have not been properly set on the basis function boundaries.
    Kink means a single minimum_step length step, which actually just increases the
    step error.

    The steps initially are start_step - kink_start - kink_end - final_step
    This function checks if the combined error would be reduced if the kink is
    contracted into a single step, like so: start_step - kink_mean - final_step
    """
    curr_path_len = path_len[i_los]
    final_step = path_steps[curr_path_len-1,i_los]
    kink_end = path_steps[curr_path_len-2,i_los]
    final_mean = kink_end + 0.5 * (final_step - kink_end)
    kink_start = path_steps[curr_path_len-3,i_los]
    kink_mean = kink_start + 0.5 * (kink_end - kink_start)
    start_step = path_steps[curr_path_len-4,i_los]
    start_mean = start_step + 0.5 * (kink_start - start_step)
    n_reassess = 20

    # Original interp errors
    orig_err = 0.0
    orig_err += reassessment_error(start_step, start_mean, kink_start, weights_start, weights_mid, weights_end, i_los)
    orig_err += reassessment_error(kink_start, kink_mean, kink_end, weights_start, weights_mid, weights_end, i_los)
    orig_err += reassessment_error(kink_end, final_mean, final_step, weights_start, weights_mid, weights_end, i_los)

    reassess_err_best = orig_err
    i_best = 0
    for i in range(n_reassess):
        kink_test = kink_start + (i / n_reassess) * (kink_end - kink_start)
        reassess_err = 0.0
        reassess_err += 1.5 * reassessment_error(start_step, 0.5 * (start_step + kink_test), kink_test, weights_start, weights_mid, weights_end, i_los)
        reassess_err += 1.5 * reassessment_error(kink_test,  0.5 * (final_step + kink_test), final_step, weights_start, weights_mid, weights_end, i_los)
        # the above 1.5 multiplier is due to the fact that we have to weight the error appropriately for comparison
        # because there's only two steps instead of three
        if reassess_err < reassess_err_best:
            reassess_err_best = reassess_err
            i_best = i

    if reassess_err_best < orig_err:
        #print('reassess_err_best: ',reassess_err_best,'orig err: ',orig_err)
        kink_mean = kink_start + (i_best / n_reassess) * (kink_end - kink_start)
        #print('final_step',final_step[0],final_step[1],final_step[2])
        #if True:
        path_steps[curr_path_len-3,i_los] = kink_mean
        clear_arr(weights_start,i_los)
        basis_contrib(kink_mean,weights_start,basis_type,basis_pos,basis_lin,basis_param,i_los)
        path_steps[curr_path_len-2,i_los] = final_step
        for i_pos in range(n_pos):
            path_basis[curr_path_len-3,i_pos,i_los] = weights_start[i_pos,i_los]
            path_basis[curr_path_len-2,i_pos,i_los] = path_basis[curr_path_len-1,i_pos,i_los]
            path_basis[curr_path_len-1,i_pos,i_los] = 0.0
        path_steps[curr_path_len-1,i_los] = tm.vec3(0.0,0.0,0.0)
        node_dist[i_los,path_len[i_los] - 2,i_los,path_len[i_los] - 1] = 0.0
        node_dist[i_los,path_len[i_los] - 3,i_los,path_len[i_los] - 2] = tm.length(final_step - kink_mean)
        node_dist[i_los,path_len[i_los] - 4,i_los,path_len[i_los] - 3] = tm.length(kink_mean - start_step)
        path_len[i_los] -= 1

@ti.kernel
def test_step():
    r_start = tm.vec3(6371.50, 288.008362, 0.0)
    r_start = tm.vec3(6371.50, 0.0, 0.0)
    basis_contrib(r_start,basis_weights,basis_type,basis_pos,basis_lin,basis_param)
    step = find_step(r_start,
                     basis_weights,
                     basis_mid,
                     basis_end,
                     basis_type,
                     step_dir=tm.vec3(0.0,1.0,0.0),
                     minimum_step=0.1,
                     interp_error=0.01)

    print(step)

@ti.func
def integrate_step_basis(i_node,path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los):
    N_delta = 10
    clear_arr(basis_mid,i_los)
    """
    NOTE: this will cause trouble if the los is not a straight line!!!
    """
    start_point = (path_steps[i_node-1,i_los] + path_steps[i_node,i_los]) * 0.5
    end_point = (path_steps[i_node,i_los] + path_steps[i_node+1,i_los]) * 0.5
    total_delta = end_point - start_point
    for i_delta in range(N_delta+1):
        point = i_delta / N_delta * total_delta + start_point
        clear_arr(basis_end,i_los)
        basis_contrib(point,basis_end,basis_type,basis_pos,basis_lin,basis_param,i_los)
        for i_pos in range(n_pos):
            basis_mid[i_pos,i_los] += basis_end[i_pos,i_los]
    for i_pos in range(n_pos):
        basis_mid[i_pos,i_los] *= 1/(N_delta+1)


@ti.func
def recompute_path_basis(path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los):
    """

    NOTE: This function computes the basis function at each of the path nodes.
    The problem is that the path_basis should be filled appropriately during the
    trace_path function. This is does NOT happen for some reason. The problem
    arises when the taichi kernel is parallelized, but not when it is serialized.

    Investigate this phenomenon further, because it might cause bugs later on...

    NOTE 2: This is now repurposed to set up the basis as a cumulative contribution
    of basis functions at a certain point

    NOTE 3: This needs some heavy refactoring. Final could also be an external function
    to create the cumulative basis contributions.
    """



    clear_arr(basis_mid,i_los)
    basis_contrib(path_steps[0,i_los],basis_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
    dist = node_dist[i_los,0,i_los,1]
    last_node = path_len[i_los]-1
    for i_pos in range(n_pos):
        path_basis[0,i_pos,i_los] = basis_mid[i_pos,i_los] * dist * 0.5
        avg_scatter_mat[0,i_los] += avg_scattering[i_pos] * basis_mid[i_pos,i_los]# * dist * 0.5
    for i_node in range(1,last_node):
        clear_arr(basis_mid,i_los)
        #if old_basis_integral_mode or i_node == (path_len[i_los]-1):
        basis_contrib(path_steps[i_node,i_los],basis_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
        #else:
        #    integrate_step_basis(i_node,path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los)
        dist = (node_dist[i_los,i_node-1,i_los,i_node] + node_dist[i_los,i_node,i_los,i_node+1]) * 0.5
        for i_pos in range(n_pos):
            avg_scatter_mat[i_node,i_los] += avg_scattering[i_pos] * basis_mid[i_pos,i_los]# * dist
            path_basis[i_node,i_pos,i_los] = basis_mid[i_pos,i_los] * dist
    dist = node_dist[i_los,last_node-1,i_los,last_node]
    clear_arr(basis_mid,i_los)
    #if old_basis_integral_mode or i_node == (path_len[i_los]-1):
    basis_contrib(path_steps[last_node,i_los],basis_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
    for i_pos in range(n_pos):
        #print(i_los,i_pos,basis_mid[i_pos,i_los])
        path_basis[last_node,i_pos,i_los] = basis_mid[i_pos,i_los] * dist * 0.5
        avg_scatter_mat[last_node,i_los] += avg_scattering[i_pos] * basis_mid[i_pos,i_los]# * dist * 0.5
        #print(avg_scattering[i_pos],basis_mid[i_pos,i_los])
    #print(last_node,avg_scatter_mat[last_node,i_los])
    for i_node in range(48,path_len[i_los]):
        for i_pos in range(n_pos):
            pass
            #print(i_node,i_pos,path_basis[i_node,i_pos,i_los],avg_scattering[i_pos])
            #print(i_node,i_los,avg_scatter_mat[i_node,i_los]) 0.03721313

@ti.func
def recompute_path_basis_old(path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los):
    """


    NOTE: This function computes the basis function at each of the path nodes.
    The problem is that the path_basis should be filled appropriately during the
    trace_path function. This is does NOT happen for some reason. The problem
    arises when the taichi kernel is parallelized, but not when it is serialized.

    Investigate this phenomenon further, because it might cause bugs later on...

    NOTE 2: This is now repurposed to set up the basis as a cumulative contribution
    of basis functions at a certain point

    NOTE 3: This needs some heavy refactoring. Final could also be an external function
    to create the cumulative basis contributions.
    """



    clear_arr(basis_mid,i_los)
    basis_contrib(path_steps[0,i_los],basis_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
    dist = node_dist[i_los,0,i_los,1]
    for i_pos in range(n_pos):
        path_basis[0,i_pos,i_los] = basis_mid[i_pos,i_los] * dist
        avg_scatter_mat[0,i_los] = avg_scattering[i_pos] * basis_mid[i_pos,i_los] * dist
    for i_node in range(1,path_len[i_los]):
        clear_arr(basis_mid,i_los)
        if old_basis_integral_mode or i_node == (path_len[i_los]-1):
            basis_contrib(path_steps[i_node,i_los],basis_mid,basis_type,basis_pos,basis_lin,basis_param,i_los)
        else:
            integrate_step_basis(i_node,path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los)
        dist = node_dist[i_los,i_node-1,i_los,i_node]
        for i_pos in range(n_pos):
            avg_scatter_mat[i_node,i_los] += avg_scattering[i_pos] * basis_mid[i_pos,i_los] * dist
            path_basis[i_node,i_pos,i_los] = basis_mid[i_pos,i_los] * dist

@ti.func
def recompute_path_basis_cumulative(i_los):
    dist = node_dist[i_los,0,i_los,1]
    for i_pos in range(n_pos):
        path_basis[0,i_pos,i_los] = path_basis[0,i_pos,i_los]
    for i_node in range(1,path_len[i_los]):
        dist = node_dist[i_los,i_node-1,i_los,i_node]
        for i_pos in range(n_pos):
            path_basis[i_node,i_pos,i_los] = path_basis[i_node,i_pos,i_los] + path_basis[i_node-1,i_pos,i_los]

"""
TODO: set up a stoppers-struct!
"""

@ti.func
def bound_fun(r,param,inner):
    mult = -1.0
    if inner:
        mult = 1.0
    val = 0.0
    if spherical:
        val = mult*(param - tm.length(r))
    else:
        val = mult*(param - r[2])
    return val

@ti.func
def find_closest_boundary(r):
    #returns 0 if inner is closer, 1 if outer is closer
    rlen = 0.0
    if spherical:
        rlen = tm.length(r)
    else:
        rlen = r[2]
    closeval = -1
    dist_outer = ti.abs(stopper_outer.size_parameter - rlen)
    dist_inner = ti.abs(rlen - stopper_inner.size_parameter)
    if dist_outer > dist_inner:
        closeval = 0
    else:
        closeval = 1
    return closeval

@ti.func
def find_crossing_point(point,direction,minimum_step):
    # this function will return the point very close to the boundary, on the positive side.
    # we assume that point is quite close to the stopper i.e. single step will
    # get us across the boundary.

    param = 0.0
    closest = find_closest_boundary(point)
    incident = closest
    if closest == 1:
        param = stopper_outer.size_parameter
    else:
        param = stopper_inner.size_parameter
    crossing = tm.vec3(0.0)
    t_upper = minimum_step
    t_lower = -minimum_step
    in_bounds = False
    extra_splits = 0
    while not in_bounds:
        cross_upp = point + t_upper * direction
        cross_low = point + t_lower * direction
        if bound_fun(cross_upp,param,incident) * bound_fun(cross_low,param,incident) < 0.0:
            in_bounds = True
        else:
            t_upper *= 2
            t_lower *= 2
            extra_splits += 1
    N_splits = 16 + extra_splits
    for i in range(N_splits):
        t_mid = 0.5 * (t_lower + t_upper)
        cross_upp = point + t_upper * direction
        cross_low = point + t_lower * direction
        cross_mid = point + t_mid * direction
        if bound_fun(cross_upp,param,incident) * bound_fun(cross_mid,param,incident) < 0.0:
            # the point and the cross_mid are on the different sides so we went too far
            t_lower = t_mid
        elif bound_fun(cross_low,param,incident) * bound_fun(cross_mid,param,incident) < 0.0:
            # the point and the cross_mid are on the same side, let's go further
            t_upper = t_mid
        else:
            # the midpoint is exactly at the boundary. let's reduce both
            t_upper = 0.5 * (t_mid + t_upper)
            t_lower = 0.5 * (t_mid + t_lower)
    # at this point, the lower and the upper are very close, but on the different sides
    # of the boundary.

    if bound_fun(crossing,param,incident) > 0.0:
        crossing = point + t_upper * direction
        # the t_upper point is outside the medium
        # we want the t_lower
    elif bound_fun(crossing,param,incident) <= 0.0:
        crossing = point + t_lower * direction

    return crossing

@ti.func
def find_next_boundary_crossing(beam_start,beam_direction):
    t = 0.0
    t_out = 0.0
    t_in = 0.0
    if spherical:
        t_out,_ = sphere_line_intersect(beam_start,beam_direction,tm.vec3(0.0),stopper_outer.size_parameter)
        t_in,_ = sphere_line_intersect(beam_start,beam_direction,tm.vec3(0.0),stopper_inner.size_parameter)
    else:
        norm_out = tm.vec3(0.0,0.0,-1.0)
        cent_out = tm.vec3(0.0,0.0,stopper_outer.size_parameter)
        t_out = plane_line_intersect(beam_start,beam_direction,cent_out,norm_out)
        norm_in = tm.vec3(0.0,0.0,1.0)
        cent_in = tm.vec3(0.0,0.0,stopper_inner.size_parameter)
        t_in = plane_line_intersect(beam_start,beam_direction,cent_in,norm_in)
    if t_out > 0.0 and (t_out < t_in or t_in < 0.0):
        t = t_out
    elif t_in > 0.0 and (t_in < t_out or t_out < 0.0):
        t = t_in
    return beam_start + t * beam_direction

@ti.func
def trace_path_alternative(beam_start,beam_direction,stopper_inner,stopper_outer,
                                  minimum_step,interp_error,
                                  basis_weights,basis_mid,basis_end,
                                  basis_type,path_steps,path_len,
                                  path_basis,i_los):

    start_step = beam_start
    #print(start_step,beam_direction)
    #in_medium = stopper_inner.size_parameter < tm.length(start_step) < stopper_outer.size_parameter
    in_medium = is_in_medium(start_step,stopper_inner,stopper_outer)
    if not in_medium:
        start_step = find_next_boundary_crossing(beam_start,beam_direction)
        #print(start_step,t,beam_direction)
        #start_step = start_step + 1.5*minimum_step * beam_direction
        #print(i_los,is_in_medium(start_step,stopper_inner,stopper_outer))

    #return
    start_step = find_crossing_point(start_step,-beam_direction,minimum_step)
    in_medium = True
    append_path(path_basis,basis_weights,path_steps,start_step,path_len,i_los,first=True)
    new_step = start_step
    step_halver = False # this is debug variable to examine path integration
    half_step = False
    while in_medium:
        new_step = find_step_alternative(new_step,
                         basis_weights,
                         basis_mid,
                         basis_end,
                         basis_type,
                         step_dir=beam_direction,
                         minimum_step=minimum_step,
                         interp_error=interp_error,
                         i_los=i_los,
                         halver=half_step)
        if step_halver:
            half_step = not half_step

        in_medium = is_in_medium(new_step,stopper_inner,stopper_outer)

        if not in_medium:
            #print(new_step,tm.length(new_step))
            new_step = find_crossing_point(new_step,-beam_direction,minimum_step)
            #print(new_step,tm.length(new_step))
            #print(tm.length(path_steps[path_len[i_los]-1,i_los]))
        if tm.length(new_step - path_steps[path_len[i_los]-1,i_los]) > (minimum_step / 2):
            append_path(path_basis,basis_end,path_steps,new_step,path_len,i_los,first=False)


@ti.func
def trace_path(beam_start,beam_direction,stopper_inner,stopper_outer,
                                  minimum_step,interp_error,
                                  basis_weights,basis_mid,basis_end,
                                  basis_type,path_steps,path_len,
                                  path_basis,i_los):
    """
    Function trace_path will trace a path through 3D space starting from beam_start
    in the direction defined by beam_direction until function stopper is less than
    zero. This function will create steps of varying step size according to the
    integrable function: the step lengths will be of multiples of minimum_step.
    The step length will be as long as possible without resulting into an
    linear interpolation error larger than interp_error in the integrable.

    This function can also be used in tracing the paths to the source from nodes,
    and not just the line-of-sights

    TODO: refraction and other beam direction changes should be taken into
    account in this part of the RT.
    TODO: terminator effect should be checked for each of the sources.

    Inputs:
        beam_start, (3,) Numpy float array: where the beam is started to trace

        beam_direction, (3,) Numpy float array: normalized vector toward which
        the steps are taken.

        integrable, function((3,) Numpy float array) -> (M,) Numpy sparse float
        array: The contribution of each of the basis functions at a R^3 position.

        stopper, function((3,) Numpy float array) -> float: The beam is traced until
        this function is less than zero with the argument beam_position.

        minimum_step, float: the minimum step length the steps can be discretized with.

        interp_error, float: Maximum error allowed for the local interpolation error.
        If the interpolation error would get too large, the step is ended and a
        new one is started.

    Outputs:
        path_nodes, (3,N) Numpy float array: The position of nodes in the path with
        N being the total number of nodes in the path. path_nodes[:,0] will be
        beam_start.

        integrable_contributions, (N,M) Numpy sparse float array: For each of the
        nodes, the proportional contribution of each of the basis functions is.
        M is now the amount of basis functions.

        path_lens, (N-1,) Numpy float array: The lengths of the segments between
        the path nodes. path_lens[0] = np.linalg.norm(path_nodes[:,1] - path_nodes[:,0])
        etc.

        stopper_index, integer: The index of the stopper function which terminated
        this path trace.

    Notes:
        Previously in the inputs there was a function
            integrable, function((3,) Numpy float array) -> float: function which
            can be interpolated using the output points without making a local interpolation
            error larger than interp_error.
        However, it is not clear at all when considering a vector-valued function
        how to assess its interpolatedness. Is it through 1-norm? Therefore it has
        been retrofitted into a basis function contribution function.
            direct_transmittance, integer or None: Returns the radiation source index at which
            the last path segment is pointing. If the path does not point at a radiation
            source outside, then this is None.

        All the source effects are handled in the later function node_source_contributions.
        These include direct transmittance, shadowing

        TODO: Fit the boundary points onto the stopper boundary

    """
    max_step = 10000000
    curr_step = 0
    start_step = beam_start
    in_medium = is_in_medium(start_step,stopper_inner,stopper_outer)
    while not in_medium:
        start_step = start_step + minimum_step * beam_direction
        in_medium = is_in_medium(start_step,stopper_inner,stopper_outer)
        if curr_step > max_step:
            break
        else:
            curr_step += 1
    n_step_split = 20
    if curr_step > 0:
        # in this case, we've started outside the domain and moved into it.
        # let's find the exact crossing point.
        for i_split in range(n_step_split):
            split_step = start_step - (i_split / n_step_split) * minimum_step * beam_direction
            if not is_in_medium(split_step,stopper_inner,stopper_outer):
                start_step = start_step - ((i_split-1) / n_step_split) * minimum_step * beam_direction
                break
    if in_medium:
        basis_contrib(start_step,basis_weights,basis_type,basis_pos,basis_lin,basis_param,i_los)
        append_path(path_basis,basis_weights,path_steps,start_step,path_len,i_los,first=True)
    old_step = start_step
    while in_medium:
        clear_arr(basis_weights,i_los)
        basis_contrib(old_step,basis_weights,basis_type,basis_pos,basis_lin,basis_param,i_los)
        clear_arr(basis_mid,i_los)
        clear_arr(basis_end,i_los)
        new_step = find_step(old_step,
                         basis_weights,
                         basis_mid,
                         basis_end,
                         basis_type,
                         step_dir=beam_direction,
                         minimum_step=minimum_step,
                         interp_error=interp_error,
                         i_los=i_los)
        in_medium = is_in_medium(new_step,stopper_inner,stopper_outer)
        final_step_bool = False
        if not in_medium:
            for i_split in range(n_step_split):
                test_step = new_step - (i_split / n_step_split) * (new_step - path_steps[path_len[i_los]-1,i_los])
                if is_in_medium(test_step,stopper_inner,stopper_outer):
                    new_step = test_step
                    clear_arr(basis_end,i_los)
                    basis_contrib(new_step,basis_end,basis_type,basis_pos,basis_lin,basis_param,i_los)
                    final_step_bool = True
                    break

        if is_in_medium(new_step,stopper_inner,stopper_outer):

            append_path(path_basis,basis_end,path_steps,new_step,path_len,i_los,first=False)
            old_step = new_step
            if path_len[i_los] > 4:
                if node_dist[i_los,path_len[i_los] - 3,i_los,path_len[i_los] - 2] < 1.5 * minimum_step:
                    reassess_steps(path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los,basis_weights,basis_mid,basis_end)
                elif final_step_bool:
                    reassess_final(path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los,basis_weights,basis_mid,basis_end,minimum_step)

    # end: trim steps so that if two minimum step points are next to each other
    # (and far away from others), then check if mean of the points is better
    # than the sum of the errors from the separate steps. Käytännössä siis
    # että yhistät kaksi pistettä jos ne on jonkin funktion jakopisteen ympärillä!


@ti.func
def is_in_medium(r,stopper_1,stopper_2):
    return stopper_1.value(r) >= 0 and stopper_2.value(r) >= 0

@ti.func
def source_basis_path_trace(i_node,path_steps,path_len,source_basis,i_los,i_source,minimum_step,interp_error,
                            basis_weights,basis_mid,basis_end,basis_pos,basis_lin,basis_param,
                            basis_type):
    N_int = source_basis_integral_steps
    r_old = path_steps[i_node,i_los]

    r_new = find_step_alternative(r_old,
                     basis_weights,
                     basis_mid,
                     basis_end,
                     basis_type,
                     step_dir=-source_dir[i_source],
                     minimum_step=minimum_step,
                     interp_error=interp_error,
                     i_los=i_los,
                     halver=False)
    dist = tm.length(r_new - r_old)
    dnorm = (r_new - r_old) / dist
    int_steplen = (1 / N_int) * dist
    for i_int in range(N_int):
        r_int = r_old + int_steplen * dnorm * i_int
        clear_arr(basis_weights,i_los)
        basis_contrib(r_int,basis_weights,basis_type,basis_pos,basis_lin,basis_param,i_los)
        #print(tm.length(r_int))
        for i_pos in range(n_pos):
            #print(basis_weights[i_pos,i_los])
            source_basis[i_node, i_pos, i_los, i_source] += int_steplen * basis_weights[i_pos,i_los]

    #while tm.length(r_new) < stopper_outer.size_parameter:
    while is_in_medium(r_new,stopper_inner,stopper_outer):
        # change maybe to is_in_medium(r_new,stopper_inner,stopper_outer)
        r_old = r_new
        r_new = find_step_alternative(r_old,
                         basis_weights,
                         basis_mid,
                         basis_end,
                         basis_type,
                         step_dir=-source_dir[i_source],
                         minimum_step=minimum_step,
                         interp_error=interp_error,
                         i_los=i_los,
                         halver=False)

        dist = tm.length(r_new - r_old)
        dnorm = (r_new - r_old) / dist
        int_steplen = (1 / N_int) * dist
        for i_int in range(N_int):
            r_int = r_old + int_steplen * dnorm * i_int
            clear_arr(basis_weights,i_los)
            basis_contrib(r_int,basis_weights,basis_type,basis_pos,basis_lin,basis_param,i_los)
            for i_pos in range(n_pos):
                source_basis[i_node, i_pos, i_los, i_source] += int_steplen * basis_weights[i_pos,i_los]

@ti.func
def source_basis_contrib(path_steps,path_len,source_basis,i_los,i_source,minimum_step,interp_error,
                           basis_weights,basis_mid,basis_end,basis_pos,basis_lin,basis_param,
                            basis_type):
    # NOTE: Should a node have minimum scattering efficiency before it is traced
    # back to the source?
    if scatter_mode == 0:
        # just surface reflection; the last point is considered
        i_node = path_len[i_los]-1
        source_basis_path_trace(i_node,path_steps,path_len,source_basis,i_los,i_source,minimum_step,interp_error,
                                   basis_weights,basis_mid,basis_end,basis_pos,basis_lin,basis_param,
                                    basis_type)
    else:
        # single- or multiple scattering
        for i_node in range(path_len[i_los]):
            source_basis_path_trace(i_node,path_steps,path_len,source_basis,i_los,i_source,minimum_step,interp_error,
                                       basis_weights,basis_mid,basis_end,basis_pos,basis_lin,basis_param,
                                        basis_type)
@ti.kernel
def test_trace(minimum_step: float):

    for i_los in instr_pos:
        #i_los = 0
        #if True:
        beam_start = instr_pos[i_los]
        beam_dir = instr_view[i_los]
        #print(i_los,beam_dir)
        interp_error = 0.01
        accurate_step_trace = True
        if accurate_step_trace:
            trace_path_alternative(beam_start,beam_dir,stopper_inner,stopper_outer,minimum_step,interp_error,
            basis_weights,basis_mid,basis_end,basis_type,path_steps,path_len,
            path_basis,i_los)
        else:
            # NOTE: this method here increases the compile to drastically, so
            # the trace_path here is just commented.
            pass
            #trace_path(beam_start,beam_dir,stopper_inner,stopper_outer,minimum_step,interp_error,
            #basis_weights,basis_mid,basis_end,basis_type,path_steps,path_len,
            #path_basis,i_los)

        recompute_path_basis(path_steps,path_len,path_basis,basis_type,basis_pos,basis_lin,basis_param,i_los)
        #if final_trace:
        #    recompute_path_basis_cumulative(i_los)
    #for i_los in range(n_los):
    #    if path_len[i_los] == max_path_len:
    #        print(f"Warning: Path length of line-of-sight of index {i_los} is at max_path_len. Consider increasing max_path_len.")


@ti.kernel
def cumulate_path_basis():
    for i_los in instr_pos:
        recompute_path_basis_cumulative(i_los)

@ti.kernel
def test_basis_contrib():
    r = tm.vec3(6.45097119e+03, -2.39791298e+00, -2.39789701e+00)
    basis_contrib(r,basis_end,basis_type,basis_pos,basis_lin,basis_param)

@ti.kernel
def test_source_basis(minimum_step: float):
    interp_error = 0.01
    i_source = 0
    minimum_step_integral = minimum_step
    # to make this integrating much more finer
    for i_los in instr_pos:
        #for i_los in range(1):
        source_basis_contrib(path_steps,
        path_len,source_basis,i_los,i_source,minimum_step_integral,interp_error,
        basis_weights,basis_mid,basis_end,basis_pos,basis_lin,basis_param,basis_type)

@ti.func
def calc_source_transmittance(i_node,i_wl,i_los,i_source):
    tau = 0.0
    for i_pos in range(n_pos):
        tau += source_basis[i_node,i_pos,i_los,i_source] * extinction[i_pos,i_wl]
    return tm.exp(-tau)

@ti.func
def calc_path_transmittance(end_node,i_wl,i_los):
    tau = 0.0
    for i_pos in range(n_pos):
        tau += path_basis[end_node,i_pos,i_los] * extinction[i_pos,i_wl]
    return tm.exp(-tau)
