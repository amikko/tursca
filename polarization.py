
phase_angles = ti.field(ti.f32, shape=(particle_muller_length))
phase_function_table = ti.field(ti.f32, shape=(particle_muller_length))
if polariz_mode == 4:
    phase_matrix_table = ti.Matrix.field(n=4, m=4, dtype=ti.f32, shape=(particle_muller_length))

def set_up_muller_scattering():
    scaling = 1 / (2 * np.pi)
    muller_data = np.genfromtxt(particle_muller_filename)
    #print(muller_data)
    #print(particle_muller_length)
    for i in range(particle_muller_length):
        phase_angles[i] = tm.acos(muller_data[i,0])
        #var = phase_angles[i] - phase_angles[tm.max(i-1,0)]
        #print(var)
    for i in range(particle_muller_length):
        phase_function_table[i] = scaling * 0.5 * (muller_data[i,1] + muller_data[i,6])
    if polariz_mode == 4:
        for i in range(particle_muller_length):
            phase_matrix_table[i][0,0] = scaling * muller_data[i,1]
            phase_matrix_table[i][0,1] = scaling * muller_data[i,2]
            phase_matrix_table[i][0,2] = scaling * muller_data[i,3]
            phase_matrix_table[i][0,3] = scaling * muller_data[i,4]
            phase_matrix_table[i][1,0] = scaling * muller_data[i,5]
            phase_matrix_table[i][1,1] = scaling * muller_data[i,6]
            phase_matrix_table[i][1,2] = scaling * muller_data[i,7]
            phase_matrix_table[i][1,3] = scaling * muller_data[i,8]
            phase_matrix_table[i][2,0] = scaling * muller_data[i,9]
            phase_matrix_table[i][2,1] = scaling * muller_data[i,10]
            phase_matrix_table[i][2,2] = scaling * muller_data[i,11]
            phase_matrix_table[i][2,3] = scaling * muller_data[i,12]
            phase_matrix_table[i][3,0] = scaling * muller_data[i,13]
            phase_matrix_table[i][3,1] = scaling * muller_data[i,14]
            phase_matrix_table[i][3,2] = scaling * muller_data[i,15]
            phase_matrix_table[i][3,3] = scaling * muller_data[i,16]

def normalize_table():
    integral = 0.0
    ray_int = 0.0
    def rayleigh(cost):
        return 0.75 / (4 * np.pi) * (1 + cost ** 2)
    for i in range(1,particle_muller_length):
        sincoef = np.sin((phase_angles[i-1] - phase_angles[i]))
        integral += sincoef*(phase_angles[i-1] - phase_angles[i]) * (phase_function_table[i-1] + phase_function_table[i]) * 0.5
        costermim1 = tm.cos(phase_function_table[i-1])
        costermi = tm.cos(phase_function_table[i])
        ray_int += sincoef*(phase_angles[i-1] - phase_angles[i]) * (rayleigh(costermim1) + rayleigh(costermi)) * 0.5

    print('aerosol phase fun integral:',integral)
    print('rayleigh phase fun integral:',ray_int)

@ti.func
def phase_function(i_sca,dir_in,dir_out):
    costerm = tm.dot(dir_in,dir_out)
    ph = 0.0
    if i_sca == 0:
        ph = 0.75 / (4 * tm.pi) * (1 + costerm ** 2)
    elif i_sca == 1:
        if costerm > 1.0:
            costerm = 1.0
        elif costerm < -1.0:
            costerm = -1.0
        ang = tm.acos(costerm)
        angle_epsilon = 0.01 / 180 * tm.pi
        if ang < angle_epsilon:
            ph = phase_function_table[particle_muller_length-1]
        elif tm.pi - ang < angle_epsilon:
            ph = phase_function_table[0]
        else:
            for i in range(1,particle_muller_length):
                if phase_angles[i] < ang:
                    d_up = ang - phase_angles[i]
                    d_down = phase_angles[i-1] - ang
                    ph_up = phase_function_table[i]
                    ph_down = phase_function_table[i-1]
                    denom = d_up + d_down
                    if denom > 1e-6:
                        #print(1,phase_angles[i],costerm,phase_angles[i-1])
                        # this can happen in very finely divided aerosol phase tables
                        # and because floats we operate here are only single precision.
                        ph = (ph_up * d_down + ph_down * d_up) / (d_up + d_down)
                    else:
                        # when the distance between the nodes is "thinner than an atom"
                        # then average is good enough :)
                        ph = 0.5 * (ph_up + ph_down)
                    break
            #ph *= 1.4
    #print(costerm,ph)
    return ph

@ti.func
def phase_matrix(i_sca,dir_in,dir_out):
    costerm = tm.dot(dir_in,dir_out)
    ph_mat = tm.mat4(0)
    if i_sca == 0:
        """
        Rayleigh scattering with depolarization factor.
        Chandrasekhar p. 49
        Originally in Siro by Liisa Oikarinen on 8.12.1998
        """
        depolarization = 0.0
        gamma = depolarization / (2.0 - depolarization)
        coeff = 3.0 / (8.0 * tm.pi) * (1.0 / (1.0 + 2.0 * depolarization))
        M00 = (costerm ** 2 * (1.0 - gamma) + gamma)
        M22 = coeff * costerm * (1.0 - gamma)
        M33 = coeff * costerm * (1.0 - 3.0 * gamma)
        ph_mat[0,0] = coeff * M00
        ph_mat[0,1] = gamma * coeff
        ph_mat[1,0] = gamma * coeff
        ph_mat[1,1] = coeff
        ph_mat[2,2] = coeff * M22
        ph_mat[3,3] = coeff * M33
    elif i_sca == 1:
        for i in range(particle_muller_length):
            # if you want to be fancy, do a linear interp. here
            ang = tm.acos(costerm)
            if phase_angles[i] < ang:
                ph_mat = phase_matrix_table[i]
                break
    return ph_mat

"""
def muller_mie(theta, muller_data):
    costheta = np.cos(theta)
    costheta_idx = np.argmax(muller_data[:,0] >= costheta)
    Mu = muller_data[costheta_idx,:]
    # TODO: interpolation here, if needed
    S = np.array([[Mu[1],  Mu[2],  Mu[3],  Mu[4]],
                  [Mu[5],  Mu[6],  Mu[7],  Mu[8]],
                  [Mu[9],  Mu[10], Mu[11], Mu[12]],
                  [Mu[13], Mu[14], Mu[15], Mu[16]]])
    return 1 / (2 * np.pi) * S

def phase_function_mie(theta, muller_data):
    costheta = np.cos(theta)
    costheta_idx = np.argmax(muller_data[:,0] >= costheta)
    # The muller data ought to be in IIUV basis, so the phase function is
    # given by the sum of M11 and M22.
    phase = 1 / (4 * np.pi) * muller_data(costheta_idx,1) + muller_data(costheta_idx,6)
    return phase
"""

@ti.func
def matrix_vec_mult(b,A,x):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            b[i] += A[i,j] * x[j]

@ti.func
def matrix_vec_mult_4(A,x):
    return tm.vec4(A[0,0] * x[0] + A[0,1] * x[1] + A[0,2] * x[2] + A[0,3] * x[3],
                   A[1,0] * x[0] + A[1,1] * x[1] + A[1,2] * x[2] + A[1,3] * x[3],
                   A[2,0] * x[0] + A[2,1] * x[1] + A[2,2] * x[2] + A[2,3] * x[3],
                   A[3,0] * x[0] + A[3,1] * x[1] + A[3,2] * x[2] + A[3,3] * x[3])

@ti.func
def matrix_vec_mult_4_2(A,x):
    return tm.vec4(A[0,0] * x[0,0] + A[0,1] * x[1,0] + A[0,2] * x[2,0] + A[0,3] * x[3,0],
                   A[1,0] * x[0,0] + A[1,1] * x[1,0] + A[1,2] * x[2,0] + A[1,3] * x[3,0],
                   A[2,0] * x[0,0] + A[2,1] * x[1,0] + A[2,2] * x[2,0] + A[2,3] * x[3,0],
                   A[3,0] * x[0,0] + A[3,1] * x[1,0] + A[3,2] * x[2,0] + A[3,3] * x[3,0])

@ti.func
def matrix_mult_3(A,B):
    resmat = tm.mat3(0.0)
    resmat[0,0] = A[0,0] * B[0,0] + A[0,1] * B[1,0] + A[0,2] * B[2,0]
    resmat[1,0] = A[1,0] * B[0,0] + A[1,1] * B[1,0] + A[1,2] * B[2,0]
    resmat[2,0] = A[2,0] * B[0,0] + A[2,1] * B[1,0] + A[2,2] * B[2,0]

    resmat[0,1] = A[0,0] * B[0,1] + A[0,1] * B[1,1] + A[0,2] * B[2,1]
    resmat[1,1] = A[1,0] * B[0,1] + A[1,1] * B[1,1] + A[1,2] * B[2,1]
    resmat[2,1] = A[2,0] * B[0,1] + A[2,1] * B[1,1] + A[2,2] * B[2,1]

    resmat[0,2] = A[0,0] * B[0,2] + A[0,1] * B[1,2] + A[0,2] * B[2,2]
    resmat[1,2] = A[1,0] * B[0,2] + A[1,1] * B[1,2] + A[1,2] * B[2,2]
    resmat[2,2] = A[2,0] * B[0,2] + A[2,1] * B[1,2] + A[2,2] * B[2,2]
    return resmat

@ti.func
def matrix_vec_mult_3(A,x):
    return tm.vec3(A[0,0] * x[0] + A[0,1] * x[1] + A[0,2] * x[2],
                   A[1,0] * x[0] + A[1,1] * x[1] + A[1,2] * x[2],
                   A[2,0] * x[0] + A[2,1] * x[1] + A[2,2] * x[2])

@ti.func
def matrix_mult_4(A,B):
    resmat = tm.mat4(0)
    resmat[0,0] = A[0,0] * B[0,0] + A[0,1] * B[1,0] + A[0,2] * B[2,0] + A[0,3] * B[3,0]
    resmat[1,0] = A[1,0] * B[0,0] + A[1,1] * B[1,0] + A[1,2] * B[2,0] + A[1,3] * B[3,0]
    resmat[2,0] = A[2,0] * B[0,0] + A[2,1] * B[1,0] + A[2,2] * B[2,0] + A[2,3] * B[3,0]
    resmat[3,0] = A[3,0] * B[0,0] + A[3,1] * B[1,0] + A[3,2] * B[2,0] + A[3,3] * B[3,0]
    resmat[0,1] = A[0,0] * B[0,1] + A[0,1] * B[1,1] + A[0,2] * B[2,1] + A[0,3] * B[3,1]
    resmat[1,1] = A[1,0] * B[0,1] + A[1,1] * B[1,1] + A[1,2] * B[2,1] + A[1,3] * B[3,1]
    resmat[2,1] = A[2,0] * B[0,1] + A[2,1] * B[1,1] + A[2,2] * B[2,1] + A[2,3] * B[3,1]
    resmat[3,1] = A[3,0] * B[0,1] + A[3,1] * B[1,1] + A[3,2] * B[2,1] + A[3,3] * B[3,1]
    resmat[0,2] = A[0,0] * B[0,2] + A[0,1] * B[1,2] + A[0,2] * B[2,2] + A[0,3] * B[3,2]
    resmat[1,2] = A[1,0] * B[0,2] + A[1,1] * B[1,2] + A[1,2] * B[2,2] + A[1,3] * B[3,2]
    resmat[2,2] = A[2,0] * B[0,2] + A[2,1] * B[1,2] + A[2,2] * B[2,2] + A[2,3] * B[3,2]
    resmat[3,2] = A[3,0] * B[0,2] + A[3,1] * B[1,2] + A[3,2] * B[2,2] + A[3,3] * B[3,2]
    resmat[0,3] = A[0,0] * B[0,3] + A[0,1] * B[1,3] + A[0,2] * B[2,3] + A[0,3] * B[3,3]
    resmat[1,3] = A[1,0] * B[0,3] + A[1,1] * B[1,3] + A[1,2] * B[2,3] + A[1,3] * B[3,3]
    resmat[2,3] = A[2,0] * B[0,3] + A[2,1] * B[1,3] + A[2,2] * B[2,3] + A[2,3] * B[3,3]
    resmat[3,3] = A[3,0] * B[0,3] + A[3,1] * B[1,3] + A[3,2] * B[2,3] + A[3,3] * B[3,3]
    return resmat

@ti.func
def matrix_inverse(A):
    """
    This is written out a determinant adjugate method of computing an inverse matrix for a 4x4.
    Copied from here: http://semath.info/src/inverse-cofactor-ex4.html

    NOTE: in the future when translating methods from 1-indexed source onto 0-indexed code,
    I would suggest you to type it out onto another document and then find-and-replace
    the index numbers accordingly. This was painful otherwise.
    """
    Ainv = tm.mat4(0)
    det = (A[0,0] * (A[1,1] * A[2,2] * A[3,3] + A[1,2] * A[2,3] * A[3,1] + A[1,3] * A[2,1] * A[3,2] - A[1,3] * A[2,2] * A[3,1] - A[1,2] * A[2,1] * A[3,3] - A[1,1] * A[2,3] * A[3,2])
    -      A[1,0] * (A[0,1] * A[2,2] * A[3,3] + A[0,2] * A[2,3] * A[3,1] + A[0,3] * A[2,1] * A[3,2] - A[0,3] * A[2,2] * A[3,1] - A[0,2] * A[2,1] * A[3,3] - A[0,1] * A[2,3] * A[3,2])
    +      A[2,0] * (A[0,1] * A[1,2] * A[3,3] + A[0,2] * A[1,3] * A[3,1] + A[0,3] * A[1,1] * A[3,2] - A[0,3] * A[1,2] * A[3,1] - A[0,2] * A[1,1] * A[3,3] - A[0,1] * A[1,3] * A[3,2])
    -      A[3,0] * (A[0,1] * A[1,2] * A[2,3] + A[0,2] * A[1,3] * A[2,1] + A[0,3] * A[1,1] * A[2,2] - A[0,3] * A[1,2] * A[2,1] - A[0,2] * A[1,1] * A[2,3] - A[0,1] * A[1,3] * A[2,2])
    )
    inv_det = 1 / det
    Ainv[0,0] =  inv_det * (A[1,1]*A[2,2]*A[3,3] + A[1,2]*A[2,3]*A[3,1] + A[1,3]*A[2,1]*A[3,2] - A[1,3]*A[2,2]*A[3,1] - A[1,2]*A[2,1]*A[3,3] - A[1,1]*A[2,3]*A[3,2])
    Ainv[0,1] = -inv_det * (A[0,1]*A[2,2]*A[3,3] + A[0,2]*A[2,3]*A[3,1] + A[0,3]*A[2,1]*A[3,2] - A[0,3]*A[2,2]*A[3,1] - A[0,2]*A[2,1]*A[3,3] - A[0,1]*A[2,3]*A[3,2])
    Ainv[0,2] =  inv_det * (A[0,1]*A[1,2]*A[3,3] + A[0,2]*A[1,3]*A[3,1] + A[0,3]*A[1,1]*A[3,2] - A[0,3]*A[1,2]*A[3,1] - A[0,2]*A[1,1]*A[3,3] - A[0,1]*A[1,3]*A[3,2])
    Ainv[0,3] = -inv_det * (A[0,1]*A[1,2]*A[2,3] + A[0,2]*A[1,3]*A[2,1] + A[0,3]*A[1,1]*A[2,2] - A[0,3]*A[1,2]*A[2,1] - A[0,2]*A[1,1]*A[2,3] - A[0,1]*A[1,3]*A[2,2])

    Ainv[1,0] = -inv_det * (A[1,0]*A[2,2]*A[3,3] + A[1,2]*A[2,3]*A[3,0] + A[1,3]*A[2,0]*A[3,2] - A[1,3]*A[2,2]*A[3,0] - A[1,2]*A[2,0]*A[3,3] - A[1,0]*A[2,3]*A[3,2])
    Ainv[1,1] =  inv_det * (A[0,0]*A[2,2]*A[3,3] + A[0,2]*A[2,3]*A[3,0] + A[0,3]*A[2,0]*A[3,2] - A[0,3]*A[2,2]*A[3,0] - A[0,2]*A[2,0]*A[3,3] - A[0,0]*A[2,3]*A[3,2])
    Ainv[1,2] = -inv_det * (A[0,0]*A[1,2]*A[3,3] + A[0,2]*A[1,3]*A[3,0] + A[0,3]*A[1,0]*A[3,2] - A[0,3]*A[1,2]*A[3,0] - A[0,2]*A[1,0]*A[3,3] - A[0,0]*A[1,3]*A[3,2])
    Ainv[1,3] =  inv_det * (A[0,0]*A[1,2]*A[2,3] + A[0,2]*A[1,3]*A[2,0] + A[0,3]*A[1,0]*A[2,2] - A[0,3]*A[1,2]*A[2,0] - A[0,2]*A[1,0]*A[2,3] - A[0,0]*A[1,3]*A[2,2])

    Ainv[2,0] =  inv_det * (A[1,0]*A[2,1]*A[3,3] + A[1,1]*A[2,3]*A[3,0] + A[1,3]*A[2,0]*A[3,1] - A[1,3]*A[2,1]*A[3,0] - A[1,1]*A[2,0]*A[3,3] - A[1,0]*A[2,3]*A[3,1])
    Ainv[2,1] = -inv_det * (A[0,0]*A[2,1]*A[3,3] + A[0,1]*A[2,3]*A[3,0] + A[0,3]*A[2,0]*A[3,1] - A[0,3]*A[2,1]*A[3,0] - A[0,1]*A[2,0]*A[3,3] - A[0,0]*A[2,3]*A[3,1])
    Ainv[2,2] =  inv_det * (A[0,0]*A[1,1]*A[3,3] + A[0,1]*A[1,3]*A[3,0] + A[0,3]*A[1,0]*A[3,1] - A[0,3]*A[1,1]*A[3,0] - A[0,1]*A[1,0]*A[3,3] - A[0,0]*A[1,3]*A[3,1])
    Ainv[2,3] = -inv_det * (A[0,0]*A[1,1]*A[2,3] + A[0,1]*A[1,3]*A[2,0] + A[0,3]*A[1,0]*A[2,1] - A[0,3]*A[1,1]*A[2,0] - A[0,1]*A[1,0]*A[2,3] - A[0,0]*A[1,3]*A[2,1])

    Ainv[3,0] = -inv_det * (A[1,0]*A[2,1]*A[3,2] + A[1,1]*A[2,2]*A[3,0] + A[1,2]*A[2,0]*A[3,1] - A[1,2]*A[2,1]*A[3,0] - A[1,1]*A[2,0]*A[3,2] - A[1,0]*A[2,2]*A[3,1])
    Ainv[3,1] =  inv_det * (A[0,0]*A[2,1]*A[3,2] + A[0,1]*A[2,2]*A[3,0] + A[0,2]*A[2,0]*A[3,1] - A[0,2]*A[2,1]*A[3,0] - A[0,1]*A[2,0]*A[3,2] - A[0,0]*A[2,2]*A[3,1])
    Ainv[3,2] = -inv_det * (A[0,0]*A[1,1]*A[3,2] + A[0,1]*A[1,2]*A[3,0] + A[0,2]*A[1,0]*A[3,1] - A[0,2]*A[1,1]*A[3,0] - A[0,1]*A[1,0]*A[3,2] - A[0,0]*A[1,2]*A[3,1])
    Ainv[3,3] =  inv_det * (A[0,0]*A[1,1]*A[2,2] + A[0,1]*A[1,2]*A[2,0] + A[0,2]*A[1,0]*A[2,1] - A[0,2]*A[1,1]*A[2,0] - A[0,1]*A[1,0]*A[2,2] - A[0,0]*A[1,2]*A[2,1])

@ti.func
def rotation_matrix(cosphi,sinphi):
    #Tarkistettu 23.09.2024 t. Antti
    cosphi2 = cosphi ** 2
    sinphi2 = sinphi ** 2
    cos2phi = 2.0 * cosphi2 - 1.0
    sin2phi = 2.0 * sinphi * cosphi

    R_mat = tm.mat4(0)
    """
    R_mat = np.array(
            [[ cosphi2,sinphi2, 0.5 * sin2phi,0.0],
             [ sinphi2,cosphi2,-0.5 * sin2phi,0.0],
             [-sin2phi,sin2phi,       cos2phi,0.0],
             [     0.0,    0.0,           0.0,1.0]])
    """
    R_mat[0,0] = cosphi2
    R_mat[1,0] = sinphi2
    R_mat[2,0] =-sin2phi
    R_mat[3,0] = 0.0

    R_mat[0,1] = sinphi2
    R_mat[1,1] = cosphi2
    R_mat[2,1] = sin2phi
    R_mat[3,1] = 0.0

    R_mat[0,2] = sin2phi * 0.5
    R_mat[1,2] =-sin2phi * 0.5
    R_mat[2,2] = cos2phi
    R_mat[3,2] = 0.0

    R_mat[3,0] = 0.0
    R_mat[3,1] = 0.0
    R_mat[3,2] = 0.0
    R_mat[3,3] = 1.0
    return R_mat

@ti.func
def incident_stokes(i_wl):
    # this is naturally polarized light
    return tm.vec4(0.5, 0.5, 0.0, 0.0)

@ti.func
def pointwise_product_4(u,v):
    return tm.vec4(u[0] * v[0], u[1] * v[1], u[2] * v[2], u[3] * v[3])

@ti.func
def forward_rotation(dir_in,dir_out):
    #dir_in and dir_out are now reversed from Siro formulation.
    # This rotates the polarization vectors so that they are properly aligned
    eps = 0.1 / 180.0 * tm.pi

    # dir_in is parallel to the z-axis
    cond_parallel_z = abs(abs(dir_in[2]) - 1.0) < eps

    # dir_in and dir_out are parallel
    cond_parallel = tm.dot(dir_in - dir_out, dir_in - dir_out) < eps
    cond_antiparallel = tm.dot(dir_in + dir_out, dir_in + dir_out) < eps

    rotmat = tm.mat4(0)
    if cond_parallel_z or cond_parallel or cond_antiparallel:
        rotmat[0,0] = 1.0
        rotmat[1,1] = 1.0
        rotmat[2,2] = 1.0
        rotmat[3,3] = 1.0
    else:
        # TODO: Figure out what is going on in here
        dotprod = tm.dot(dir_in, dir_out)
        proj1 = 1.0 / tm.sqrt(1.0 - dotprod ** 2)
        proj2 = 1.0 / tm.sqrt(dir_in[0] ** 2 + dir_in[1] ** 2)
        norma = proj1 * proj2
        cosphi = norma * (dir_out[0] * dir_in[1] - dir_out[1] * dir_in[0])
        sinphi = norma * (-dir_out[2] + dir_in[2] * dotprod)
        rotmat = rotation_matrix(cosphi,sinphi)
    return rotmat

@ti.func
def cossinalphas(dir_in,dir_out,costheta):
    # NOTE: apologies about this function, this is implemented
    # directly from Siro's routines.f90, scattering_angle subroutine
    # which is not as clear as it could be.
    """
    Here we select a vector in Siro style to find the cosalpha and sinalpha.
    Assume that forward and backward scattering have been checked already.
    """

    plane_x = tm.vec3(0.0)
    plane_x[0] = dir_in[1]
    plane_x[1] = dir_in[0]
    plane_x_norm = tm.length(plane_x)

    if plane_x_norm < 0.001:
        plane_x[0] = dir_in[2]
        plane_x[1] = 0.0
        plane_x[2] = -dir_in[0]
        plane_x_norm = tm.length(plane_x)
    plane_x = plane_x / plane_x_norm

    plane_y = tm.cross(dir_in, plane_x)
    plane_y = plane_y / tm.length(plane_y)
    # at this point, we project dir_out onto the plane defined by
    # plane_x and plane_y
    proj_do = dir_out - tm.dot(dir_in,dir_out) * dir_in
    proj_do = proj_do / tm.length(proj_do)
    cosalpha = tm.dot(plane_x,proj_do)
    sinalpha = tm.dot(plane_y,proj_do)
    #print('cossinalpha',cosalpha,sinalpha,plane_x,plane_y,proj_do)
    return cosalpha, sinalpha

@ti.func
def rotate_polarization(dir_in,dir_out,S):
    # This is the function polaris from polarisation.f90
    eps = 0.1 / 180.0 * tm.pi
    costheta = tm.dot(dir_in,dir_out)
    cond_forward_sca = False
    cond_backward_sca = False
    cosalpha = 0.0
    sinalpha = 0.0
    if ti.abs(1 - costheta) < eps:
        cond_forward_sca = True
    elif ti.abs(1 + costheta) < eps:
        cond_backward_sca = True
    else:
        cosalpha, sinalpha = cossinalphas(dir_in,dir_out,costheta)
        #print("was in else")

    #print(cond_forward_sca,cond_backward_sca,tm.length(dir_in - dir_out),cosalpha,sinalpha,costheta)
    cosbeta, sinbeta = 0.0, 0.0
    rotated_S = tm.mat4(0.0)
    if cond_forward_sca or cond_backward_sca:
        rotated_S = S
    else:
        if abs(abs(dir_in[2]) - 1.0) < eps:
            #print(2)
            # The incoming direction is parallel to the z-axis
            cosbeta = 0.0
            sinbeta = 1.0
        elif abs(dir_out[2] - 1.0) < eps:
            #print(3)
            # The outgoing radiation is toward the positive z-axis
            norm = tm.sqrt(1.0 - dir_in[2] ** 2)
            cosbeta = -dir_in[0] / norm
            sinbeta =  dir_in[1] / norm
        elif abs(dir_out[2] + 1.0) < eps:
            #print(4)
            # The outgoing radiation is toward the negative z-axis
            norm = tm.sqrt(1.0 - dir_in[2] ** 2)
            cosbeta = -dir_in[0] / norm
            sinbeta = -dir_in[1] / norm
        else:
            determinant = dir_in[0] * dir_out[1] - dir_in[1] * dir_out[0]
            if abs(determinant) < eps:
                #print(5)
                cosbeta = 0.0
                sinbeta = 1.0 # TODO: In the original Siro implementation
                # this was commented with "should this be -1?". Research why
                # it should be so.
            else:
                #print(6,dir_in,dir_out,determinant,cosalpha,sinalpha)
                cosbeta = -cosalpha * tm.sqrt(dir_in[0] ** 2 + dir_in[1] ** 2) / tm.sqrt(dir_out[0] ** 2 + dir_out[1] ** 2)
                sinbeta = cosbeta * (dir_in[2] - dir_out[2] * costheta) / determinant
                #print(6,cosbeta,sinbeta)

        if sinalpha == 0.0 and cosalpha == 0.0:
            print('aargh')
        #print(S)
        R_alpha = rotation_matrix(cosalpha,sinalpha)
        #print(R_alpha)
        R_beta = rotation_matrix(cosbeta,sinbeta)
        #print(R_beta)
        SR_beta = matrix_mult_4(S,R_beta)
        rotated_S = matrix_mult_4(R_alpha,SR_beta)
        if False:
            print(S)
            print('^-- orig | rotated --v')
            print(rotated_S)
            print("")
    return rotated_S
