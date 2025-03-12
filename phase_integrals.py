#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 20:37:55 2023

@author: mikkonea
"""

import numpy as np


phase_func_w = ti.field(dtype=ti.f32, shape=(n_sca,n_pos,n_fibo_dirs))
if polariz_mode == 4:
    phase_matrices_w = ti.Matrix.field(n=4, m=4, dtype=ti.f32, shape=(n_sca,n_pos,n_fibo_dirs))
phase_dir_w = ti.Vector.field(n=3, dtype=ti.f32, shape=(n_fibo_dirs,))
phase_area_w = ti.field(dtype=ti.f32, shape=(n_fibo_dirs,))
phase_neigh_w = ti.Vector.field(n=2, dtype=ti.i32, shape=(n_fibo_dirs*n_fibo_dirs,))
n_phase_neigh = ti.field(dtype=ti.i32, shape=())

# this is then multiplied in the A_obs and A_coup functions to calculate the weights
dir_areas = ti.field(dtype=ti.i32, shape=(n_fibo_dirs,max_scatter_couplings,2))
area_amt = ti.field(dtype=ti.i32, shape=(n_fibo_dirs,max_scatter_couplings))

# this is for b_coup
dir_areas_b = ti.field(dtype=ti.i32, shape=(n_fibo_dirs,n_los,max_path_len,2))

# these following arrays are set up in the scattering.py
scatter_coupling_amt = ti.field(dtype=ti.i32,shape=())
node_idx_inv = ti.Vector.field(n=4, dtype=ti.i32, shape=(max_scatter_couplings,))
coup_neigh_amt = ti.field(dtype=ti.i32,shape=()) #these could be u32 or something larger
coup_idx_neighbour = ti.Vector.field(n=2, dtype=ti.i32) # these cause the neighbours to have a ~46k limit
coup_neigh_block1 = ti.root.pointer(ti.i,(max_scatter_couplings,))
coup_neigh_block2 = coup_neigh_block1.pointer(ti.i,(max_scatter_couplings,))
coup_neigh_block2.place(coup_idx_neighbour)

from scipy.spatial import ConvexHull
import math

def fibonacci_sphere(samples=1000):
    #stolen from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

def gen_sphere(n):
    points = np.array(fibonacci_sphere(n))

    hull = ConvexHull(points)
    tris = hull.simplices

    return points, tris

def tri_areas(tris,points):
    n_tri = tris.shape[0]
    A = np.zeros((n_tri,))
    for i in range(n_tri):
        p0 = points[tris[i,0]]
        p1 = points[tris[i,1]]
        p2 = points[tris[i,2]]
        #print(np.abs(np.cross(p1-p0,p2-p0))/2)
        A[i] = np.linalg.norm(np.cross(p1-p0,p2-p0))/2
    return A

def point_areas(tris,world_points):
    n_tri = tris.shape[0]
    n_point = world_points.shape[0]
    At = tri_areas(tris,world_points)
    Ap = np.zeros((n_point,))
    for i_t in range(n_tri):
        for i_p in range(3):
            Ap[tris[i_t,i_p]] += At[i_t] / 3
    # in theory this area should sum up to be 4pi
    # due to the construction method, this is always less than that
    # scaling for good measure
    total = np.sum(Ap)
    Ap *= (4*np.pi) / total
    return Ap

def populate_phase_arrays(i_wl):
    points,tris = gen_sphere(n_fibo_dirs)
    areas = point_areas(tris,points)
    for i in range(points.shape[0]):
        phase_dir_w[i][0] = points[i,0]
        phase_dir_w[i][1] = points[i,1]
        phase_dir_w[i][2] = points[i,2]
        phase_area_w[i] = areas[i]
    #eval phase funs at scattering angles at all
    populate_phase_neigh(tris)
    populate_phase_funs(i_wl)
    if polariz_mode == 4:
        populate_phase_mats(i_wl)

def populate_phase_neigh(tris):
    #optimoi tää, kestää pitkään!!
    #normipyttistä, varmaan joku kaava jo olemassa
    #puske kaikki vaan ärreihin ja sitten np.unique tms etc.
    n_tri = tris.shape[0]
    pair_list = []
    for i_t in range(n_tri):
        tri = tris[i_t]
        pairs = [(tri[0],tri[1]),(tri[1],tri[0]),(tri[2],tri[1]),(tri[1],tri[2]),(tri[0],tri[2]),(tri[2],tri[0])]
        pair_list += pairs
    arr = np.array(pair_list)
    uarr = np.unique(arr,axis=0)
    n_phase_neigh[None] = uarr.shape[0]
    for i in range(n_phase_neigh[None]):
        phase_neigh_w[i][0] = uarr[i,0]
        phase_neigh_w[i][1] = uarr[i,1]


@ti.func
def find_closest_fibodir(dir):
    min_idx = -1
    min_val = 10.0 # this is practically inf, since the distance value should be
    # from 0 to 2
    for i_f in range(n_fibo_dirs):
        d = 1.0 - tm.dot(phase_dir_w[i_f],dir)
        if d < min_val:
            min_idx = i_f
            min_val = d
    return min_idx

@ti.func
def generate_vec(theta,phi):
    return tm.vec3(tm.cos(theta),
                   tm.sin(theta) * tm.cos(phi),
                   tm.sin(theta) * tm.sin(phi))

@ti.kernel
def populate_phase_funs(i_wl:int):
    dir_in = tm.vec3(0.0, 1.0, 0.0) # this is the (theta,phi) = (0,0) vector
    for i_pos in range(n_pos):
        for i_sca in range(n_sca):
            sca = scattering[i_sca,i_pos,i_wl]
            for i_fibo in range(n_fibo_dirs):
                dir_out = phase_dir_w[i_fibo]
                phase_func_w[i_sca,i_pos,i_fibo] = sca * phase_function(i_sca,dir_in,dir_out)


@ti.kernel
def populate_phase_mats(i_wl:int):
    dir_in = tm.vec3(0.0, 1.0, 0.0) # this is the (theta,phi) = (0,0) vector
    for i_pos in range(n_pos):
        for i_sca in range(n_sca):
            sca = scattering[i_sca,i_pos,i_wl]
            for i_fibo in range(n_fibo_dirs):
                dir_out = phase_dir_w[i_fibo]
                phasemat = phase_matrix(i_sca,dir_in,dir_out)
                phase_matrices_w[i_sca,i_pos,i_fibo] = phasemat

@ti.func
def flood_fill(i_coup):
    # flood fills the dir_areas
    timeout = 0
    zeros = 0
    i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
    if i_los == 0 and i_node == 0 and j_los == 0 and j_node == 0:
        # the flood fill is called to uninitialized coupling.
        # let's not increment zeros so that we can exit smoothly
        pass
    else:
        for i_f in range(n_fibo_dirs):
            if dir_areas[i_f,i_coup,0] == 0:
                zeros += 1
    while zeros > 0:
        for i_n in range(n_phase_neigh[None]):
            i = phase_neigh_w[i_n][0]
            j = phase_neigh_w[i_n][1]
            if dir_areas[i,i_coup,0] == 0 and dir_areas[j,i_coup,0] > 0:
                dir_areas[i,i_coup,1] = dir_areas[j,i_coup,0]
        zeros = 0
        for i_f in range(n_fibo_dirs):
            dir_areas[i_f,i_coup,0] = dir_areas[i_f,i_coup,1]
        timeout += 1
        if timeout > n_fibo_dirs:
            zeros = 0
            print("error in fyllning")
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            print(i_coup,i_los,i_node,j_los,j_node)
            for i_f in range(n_fibo_dirs):
                #print(i_f,dir_areas[i_f,i_coup,0])
                pass
            for i_n in range(n_phase_neigh[None]):
                i = phase_neigh_w[i_n][0]
                j = phase_neigh_w[i_n][1]
                #print(i,j)

@ti.func
def flood_fill_b(i_coup):
    # this could be refactored into flood_fill, probably
    timeout = 0
    zeros = 0
    i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
    if i_los == 0 and i_node == 0 and j_los == 0 and j_node == 0:
        # the flood fill is called to uninitialized coupling.
        # let's not increment zeros so that we can exit smoothly
        pass
    else:
        for i_f in range(n_fibo_dirs):
            if dir_areas_b[i_f,i_los,i_node,0] == 0:
                zeros += 1
    while zeros > 0:
        for i_n in range(n_phase_neigh[None]):
            i = phase_neigh_w[i_n][0]
            j = phase_neigh_w[i_n][1]
            if dir_areas_b[i,i_los,i_node,0] == 0 and dir_areas_b[j,i_los,i_node,0] > 0:
                dir_areas_b[i,i_los,i_node,1] = dir_areas_b[j,i_los,i_node,0]
        zeros = 0
        for i_f in range(n_fibo_dirs):
            dir_areas_b[i_f,i_los,i_node,0] = dir_areas_b[i_f,i_los,i_node,1]
            if dir_areas_b[i_f,i_los,i_node,0] == 0:
                zeros += 1
        timeout += 1
        if timeout > n_fibo_dirs:
            zeros = 0
            print("error in bfyllning")

@ti.func
def rotmat(v1,v2):
    #returns the rotation matrix to transform unit vector v1 into v2
    # Source: https://math.stackexchange.com/a/476311
    R = tm.mat3(0.0)
    c = tm.dot(v1,v2)
    if c == -1.0:
        R[0,0] = -1.0
        R[1,1] = -1.0
        R[2,2] = -1.0
    else:
        x = tm.cross(v1,v2)
        s = tm.length(x)

        vx = tm.mat3(0.0)
        vx[1,0] = x[2]
        vx[2,0] = -x[1]
        vx[2,1] = x[0]
        vx[0,1] = -x[2]
        vx[0,2] = x[1]
        vx[1,2] = -x[0]

        vx2 = matrix_mult_3(vx,vx)

        I = tm.mat3(0.0)
        I[0,0] = 1.0
        I[1,1] = 1.0
        I[2,2] = 1.0

        one_per_1plusc = 1 / (1 + c)

        R[0,0] = I[0,0] + vx[0,0] + vx2[0,0] * one_per_1plusc
        R[1,0] = I[1,0] + vx[1,0] + vx2[1,0] * one_per_1plusc
        R[2,0] = I[2,0] + vx[2,0] + vx2[2,0] * one_per_1plusc
        R[0,1] = I[0,1] + vx[0,1] + vx2[0,1] * one_per_1plusc
        R[1,1] = I[1,1] + vx[1,1] + vx2[1,1] * one_per_1plusc
        R[2,1] = I[2,1] + vx[2,1] + vx2[2,1] * one_per_1plusc
        R[0,2] = I[0,2] + vx[0,2] + vx2[0,2] * one_per_1plusc
        R[1,2] = I[1,2] + vx[1,2] + vx2[1,2] * one_per_1plusc
        R[2,2] = I[2,2] + vx[2,2] + vx2[2,2] * one_per_1plusc
    return R

@ti.kernel
def set_up_dir_source(i_source : int):
    dir_in = source_dir[i_source]
    origin = tm.vec3(0.0, 1.0, 0.0)
    R = rotmat(dir_in,origin)
    for i_coup in range(scatter_coupling_amt[None]):
        i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
        dir_out = path_steps[j_node,j_los] - path_steps[i_node,i_los]
        dir_out /= tm.length(dir_out)
        out_rot = matrix_vec_mult_3(R,dir_out)
        i_f = find_closest_fibodir(out_rot)
        if dir_areas_b[i_f,i_coup,0] == 0:
            color = 0
            for i_f in range(n_fibo_dirs):
                if dir_areas_b[i_f,i_los,i_node,0] > color:
                    color = dir_areas_b[i_f,i_los,i_node,0]
            dir_areas_b[i_f,i_coup,0] = color+1
    for i_coup in range(scatter_coupling_amt[None]):
        flood_fill_b(i_coup)

@ti.func
def paint_in_assured_area(i_f,i_coup,dist):
    A = tm.log(2)
    # assured cosine, this variable tells how much of the cosine is at dist = 1.
    # for example, A = log(2) means that assured cosine is 0.5 at the distance 1.
    ass = tm.exp(-A * dist)
    for j_fibo in range(n_fibo_dirs):
        if j_fibo == i_f:
            continue
        cos = tm.dot(phase_dir_w[j_fibo],phase_dir_w[i_f])
        if cos > ass:
            dir_areas[j_fibo,i_coup,0] = -1

@ti.kernel
def set_up_dir_areas():
    origin = tm.vec3(0.0, 1.0, 0.0)
    if False:
        for i_coup in range(scatter_coupling_amt[None]):
            i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
            if i_los == 0 and i_node == 0 and j_los == 0 and j_node == 0:
                continue
            #print(i_coup,i_los,i_node,j_los,j_node)
            dir_in = path_steps[j_node,j_los] - path_steps[i_node,i_los]
            dir_in /= tm.length(dir_in)
            R = rotmat(dir_in,origin)
            color = 1
            # A_obs weights
            dir_out = tm.vec3(0.0)
            dist_out = 1.0
            if j_node > 0:
                dir_out = path_steps[j_node-1,j_los] - path_steps[j_node,j_los]
                dist_out = tm.length(dir_out)
                dir_out /= dist_out
            else:
                dir_out = -instr_view[j_los]
            out_rot = matrix_vec_mult_3(R,dir_out)
            i_f = find_closest_fibodir(out_rot)
            if i_f == -1:
                print(path_steps[j_node,j_los],path_steps[i_node,i_los])
                print(i_los,i_node,j_los,j_node)
                print(dir_in,R)
                print(dir_out,out_rot)
            dir_areas[i_f,i_coup,0] = color
            paint_in_assured_area(i_f,i_coup,dist_out)
            area_amt[color-1,i_coup] = 1
            color += 1
    # A_coup weights
    ti.loop_config(serialize=True)
    for i_cc in range(coup_neigh_amt[None]):
        i_coup, j_coup = coup_idx_neighbour[i_cc]
        i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
        _, _, k_los, k_node = node_idx_inv[j_coup]
        dir_in = path_steps[j_node,j_los] - path_steps[i_node,i_los]
        dir_in /= tm.length(dir_in)
        dir_out = path_steps[k_node,k_los] - path_steps[j_node,j_los]
        dir_out /= tm.length(dir_out)
        R = rotmat(dir_in,origin)
        out_rot = matrix_vec_mult_3(R,dir_out)
        color = 1
        for i_f in range(n_fibo_dirs):
            if dir_areas[i_f,i_coup,0] > color:
                color = dir_areas[i_f,i_coup,0]
        i_f = find_closest_fibodir(out_rot)
        if dir_areas[i_f,i_coup,0] == 0:
            dir_areas[i_f,i_coup,0] = color+1
            area_amt[color-1,i_coup] = 1
        else:
            # the area is already colored in
            _color = dir_areas[i_f,i_coup,0]
            #area_amt[_color-1,i_coup] += 1

    for i_coup in range(scatter_coupling_amt[None]):
        flood_fill(i_coup)

@ti.func
def get_color_in_dir(i_coup,dir_out):
    origin = tm.vec3(0.0, 1.0, 0.0)
    i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
    dir_in = path_steps[j_node,j_los] - path_steps[i_node,i_los]
    dir_in /= tm.length(dir_in)
    R = rotmat(dir_in,origin)
    out_rot = matrix_vec_mult_3(R,dir_out)
    i_f = find_closest_fibodir(out_rot)
    if i_f == -1:
        print(i_los,i_node,j_los,j_node)
        print(dir_in,R)
        print(dir_out,out_rot)
    color = dir_areas[i_f,i_coup,1]
    return color

@ti.func
def get_color_in_dir_b(i_coup,i_source):
    origin = tm.vec3(0.0, 1.0, 0.0)
    dir_in = source_dir[i_source]
    i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
    dir_out = path_steps[j_node,j_los] - path_steps[i_node,i_los]
    dir_out /= tm.length(dir_out)
    R = rotmat(dir_in,origin)
    out_rot = matrix_vec_mult_3(R,dir_out)
    i_f = find_closest_fibodir(out_rot)
    if i_f == -1:
        print(i_los,i_node,j_los,j_node)
        print(dir_in,R)
        print(dir_out,out_rot)
    color = dir_areas_b[i_f,i_los,i_node,1]
    return color

@ti.func
def get_integrated_phasefun(i_coup,dir_out):
    i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
    phasevec = tm.vec4(0.0)
    real_scattering = True
    dir_amt = 1
    if i_los == 0 and i_node == 0 and j_los == 0 and j_node == 0:
        real_scattering = False
    if real_scattering:
        color = get_color_in_dir(i_coup,dir_out)
        _, _, j_los, j_node = node_idx_inv[i_coup]

        for i_fibo in range(n_fibo_dirs):
            if dir_areas[i_fibo,i_coup,1] == color:
                for i_pos in range(n_pos):
                    phasevec[0] += path_basis[j_node,i_pos,j_los] * phase_func_w[0,i_pos,i_fibo] * phase_area_w[i_fibo]
                    if n_sca > 1:
                        phasevec[1] += path_basis[j_node,i_pos,j_los] * phase_func_w[1,i_pos,i_fibo] * phase_area_w[i_fibo]
                        if n_sca > 2:
                            phasevec[2] += path_basis[j_node,i_pos,j_los] * phase_func_w[2,i_pos,i_fibo] * phase_area_w[i_fibo]
                            if n_sca > 3:
                                phasevec[3] += path_basis[j_node,i_pos,j_los] * phase_func_w[3,i_pos,i_fibo] * phase_area_w[i_fibo]
                    #print(j_los,j_node,i_pos,i_fibo,path_basis[j_node,i_pos,j_los],phase_func_w[i_sca,i_pos,i_fibo],phase_area_w[i_fibo])
        # the flux is divided equally between the couplings in the same color
        #dir_amt = area_amt[color-1,i_coup]
    coeff = 1/(4*tm.pi)
    #coeff = 1.0
    #print(phase,dir_amt)
    return phasevec# * coeff# / dir_amt * coeff
    #return phase# / 2#(tm.pi * 4)# / dir_amt

@ti.func
def get_integrated_phasefun_b(i_coup,i_source):
    i_los, i_node, j_los, j_node = node_idx_inv[i_coup]
    phasevec = tm.vec4(0.0)
    real_scattering = True
    if i_los == 0 and i_node == 0 and j_los == 0 and j_node == 0:
        real_scattering = False
    if real_scattering:
        color = get_color_in_dir_b(i_coup,i_source)
        _, _, j_los, j_node = node_idx_inv[i_coup]

        for i_fibo in range(n_fibo_dirs):
            if dir_areas_b[i_fibo,i_los,i_node,1] == color:
                for i_pos in range(n_pos):
                    phasevec[0] += path_basis[j_node,i_pos,j_los] * phase_func_w[0,i_pos,i_fibo] * phase_area_w[i_fibo]
                    if n_sca > 1:
                        phasevec[1] += path_basis[j_node,i_pos,j_los] * phase_func_w[1,i_pos,i_fibo] * phase_area_w[i_fibo]
                        if n_sca > 2:
                            phasevec[2] += path_basis[j_node,i_pos,j_los] * phase_func_w[2,i_pos,i_fibo] * phase_area_w[i_fibo]
                            if n_sca > 3:
                                phasevec[3] += path_basis[j_node,i_pos,j_los] * phase_func_w[3,i_pos,i_fibo] * phase_area_w[i_fibo]
    return phasevec

@ti.func
def get_integrated_phasemat(i_coup,dir_out):
    color = get_color_in_dir(i_coup,dir_out)
    _, _, j_los, j_node = node_idx_inv[i_coup]
    phasemat = tm.mat4(0.0)
    for i_fibo in range(n_fibo_dirs):
        if dir_areas[i_fibo,i_coup,1] == color:
            for i_sca in range(n_sca):
                for i_pos in range(n_pos):
                    phasemat += path_basis[j_node,i_pos,j_los] * phase_matrices_w[i_sca,i_pos,i_fibo] * phase_area_w[i_fibo]
    ph = get_integrated_phasefun(i_coup,dir_out)
    ph = 1.0
    dir_amt = area_amt[color-1,i_coup]
    return phasemat * (1 / (2*ph))# / dir_amt
