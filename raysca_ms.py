# Raysca^2
# "If Raysca is so great, how come there's no Raysca 2?"
# New name: Graysca


# Three parts:
# 1. trace optical paths and deduce sufficient integration steps
# 2. link nodes to each other with some logic
#  - logic can be pairwise computation of mean xsec_sca at endpoints divided by distance?
#  - DISORT/2Stream like structure also possible.
# 3. do the RT computation
#  - two ways possible:
#    - systematic or randoms walks through the graph
#    - solve the polarized fluxes from one element to another and aggregate the
#      transmissivity at the end.

# TODO: read other 3D RT models
    #LESS: https://www.sciencedirect.com/science/article/abs/pii/S0034425718305443
    #I3RC: https://earth.gsfc.nasa.gov/climate/model/i3rc/publicmodels
    #Trident: https://arxiv.org/pdf/2111.05862.pdf
    #DART: https://www.mdpi.com/2072-4292/11/22/2637
    #skirt: https://www.aanda.org/articles/aa/abs/2014/11/aa24747-14/aa24747-14.html
    #Shdom: https://coloradolinux.com/shdom/

# TODO: check if there are mixed dense/sparse matrix dimensions: possible in taichi!
# TODO: laadi infrastruktuurikaavio. mitkä on vaihtoehtoisia, mitkä voi ratkoa paralleelisti
# TODO: jacobian-lasku voisi olla osittain analyyttinen tai sitten taichin backpropagation
#       tästä algorakenteesta johtuen ei tarvihe uudelleenevaluoida kaikkia steppejä!
# TODO: fluoresenssi ja raman-sironta voisi olla vaan matriisimuotoinen sirontakrossari!!!
# TODO: Computationally nice way to develop arbitrary contour lines. Partially defined,
#       so that the distance criterion is cheap far away and more expensive closer
#       to the boundary? Note that no gradients are needed.
# TODO: hyperparametric optimization mode for figuring out the coupling parameter
#       or minimum step length.
# TODO: Evaluate the Muller matrices at a fixed angle grid, then LUT check from
#       there when building the large matrix system. Same for polarization
#       angle selection.

# Limitations:
# * only one phase matrix for all of the wavelengths in the band.
#   If several phases matrices are needed, then each needs their own band of computations.
# * The last node of each path is surface, or other boundary effect.
#   The path tracing and node scattering computation logic needs to be modified if this
#   is desired.
# * Radiation sources may only be situated outside the computation medium. This could
#   be circumvented by creating a small cavity of "outside" within the medium,
#   where the radiation source is placed. If the beam then hits that cavity, the
#   trace is stopped. Medium emission can be handled with emissivity.
# * Even though the fine medium details are handled in trace_path, the coupled
#   node scattering paths are now just linearly interpolated from each end points.
# * Only unpolarized emission and absorption is considered. Scattering and reflectivity
#   however will polarize. Birefrigence and dichroism are not possible either.

def trace_path(beam_start,beam_direction,integrable,stopper,
                                  source_function,minimum_step,interp_error):
    """
    Function trace_path will trace a path through 3D space starting from beam_start
    in the direction defined by beam_direction until function stopper is less than
    zero. This function will create steps of varying step size according to the
    integrable function: the step lengths will be of multiples of minimum_step.
    The step length will be as long as possible without resulting into an
    linear interpolation error larger than interp_error in the integrable.

    TODO: refraction and other beam direction changes should be taken into
    account in this part of the RT.
    TODO: terminator effect should be checked for each of the sources.

    Inputs:
        beam_start, (3,) Numpy float array: where the beam is started to trace

        beam_direction, (3,) Numpy float array: normalized vector toward which
        the steps are taken.

        integrable, function((3,) Numpy float array) -> float: function which
        can be interpolated using the output points without making a local interpolation
        error larger than interp_error.

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

        path_lens, (N-1,) Numpy float array: The lengths of the segments between
        the path nodes. path_lens[0] = np.linalg.norm(path_nodes[:,1] - path_nodes[:,0])
        etc.

        direct_transmittance, integer or None: Returns the radiation source index at which
        the last path segment is pointing. If the path does not point at a radiation
        source outside, then this is None.
    """
    pass

def link_scattering_nodes(path_nodes_list,path_lens_list,coupler,stopper,minimum_coupling):
    """
    Function link_scattering_nodes will create an adjacency matrix of the nodes
    between each of the paths as long as the coupler function value is larger than
    minimum_coupling.

    Note: On a single path, two sequential nodes are always linked. Even though
    the scattering coupling would not be high enough for coupling, the flux backward
    the path is required for the total observed radiance computations.

    Inputs:
        path_nodes_list, list of path_nodes as defined in trace_path

        path_lens_list, list of path_lens as defined in trace_path

        coupler, function((3,) Numpy float array, (3,) Numpy float array) -> float:
        Yields the degree of coupling between two points in the two nodes.
        Should be symmetric. Initially this could be just the sum of mean scattering
        cross-section of the nodes divided by the distance between them. This could
        also take into accout the phase function

        stopper, function((3,) Numpy float array) -> float: The beam is traced until
        this function is less than zero with the argument beam_position. Needed
        within non-convex media.

        minimum_coupling, float: How large the coupling should be between two
        nodes before it is considered

    Outputs:
        coupling_matrix, (M,M) Numpy sparse float array: If coupling_matrix[m,n] is
        set, then node with index m is linked with node with index n and the
        array value is the distance between the nodes. The node
        indices are global and not specific to certain path. The indices are
        derived sequentially from first path's first node onwards with second path's
        first node coming directly after first path's last node and so on.
            The scattering length used in the calculations is half of the distance
        between the nodes. For example, radiance from node A is scattered by
        node B with dist(A,B) / 2 length with the scattering cross-sections at
        node B. Likewise the radiance from B is scattered by A with the same
        scattering distance, but using the scattering cross-section of A.
            NOTE 7.7.2023: The scattering length instead needs to be the total
        length between the nodes. This was proven by formula twisting on the whiteboard.

        weighing_matrix, (M,M) Numpy sparse float array: Because the linking between
        the nodes are not necessarily uniformly distributed onto the unit sphere, a
        way to estimate each contribution needs to be weighed appropriately.
            One way to estimate this is to compute the Voronoi areas of each
        direction vector on the sphere. A fast way to compute these is needed.
        What kind of selection is done if two fluxes are pointing exactly into
        same direction?
            Initially this could just be a uniform distribution and more fancier
        is implemented if problems arise.


    Notes:

    """
    pass

def node_scattering_properties(path_nodes_list,coupling_matrix,phase_function,
                                            scattering_xsec,reflectance_function,
                                            albedo_function):
    """
    To prepare the RT computation, scattering properties in each of the nodes is
    needed.

    Inputs:
        coupling matrix, (M,M) Numpy sparse float array: as defined in function
        link_scattering_nodes.

        phase_function, function((3,) Numpy float array, (3,) Numpy float array,
        (3,) Numpy float array) -> (P,P) Numpy float array OR float: Returns
        the scalar phase function or the polarized phase matrix at position
        defined by the first argument from the direction defined by the second argument
        into the direction defined by the third argument.

        scattering_xsec, function((3,) Numpy float array) -> (W,) Numpy float array:
        The wavelength-dependent scattering cross-section at arbitrary 3D position.
        W is the wavelength amount.

        reflectance_function, function((3,) Numpy float array, (3,) Numpy float array,
        (3,) Numpy float array) -> (P,P) Numpy float array OR float: Returns
        the scalar reflectance function or the polarized phase matrix at position
        defined by the first argument from the direction defined by the second argument
        into the direction defined by the third argument. The last node in each
        of the paths will have the sum of the reflectance function and the atmospheric
        scattering function.

        albedo_function, function((3,) Numpy float array) -> (W,) Numpy float array:
        The wavelength-dependent reflectance at arbitrary 3D position.
        W is the amount of wavelengths.

    Outputs:
        scatter_matrix, (M,M,P,P) Numpy sparse float array: The Muller matrix
        when scattering from one node to another with appropriate rotations
        applied. Despite the name, this includes the surface reflectances too.

        scaxsec_matrix, (M,W) Numpy float array: The scattering spectrum
        of each node. Despite the name, this includes the surface reflectances too.

    """
    pass

def node_emissivity_properties(path_nodes_list,coupling_matrix,emissivity):
    """
    This function creates the unpolarized emission terms for the solve_fluxes
    function.

    Inputs:
        path_nodes_list, list of path_nodes as defined in trace_path.

        coupling_matrix, (M,M) Numpy sparse float array: as defined in function
        link_scattering_nodes.

        emissivity, function((3,) Numpy float array) -> (W,) Numpy float array:
        This function yields the spectral emissivity at a point within the medium.

    Outputs:
        emissivity_matrix(M,W) Numpy float array: The element [m,w] is the
        emissivity of node m at the wavelength index w. The total emission volume
        is computed with the total length of connecting paths divided by 2 and for
        individual compontents, the...
        NOTE 7.7.2023: The emissivity of each path needs to be the path length and
        not a volumetric property. This was proven by formula twisting in the whiteboard.

    """
    pass

def path_transmissivity_properties(path_nodes_list,coupling_matrix,extinction_xsec):
    """
    The spectral transmissivity of each of the paths between nodes is needed.

    Inputs:
        path_nodes_list, list of path_nodes as defined in trace_path.

        coupling_matrix, (M,M) Numpy sparse float array: as defined in function
        link_scattering_nodes.

        extinction_xsec, function((3,) Numpy float array) -> (W,) Numpy float
        array: The function to compute extinction in the medium at arbitrary 3D
        coordinates.

    Outputs:
        transmissivity_matrix, (M,M,W) Numpy sparse float array: The element [m,n,w]
        is the transmissivity between nodes indexed m and n at the wavelength index
        w.
    """
    pass

def node_source_contributions(path_nodes_list,phase_function,source_function_list,extinction_xsec):
    """
    In addition to the fluxes due to the scattering, the incident radiation fluxes
    are present in the nodes. The transmittances are computed through the medium
    from the sources to the nodes. All the source effects are here, which are
    i.e. shadowing, terminator effects, direct transmittances and such.

    TODO: This function also needs to compute the path transmittances for each
    nodes and source points

    Inputs:
        path_nodes_list, list of path_nodes as defined in trace_path.

        phase_function, as defined in node_scattering_properties.

        source_function_list, list of function((3,) Numpy float array)
         -> ((3,) Numpy float array, (W,P) Numpy float array): The incident
         radiation spectrum at specific location and its incident direction for
         each of the radiation sources.

    Outputs:
        source_matrix, (M,M,S,W,P) Numpy sparse float array

    The incident radiation in each of the nodes is computed here. Transmissivity
    from each of the sources into each of the nodes is estimated
    (M,M,S,W,P) into each direction at each node.

    Notes:
        In both node_source_contributions and node_scattering_properties the
        vectors between coupled nodes are computed. This is redundant and could
        be optmized.
    """
    pass

def solve_fluxes(transmissivity_matrix, scatter_matrix,scattering_xsec, source_matrix, emissivity_matrix):
    """
    Solve the fluxes at each of the nodes. Due to the precomputations we can obtain
    the fluxes with solving a large linear system of form Ax = b, where b contains
    the source and emission terms, A contains information about the transmissivities
    between the nodes and their corresponding scattering matrices and cross-sections,
    and x is the polarized fluxes between each of the nodes.

    Inputs:
        transmissivity_matrix, as defined in path_transmissivity_properties.

        scatter_matrix, as defined in node_scattering_properties.

        scattering_xsec, as defined in node_scattering_properties.

        source_matrix, as defined in node_source_contributions.

        emissivity_matrix, as defined in node_emissivity_properties.
    Outputs:
        radiance_fluxes, (M,M,W,P) Numpy sparse float array: The polarized radiance
        spectra between each of the linked nodes.

    """
    pass

def compute_observed_radiances(path_nodes_list, transmissivity_matrix, radiance_fluxes):
    """
    When the fluxes are solved, then the observed radiances are computed by
    integrating the fluxes backward to the path starting nodes. The fluxes in question
    are the ones linked backward along the path with their transmittances computed
    along the path.

    NOTE: If the scattering coupling is so high that some node is coupled with
    several nodes on the same path, are those fluxes computed here too? Arguably,
    no. If the node is coupled with another node previously on the path but not
    directly the one before, then its effect on the total fluxes are taken into
    account when the linked flux is traced back to the observation point.
        This might be an issue when weighing the node fluxes.
    UPDATE 7.7.2023
        If a node is coupled with several other nodes on the same path, that is
    not a problem. This is because of the fact that the scattered energy is
    divided between the several paths. This is okay and it should be included.
    However, scattering in the "same direction" should depend on the instrument's
    observational capabilities. and its pixels' angular width.

    Inputs:
        path_nodes_list, list of path_nodes as defined in trace_path.

        transmissivity_matrix, as defined in path_transmissivity_properties.

        radiance_fluxes, as defined in solve_fluxes.
    Outputs:
        observed_radiances, (N_path,W,P) Numpy float array: The observed radiance
        spectra at the start of each path.
    """
    pass

def walk_radiance_graph():
    """
    Alternatively the radiative transfer problem can be computed by walking the
    scattering graph

    TODO: this function
    """
    pass
