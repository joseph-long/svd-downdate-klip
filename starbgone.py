import warnings
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import aslinearoperator, svds
from scipy.linalg import lapack
import skimage.transform
# GPU computing libs are wrapped in a try/except so that
# this file is still importable when they're absent, but
# that's not a recommended way to use this library!
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    warnings.warn('PyTorch is unavailable, attempts to use PyTorch solvers will error')
    torch = None
    HAVE_TORCH = False
try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    warnings.warn('CuPy is unavailable, attempts to use GPU solvers will error')
    class cupy():
        @staticmethod
        def get_array_module(array):
            '''Stub for missing cupy.get_array_module that always
            returns NumPy for CPU/GPU generic code that calls
            this function'''
            return np
    cp = cupy
    HAVE_CUPY = False


def torch_svd(array, full_matrices=False, n_modes=None):
    '''Wrap `torch.svd` to handle conversion between NumPy/CuPy arrays
    and Torch tensors. Returns U s V such that
    allclose(U @ diag(s) @ V.T, array) (with some tolerance).

    Parameters
    ----------
    array : (m, n) array
    full_matrices : bool (default False)
        Whether to return full m x m U and full n x n V,
        otherwise U is m x r and V is n x r
        where r = min(m,n,n_modes)
    n_modes : int or None
        Whether to truncate the decomposition, keeping the top
        `n_modes` greatest singular values and corresponding vectors

    Returns
    -------
    mtx_u
    diag_s
    mtx_v
    '''
    xp = cp.get_array_module(array)
    torch_array = torch.as_tensor(array)
    # Note: use of the `out=` argument for torch.svd and preallocated
    # output tensors proved not to save any runtime, so for simplicity
    # they're not retained.
    torch_mtx_u, torch_diag_s, torch_mtx_v = torch.svd(torch_array, some=not full_matrices)
    mtx_u = xp.asarray(torch_mtx_u)
    diag_s = xp.asarray(torch_diag_s)
    mtx_v = xp.asarray(torch_mtx_v)
    if n_modes is not None:
        return mtx_u[:,:n_modes], diag_s[:n_modes], mtx_v[:,:n_modes]
    else:
        return mtx_u, diag_s, mtx_v


def torch_symeig(array, n_modes=None):
    '''Wrap `torch.symeig` to handle conversion between NumPy/CuPy arrays
    and Torch tensors. Returns a tuple of `(evals, evecs)`.

    Parameters
    ----------
    array : (m, m) array
        Real symmetric matrix
    n_modes : int or None
        Whether to truncate the decomposition, keeping the top
        `n_modes` greatest singular values and corresponding vectors

    Returns
    -------
    (evals, evecs)
    '''
    xp = cp.get_array_module(array)
    torch_array = torch.as_tensor(array)
    torch_evals, torch_evecs = torch.symeig(torch_array, eigenvectors=True)
    evals = xp.asarray(torch_evals)
    evals = xp.flip(evals)
    evecs = xp.asarray(torch_evecs)
    evecs = xp.flip(evecs, axis=1)
    if n_modes is not None:
        return evals[:n_modes], evecs[:,:n_modes]
    else:
        return evals, evecs

def generic_svd(array, full_matrices=False, n_modes=None):
    '''Calls np.linalg.svd or cp.linalg.svd. Returns U s V such that
    allclose(array, U @ diag(s) @ V.T) (with some tolerance).

    Parameters
    ----------
    array : (m, n) array
    full_matrices : bool (default False)
        Whether to return full m x m U and full n x n V,
        otherwise U is m x r and V is n x r
        where r = min(m,n,n_modes)
    n_modes : int or None
        Whether to truncate the decomposition, keeping the top
        `n_modes` greatest singular values and corresponding vectors

    Returns
    -------
    mtx_u
    diag_s
    mtx_v
    '''
    xp = cp.get_array_module(array)
    mtx_u, diag_s, mtx_vt = xp.linalg.svd(array, full_matrices=full_matrices)
    mtx_v = mtx_vt.T
    return mtx_u[:,:n_modes], diag_s[:n_modes], mtx_v[:,:n_modes]

def generic_eigh(array, n_modes=None):
    '''Calls np.linalg.eigh or cp.linalg.eigh. Returns eigenvalues
    and eigenvectors of `array`, optionally truncating after the
    largest `n_modes` eigenvalues.

    Parameters
    ----------
    array : (m, m) array
        Real symmetric matrix
    n_modes : int or None
        Whether to truncate the decomposition, keeping the top
        `n_modes` greatest eigenvalues and corresponding eigenvectors

    Returns
    -------
    evals, evecs
    '''
    xp = cp.get_array_module(array)
    evals, evecs = xp.linalg.eigh(array)
    evals = xp.flip(evals)
    evecs = xp.flip(evecs, axis=1)
    if n_modes is not None:
        return evals[:n_modes], evecs[:,:n_modes]
    else:
        return evals, evecs

def cpu_top_k_svd_arpack(array, n_modes=None):
    '''Calls scipy.sparse.linalg.svds to compute top `n_modes`
    singular vectors.  Returns U s V such that
    `U @ diag(s) @ V.T` is the rank-`n_modes` SVD of `array`

    Parameters
    ----------
    array : (m, n) array
    n_modes : int or None
        Compute `n_modes` greatest singular values
        and corresponding vectors, or else default to
        ``n_modes = min(m,n)``

    Returns
    -------
    mtx_u
    diag_s
    mtx_v
    '''
    if n_modes is None:
        n_mods = min(array.shape)
    mtx_u, diag_s, mtx_vt = svds(aslinearoperator(array), k=n_modes)
    return mtx_u, diag_s, mtx_vt.T

def cpu_top_k_cov_syevr(array, n_modes=None):
    '''Calls `scipy.linalg.lapack.[ds]syevr` to compute top `n_modes`
    eigenvectors of a covariance array `array`.

    Parameters
    ----------
    array : (m, m) array
        Real symmetric matrix
    n_modes : int or None
        Compute `n_modes` largest eigenvalues
        and corresponding vectors, or else default to
        ``n_modes = m``

    Returns
    -------
    evals, evecs
    '''
    if array.dtype == np.float32:
        syevr = lapack.ssyevr
    elif array.dtype == np.float64:
        syevr = lapack.dsyevr
    upper_idx = array.shape[0]
    lower_idx = upper_idx - n_modes+1
    w, z, m, isuppz, info = syevr(
        array,
        compute_v=1,
        range='I',
        iu=upper_idx,
        il=lower_idx,
    )
    if info != 0:
        raise RuntimeError("LAPACK SYEVR returned {}".format(info))
    evals = np.flip(w)
    evecs = np.flip(z, axis=1)
    return evals[lower_idx-1:upper_idx+1], evecs


def minimal_downdate(mtx_u, diag_s, mtx_v, min_col_to_remove, max_col_to_remove, solver=generic_svd, compute_v=False):
    '''Modify an existing SVD `mtx_u @ diag(diag_s) @ mtx_v.T` to
    remove columns given by `col_idxs_to_remove`, returning
    a new diagonalization

    Parameters
    ----------
    mtx_u : array (p, r)
    diag_s : array (r,)
    mtx_v : array (q, r)
    col_idxs_to_remove : iterable of integers
    solver : callable (default torch_svd)
        Function to use for internal
        rediagonalization of the low-rank SVD,
        must accept `array` and return `mtx_u, diag_s, mtx_v`
    compute_v : bool (default False)
        Multiply out the new right singular vectors instead
        of discarding their rotation

    Returns
    -------
    new_mtx_u : array (p, r)
    new_diag_s : array (r,)
    new_mtx_v : array (q, r) or None
        If `compute_v` is True this is the updated V matrix,
        otherwise None.
    '''
    xp = cp.get_array_module(mtx_u)
    gpu = xp == cp
    dim_p, dim_q = mtx_u.shape[0], mtx_v.shape[0]
    dim_r = diag_s.shape[0]
    assert mtx_u.shape[1] == dim_r
    assert mtx_v.shape[1] == dim_r

    # Omit computation of mtx_a and mtx_b as their products
    # mtx_uta and mtx_vtb can be expressed without the intermediate
    # arrays.
    dim_c = max_col_to_remove - min_col_to_remove

    # Omit computation of P, R_A, Q, R_B
    # as they represent the portion of the update matrix AB^T
    # not captured in the original basis and we're making
    # the assumption that downdating our (potentially truncated)
    # SVD doesn't require new basis vectors, merely rotating the
    # existing ones. Indeed, P, R_A, Q, and R_B are very close to
    # machine zero

    # "Eigen-code" the update matrices from both sides
    # into the space where X is diagonalized (and truncated)
    #
    # This is just the first part of the product that would have been
    # formed to make mtx_a:
    mtx_uta = -(xp.diag(diag_s) @ mtx_v[min_col_to_remove:max_col_to_remove].T)
    # and just the rows of V corresponding to removed columns:
    mtx_vtb = mtx_v[min_col_to_remove:max_col_to_remove].T

    # Additive modification to inner diagonal matrix
    mtx_k = xp.diag(diag_s)
    mtx_k += mtx_uta @ mtx_vtb.T  # U^T A is r x c, (V^T B)^T is c x r, O(r c r) -> r x r

    # Smaller (dimension r x r) SVD to re-diagonalize, giving
    # rotations of the left and right singular vectors and
    # updated singular values
    mtx_uprime, diag_sprime, mtx_vprime = solver(mtx_k)

    # Compute new SVD by applying the rotations
    new_mtx_u = mtx_u @ mtx_uprime
    new_diag_s = diag_sprime
    if compute_v:
        new_mtx_v = mtx_v @ mtx_vprime
        # columns of X become rows of V, delete the dropped ones
        new_mtx_v = drop_idx_range_rows(new_mtx_v, min_col_to_remove, max_col_to_remove)
    else:
        new_mtx_v = None
    return new_mtx_u, new_diag_s, new_mtx_v

def unwrap_cube(cube, good_pix_mask):
    '''Unwrap a shape (planes, m, n) `cube` and transpose into
    a (pix, planes) matrix, where `pix` is the number of *True*
    entries in a (m, n) `mask` (i.e. False entries are removed)

    Parameters
    ----------
    cube : array (planes, m, n)
    good_pix_mask : array (m, n)
        Pixels to include in `matrix`

    Returns
    -------
    matrix : array (pix, planes)
        Vectorized images, one per column
    x_indices : array (pix,)
        The x indices into the original image that correspond
        to each entry in the vectorized image
    y_indices : array (pix,)
        The y indices into the original image that correspond
        to each entry in the vectorized image
    '''
    xp = cp.get_array_module(cube)
    yy, xx = xp.indices(cube.shape[1:])
    x_indices = xx[good_pix_mask]
    y_indices = yy[good_pix_mask]
    return cube[:,good_pix_mask].T, x_indices, y_indices

def unwrap_image(image, good_pix_mask):
    '''Unwrap a shape (m, n) `image` and transpose into a (pix,)
    vector, where `pix` is the number of *True* entries in a (m, n)
    `mask` (i.e. False entries are removed)

    Parameters
    ----------
    image : array (m, n)
    good_pix_mask : array (m, n)
        Pixels to include in `vector`

    Returns
    -------
    vector : array (pix,)
        Vectorized image
    x_indices : array (pix,)
        The x indices into the original image that correspond
        to each entry in the vectorized image
    y_indices : array (pix,)
        The y indices into the original image that correspond
        to each entry in the vectorized image
    '''
    xp = cp.get_array_module(image)
    cube, x_indices, y_indices = unwrap_cube(image[xp.newaxis,:,:], good_pix_mask)
    return cube[:,0], x_indices, y_indices


def wrap_matrix(matrix, shape, x_indices, y_indices):
    '''Wrap a (planes, pix) matrix into a shape `shape`
    data cube, where pix is the number of entries in `x_indices`
    and `y_indices`

    Parameters
    ----------
    matrix
    shape
    x_indices
    y_indices

    Returns
    -------
    cube
    '''
    xp = cp.get_array_module(matrix)
    cube = xp.zeros(shape)
    cube[:,y_indices,x_indices] = matrix.T
    return cube

def wrap_vector(image_vec, shape, x_indices, y_indices):
    '''Wrap a (pix,) vector into a shape `shape` image,
    where pix is the number of entries in `x_indices`
    and `y_indices`

    Parameters
    ----------
    vector
    shape
    x_indices
    y_indices

    Returns
    -------
    vector
    '''
    xp = cp.get_array_module(image_vec)
    matrix = image_vec[:,xp.newaxis]
    cube = wrap_matrix(matrix, (1,) + shape, x_indices, y_indices)
    return cube[0]


def mean_subtract_vecs(image_vecs):
    xp = cp.get_array_module(image_vecs)
    mean_vec = xp.average(image_vecs, axis=1)
    image_vecs_meansub = image_vecs - mean_vec[:,xp.newaxis]
    return image_vecs_meansub, mean_vec

class Decomposer:
    def __init__(self, image_vecs, n_modes, solver=None):
        self.image_vecs = image_vecs
        self.meansub_image_vecs, self.mean_vec = mean_subtract_vecs(image_vecs)
        self.n_modes = n_modes
        self.xp = cp.get_array_module(image_vecs)
        self.idxs = self.xp.arange(self.meansub_image_vecs.shape[1])
        self.solver = solver
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        raise NotImplementedError()


def drop_idx_range_cols(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    xp = cp.get_array_module(arr)
    rows, cols = arr.shape
    n_drop = max_excluded_idx - min_excluded_idx
    out_shape = (rows, cols - n_drop)
    out = xp.empty(out_shape, dtype=arr.dtype)
    # L | |  R

    # L
    out[:,:min_excluded_idx] = arr[:,:min_excluded_idx]
    # R
    out[:,min_excluded_idx:] = arr[:,max_excluded_idx:]
    return out

def drop_idx_range_rows(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    xp = cp.get_array_module(arr)
    rows, cols = arr.shape
    n_drop = max_excluded_idx - min_excluded_idx
    out_shape = (rows - n_drop, cols)
    out = xp.empty(out_shape, dtype=arr.dtype)
    #  U
    # ===
    #  L

    # U
    out[:min_excluded_idx] = arr[:min_excluded_idx]
    # L
    out[min_excluded_idx:] = arr[max_excluded_idx:]
    return out

class SVDDecomposer(Decomposer):
    def __init__(self, image_vecs, n_modes, solver=None):
        super().__init__(image_vecs, n_modes, solver=solver)
        if self.solver is None:
            self.solver = generic_svd
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        ref = drop_idx_range_cols(self.meansub_image_vecs, min_excluded_idx, max_excluded_idx)
        u, s, v = self.solver(ref, n_modes=self.n_modes)
        return u[:,:self.n_modes]

class MinimalDowndateSVDDecomposer(Decomposer):
    def __init__(self, image_vecs, n_modes, solver=None, extra_modes=1, initial_solver=None):
        super().__init__(image_vecs, n_modes, solver=solver)
        if self.solver is None:
            self.solver = generic_svd
        if initial_solver is None:
            self.initial_solver = self.solver
        self.n_modes = n_modes
        self.mtx_u, self.diag_s, self.mtx_v = self.initial_solver(self.meansub_image_vecs, n_modes=n_modes+extra_modes)
        self.idxs = self.xp.arange(image_vecs.shape[1])
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        new_u, new_s, new_v = minimal_downdate(
            self.mtx_u,
            self.diag_s,
            self.mtx_v,
            min_col_to_remove=min_excluded_idx,
            max_col_to_remove=max_excluded_idx,
            solver=self.solver
        )
        return new_u[:,:self.n_modes]

class ReuseSVDDecomposer(Decomposer):
    def __init__(self, image_vecs, n_modes, solver=None):
        super().__init__(image_vecs, n_modes, solver=solver)
        if self.solver is None:
            self.solver = generic_svd
        self.n_modes = n_modes
        self.mtx_u, self.diag_s, self.mtx_v = self.solver(self.meansub_image_vecs, n_modes=n_modes)
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        return self.mtx_u[:,:self.n_modes]

class RandomizedSVDDecomposer(Decomposer):
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        model = sklearn.decomposition.PCA(
            n_components=self.n_modes,
            copy=False,
            whiten=False,
            svd_solver='randomized'
        )
        data_subset = np.delete(self.meansub_image_vecs, slice(min_excluded_idx, max_excluded_idx), axis=1)
        model.fit(data_subset.T)
        return model.components_.T

def drop_idx_range_rows_cols(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    xp = cp.get_array_module(arr)
    rows, cols = arr.shape
    assert rows == cols
    n_drop = max_excluded_idx - min_excluded_idx
    out_shape = (rows - n_drop, cols - n_drop)
    out = xp.empty(out_shape, dtype=arr.dtype)
    # UL | | U R
    # ===   ====
    # LL | | L R

    # UL
    out[:min_excluded_idx,:min_excluded_idx] = arr[:min_excluded_idx,:min_excluded_idx]
    # UR
    out[:min_excluded_idx,min_excluded_idx:] = arr[:min_excluded_idx,max_excluded_idx:]
    # LL
    out[min_excluded_idx:,:min_excluded_idx] = arr[max_excluded_idx:,:min_excluded_idx]
    # LR
    out[min_excluded_idx:,min_excluded_idx:] = arr[max_excluded_idx:,max_excluded_idx:]
    return out

class CovarianceDecomposition(Decomposer):
    def __init__(self, image_vecs, n_modes, solver=None):
        super().__init__(image_vecs, n_modes, solver=solver)
        self.covariance = self.meansub_image_vecs.T @ self.meansub_image_vecs
        if self.solver is None:
            self.solver = generic_eigh
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        xp = cp.get_array_module(self.meansub_image_vecs)
        temp_covar = drop_idx_range_rows_cols(self.covariance, min_excluded_idx, max_excluded_idx)
        evals, evecs = self.solver(temp_covar, n_modes=self.n_modes)
        indices = xp.arange(self.meansub_image_vecs.shape[1])
        mask = (indices < min_excluded_idx) | (indices >= max_excluded_idx)
        reference = self.meansub_image_vecs[:,mask]
        Z_KL = reference @ (evecs * np.power(evals, -1/2))
        # Truncate
        Z_KL_truncated = Z_KL[:,:self.n_modes]
        return Z_KL_truncated

def klip_frame(target, decomposer, exclude_idx_min, exclude_idx_max):
    eigenimages = decomposer.eigenimages(exclude_idx_min, exclude_idx_max)
    meansub_target = target - decomposer.mean_vec
    return meansub_target - eigenimages @ (eigenimages.T @ meansub_target)


def klip_to_modes(image_vecs, decomp_class, n_modes, solver=None, exclude_nearest=0):
    '''
    Parameters
    ----------
    image_vecs : array (m, n)
        a series of n images arranged into columns of m pixels each
        and mean subtracted such that each pixel-series mean is 0
    decomp_class : Decomposer subclass
        Must accept image_vecs, n_modes, solver in __init__ and implement
        eigenimages() method
    n_modes : int
        Rank of low-rank decomposition
    solver : callable (default None)
        Explicitly select the solver for SVD or symmetric eigenproblem
        (i.e. for swapping between cuSolver and MAGMA) If `None` then
        let the `decomp_class` decide (it will receive solver=None
        when instantiated)
    exclude_nearest : int
        In addition to excluding the current frame, exclude
        this many adjacent frames. (Note the first and last
        few frames won't exclude the same number of frames;
        they will just go to the ends of the dataset.)
    '''
    xp = cp.get_array_module(image_vecs)
    gpu = xp == cp
    _, n_frames = image_vecs.shape

    output = xp.zeros_like(image_vecs)
    idxs = xp.arange(image_vecs.shape[1])
    decomposer = decomp_class(image_vecs, n_modes, solver=solver)
    for i in range(image_vecs.shape[1]):
        output[:,i] = klip_frame(
            image_vecs[:,i],
            decomposer,
            exclude_idx_min=max(i - exclude_nearest, 0),
            exclude_idx_max=min(i+1+exclude_nearest, n_frames)
        )
    return output

def quick_derotate(cube, angles):
    xp = cp.get_array_module(cube)
    outimg = np.zeros(cube.shape[1:])

    for i in range(cube.shape[0]):
        if xp == cp:
            outimg += skimage.transform.rotate(cube[i].get(), -angles[i])
        else:
            outimg += skimage.transform.rotate(cube[i], -angles[i])

    return outimg

def simple_aperture_locations(r_px, pa_deg, resolution_element_px,
                              exclude_nearest=0, exclude_planet=False):
    '''Aperture centers (x, y) in a ring of radius `r_px` and starting
    at angle `pa_deg` E of N. Unless `exclude_planet` is True,
    the first (x, y) pair gives the planet location (signal aperture).

    Specifying `exclude_nearest` > 0 will skip that many apertures
    from either side of the signal aperture's location'''
    circumference = 2 * r_px * np.pi
    aperture_pixel_diameter = resolution_element_px
    n_apertures = int(circumference / aperture_pixel_diameter)
    start_theta = np.deg2rad(pa_deg + 90)
    delta_theta = np.deg2rad(360 / n_apertures)
    idxs = np.arange(1 + exclude_nearest, n_apertures - exclude_nearest)
    if not exclude_planet:
        idxs = np.concatenate(([0,], idxs))
    offset_x = r_px * np.cos(start_theta + idxs * delta_theta)
    offset_y = r_px * np.sin(start_theta + idxs * delta_theta)
    return np.stack((offset_x, offset_y), axis=-1)

def add_colorbar(mappable):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def show_simple_aperture_locations(image, resolution_element_px, r_px, pa_deg,
                                   exclude_nearest=0, exclude_planet=False, ax=None):
    '''Plot `image` and overplot the circular apertures of diameter
    `resolution_element_px` in a ring at radius `r_px`
    starting at `pa_deg` E of N.
    '''
    if ax is None:
        ax = plt.gca()
    ctr = (image.shape[0] - 1) / 2
    im = ax.imshow(image)
    add_colorbar(im)
    ax.axhline(ctr, color='w', linestyle=':')
    ax.axvline(ctr, color='w', linestyle=':')
    planet_dx, planet_dy = r_px * np.cos(np.deg2rad(90 + pa_deg)), r_px * np.sin(np.deg2rad(90 + pa_deg))
    ax.arrow(ctr, ctr, planet_dx, planet_dy, color='w', lw=2)
    for offset_x, offset_y in simple_aperture_locations(r_px, pa_deg, resolution_element_px, exclude_nearest=exclude_nearest, exclude_planet=exclude_planet):
        ax.add_artist(plt.Circle(
            (ctr + offset_x, ctr + offset_y),
            radius=resolution_element_px/2,
            edgecolor='orange',
            facecolor='none',
        ))

def calc_snr_mawet(signal, noises):
    '''Calculate signal to noise following the
    two-sample t test as defined in Mawet 2014'''
    return (
        signal - np.average(noises)
    ) / (
        np.std(noises) * np.sqrt(1 + 1/len(noises))
    )

def cartesian_coords(center, data_shape):
    '''center in x,y order; returns coord arrays xx, yy of data_shape'''
    yy, xx = np.indices(data_shape, dtype=float)
    center_x, center_y = center
    yy -= center_y
    xx -= center_x
    return xx, yy

def reduce_apertures(image, r_px, starting_pa_deg, resolution_element_px, operation,
                     exclude_nearest=0, exclude_planet=False):
    '''apply `operation` to the pixels within radius `resolution_element_px`/2 of the centers
    of the simple aperture locations for a planet at `r_px` and `starting_pa_deg`, returning
    the locations and the results as a tuple with the first location and result corresponding
    to the planet aperture'''
    center = (image.shape[0] - 1) / 2, (image.shape[0] - 1) / 2
    xx, yy = cartesian_coords(center, image.shape)
    locations = list(simple_aperture_locations(r_px, starting_pa_deg, resolution_element_px, exclude_nearest=exclude_nearest, exclude_planet=exclude_planet))
    simple_aperture_radius = resolution_element_px / 2
    results = []
    for offset_x, offset_y in locations:
        dist = np.sqrt((xx - offset_x)**2 + (yy - offset_y)**2)
        mask = dist <= simple_aperture_radius
        results.append(operation(image[mask] / np.count_nonzero(mask & np.isfinite(image))))
    return locations, results

SOLVER_COMBINATIONS = ()

GOOD_COMBOS_CPU = (
    (np, MinimalDowndateSVDDecomposer, generic_svd),
    (np, SVDDecomposer, generic_svd),
    (np, SVDDecomposer, cpu_top_k_svd_arpack),
    (np, RandomizedSVDDecomposer, None),
    (np, CovarianceDecomposition, generic_eigh),
    (np, CovarianceDecomposition, cpu_top_k_cov_syevr),
)
SOLVER_COMBINATIONS += GOOD_COMBOS_CPU

GOOD_COMBOS_CPU_TORCH = (
    (np, MinimalDowndateSVDDecomposer, torch_svd),
    (np, SVDDecomposer, torch_svd),
    (np, CovarianceDecomposition, torch_symeig),
)
SOLVER_COMBINATIONS += GOOD_COMBOS_CPU_TORCH

GOOD_COMBOS_GPU = (
    (cp, MinimalDowndateSVDDecomposer, generic_svd),
    (cp, SVDDecomposer, generic_svd),
    (cp, CovarianceDecomposition, generic_eigh),
)
SOLVER_COMBINATIONS += GOOD_COMBOS_GPU

GOOD_COMBOS_GPU_TORCH = (
    (cp, MinimalDowndateSVDDecomposer, torch_svd),
    (cp, SVDDecomposer, torch_svd),
    (cp, CovarianceDecomposition, torch_symeig),
)
SOLVER_COMBINATIONS += GOOD_COMBOS_GPU_TORCH

BAD_SOLVER_COMBINATIONS = (
    (np, ReuseSVDDecomposer, generic_svd),
)
SOLVER_COMBINATIONS += BAD_SOLVER_COMBINATIONS

GOOD_SOLVER_COMBINATIONS = GOOD_COMBOS_CPU
if HAVE_TORCH:
    GOOD_SOLVER_COMBINATIONS += GOOD_COMBOS_CPU_TORCH
    if HAVE_CUPY:
        GOOD_SOLVER_COMBINATIONS += GOOD_COMBOS_GPU
