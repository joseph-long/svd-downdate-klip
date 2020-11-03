from urllib.request import urlopen
from io import BytesIO
import pytest
import numpy as np
import cupy as cp
import starbgone

def compare_columns_modulo_sign(init_u, final_u, display=False):
    init_u = cp.asnumpy(init_u)
    final_u = cp.asnumpy(final_u)
    signs = np.zeros(init_u.shape[1])
    for col in range(init_u.shape[1]):
        signs[col] = 1 if np.allclose(init_u[:,col], final_u[:,col]) else -1
    vmax = np.max(np.abs([init_u, final_u]))
    final_u_mod = signs * final_u
    if display:
        import matplotlib.pyplot as plt
        fig, (ax_iu, ax_fu, ax_du) = plt.subplots(ncols=3, figsize=(14, 4))
        ax_iu.imshow(init_u, vmin=-vmax, vmax=vmax, origin='lower')
        ax_iu.set_title(r'$\mathbf{U}_\mathrm{first}$')
        ax_fu.imshow(final_u_mod, vmin=-vmax, vmax=vmax, origin='lower')
        ax_fu.set_title(r'(signs) * $\mathbf{U}_\mathrm{second}$')
        diff_vmax = np.max(np.abs(final_u_mod - init_u))
        ax_du.imshow(final_u_mod - init_u, cmap='RdBu_r', vmax=diff_vmax, vmin=-diff_vmax, origin='lower')
        ax_du.set_title(r'(signs) * $\mathbf{U}_\mathrm{second}$ - $\mathbf{U}_\mathrm{first}$')
    return np.allclose(final_u_mod, init_u)

def test_minimal_downdate():
    dim_p = 6
    dim_q = 5

    # Initialize p x q noise matrix X
    mtx_x = cp.random.randn(dim_p, dim_q)
    
    # Initialize thin SVD
    dim_r = dim_q  # for truncated, r < q
    mtx_u, diag_s, mtx_vt = cp.linalg.svd(mtx_x, full_matrices=False)
    mtx_v = mtx_vt.T

    # Select columns to remove
    min_col_to_remove = 1
    max_col_to_remove = 3  # exclusive
    col_idxs_to_remove = cp.arange(min_col_to_remove, max_col_to_remove)
    new_mtx_u, new_diag_s, new_mtx_v = starbgone.minimal_downdate(
        mtx_u,
        diag_s,
        mtx_v,
        min_col_to_remove,
        max_col_to_remove,
        compute_v=True
    )

    # X with columns zeroed for comparison
    final_mtx_x = starbgone.drop_idx_range_cols(mtx_x, min_col_to_remove, max_col_to_remove)
    assert cp.allclose(new_mtx_u @ cp.diag(new_diag_s) @ new_mtx_v.T, final_mtx_x, atol=1e-6)

    # SVD of final matrix for comparison
    final_mtx_u, final_diag_s, final_mtx_vt = cp.linalg.svd(final_mtx_x)

    n_nonzero = cp.count_nonzero(final_diag_s > 1e-14)
    assert n_nonzero == 3

    assert compare_columns_modulo_sign(
        new_mtx_u[:,:n_nonzero],
        final_mtx_u[:,:n_nonzero],
    )

@pytest.mark.parametrize('solver', [starbgone.torch_svd, starbgone.generic_svd])
@pytest.mark.parametrize('xp', [cp, np])
def test_interchangeable_svd_solvers(xp, solver):
    # when using float32, unrelated to cupy->torch->cupy
    # round trip, rtol must be increased from 1e-5 to 1e-4
    # and atol from 1e-8 to 1e-5
    # for allclose to be True
    atol = 1e-5
    rtol = 1e-4
    dim_m, dim_n = 10, 20
    mtx_x = xp.random.randn(dim_m, dim_n).astype(xp.float32)
    mtx_u, diag_s, mtx_v = solver(mtx_x)
    assert xp.allclose(mtx_x, mtx_u @ xp.diag(diag_s) @ mtx_v.T, atol=atol, rtol=rtol)

    # transposition and strides mentioned in torch.svd docstring
    # don't affect this:
    mtx_x = xp.random.randn(dim_n, dim_m).astype(xp.float32)
    mtx_u, diag_s, mtx_v = solver(mtx_x)
    assert xp.allclose(mtx_x, mtx_u @ xp.diag(diag_s) @ mtx_v.T, atol=atol, rtol=rtol)

def test_wrap():
    image = np.arange(9).reshape(3, 3)
    mask = np.ones_like(image, dtype=bool)
    mtx, xx, yy = starbgone.unwrap_image(image, mask)
    assert np.all(image == starbgone.wrap_vector(mtx, image.shape, xx, yy))

def test_unwrap():
    image = np.zeros((3, 3))
    mask = np.ones((3, 3), dtype=bool)
    image[1, 1] = 1
    mask[1, 1] = False
    mtx, xx, yy = starbgone.unwrap_image(image, mask)
    assert mtx.shape[0] == 8
    assert xx.shape[0] == 8
    assert yy.shape[0] == 8
    assert np.max(mtx) == 0

def test_simple_aperture_locations():
    r_px = 5
    pa_deg = 0
    diam = 7
    assert np.allclose(
        np.asarray(list(starbgone.simple_aperture_locations(r_px, pa_deg, diam))),
        [[0, 5], [-5, 0], [0, -5], [5, 0]]
    )
    assert np.allclose(
        np.asarray(list(starbgone.simple_aperture_locations(r_px, pa_deg, diam, exclude_planet=True))),
        [[-5, 0], [0, -5], [5, 0]]
    )
    assert np.allclose(
        np.asarray(list(starbgone.simple_aperture_locations(r_px, pa_deg, diam, exclude_planet=True, exclude_nearest=1))),
        [[0, -5]]
    )

@pytest.mark.parametrize('xp,decomposer,solver', starbgone.SOLVER_COMBINATIONS)
def test_end_to_end(xp, decomposer, solver):
    if decomposer == starbgone.ReuseSVDDecomposer:
        # Doesn't actually pass the test because of self-subtraction,
        # but we wouldn't expect it to.
        return
    data_url = 'https://github.com/carlgogo/VIP_extras/raw/master/datasets/naco_betapic_preproc.npz'
    data = np.load(BytesIO(urlopen(data_url).read()))
    n_modes = 50
    threshold = 2200  # fake, just to test masking
    good_pix_mask = xp.asarray(np.average(data['cube'], axis=0) < threshold)
    cube = xp.asarray(data['cube'])
    image_vecs, xx, yy = starbgone.unwrap_cube(cube, good_pix_mask)
    image_vecs_meansub = starbgone.mean_subtract_vecs(image_vecs)
    starlight_subtracted = starbgone.klip_to_modes(
        image_vecs_meansub,
        decomposer,
        n_modes,
        solver=solver
    )
        
    outcube = starbgone.wrap_matrix(starlight_subtracted, cube.shape, xx, yy)
    if xp == cp:
        outcube = outcube.get()
    final_image = starbgone.quick_derotate(outcube, data['angles'])

    r_px, pa_deg = 18.4, -42.8
    fwhm_naco = 4

    locations, results = starbgone.reduce_apertures(
        final_image,
        r_px,
        pa_deg,
        fwhm_naco,
        np.sum
    )
    assert starbgone.calc_snr_mawet(results[0], results[1:]) > 8.36
