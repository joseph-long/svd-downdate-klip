from urllib.request import urlopen
from io import BytesIO
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
import cupy as cp
import starbgone

STDDEV_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
FWHM_TO_STDDEV = 1. / STDDEV_TO_FWHM

def log_and_print(fh, msg):
    fh.write(f'{msg}\n')
    print(msg)

def calc_snr(cube, angles, decomposer, solver, k_modes):
    good_pix_mask = np.ones_like(cube[0], dtype=bool)
    image_vecs, xx, yy = starbgone.unwrap_cube(cube, good_pix_mask)
    starlight_subtracted = starbgone.klip_to_modes(
        image_vecs,
        decomposer,
        k_modes,
        solver=solver
    )
        
    outcube = starbgone.wrap_matrix(starlight_subtracted, cube.shape, xx, yy)
    final_image = starbgone.quick_derotate(outcube, angles)

    r_px, pa_deg = 18.4, -42.8
    fwhm_naco = 4

    _, results = starbgone.reduce_apertures(
        final_image,
        r_px,
        pa_deg,
        fwhm_naco,
        np.sum
    )
    return starbgone.calc_snr_mawet(results[0], results[1:])

def modes_vs_snr(cube, angles, k_modes_vals, decomposer, solver, verbose=False):
    snrs = []
    for k_modes in k_modes_vals:
        the_snr = calc_snr(cube, angles, decomposer, solver, k_modes=k_modes)
        if verbose:
            print(k_modes, the_snr)
        snrs.append(the_snr)
    return np.asarray(snrs)

def main():
    data_url = 'https://github.com/carlgogo/VIP_extras/raw/master/datasets/naco_betapic_preproc.npz'
    data = np.load(BytesIO(urlopen(data_url).read()))
    cube_host = data['cube'].astype('=f4')
    angles = data['angles']
    n_frames = cube_host.shape[0]
    p_pixels = cube_host.shape[1] * cube_host.shape[2]
    min_k_modes = 1
    max_k_modes = n_frames - 1
    k_modes_vals = np.arange(min_k_modes, max_k_modes)
    outfilename = 'out_benchmark_snr.csv'
    outfh = open(outfilename, 'w')
    log_and_print(outfh, 'n_frames,k_modes,p_pixels,device,decomposer,solver,gaussian_fwhm,snr')
    for xp, decomposer, solver in starbgone.SOLVER_COMBINATIONS:
        if xp == cp:
            device = 'gpu'
        else:
            device = 'cpu'
        decomposer_name = decomposer.__name__
        solver_name = solver.__name__ if solver is not None else 'None'
        for gaussian_fwhm in [0, 1, 2, 3, 4, 5]:
            if gaussian_fwhm != 0:
                kernel = Gaussian2DKernel(FWHM_TO_STDDEV * gaussian_fwhm)
                conv_cube = cube_host.copy()
                for i in range(conv_cube.shape[0]):
                    conv_cube[i] = convolve(cube_host[i], kernel)
            else:
                conv_cube = cube_host
            if device == 'gpu':
                cube = cp.asarray(conv_cube)
            else:
                cube = conv_cube
            the_snrs = modes_vs_snr(cube, angles, k_modes_vals, decomposer, solver)
            for idx, snr_val in enumerate(the_snrs):
                k_modes = k_modes_vals[idx]
                log_and_print(outfh,
                    f'{n_frames},{k_modes},{p_pixels},{device},{decomposer_name},{solver_name},{gaussian_fwhm},{snr_val}'
                )

if __name__ == "__main__":
    main()
