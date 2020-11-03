#/usr/bin/env python
# benchmark_scaling.py \
#     big_cube.fits \
#     goodpixmask.fits \
#     --n-frames=500 \
#     --k-modes=10 \
#     --iterations=3
import sys
import time
import argparse
import numpy as np
import cupy as cp
from astropy.io import fits
import starbgone

def log_and_print(fh, msg):
    fh.write(f'{msg}\n')
    print(msg)

def main():
    # parse args
    parser = argparse.ArgumentParser(description='Benchmark competing KLIP eigenvectorization schemes')
    parser.add_argument('datafile')
    parser.add_argument('goodpixmask')
    parser.add_argument('--n-frames', '-n', default=50, type=int, help='Number of frames to keep from datafile')
    parser.add_argument('--k-modes', '-k', default=25, type=int, help='Number of modes to solve for')
    parser.add_argument('--iterations', '-i', default=1, type=int, help='Number of iterations per solver (after a warmup iteration)')
    parser.add_argument('--combo', '-c', type=int, help='Solver combination to try, specify more than once for multiple (default: all)', action='append')
    # list combinations to try
    print('# Solver combinations:')
    for idx, combo in enumerate(starbgone.SOLVER_COMBINATIONS):
        print(f'# [{idx:}]', *map(lambda x: x.__name__ if x is not None else 'None', combo))
    args = parser.parse_args()
    datafile = args.datafile
    goodpixmask = args.goodpixmask
    iterations = args.iterations
    n_frames = args.n_frames
    k_modes = args.k_modes
    if args.combo is None:
        combination_idxs = range(len(starbgone.SOLVER_COMBINATIONS))
        combination_filename_part = ''
    else:
        combination_idxs = args.combo
        combination_filename_part = '_using_{}'.format(','.join(map(str, combination_idxs)))
    print('# Trying:')
    for idx, combo in enumerate(starbgone.SOLVER_COMBINATIONS):
        if idx in combination_idxs:
            print(f'# [{idx:}]', *map(lambda x: x.__name__ if x is not None else 'None', combo))
    # load data
    with open(datafile, 'rb') as f:
        cube_host = fits.getdata(f).astype('=f4')
    with open(goodpixmask, 'rb') as f:
        goodpixmask_host = fits.getdata(f) != 0  # convert to bool, nonzero pixels are kept
    # slice data
    cube_host = cube_host[:n_frames]
    image_vecs_host, _, _ = starbgone.unwrap_cube(cube_host, goodpixmask_host)
    p_pixels = image_vecs_host.shape[0]
    image_vecs_meansub_host = starbgone.mean_subtract_vecs(image_vecs_host)
    image_vecs_meansub_gpu = cp.asarray(image_vecs_meansub_host)
    # open log outfile
    outfile = f'./out_benchmark_scaling_{n_frames}_{k_modes}{combination_filename_part}.csv'
    print('# writing to', outfile)
    outfh = open(outfile, 'w')
    log_and_print(outfh, 'n_frames,k_modes,p_pixels,device,decomposer,solver,duration_s')
    for idx, (xp, decomposer, solver) in enumerate(starbgone.SOLVER_COMBINATIONS):
        if idx not in combination_idxs:
            continue
        if xp == cp:
            image_vecs = image_vecs_meansub_gpu
            device = 'gpu'
        else:
            image_vecs = image_vecs_meansub_host
            device = 'cpu'
        # warmup
        starbgone.klip_to_modes(
            image_vecs,
            decomposer,
            k_modes,
            solver=solver
        )
        for i in range(iterations):
            # start timer
            start = time.perf_counter()
            # do it
            starbgone.klip_to_modes(
                image_vecs,
                decomposer,
                k_modes,
                solver=solver
            )
            # end timer
            duration_s = time.perf_counter() - start
            # write results
            decomposer_name = decomposer.__name__
            solver_name = solver.__name__ if solver is not None else 'None'
            log_and_print(outfh, f'{n_frames},{k_modes},{p_pixels},{device},{decomposer_name},{solver_name},{duration_s}')
    return 0

if __name__ == "__main__":
    sys.exit(main())
