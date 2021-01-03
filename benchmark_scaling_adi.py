#/usr/bin/env python
# benchmark_scaling_adi.py \
#     big_cube.fits \
#     goodpixmask.fits \
#     --n-frames=500 \
#     --iterations=3
import sys
import time
import argparse
import numpy as np
from astropy.io import fits
from skimage.transform import rotate

def log_and_print(fh, msg):
    fh.write(f'{msg}\n')
    print(msg)

def main():
    # parse args
    parser = argparse.ArgumentParser(description='Benchmark competing KLIP eigenvectorization schemes')
    parser.add_argument('datafile')
    parser.add_argument('goodpixmask')
    parser.add_argument('--iterations', '-i', default=1, type=int, help='Number of iterations to time')
    # list combinations to try
    args = parser.parse_args()
    datafile = args.datafile
    goodpixmask = args.goodpixmask
    # load data
    with open(datafile, 'rb') as f:
        cube_host = fits.getdata(f).astype('=f4')
        angles = fits.getdata(f, 'ANGLES')
    with open(goodpixmask, 'rb') as f:
        goodpixmask_host = fits.getdata(f) != 0  # convert to bool, nonzero pixels are kept
    p_pixels = np.count_nonzero(goodpixmask_host)
    # open log outfile
    outfile = f'./out_benchmark_scaling_adi.csv'
    print('# writing to', outfile)
    outfh = open(outfile, 'w')
    log_and_print(outfh, 'n_frames,p_pixels,duration_s')
    for n_frames in [300, 400, 500, 1000, 2000, 10000]:
        for iteration in range(args.iterations):
            # start timer
            start = time.perf_counter()
            # do it
            canvas = np.zeros_like(cube_host[0])
            for i in range(n_frames):
                canvas += rotate(cube_host[i], angles[i])
            # end timer
            duration_s = time.perf_counter() - start
            # write results
            log_and_print(outfh, f'{n_frames},{p_pixels},{duration_s}')
    return 0

if __name__ == "__main__":
    sys.exit(main())
