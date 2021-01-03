# svd-downdate-klip

Code to accompany "Unlocking starlight subtraction in full data rate exoplanet imaging by efficiently updating Karhunen-Loève eigenimages" by Joseph D. Long (@joseph-long) and Jared R. Males (@jaredmales) (submitted).

## Usage

**This code is provided in hope that it will be useful and informative, but without documentation beyond that included in the source (and, of course, the paper).** We would recommend implementing the algorithm from the text yourself, as there are many more features you would want in a research tool that would only complicate a proof of concept and benchmarking tool.

You will need several Python packages and a CUDA-capable NVIDIA GPU to run the tests and benchmarks. The conda environment used to perform the benchmarks is serialized as `klipbenchmark_env.yml`. If you use conda, `conda create -f ./klipbenchmark_env.yml` should leave you able to `conda activate klipbenchmark` and get an equivalent environment to the one we used. You can then run `pytest` from this directory and test the implementation. `benchmark_snr.py` is also relatively self-contained, and will download the example beta Pic NACO dataset itself when run.

Our implementation of the "SVD downdate" can be found in the `minimal_downdate()` function in `starbgone.py`. An example of an end-to-end calculation from data cube to SNR can be found in `calc_snr()` in `benchmark_snr.py`.

Benchmarks were conducted on the University of Arizona HPC cluster _Ocelote_, and PBS batch scripts to reproduce `combined_out_benchmark_scaling.csv` and `out_benchmark_snr.csv` are available upon request.

## Example data

The [example data](https://github.com/carlgogo/VIP_extras/raw/master/datasets/naco_betapic_preproc.npz) of beta Pictoris b (`naco_betapic_preproc_absil2013_gonzalez2017.npz`) was initially published as [Absil et al. 2013](http://doi.org/10.1051/0004-6361/201322748) and made available as part of [https://github.com/carlgogo/VIP_extras](https://github.com/carlgogo/VIP_extras) ([Gonzalez et al. 2017](http://doi.org/10.3847/1538-3881/aa73d7)). It is stored in this repository to make the examples self-contained, but is not covered by the GPLv3 license. The original papers must be cited if it is reused.
