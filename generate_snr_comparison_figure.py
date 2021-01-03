import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'Greys_r',
    'text.usetex': True,
    'font.family': 'serif'
})
from astropy.visualization import quantity_support
quantity_support()
# n_frames,k_modes,p_pixels,device,decomposer,solver,gaussian_fwhm,snr
benchmark_data = np.genfromtxt(
    './out_benchmark_snr.csv', 
    delimiter=',',
    dtype=[('n_frames', '<f8'), ('k_modes', '<f8'), ('p_pixels', '<f8'), ('device', 'S3'),
            ('decomposer', 'S33'), ('solver', 'S25'), ('gaussian_fwhm', '<f8'), ('snr', '<f8')],
    skip_header=1,
)
p_pixels = int(np.unique(benchmark_data['p_pixels'])[0])
benchmark_data['gaussian_fwhm'][np.isnan(benchmark_data['gaussian_fwhm'])] = 0
def one_snr_curve(decomposer_name, solver_name, device, gaussian_fwhm):
    mask = (
        (benchmark_data['decomposer'] == decomposer_name.encode('utf8')) &
        (benchmark_data['solver'] == solver_name.encode('utf8')) &
        (benchmark_data['device'] == device.encode('utf8')) &
        (benchmark_data['gaussian_fwhm'] == gaussian_fwhm)
    )
    k_modes = benchmark_data[mask]['k_modes']
    snrs = benchmark_data[mask]['snr']
    sorter = np.argsort(k_modes)
    k_modes = k_modes[sorter]
    snrs = snrs[sorter]
    return k_modes, snrs

plt.figure(figsize=(4,3))
plt.plot(*one_snr_curve('MinimalDowndateSVDDecomposer', 'torch_svd', 'cpu', 2), c='C0', alpha=1, ls='-', label='New $Z^{KL}_k$ by SVD modification\n(this work)')
plt.plot(*one_snr_curve('CovarianceDecomposition', 'torch_symeig', 'cpu', 2), c='C1', alpha=1, label='New $Z^{KL}_k$ computed per-frame')
plt.plot(*one_snr_curve('ReuseSVDDecomposer', 'generic_svd', 'cpu', 0), c='C2', alpha=1, ls=':', label='Single $Z^{KL}_k$ for all frames')
plt.ylim(0, 12)
plt.xlim(0)
plt.ylabel('Signal to noise ratio')
plt.xlabel('Number of modes $k$ used in starlight subtraction')
plt.grid()
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('./figures/snr_comparison.pdf')
