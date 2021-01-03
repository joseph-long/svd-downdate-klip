from collections import defaultdict
from scipy import optimize
import starbgone
from io import BytesIO
from urllib.request import urlopen
import astropy.units as u
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

benchmark_data = np.genfromtxt(
    './combined_out_benchmark_scaling.csv', 
    delimiter=',',
    dtype=[('n_frames', '<f8'), ('k_modes', '<f8'), ('p_pixels', '<f8'), ('device', 'S3'),
            ('decomposer', 'S25'), ('solver', 'S25'), ('duration_s', '<f8')],
    skip_header=1,
)
p_pixels = int(np.unique(benchmark_data['p_pixels'])[0])
def plot(benchmark_data, ls='-', x='n_frames'):
    for xp, decomposer, solver in starbgone.SOLVER_COMBINATIONS:
        device = b'gpu' if xp.__name__ == 'cupy' else b'cpu'
        decomposer_name = decomposer.__name__.encode('utf8')
        solver_name = solver.__name__.encode('utf8') if solver is not None else b'None'
        mask = (
            (benchmark_data['decomposer'] == decomposer_name) &
            (benchmark_data['solver'] == solver_name) &
            (benchmark_data['device'] == device)
        )
        if len(benchmark_data[mask]):
            n_frames = benchmark_data[mask][x]
            durations = benchmark_data[mask]['duration_s']
            sorter = np.argsort(n_frames)
            n_frames = n_frames[sorter]
            durations = durations[sorter]
            plt.plot(
                n_frames,
                durations,
                label=f'{device.decode("utf8")} '
                    f'{decomposer_name.decode("utf8")} '
                    f'{solver_name.decode("utf8")}',
                marker='o', markersize=5, ls=ls)
    plt.legend(loc=(0, 1.15), ncol=2)
    plt.ylabel('duration [s]')
    plt.xlabel(x)
    plt.title('KLIP with changing reference sets p\\_{{pixels}} = {p_pixels}\n(Lower is better)'.format(p_pixels=p_pixels))
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(which='both')

def powerlaw_fit(xdata, ydata):
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    pinit = [1, 1]
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy))
    pfinal = out[0]
    index = pfinal[1]
    amp = 10**pfinal[0]
    return amp, index

def plot_powerlaw_fit(device, n_modes, decomposer, solver, label=None, extrapolation_label=None, marker='o'):
    mask = (
        (benchmark_data['device'] == device.encode('utf8')) &
        (benchmark_data['k_modes'] == n_modes) &
        (benchmark_data['decomposer'] == decomposer.encode('utf8')) &
        (benchmark_data['solver'] == solver.encode('utf8'))
    )
    n_frames = benchmark_data[mask]['n_frames']
    durations = benchmark_data[mask]['duration_s']
    sorter = np.argsort(n_frames)
    n_frames = n_frames[sorter]
    durations = durations[sorter]

    amp, index = powerlaw_fit(n_frames, durations)
    index_str = f'{index:1.1f}'
    extrapolate_n_frames = np.linspace(1e2, 1e7)
    (the_line,) = plt.plot(
        extrapolate_n_frames, 
        amp * extrapolate_n_frames**index, 
        label="$O(N^{" + index_str + "})$", 
        ls='--',
        alpha=0.5,
    )
    label = label if label is not None else decomposer
    plt.plot(
        n_frames,
        durations,
        label=label,
        c=the_line.get_c(),
        marker=marker
    )
    return amp, index

def pull(device, decomposer, solver, k_modes=200):
    mask = (
        (benchmark_data['device'] == device) &
        (benchmark_data['k_modes'] == k_modes) &
        (benchmark_data['decomposer'] == decomposer) &
        (benchmark_data['solver'] == solver)
    )
    n_frames = benchmark_data[mask]['n_frames']
    durations = benchmark_data[mask]['duration_s']
    sorter = np.argsort(n_frames)
    n_frames = n_frames[sorter]
    durations = durations[sorter]
    return n_frames, durations

adi_benchmark_data = np.genfromtxt(
    './out_benchmark_scaling_adi.csv', 
    delimiter=',',
    dtype=[('n_frames', '<f8'), ('p_pixels', '<f8'), ('duration_s', '<f8')],
    skip_header=1,
)
adi_nframes, adi_t = adi_benchmark_data['n_frames'], adi_benchmark_data['duration_s']
adi_nframes = adi_nframes.reshape(adi_nframes.size // 3, 3).min(axis=1)
adi_t = adi_t.reshape(adi_t.size // 3, 3).min(axis=1)

plt.figure(figsize=(8,4))
extrapolate_nframes = np.linspace(1e2, 1e7)
plot_powerlaw_fit('cpu', 200, 'SVDDecomposer', 'torch_svd', label='SVD/hybrid/MAGMA', marker='d') # best
plot_powerlaw_fit('gpu', 200, 'CovarianceDecomposition', 'generic_eigh', label='Covariance/GPU/cuSolver', marker='^') # best
plot_powerlaw_fit('cpu', 200, 'RandomizedSVDDecomposer', 'None', label='Rand. SVD/CPU', marker='s') # only
plot_powerlaw_fit('cpu', 200, 'DowndateSVDDecomposer', 'torch_svd', label='Mod. SVD/hybrid/MAGMA\n(this work)', marker='o') # best
plt.plot(extrapolate_nframes, adi_t[1] * (extrapolate_nframes / adi_nframes[1]),
         ls='--', alpha=0.5, label='$O(N)$', color='C4')
plt.plot(adi_nframes, adi_t, label='Classic ADI', marker='x', color='C4')

plt.grid()
plt.xlim(1e2, 1e5)
plt.ylim(1e-1, 1e6)


plt.legend(loc=(1.02, 0))

for text, unit in [('1 week', u.week), ('1 day', u.day), ('1 hour', u.hour), ('1 minute', u.minute)]:
    val_sec = (1 * unit).to(u.s).value
    plt.axhline(val_sec, alpha=0.25, c='k', ls=':')
    plt.annotate(
        text, 
        (1e2, val_sec),
        xytext=(5, -6), 
        textcoords='offset points',
        bbox={'boxstyle': 'square', 'edgecolor': 'none', 'facecolor': 'w', 'alpha': 0.75},
    )


plt.xlabel('$N$ frames')
plt.ylabel('Time to process $N$ frames\n[seconds]')
plt.tight_layout()
plt.xscale('log')
plt.yscale('log')
plt.savefig('./benchmark.pdf')
