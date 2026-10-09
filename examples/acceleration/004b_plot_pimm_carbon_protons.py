"""Plot saved PIMM carbon/proton pickup and acceleration data.

Run 004a_pimm_carbon_protons.py first. This script only needs NumPy and
Matplotlib; it does not import Xsuite, compile kernels, or track particles.

    python 004b_plot_pimm_carbon_protons.py
    python 004b_plot_pimm_carbon_protons.py /path/to/004_pimm_carbon_protons.npz --no-show
"""

import argparse
from pathlib import Path

import numpy as np


COLORS = ('#007f86', '#d65b32')
DEFAULT_DATA = (Path(__file__).with_name('004_pimm_carbon_protons')
                / '004_pimm_carbon_protons.npz')


def plot_pickup(result, output_dir, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_us = result['time_s'].ravel() * 1e6
    currents = result['current_A'] * 1e3  # mA
    period_us = result['periods_s'][0] * 1e6
    n_turns = result['num_carbon_turns']
    labels = (r'$^{12}$C$^{6+}$', 'protons')
    freqs = result['frequencies_Hz'] / 1e3

    with plt.rc_context({'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'}):
        fig = plt.figure(figsize=(9, 7), dpi=100, layout='constrained')
        grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.45))
        fig.suptitle('Two bunches, one magnetic rigidity', fontsize=15, weight='bold')
        first = fig.add_subplot(grid[0, :])
        last = fig.add_subplot(grid[1, :], sharey=first)
        for ax, start, title in (
                (first, 0., 'Initial passages'),
                (last, result['acquisition_duration_s']*1e6 - 8*period_us, 'Later passages')):
            selection = (time_us >= start) & (time_us <= start + 8*period_us)
            for ii in range(2):
                ax.plot(time_us[selection], currents[ii].ravel()[selection],
                        color=COLORS[ii], lw=1.25,
                        label=f'{labels[ii]}  |  {freqs[ii]:.1f} kHz')
            ax.set(xlim=(start, start + 8*period_us),
                   xlabel='Laboratory time [us]', ylabel='Pickup current [mA]')
            ax.set_title(title, loc='left', weight='bold')
            ax.grid(alpha=.15)
        first.legend(loc='upper right', ncols=2, frameon=False)
        first.text(.005, .96,
                   f'Injection: C 7 MeV/u   |   p '
                   f'{result["proton_kinetic_energy_eV"]/1e6:.2f} MeV\n'
                   f'PIMM: 75.24 m   |   f_p / f_C = {freqs[1]/freqs[0]:.3f}',
                   transform=first.transAxes, va='top', fontsize=7.5)
        first.set_ylim(0, 1.32*currents.max())

        # Choose one fixed phase origin for the folded pickup image. Use the
        # circular mean to handle profiles that straddle the turn boundary.
        # Shift both species equally and retain the carbon centroid oscillation.
        num_slices = currents.shape[-1]
        slice_phase = 2*np.pi*((np.arange(num_slices) + .5)/num_slices - .5)
        carbon_profile = result['counts'][0].sum(axis=0)
        carbon_phase = np.angle(np.sum(carbon_profile*np.exp(1j*slice_phase)))
        shift = -int(np.rint(carbon_phase*num_slices/(2*np.pi)))
        # Shift the continuous record so bins crossing a turn boundary move
        # into the adjacent row. Do not wrap the ends of the acquisition.
        flat_currents = currents.reshape(2, -1)
        centered_flat = np.full_like(flat_currents, np.nan)
        if shift > 0:
            centered_flat[:, shift:] = flat_currents[:, :-shift]
        elif shift < 0:
            centered_flat[:, :shift] = flat_currents[:, -shift:]
        else:
            centered_flat[:] = flat_currents
        centered_currents = centered_flat.reshape(currents.shape)

        vmax = currents.max()
        pickup_grid = grid[2, :].subgridspec(1, 3, width_ratios=(1, .025, .025))
        ax = fig.add_subplot(pickup_grid[0, 0])
        ax.set_title('Both species — all recorded passages', loc='left', weight='bold')
        ax.set(xlabel='Arrival coordinate relative to mean carbon [m]',
               ylabel='Carbon reference-turn index')
        for ii in range(2):
            # Empty bins must be transparent so neither species hides the other.
            cmap = LinearSegmentedColormap.from_list('species',
                [to_rgba(COLORS[ii], 0), to_rgba(COLORS[ii], 1)])
            picture = ax.imshow(centered_currents[ii], origin='lower', aspect='auto',
                extent=(-.5*result['circumference_m'], .5*result['circumference_m'],
                        -.5, n_turns + .5),
                cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            cax = fig.add_subplot(pickup_grid[0, ii + 1])
            fig.colorbar(picture, cax=cax, label=f'{labels[ii]} current [mA]')
        fig.savefig(output_dir / '004_pimm_carbon_protons.png', dpi=180)
        fig.savefig(output_dir / '004_pimm_carbon_protons.pdf')

        energy_fig, axes = plt.subplots(2, 1, figsize=(9, 5), dpi=100,
                                       sharex=True, layout='constrained')
        energy_fig.suptitle('Acceleration and beam survival', fontsize=15, weight='bold')
        history = result['history']
        for ii in range(2):
            scale = (12., 1.)[ii]*1e6
            mass = result['species_masses_eV'][ii]
            pc = result['ramp_p0c']*result['species_charge_ratios'][ii]
            target = (np.sqrt(pc**2 + mass**2) - mass)/scale
            mean, sigma = history[:, 2 + 4*ii]/scale, history[:, 3 + 4*ii]/scale
            ax = axes[ii]
            ax.plot(result['ramp_times_s']*1e6, target, 'k--', lw=1, label='Programmed')
            ax.plot(history[:, 0]*1e6, mean, color=COLORS[ii], label='Bunch mean')
            ax.fill_between(history[:, 0]*1e6, mean-sigma, mean+sigma,
                            color=COLORS[ii], alpha=.25, label='Bunch rms spread')
            ax.set_ylabel(('Carbon [MeV/u]', 'Proton [MeV]')[ii])
            ax.set_title(f'{labels[ii]}: {history[-1, 1+4*ii]:.1%} surviving', loc='left')
            ax.grid(alpha=.15)
        axes[0].legend(ncols=3, frameon=False)
        axes[1].set_xlabel('Laboratory time [us]')
        axes[1].set_xlim(0, history[-1, 0]*1e6)
        energy_fig.savefig(output_dir / '004_pimm_carbon_protons_energy.png', dpi=180)
        energy_fig.savefig(output_dir / '004_pimm_carbon_protons_energy.pdf')

    print(f'Plots saved in {output_dir.resolve()}')
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_file', nargs='?', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path,
                        help='Defaults to the directory containing the data')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    with np.load(args.data_file, allow_pickle=False) as data:
        result = dict(data)
    plot_pickup(result, args.output_dir or args.data_file.parent,
                show=not args.no_show)
