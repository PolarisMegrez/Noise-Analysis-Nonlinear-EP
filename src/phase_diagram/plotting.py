import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence
from scipy.signal import welch, periodogram, get_window

def plot_phase_diagram(X, Y, U, V, title="Phase Diagram", xlabel="Re(α)", ylabel="Im(α)", *, save_path: Optional[str] = None, show: bool = True):
    plt.figure(figsize=(7, 7))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.streamplot(X, Y, U, V, density=1.2, linewidth=1)
    plt.scatter([0], [0], marker='x')  # mark origin for reference
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.grid(True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

def plot_trajectory(sol, title="Trajectory in Complex Plane", *, save_path: Optional[str] = None, show: bool = True):
    plt.plot(sol.y[0], sol.y[1])  # trajectory in complex plane
    plt.plot(sol.y[0, -1], sol.y[1, -1], 'o')  # endpoint marker
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("Re(α)")
    plt.ylabel("Im(α)")
    plt.grid(True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

def plot_amplitude_time_series(sol, title="Amplitude Evolution", *, save_path: Optional[str] = None, show: bool = True):
    amp = np.hypot(sol.y[0], sol.y[1])
    plt.figure(figsize=(8, 3.5))
    plt.plot(sol.t, amp)
    plt.xlabel("Time")
    plt.ylabel("|α(t)|")
    plt.title(title)
    plt.grid(True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

def plot_phase_trajectories(solutions, var_index=0, labels: Optional[Sequence[str]] = None, title="Phase trajectories", *, save_path: Optional[str] = None, show: bool = True):
    """
    在同一张相图上绘制指定复变量 var_index 的多条相轨迹。
    每个解的 Re(z[var_index]) 与 Im(z[var_index]) 分别在 y 的 2*idx 与 2*idx+1 行。
    """
    plt.figure(figsize=(7, 7))
    for k, sol in enumerate(solutions):
        re = sol.y[2*var_index, :]
        im = sol.y[2*var_index + 1, :]
        lab = labels[k] if labels and k < len(labels) else f"IC {k+1}"
        # main line: capture the line handle to reuse its color
        line, = plt.plot(re, im, label=lab, alpha=0.9)
        color = line.get_color()
        # start marker: hollow triangle
        plt.plot(re[0], im[0], marker='^', mfc='none', mec=color, ms=7)
        # end marker: hollow x (cross)
        plt.plot(re[-1], im[-1], marker='x', mfc='none', mec=color, ms=7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel(f"Re(z[{var_index}])")
    plt.ylabel(f"Im(z[{var_index}])")
    plt.grid(True)
    plt.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

def plot_modulus_phase_trajectories(solutions, i=0, j=1, labels: Optional[Sequence[str]] = None, title="|z_i| vs |z_j|", *, save_path: Optional[str] = None, show: bool = True):
    """
    绘制指定两复变量索引 i, j 的模长相图：横轴 |z_i|，纵轴 |z_j|。
    """
    plt.figure(figsize=(7, 5))
    for k, sol in enumerate(solutions):
        rei, imi = sol.y[2*i, :], sol.y[2*i + 1, :]
        rej, imj = sol.y[2*j, :], sol.y[2*j + 1, :]
        ai = np.hypot(rei, imi)
        aj = np.hypot(rej, imj)
        lab = labels[k] if labels and k < len(labels) else f"IC {k+1}"
        line, = plt.plot(ai, aj, label=lab, alpha=0.9)
        color = line.get_color()
        # start marker: hollow triangle
        plt.plot(ai[0], aj[0], marker='^', mfc='none', mec=color, ms=7)
        # end marker: hollow cross
        plt.plot(ai[-1], aj[-1], marker='x', mfc='none', mec=color, ms=7)
    plt.xlabel(f"|z[{i}]|")
    plt.ylabel(f"|z[{j}]|")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def compute_modes_amplitudes(sol):
    """Return amplitudes |z_i| for all modes from a solution with real-expanded y."""
    m = sol.y.shape[0] // 2
    amps = []
    for i in range(m):
        re = sol.y[2 * i, :]
        im = sol.y[2 * i + 1, :]
        amps.append(np.hypot(re, im))
    return np.array(amps), np.asarray(sol.t)


def compute_psd_modes_average_welch(solutions, mode_indices=None, *, nperseg: int = 256, window: str | tuple | None = 'hann', noverlap: int | None = None, detrend: str | None = 'constant'):
    """
    Compute Welch PSD for |z_i| of each mode, averaged across all solutions (ICs).
    Returns freqs and PSD array with shape (num_modes, len(freqs)).
    """
    # assume uniform dt based on first solution
    amps0, t0 = compute_modes_amplitudes(solutions[0])
    dt = float(t0[1] - t0[0]) if len(t0) > 1 else 1.0
    fs = 1.0 / dt
    m_total = amps0.shape[0]
    if mode_indices is None:
        mode_indices = list(range(m_total))
    psd_accum = None
    count = 0
    f_ref = None
    for sol in solutions:
        amps, t = compute_modes_amplitudes(sol)
        dt_i = float(t[1] - t[0]) if len(t) > 1 else dt
        fs_i = 1.0 / dt_i
        for idx_m, mi in enumerate(mode_indices):
            seg = min(nperseg, amps[mi].size)
            f, Pxx = welch(
                amps[mi] - np.mean(amps[mi]),
                fs=fs_i,
                window=window,
                nperseg=seg,
                noverlap=noverlap if (noverlap is not None and noverlap < seg) else None,
                detrend=detrend,
            )
            if psd_accum is None:
                psd_accum = np.zeros((len(mode_indices), len(f)))
                f_ref = f
            else:
                # If segment length differs, resample/skip; here we enforce equal by nperseg+fs
                if len(f) != psd_accum.shape[1]:
                    # simple fallback: interpolate to f_ref
                    Pxx = np.interp(f_ref, f, Pxx)
            psd_accum[idx_m] += Pxx
        count += 1
    psd_avg = psd_accum / max(count, 1)
    return f_ref, psd_avg


def compute_psd_modes_average_multi_traj(solutions, mode_indices=None, *, trajectories: int | None = None, window: str | tuple | None = 'boxcar', detrend: str | None = None):
    """
    Compute PSD by averaging periodograms across multiple trajectories (ICs).
    - trajectories: number of trajectories to include (from the start); if None, use all.
    - window: window applied before FFT; default 'boxcar' (no window).
    - detrend: optional detrending ('constant' or 'linear') before PSD.
    Returns freqs and PSD array with shape (num_modes, len(freqs)).
    """
    amps0, t0 = compute_modes_amplitudes(solutions[0])
    dt = float(t0[1] - t0[0]) if len(t0) > 1 else 1.0
    fs = 1.0 / dt
    m_total = amps0.shape[0]
    if mode_indices is None:
        mode_indices = list(range(m_total))
    psd_accum = None
    count = 0
    f_ref = None
    n_use = len(solutions) if trajectories is None else min(trajectories, len(solutions))
    for sol in solutions[:n_use]:
        amps, t = compute_modes_amplitudes(sol)
        dt_i = float(t[1] - t[0]) if len(t) > 1 else dt
        fs_i = 1.0 / dt_i
        for idx_m, mi in enumerate(mode_indices):
            sig = amps[mi]
            if detrend:
                if detrend == 'constant':
                    sig = sig - np.mean(sig)
                elif detrend == 'linear':
                    x = np.arange(sig.size)
                    A = np.vstack([x, np.ones_like(x)]).T
                    a, b = np.linalg.lstsq(A, sig, rcond=None)[0]
                    sig = sig - (a * x + b)
            win = get_window(window, sig.size) if window is not None else 1.0
            f, Pxx = periodogram(sig * win, fs=fs_i, scaling='density')
            if psd_accum is None:
                psd_accum = np.zeros((len(mode_indices), len(f)))
                f_ref = f
            else:
                if len(f) != psd_accum.shape[1]:
                    Pxx = np.interp(f_ref, f, Pxx)
            psd_accum[idx_m] += Pxx
        count += 1
    psd_avg = psd_accum / max(count, 1)
    return f_ref, psd_avg


def plot_psd_modes(
    solutions,
    mode_indices=None,
    labels: Optional[Sequence[str]] = None,
    title: str = "PSD of |z_i|",
    *,
    method: str = 'welch',
    params: Optional[dict] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    params = params or {}
    method = (method or 'welch').lower()
    if method == 'welch':
        f, S = compute_psd_modes_average_welch(solutions, mode_indices=mode_indices, **{
            k: v for k, v in params.items() if k in {'nperseg', 'window', 'noverlap', 'detrend'}
        })
    elif method in ('multi-trajectory', 'multi_traj', 'multi'):
        f, S = compute_psd_modes_average_multi_traj(solutions, mode_indices=mode_indices, **{
            k: v for k, v in params.items() if k in {'trajectories', 'window', 'detrend'}
        })
    else:
        raise ValueError(f"Unknown PSD method: {method}")
    if mode_indices is None:
        mode_indices = list(range(S.shape[0]))
    plt.figure(figsize=(8, 5))
    for k in range(S.shape[0]):
        lab = labels[k] if labels and k < len(labels) else f"mode {mode_indices[k]}"
        plt.loglog(f, S[k], label=lab)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [arb. units]")
    plt.title(title)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()