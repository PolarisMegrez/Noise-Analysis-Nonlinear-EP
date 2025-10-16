import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence
from scipy.signal import welch, periodogram, get_window


def _interp_to_uniform_time(t: np.ndarray, y: np.ndarray, *, kind: str = 'linear'):
    """
    Given a possibly non-uniform time grid t (shape [T]) and signal y (shape [T] or [M,T]),
    interpolate onto a uniform grid over [t[0], t[-1]]. Returns (t_uniform, y_uniform).
    """
    if t.size < 2:
        return t, y
    # detect uniformity
    diffs = np.diff(t)
    if np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-12):
        return t, y
    T = t.size
    # choose N equal to original length for comparable frequency resolution
    t_uniform = np.linspace(t[0], t[-1], T)
    if y.ndim == 1:
        if np.iscomplexobj(y):
            y_uniform = np.interp(t_uniform, t, y.real) + 1j * np.interp(t_uniform, t, y.imag)
        else:
            y_uniform = np.interp(t_uniform, t, y)
    else:
        # broadcast over rows
        if np.iscomplexobj(y):
            y_uniform = np.vstack([
                np.interp(t_uniform, t, row.real) + 1j * np.interp(t_uniform, t, row.imag)
                for row in y
            ])
        else:
            y_uniform = np.vstack([np.interp(t_uniform, t, row) for row in y])
    return t_uniform, y_uniform

def plot_phase_diagram(X, Y, U, V, title="Phase Diagram", xlabel="Re(α)", ylabel="Im(α)", *, save_path: Optional[str] = None, show: bool = True):
    plt.figure(figsize=(7, 7))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.streamplot(X, Y, U, V, density=1.2, linewidth=1)
    plt.scatter([0], [0], marker='x')  # mark origin for reference
    # Let Matplotlib choose axis limits based on data automatically
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


def compute_modes_complex(sol):
    """Return complex z_i(t) for all modes from a solution with real-expanded y."""
    m = sol.y.shape[0] // 2
    Z = []
    for i in range(m):
        re = sol.y[2 * i, :]
        im = sol.y[2 * i + 1, :]
        Z.append(re + 1j * im)
    return np.array(Z, dtype=complex), np.asarray(sol.t)


def compute_psd_modes_average_welch(
    solutions,
    mode_indices=None,
    *,
    nperseg: int = 256,
    window: str | tuple | None = 'hann',
    noverlap: int | None = None,
    detrend: str | None = 'constant',
    signal: str = 'amplitude',
):
    """
    Compute two-sided Welch PSD for complex z_i(t) of each mode, averaged across solutions (ICs).
    Returns freqs (two-sided, shifted) and PSD array with shape (num_modes, len(freqs)).
    """
    # assume uniform dt based on first solution
    Z0, t0 = compute_modes_complex(solutions[0])
    # Interpolate to uniform if needed for the first solution to get base dt
    t0_u, Z0_u = _interp_to_uniform_time(t0, Z0)
    dt = float(t0_u[1] - t0_u[0]) if len(t0_u) > 1 else 1.0
    fs = 1.0 / dt
    m_total = Z0.shape[0]
    if mode_indices is None:
        mode_indices = list(range(Z0.shape[0]))
    psd_accum = None
    count = 0
    f_ref = None
    sig_mode = (signal or 'amplitude').lower()
    for sol in solutions:
        Zi, t = compute_modes_complex(sol)
        # Interpolate to uniform if needed
        t_u, Zi_u = _interp_to_uniform_time(t, Zi)
        dt_i = float(t_u[1] - t_u[0]) if len(t_u) > 1 else dt
        fs_i = 1.0 / dt_i
        for idx_m, mi in enumerate(mode_indices):
            # Choose signal: amplitude |z| or complex z
            if sig_mode == 'intensity':
                sig_i = Zi_u[mi]
            else:  # 'amplitude' default
                sig_i = np.abs(Zi_u[mi])
            seg = min(nperseg, sig_i.size)
            f, Pxx = welch(
                sig_i - np.mean(sig_i) if detrend == 'constant' else sig_i,
                fs=fs_i,
                window=window,
                nperseg=seg,
                noverlap=noverlap if (noverlap is not None and noverlap < seg) else None,
                detrend=detrend,
                return_onesided=False,
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
    # Shift for nicer two-sided plotting (negative freqs first)
    f_out = np.fft.fftshift(f_ref)
    S_out = np.fft.fftshift(psd_avg, axes=1)
    return f_out, S_out


def compute_psd_modes_average_multi_traj(
    solutions,
    mode_indices=None,
    *,
    trajectories: int | None = None,
    window: str | tuple | None = 'boxcar',
    detrend: str | None = None,
    signal: str = 'amplitude',
):
    """
    Compute PSD by averaging periodograms across multiple trajectories (ICs).
    - trajectories: number of trajectories to include (from the start); if None, use all.
    - window: window applied before FFT; default 'boxcar' (no window).
    - detrend: optional detrending ('constant' or 'linear') before PSD.
    Returns freqs and PSD array with shape (num_modes, len(freqs)).
    """
    Z0, t0 = compute_modes_complex(solutions[0])
    dt = float(t0[1] - t0[0]) if len(t0) > 1 else 1.0
    fs = 1.0 / dt
    m_total = Z0.shape[0]
    if mode_indices is None:
        mode_indices = list(range(m_total))
    n_use = len(solutions) if trajectories is None else min(trajectories, len(solutions))

    # Check if all solutions share identical time grids for vectorization
    same_grid = True
    T_len = len(t0)
    for sol in solutions[:n_use]:
        if len(sol.t) != T_len:
            same_grid = False
            break
        if np.max(np.abs(np.diff(sol.t) - np.diff(t0))) > 1e-12:
            same_grid = False
            break

    sig_mode = (signal or 'amplitude').lower()
    if same_grid:
        # Vectorized path: build array (n_use, modes, T)
        sig_all = []
        for sol in solutions[:n_use]:
            Zi, _ = compute_modes_complex(sol)
            if sig_mode == 'intensity':
                S_i = Zi
            else:
                S_i = np.abs(Zi)
            sig_all.append(S_i)
        A = np.stack(sig_all, axis=0)  # (n_use, m_total, T)
        # Detrend and window per-replicate
        if detrend == 'constant':
            A = A - A.mean(axis=-1, keepdims=True)
        elif detrend == 'linear':
            x = np.arange(T_len)
            X = np.vstack([x, np.ones_like(x)]).T  # (T,2)
            # Solve per (replicate, mode)
            a = np.empty((n_use, m_total))
            b = np.empty((n_use, m_total))
            for i in range(n_use):
                for m in range(m_total):
                    coeffs = np.linalg.lstsq(X, A[i, m].real, rcond=None)[0]
                    a[i, m], b[i, m] = coeffs
            trend = (a[..., None] * x) + b[..., None]
            A = A - trend  # trend on real part only as a simple option
        win = get_window(window, T_len) if window is not None else 1.0
        A = A * win  # broadcasting over T
        # Compute full FFT and two-sided periodogram, average over replicates
        F = np.fft.fftfreq(T_len, d=dt)
        S_accum = np.zeros((len(mode_indices), F.size))
        for idx_m, mi in enumerate(mode_indices):
            Y = np.fft.fft(A[:, mi, :], axis=-1)
            # Density-like scaling; for two-sided, no factor 2
            P = (np.abs(Y) ** 2) / (T_len * fs)
            S_accum[idx_m] = P.mean(axis=0)
        # Shift for nicer two-sided plotting
        return np.fft.fftshift(F), np.fft.fftshift(S_accum, axes=1)
    else:
        # Fallback: loop per solution and average
        psd_accum = None
        count = 0
        f_ref = None
        for sol in solutions[:n_use]:
            Zi, t = compute_modes_complex(sol)
            dt_i = float(t[1] - t[0]) if len(t) > 1 else dt
            fs_i = 1.0 / dt_i
            for idx_m, mi in enumerate(mode_indices):
                sig = Zi[mi] if sig_mode == 'intensity' else np.abs(Zi[mi])
                if detrend:
                    if detrend == 'constant':
                        sig = sig - np.mean(sig)
                    elif detrend == 'linear':
                        x = np.arange(sig.size)
                        A_ = np.vstack([x, np.ones_like(x)]).T
                        a, b = np.linalg.lstsq(A_, sig.real, rcond=None)[0]
                        sig = sig - (a * x + b)
                win = get_window(window, sig.size) if window is not None else 1.0
                f, Pxx = periodogram(sig * win, fs=fs_i, scaling='density', return_onesided=False)
                if psd_accum is None:
                    psd_accum = np.zeros((len(mode_indices), len(f)))
                    f_ref = f
                else:
                    if len(f) != psd_accum.shape[1]:
                        Pxx = np.interp(f_ref, f, Pxx)
                psd_accum[idx_m] += Pxx
            count += 1
        psd_avg = psd_accum / max(count, 1)
        return np.fft.fftshift(f_ref), np.fft.fftshift(psd_avg, axes=1)


def plot_psd_modes(
    solutions,
    mode_indices=None,
    labels: Optional[Sequence[str]] = None,
    title: str = "PSD of |z_i|",
    *,
    method: str = 'welch',
    params: Optional[dict] = None,
    signal: str = 'amplitude',
    axis_scale: str = 'linear',
    freq_range: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    params = params or {}
    method = (method or 'welch').lower()
    if method == 'welch':
        p = {k: v for k, v in params.items() if k in {'nperseg', 'window', 'noverlap', 'detrend'}}
        f, S = compute_psd_modes_average_welch(solutions, mode_indices=mode_indices, signal=signal, **p)
    elif method in ('multi-trajectory', 'multi_traj', 'multi'):
        # Accept both 'replicates' and 'trajectories'
        p = dict(params)
        if 'trajectories' not in p and 'replicates' in p:
            p['trajectories'] = p['replicates']
        p2 = {k: v for k, v in p.items() if k in {'trajectories', 'window', 'detrend'}}
        f, S = compute_psd_modes_average_multi_traj(solutions, mode_indices=mode_indices, signal=signal, **p2)
    else:
        raise ValueError(f"Unknown PSD method: {method}")
    if mode_indices is None:
        mode_indices = list(range(S.shape[0]))
    plt.figure(figsize=(8, 5))
    THRESH = 1e-10
    # Normalize axis scale
    axis_scale = (axis_scale or 'linear').lower()
    if axis_scale not in {'linear', 'semilogy', 'semilogx', 'loglog'}:
        print(f"[warn] Unknown axis_scale '{axis_scale}', falling back to 'linear'.")
        axis_scale = 'linear'
    # Prepare frequency range mask
    use_freq_mask = False
    if isinstance(freq_range, (list, tuple)) and len(freq_range) >= 2:
        fmin = min(float(freq_range[0]), float(freq_range[1]))
        fmax = max(float(freq_range[0]), float(freq_range[1]))
        use_freq_mask = True
    for k in range(S.shape[0]):
        sig_mode = (signal or 'amplitude').lower()
        default_label = f"z[{mode_indices[k]}]" if sig_mode == 'intensity' else f"|z[{mode_indices[k]}]|"
        lab = labels[k] if labels and k < len(labels) else default_label
        mask = S[k] >= THRESH
        if use_freq_mask:
            mask = mask & (f >= fmin) & (f <= fmax)
        fk = f[mask]
        Sk = S[k][mask]
        if fk.size > 0 and Sk.size > 0:
            # For semilogx/loglog, the x-axis cannot include non-positive values.
            if axis_scale in {"semilogx", "loglog"}:
                pos_mask = fk > 0
                if not np.any(pos_mask):
                    # Nothing to plot on positive frequencies; skip this series.
                    continue
                if np.count_nonzero(~pos_mask) > 0:
                    # Inform about dropping non-positive freqs for log x-axis
                    pass  # keep silent to avoid clutter
                fk_plot = fk[pos_mask]
                Sk_plot = Sk[pos_mask]
            else:
                fk_plot, Sk_plot = fk, Sk
            if axis_scale == 'linear':
                plt.plot(fk_plot, Sk_plot, label=lab)
            elif axis_scale == 'semilogy':
                plt.semilogy(fk_plot, Sk_plot, label=lab)
            elif axis_scale == 'semilogx':
                plt.semilogx(fk_plot, Sk_plot, label=lab)
            elif axis_scale == 'loglog':
                plt.loglog(fk_plot, Sk_plot, label=lab)
    plt.xlabel("Frequency [Hz] (two-sided)")
    ylabel = "PSD |Z(f)|^2 [arb. units]" if (signal or 'amplitude').lower() == 'intensity' else "PSD of |z| [arb. units]"
    plt.ylabel(ylabel)
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