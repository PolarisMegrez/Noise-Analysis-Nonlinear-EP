import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence

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