"""
Paper Figure Generation Script
================================
Creates figures specifically for the short paper on STDP computational cost
and FPGA motivation.

Figures produced:
  paper_fig1_stdp_intro.pdf   - STDP learning rule explanation
  paper_fig2_simple_timing.pdf - Simple static vs STDP timing comparison
  paper_fig3_overhead_scaling.pdf - Overhead scaling with network size
  paper_fig4_multicore.pdf    - Multi-core scaling efficiency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

PROCESSED_DIR = Path("processed_data")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BLUE   = '#2E86AB'
PURPLE = '#A23B72'
ORANGE = '#F18F01'
GREEN  = '#06A77D'
RED    = '#D62246'

# ---------------------------------------------------------------------------
# Figure 1: STDP Introduction (two panels)
#   A - STDP learning window (ΔW vs Δt)
#   B - Schematic of pre/post spike pair
# ---------------------------------------------------------------------------

def _spike_trace(t_spike, t, baseline=-0.07, amp=0.10, width=1.5):
    """Return a voltage trace with one action potential at t_spike."""
    v = np.full_like(t, baseline)
    rise  = np.exp(-((t - t_spike)**2) / (0.3 * width)**2)
    fall  = np.exp(-((t - t_spike - width*0.6)**2) / (0.5 * width)**2) * 0.35
    v += amp * rise - amp * fall * 0.7
    return v


def make_fig1_stdp_intro():
    print("Creating paper_fig1_stdp_intro ...")

    fig = plt.figure(figsize=(8.5, 3.4))
    gs  = fig.add_gridspec(1, 2, wspace=0.42,
                           left=0.08, right=0.97, top=0.90, bottom=0.16)
    ax  = fig.add_subplot(gs[0])   # STDP window
    ax2 = fig.add_subplot(gs[1])   # voltage traces

    # ------------------------------------------------------------------ #
    # Panel A — STDP learning window                                       #
    # ------------------------------------------------------------------ #
    tau_p, tau_m = 20.0, 20.0
    A_p,   A_m   = 0.01, 0.0105

    dt = np.linspace(-90, 90, 1000)
    dw = np.where(dt > 0,  A_p * np.exp(-dt / tau_p),
                           -A_m * np.exp( dt / tau_m))

    # Shade first, then draw curve on top
    ax.fill_between(dt, dw, 0,
                    where=(dt > 0), alpha=0.18, color=GREEN, zorder=1)
    ax.fill_between(dt, dw, 0,
                    where=(dt < 0), alpha=0.18, color=RED,   zorder=1)

    ax.plot(dt[dt >= 0], dw[dt >= 0], color=GREEN, linewidth=2.2, zorder=3)
    ax.plot(dt[dt <= 0], dw[dt <= 0], color=RED,   linewidth=2.2, zorder=3)

    # Zero lines
    ax.axhline(0, color='#888888', linewidth=0.8, zorder=2)
    ax.axvline(0, color='#888888', linewidth=0.8, linestyle=':', zorder=2)

    # Region labels — placed inside the shaded areas, no overlap
    ax.text(48,  A_p * 0.55, 'LTP\n(strengthen)',
            color=GREEN, fontsize=8.5, ha='center', va='center',
            fontweight='bold')
    ax.text(-48, -A_m * 0.55, 'LTD\n(weaken)',
            color=RED,   fontsize=8.5, ha='center', va='center',
            fontweight='bold')

    # τ annotations with brace-style arrows
    ax.annotate('', xy=(tau_p, A_p * np.exp(-1)),
                xytext=(tau_p, 0),
                arrowprops=dict(arrowstyle='->', color=GREEN,
                                lw=1.2, linestyle='dashed'))
    ax.text(tau_p + 4, A_p * np.exp(-1) / 2,
            r'$\tau_+$', color=GREEN, fontsize=8.5)

    ax.annotate('', xy=(-tau_m, -A_m * np.exp(-1)),
                xytext=(-tau_m, 0),
                arrowprops=dict(arrowstyle='->', color=RED,
                                lw=1.2, linestyle='dashed'))
    ax.text(-tau_m - 12, -A_m * np.exp(-1) / 2,
            r'$\tau_-$', color=RED, fontsize=8.5)

    ax.set_xlabel(r'$\Delta t = t_{\mathrm{post}} - t_{\mathrm{pre}}$  (ms)',
                  fontsize=10)
    ax.set_ylabel(r'Weight change $\Delta W$', fontsize=10)
    ax.set_title('A  —  STDP Learning Window', loc='left',
                 fontweight='bold', fontsize=10)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-A_m * 1.4, A_p * 1.4)
    ax.tick_params(labelsize=8.5)
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------------ #
    # Panel B — two-row layout: LTP (top) and LTD (bottom)               #
    # ------------------------------------------------------------------ #
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-1.10, 1.10)
    ax2.axis('off')
    ax2.set_title('B  —  Spike-Pair Rule', loc='left',
                  fontweight='bold', fontsize=10)

    t = np.linspace(0, 100, 3000)

    # Each row: pre trace, post trace, spike positions
    # LTP row: pre at y=0.65, post at y=0.20 → Δt>0 (pre before post)
    # LTD row: pre at y=-0.20, post at y=-0.65 → Δt<0 (post before pre)
    rows = [
        # y_pre  y_post  t_pre  t_post  color   badge            dt_text        dt_above
        (  0.65,  0.20,    22,    52,  GREEN, 'LTP — strengthen', r'$\Delta t>0$',  True),
        ( -0.20, -0.65,    60,    30,    RED, 'LTD — weaken',     r'$\Delta t<0$', False),
    ]

    for y_pre, y_post, t_pre, t_post, color, badge, dt_txt, dt_above in rows:
        v_pre  = _spike_trace(t_pre,  t, baseline=0, amp=0.18, width=1.8) + y_pre
        v_post = _spike_trace(t_post, t, baseline=0, amp=0.18, width=1.8) + y_post

        # Shaded row background
        y_lo = min(y_pre, y_post) - 0.18
        y_hi = max(y_pre, y_post) + 0.18
        ax2.fill_betweenx([y_lo, y_hi], 2, 95,
                          color=color, alpha=0.07, zorder=0, linewidth=0)

        # Voltage traces
        ax2.plot(t, v_pre,  color=BLUE,   linewidth=1.8, zorder=2)
        ax2.plot(t, v_post, color=PURPLE, linewidth=1.8, zorder=2)

        # "Pre" / "Post" labels — left margin
        ax2.text(1.5, y_pre,  'Pre',  ha='right', va='center',
                 fontsize=8.5, color=BLUE,   fontweight='bold')
        ax2.text(1.5, y_post, 'Post', ha='right', va='center',
                 fontsize=8.5, color=PURPLE, fontweight='bold')

        # Outcome badge — right side, vertically centred in row
        y_mid = (y_pre + y_post) / 2
        ax2.text(96, y_mid, badge,
                 ha='right', va='center', fontsize=8.5,
                 color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.32',
                           facecolor=color, edgecolor='none', alpha=0.92))

        # Δt double-headed arrow — between the two spike peaks,
        # placed clearly OUTSIDE the trace band (above for LTP, below for LTD)
        if dt_above:
            y_arr = y_hi - 0.04          # just inside top of shaded box
            y_lbl = y_arr + 0.07
            va_lbl = 'bottom'
        else:
            y_arr = y_lo + 0.04          # just inside bottom of shaded box
            y_lbl = y_arr - 0.07
            va_lbl = 'top'

        ax2.annotate('', xy=(t_post, y_arr), xytext=(t_pre, y_arr),
                     arrowprops=dict(arrowstyle='<->', color=color, lw=1.6))
        ax2.text((t_pre + t_post) / 2, y_lbl, dt_txt,
                 ha='center', va=va_lbl, fontsize=9,
                 color=color, fontweight='bold')

    # Horizontal divider
    ax2.axhline(0, color='#cccccc', linewidth=0.8, linestyle='--', zorder=1)

    # Synapse arrow (single, in the middle gap between rows)
    x_syn = 80
    ax2.annotate('', xy=(x_syn, -0.14), xytext=(x_syn, 0.14),
                 arrowprops=dict(arrowstyle='->', color='#666666', lw=1.3))
    ax2.text(x_syn + 1.5, 0, 'synapse $W$',
             ha='left', va='center', fontsize=8, color='#555555')

    # Time arrow at the bottom
    ax2.annotate('', xy=(97, -1.06), xytext=(3, -1.06),
                 arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.0))
    ax2.text(50, -1.10, 'time', ha='center', va='center',
             fontsize=8, color='#aaaaaa')

    plt.savefig(FIGURES_DIR / 'paper_fig1_stdp_intro.pdf',
                bbox_inches='tight', dpi=300)
    plt.savefig(FIGURES_DIR / 'paper_fig1_stdp_intro.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  -> saved paper_fig1_stdp_intro.pdf/png")


# ---------------------------------------------------------------------------
# Figure 2: Simple timing — static vs STDP for key sizes (1 core, p=0.1, 20 Hz)
# ---------------------------------------------------------------------------

def make_fig2_simple_timing(agg):
    print("Creating paper_fig2_simple_timing ...")

    subset = agg[(agg.n_cores == 1) &
                 (agg.conn_prob == 0.1) &
                 (agg.firing_rate == 20.0)].copy()

    static = subset[subset.learning_rule == 'static'].sort_values('n_neurons')
    stdp   = subset[subset.learning_rule == 'stdp'].sort_values('n_neurons')

    # Keep only sizes present in both
    common = sorted(set(static.n_neurons) & set(stdp.n_neurons))
    static = static[static.n_neurons.isin(common)]
    stdp   = stdp[stdp.n_neurons.isin(common)]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    fig.subplots_adjust(wspace=0.40, left=0.10, right=0.97, top=0.88, bottom=0.18)

    # --- Panel A: bar chart for small sizes ---
    ax = axes[0]
    small_sizes = [n for n in common if n <= 500]
    s_s = static[static.n_neurons.isin(small_sizes)]
    s_d = stdp[stdp.n_neurons.isin(small_sizes)]

    x = np.arange(len(small_sizes))
    w = 0.35
    bars1 = ax.bar(x - w/2, s_s.T_simulate_mean, w,
                   yerr=s_s.T_simulate_sem, capsize=3,
                   color=BLUE, label='Static', alpha=0.88)
    bars2 = ax.bar(x + w/2, s_d.T_simulate_mean, w,
                   yerr=s_d.T_simulate_sem, capsize=3,
                   color=PURPLE, label='STDP', alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in small_sizes])
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Simulation time (s)')
    ax.set_title('A  –  Small networks (1 core)', loc='left', fontweight='bold')
    ax.legend(frameon=False)

    # --- Panel B: log-x linear-y for all sizes ---
    ax2 = axes[1]

    x_arr = np.array(common, dtype=float)
    y_s   = static.set_index('n_neurons').loc[common, 'T_simulate_mean'].values
    y_d   = stdp.set_index('n_neurons').loc[common, 'T_simulate_mean'].values
    yerr_s = static.set_index('n_neurons').loc[common, 'T_simulate_sem'].values
    yerr_d = stdp.set_index('n_neurons').loc[common, 'T_simulate_sem'].values

    ax2.errorbar(x_arr, y_s, yerr=yerr_s,
                 marker='o', linewidth=2, color=BLUE,
                 label='Static', capsize=3)
    ax2.errorbar(x_arr, y_d, yerr=yerr_d,
                 marker='s', linewidth=2, color=PURPLE,
                 label='STDP', capsize=3)

    ax2.fill_between(x_arr, y_s, y_d, alpha=0.20, color=ORANGE,
                     label='STDP overhead')

    ax2.set_xscale('log')
    ax2.set_xlabel('Number of neurons')
    ax2.set_ylabel('Simulation time (s)')
    ax2.set_title('B  –  All sizes, linear y-axis', loc='left', fontweight='bold')
    ax2.legend(frameon=False)

    # Annotate ratio at the three largest sizes
    for n, ys, yd in zip(x_arr[-3:], y_s[-3:], y_d[-3:]):
        ratio = yd / ys
        mid   = (ys + yd) / 2
        ax2.text(n * 1.08, mid, f'{ratio:.1f}×',
                 fontsize=7.5, color=ORANGE, va='center', fontweight='bold')

    plt.savefig(FIGURES_DIR / 'paper_fig2_simple_timing.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(FIGURES_DIR / 'paper_fig2_simple_timing.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  -> saved paper_fig2_simple_timing.pdf/png")


# ---------------------------------------------------------------------------
# Figure 3: STDP Overhead % vs network size (multi-condition)
# ---------------------------------------------------------------------------

def make_fig3_overhead_scaling(overhead):
    print("Creating paper_fig3_overhead_scaling ...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.subplots_adjust(wspace=0.40, left=0.08, right=0.96,
                        top=0.88, bottom=0.14)

    # ---- Panel A ----
    ax = axes[0]
    sub = overhead[(overhead.n_cores == 1) & (overhead.firing_rate == 20.0)]
    palette = [BLUE, ORANGE, RED]
    labels  = ['p = 0.05  (sparse)', 'p = 0.10  (medium)', 'p = 0.20  (dense)']

    for cp, color, lbl in zip(sorted(sub.conn_prob.unique()), palette, labels):
        d = (sub[sub.conn_prob == cp]
             .groupby('n_neurons')['overhead_pct']
             .agg(['mean', 'sem']).reset_index())
        ax.errorbar(d.n_neurons, d['mean'], yerr=d['sem'],
                    marker='o', linewidth=2, color=color,
                    label=lbl, capsize=3, zorder=3)

    # Threshold lines with right-edge labels
    for yval, ls, lbl in [(50, '--', '50%'), (100, ':', '100%')]:
        ax.axhline(yval, color='#bbbbbb', linestyle=ls, linewidth=0.9, zorder=1)
        ax.text(1.3e4, yval + 3, lbl, fontsize=8, color='#888888',
                va='bottom', ha='right')

    # Runaway annotation — bottom-right corner: p=0.2 ends at ~41%
    # while blue/orange are at 130-145%, so the area y<40%, x>3000 is empty
    ax.annotate(
        'Runaway spiking\n\u2192 overhead % drops\n(see text)',
        xy=(8000, 41), xytext=(5000, 15),
        fontsize=7.5, color=RED, ha='center', va='center',
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.2,
                        connectionstyle='arc3,rad=-0.2'),
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=RED,
                  alpha=0.95, linewidth=1.1)
    )

    ax.set_xscale('log')
    ax.set_ylim(-5, 200)
    ax.set_xlabel('Number of neurons', fontsize=10)
    ax.set_ylabel('STDP overhead (%)', fontsize=10)
    ax.set_title('A  --  STDP overhead vs. network size  (1 core, 20 Hz)',
                 loc='left', fontweight='bold', fontsize=9.5)
    ax.legend(title='Connection prob.', frameon=False, fontsize=8,
              title_fontsize=8, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---- Panel B ----
    ax2 = axes[1]
    sub2 = (overhead[(overhead.n_cores == 1) & (overhead.firing_rate == 20.0)
                     & (overhead.conn_prob == 0.1)]
            .groupby('n_neurons')
            .agg(T_overhead_mean=('T_overhead', 'mean'),
                 n_connections_mean=('n_connections', 'mean'))
            .reset_index())
    sub2 = sub2[sub2.n_connections_mean > 0].copy()
    sub2['us_per_conn'] = sub2.T_overhead_mean / sub2.n_connections_mean * 1e6

    bands = [
        (8,   80,  '#dde0ff', '1  Fixed call overhead', '#3344bb'),
        (80,  800, '#d4f0da', '2  In cache  (fast)',    '#226633'),
        (800, 2e5, '#ffe0e0', '3  Cache miss  (slow)',  '#cc3333'),
    ]
    for x0, x1, fc, lbl, tc in bands:
        ax2.axvspan(x0, x1, color=fc, alpha=0.60, zorder=0, linewidth=0)

    ax2.plot(sub2.n_neurons, sub2.us_per_conn,
             marker='D', linewidth=2.2, color=PURPLE, zorder=4)

    # Cache-friendly baseline (empirical minimum from zone 2)
    ax2.axhline(36, color='#999999', linestyle='--', linewidth=1.0, zorder=2)
    ax2.text(10.5, 37.5, 'Cache-friendly baseline: 36 us/synapse\n(observed min. when table fits in cache)',
             fontsize=7, color='#777777', va='bottom', linespacing=1.3)

    # Cache boundary line
    ax2.axvline(800, color='#cc3333', linestyle=':', linewidth=1.2, zorder=2)

    # Band labels — placed where data is absent in each zone:
    #   zone 1 (data ~60 µs): label below at y=27
    #   zone 2 (data ~36 µs): label above at y=54
    #   zone 3 (data ~64 µs): label below at y=27
    label_positions = [
        (np.sqrt(8   * 80),   27, 'top'),     # zone 1 — below data
        (np.sqrt(80  * 800),  65, 'bottom'),  # zone 2 — above data, higher
        (np.sqrt(800 * 3e4),  27, 'top'),     # zone 3 — below data
    ]
    for (x0, x1, fc, lbl, tc), (xm, ym, va) in zip(bands, label_positions):
        ax2.text(xm, ym, lbl, ha='center', va=va,
                 fontsize=8.5, color=tc, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white',
                           ec=tc, alpha=0.85, linewidth=0.8))

    ax2.set_xscale('log')
    ax2.set_ylim(22, 78)
    ax2.set_xlabel('Number of neurons', fontsize=10)
    ax2.set_ylabel('STDP cost per synapse (us)', fontsize=10)
    ax2.set_title('B  --  Cost per synapse  (p = 0.1, 1 core, 20 Hz)',
                  loc='left', fontweight='bold', fontsize=9.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.savefig(FIGURES_DIR / 'paper_fig3_overhead_scaling.pdf',
                bbox_inches='tight', dpi=300)
    plt.savefig(FIGURES_DIR / 'paper_fig3_overhead_scaling.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  -> saved paper_fig3_overhead_scaling.pdf/png")

def make_fig4_multicore(scaling):
    print("Creating paper_fig4_multicore ...")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.subplots_adjust(wspace=0.45, left=0.10, right=0.97, top=0.88, bottom=0.18)

    cores = [1, 2, 4, 8]

    # ------------------------------------------------------------------
    # Panel A: speedup curves for static synapses, various network sizes
    # ------------------------------------------------------------------
    ax = axes[0]
    target_sizes = [100, 500, 2000, 8000]
    palette = [BLUE, GREEN, ORANGE, RED]

    static_s = scaling[(scaling.learning_rule == 'static') &
                       (scaling.conn_prob == 0.1) &
                       (scaling.firing_rate == 20.0)]

    for size, color in zip(target_sizes, palette):
        d = static_s[static_s.n_neurons == size].sort_values('n_cores')
        if len(d) < 2:
            continue
        ax.errorbar(d.n_cores, d.speedup,
                    marker='o', linewidth=2, color=color,
                    label=f'{size:,}', capsize=3)

    ax.plot([1, 2, 4, 8], [1, 2, 4, 8], 'k--', linewidth=1.2, label='Ideal (linear)', alpha=0.6)
    ax.set_xticks(cores)
    ax.set_xlabel('Number of CPU cores')
    ax.set_ylabel('Speedup vs 1 core')
    ax.set_title('A  -  Multi-core speedup (static)', loc='left', fontweight='bold')

    leg = ax.legend(title='Neurons', frameon=False, fontsize=8,
                    title_fontsize=8, loc='upper left')

    # ------------------------------------------------------------------
    # Panel B: absolute simulation time — STDP vs Static at N=4000
    # Shows the overhead gap persists at ALL core counts
    # ------------------------------------------------------------------
    ax2 = axes[1]
    size = 4000

    data_pairs = {}
    for rule, color, label in [('static', BLUE, 'Static'),
                                ('stdp',   PURPLE, 'STDP')]:
        d = scaling[(scaling.learning_rule == rule) &
                    (scaling.n_neurons == size) &
                    (scaling.conn_prob == 0.1) &
                    (scaling.firing_rate == 20.0)].sort_values('n_cores')
        if len(d) == 0:
            continue
        data_pairs[rule] = d
        ax2.errorbar(d.n_cores, d.T_simulate,
                     yerr=d.T_simulate_sem,
                     marker='o', linewidth=2, color=color, label=label, capsize=3)
        # Dashed ideal-scaling reference from 1-core value
        t1 = d[d.n_cores == 1]['T_simulate'].values[0]
        ax2.plot([1, 2, 4, 8], [t1, t1/2, t1/4, t1/8],
                 '--', color=color, linewidth=1, alpha=0.35)

    ax2.set_yscale('log')
    ax2.set_xticks(cores)
    ax2.set_xlabel('Number of CPU cores')
    ax2.set_ylabel('Wall-clock time (s)')
    ax2.set_title(f'B  -  Overhead gap persists (N={size:,})', loc='left', fontweight='bold')
    ax2.legend(frameon=False, fontsize=8, loc='upper right')

    # Annotate the gap at 8 cores with a bracket
    if 'static' in data_pairs and 'stdp' in data_pairs:
        t_static8 = data_pairs['static'][data_pairs['static'].n_cores == 8]['T_simulate'].values[0]
        t_stdp8   = data_pairs['stdp'][data_pairs['stdp'].n_cores == 8]['T_simulate'].values[0]
        ratio = t_stdp8 / t_static8
        # Vertical arrow spanning the gap at x=8
        ax2.annotate('', xy=(8.4, t_stdp8), xytext=(8.4, t_static8),
                     arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
        ax2.text(8.6, (t_static8 * t_stdp8) ** 0.5,  # geometric midpoint on log scale
                 f'{ratio:.1f}x\noverhead',
                 fontsize=7, va='center', ha='left', color='black')

    # Note about dashed lines
    ax2.text(0.03, 0.04,
             'Dashed = ideal linear scaling\nfrom each 1-core baseline',
             transform=ax2.transAxes, fontsize=6.5, color='#666666',
             va='bottom')

    plt.savefig(FIGURES_DIR / 'paper_fig4_multicore.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(FIGURES_DIR / 'paper_fig4_multicore.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  -> saved paper_fig4_multicore.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    agg      = pd.read_pickle(PROCESSED_DIR / 'aggregated_trials.pkl')
    overhead = pd.read_pickle(PROCESSED_DIR / 'stdp_overhead.pkl')
    scaling  = pd.read_pickle(PROCESSED_DIR / 'core_scaling.pkl')

    make_fig1_stdp_intro()
    make_fig2_simple_timing(agg)
    make_fig3_overhead_scaling(overhead)
    make_fig4_multicore(scaling)

    print("\nAll paper figures saved to figures/")
