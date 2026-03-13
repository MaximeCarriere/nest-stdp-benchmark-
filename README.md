# STDP Computational Overhead Benchmark — NEST Simulator

**Paper:** *Quantifying the Computational Cost of Spike-Timing-Dependent Plasticity in the NEST Neural Simulator: A Benchmark Study Towards FPGA Acceleration*

**Author:** Maxime Carriere — Brain Language Laboratory, Freie Universität Berlin

---

## Overview

This repository contains all code and data needed to reproduce the results and figures of the paper.

We benchmark the computational overhead of Spike-Timing-Dependent Plasticity (STDP) compared to static synapses in the [NEST simulator](https://www.nest-simulator.org/) (v3.6), across 2,160 configurations spanning:

| Parameter | Values |
|-----------|--------|
| Network size | 10 – 16,000 neurons |
| Connection probability | 0.05, 0.1, 0.2 |
| Poisson input firing rate | 10, 20, 40 Hz |
| CPU cores | 1, 2, 4, 8 |
| Trials per config | 3 |

**Key findings:**
- STDP incurs up to **183% overhead** (2.83× as long) over static synapses at N = 4,000–8,000 neurons
- The overhead is driven by **CPU cache capacity**: when the synapse table exceeds cache size (~800 neurons), per-synapse cost nearly doubles
- Multi-core parallelism scales both rule types equally — the STDP overhead **persists at all core counts**
- These results motivate an FPGA-based STDP accelerator

---

## Repository Structure

```
.
├── nest_collect_data.py      # Step 1: run NEST simulations and save checkpoints
├── process_data.py           # Step 2: aggregate trials and compute metrics
├── make_paper_figures.py     # Step 3: generate all paper figures
├── make_paper_pdf.py         # Step 4: compile the paper PDF (ReportLab)
├── paper.tex                 # LaTeX source (alternative, requires tectonic/pdflatex)
├── paper.pdf                 # Final paper
│
├── processed_data/
│   ├── aggregated_trials.csv   # All 2,160 simulation results
│   ├── stdp_overhead.csv       # STDP vs static overhead per configuration
│   ├── core_scaling.csv        # Multi-core speedup metrics
│   ├── update_metrics.csv      # Per-synapse timing estimates
│   └── summary_statistics.json # Top-level summary numbers cited in the paper
│
└── figures/
    ├── paper_fig1_stdp_intro.png/pdf    # STDP learning rule explanation
    ├── paper_fig2_simple_timing.png/pdf # Static vs STDP wall-clock timing
    ├── paper_fig3_overhead_scaling.png/pdf  # Overhead % and per-synapse cost
    └── paper_fig4_multicore.png/pdf     # Multi-core speedup and gap persistence
```

---

## Reproducing the Results

### Dependencies

```bash
pip install nest-simulator matplotlib pandas numpy scipy reportlab
```

NEST v3.6 must be installed separately — see [NEST installation guide](https://nest-simulator.readthedocs.io/en/stable/installation/).

### Step-by-step

```bash
# 1. Collect raw simulation data (~hours, requires NEST)
python nest_collect_data.py

# 2. Process and aggregate results
python process_data.py

# 3. Generate paper figures (figures/ directory)
python make_paper_figures.py

# 4. Compile the paper PDF
python make_paper_pdf.py
# or with LaTeX:
tectonic paper.tex
```

If you only want to regenerate figures from the pre-computed data (no NEST required), start from step 3 — the `processed_data/` CSVs are included in this repository.

---

## NEST Simulation Parameters

Neurons: `iaf_psc_alpha` with default NEST parameters
Synapses (static): `static_synapse`, weight = 1.0, delay = 1.0 ms
Synapses (STDP): `stdp_synapse`, additive rule, τ± = 20 ms, λ = 0.01, w ∈ [0, 1]
Input: `poisson_generator` → all neurons, rate as specified
Simulation duration: 1,000 ms biological time per trial

---

## Citation

If you use this code or data, please cite:

```
Carriere, M. (2025). Quantifying the Computational Cost of Spike-Timing-Dependent
Plasticity in the NEST Neural Simulator: A Benchmark Study Towards FPGA Acceleration.
Brain Language Laboratory, Freie Universität Berlin.
https://github.com/MaximeCarriere/nest-stdp-benchmark-
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
