"""
Generate paper.pdf using ReportLab
===================================
Produces a two-column, publication-style PDF from the STDP benchmark results.
Run:  python make_paper_pdf.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    KeepTogether, HRFlowable, PageBreak, FrameBreak
)
from reportlab.platypus.doctemplate import BaseDocTemplate, PageTemplate, Frame
from reportlab.platypus.flowables import Flowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

FIGURES = Path("figures")
OUT = Path("paper.pdf")

# ---------------------------------------------------------------------------
# Page layout — two columns
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
MARGIN_OUTER = 1.8 * cm
MARGIN_INNER = 0.9 * cm
COL_GAP      = 0.5 * cm

col_w = (PAGE_W - 2*MARGIN_OUTER - COL_GAP) / 2

left_frame = Frame(
    MARGIN_OUTER, MARGIN_OUTER,
    col_w, PAGE_H - 2*MARGIN_OUTER - 1.5*cm,
    id='left', leftPadding=0, rightPadding=0,
    topPadding=0, bottomPadding=0,
)
right_frame = Frame(
    MARGIN_OUTER + col_w + COL_GAP, MARGIN_OUTER,
    col_w, PAGE_H - 2*MARGIN_OUTER - 1.5*cm,
    id='right', leftPadding=0, rightPadding=0,
    topPadding=0, bottomPadding=0,
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
SS = getSampleStyleSheet()

BASE = 9   # pt

def S(name, **kw):
    return ParagraphStyle(name, **kw)

title_style = S('Title2',
    fontName='Helvetica-Bold', fontSize=13, leading=16,
    alignment=TA_CENTER, spaceAfter=4)

author_style = S('Author',
    fontName='Helvetica', fontSize=9, leading=12,
    alignment=TA_CENTER, spaceAfter=2)

date_style = S('Date',
    fontName='Helvetica-Oblique', fontSize=8, leading=10,
    alignment=TA_CENTER, spaceAfter=8)

abstract_style = S('Abstract',
    fontName='Helvetica', fontSize=8, leading=11,
    alignment=TA_JUSTIFY, leftIndent=6, rightIndent=6,
    spaceBefore=4, spaceAfter=6)

abstract_title_style = S('AbstractTitle',
    fontName='Helvetica-Bold', fontSize=8, leading=11,
    alignment=TA_CENTER, spaceAfter=2)

section_style = S('Section',
    fontName='Helvetica-Bold', fontSize=10, leading=13,
    spaceBefore=10, spaceAfter=3)

subsection_style = S('Subsection',
    fontName='Helvetica-Bold', fontSize=9, leading=12,
    spaceBefore=6, spaceAfter=2)

body_style = S('Body2',
    fontName='Helvetica', fontSize=BASE, leading=12,
    alignment=TA_JUSTIFY, spaceAfter=4)

body_italic = S('BodyItalic',
    fontName='Helvetica-Oblique', fontSize=BASE, leading=12,
    alignment=TA_JUSTIFY, spaceAfter=4)

caption_style = S('Caption',
    fontName='Helvetica', fontSize=7.5, leading=10,
    alignment=TA_JUSTIFY, spaceAfter=6)

ref_style = S('Ref',
    fontName='Helvetica', fontSize=7.5, leading=10,
    alignment=TA_JUSTIFY, spaceAfter=2, leftIndent=10, firstLineIndent=-10)

bullet_style = S('Bullet',
    fontName='Helvetica', fontSize=BASE, leading=12,
    alignment=TA_LEFT, leftIndent=14, firstLineIndent=-8,
    spaceAfter=2)

# ---------------------------------------------------------------------------
# Helper: full-width figure spanning both columns
# ---------------------------------------------------------------------------
def fig(filename, caption_text, width_frac=0.95):
    path = str(FIGURES / filename)
    w = (col_w * 2 + COL_GAP) * width_frac
    img = Image(path, width=w, height=w * 0.46)
    cap = Paragraph(caption_text, caption_style)
    return KeepTogether([img, cap, Spacer(1, 4)])

def fig_col(filename, caption_text, width_frac=0.98):
    """Single-column figure."""
    path = str(FIGURES / filename)
    w = col_w * width_frac
    img = Image(path, width=w, height=w * 0.47)
    cap = Paragraph(caption_text, caption_style)
    return KeepTogether([img, cap, Spacer(1, 4)])

def bold(text):
    return f'<b>{text}</b>'

def it(text):
    return f'<i>{text}</i>'

def p(text, style=None):
    return Paragraph(text, style or body_style)

def sec(text):
    return Paragraph(text, section_style)

def subsec(text):
    return Paragraph(text, subsection_style)

def bullet(text):
    return Paragraph(f'• {text}', bullet_style)

def sp(h=4):
    return Spacer(1, h)

def hr():
    return HRFlowable(width='100%', thickness=0.5, color=colors.grey, spaceAfter=4)

# ---------------------------------------------------------------------------
# Build story
# ---------------------------------------------------------------------------

def build_story():
    story = []

    # --- Title block (full width via spanning table trick) ---
    story += [
        Paragraph(
            'Quantifying the Computational Cost of STDP<br/>'
            'in Spiking Neural Networks:<br/>'
            'A Benchmark Study Towards FPGA Acceleration',
            title_style),
        Paragraph('Maxime Carriere — Brain Language Laboratory, Freie Universität Berlin', author_style),
        Paragraph('March 2026 — Preliminary results', date_style),
        hr(),
        Paragraph(bold('Abstract'), abstract_title_style),
        Paragraph(
            'Spiking Neural Networks (SNNs) trained with Spike-Timing Dependent '
            'Plasticity (STDP) offer a biologically plausible model of learning. '
            'Their simulation, however, carries a heavy computational cost that '
            'grows super-linearly with network size. '
            'Using the NEST simulator across 2,160 configurations '
            '(network size, connection density, firing rate, and core count), '
            'we show that STDP can more than <b>double</b> total simulation time '
            'compared to static synapses—an overhead reaching <b>183%</b> '
            'for large networks. '
            'Multi-core CPU parallelism scales both synaptic rules equally '
            'but cannot close the STDP overhead gap: STDP remains ~2x '
            'slower than static at all core counts. '
            'These findings motivate a hardware-accelerated solution. '
            'We present these benchmarks as a foundation for designing an '
            'FPGA-based STDP co-processor.',
            abstract_style),
        hr(),
        sp(6),
    ]

    # === Section 1: Introduction ===
    story += [
        sec('1. Introduction'),
        p('The human brain processes sensory information using approximately '
          '10<super>11</super> neurons connected by roughly 10<super>15</super> synapses [1]. '
          'Unlike the artificial neural networks behind today\'s deep-learning '
          'revolution—which use smooth continuous activations and '
          'gradient-based training—biological neurons communicate through '
          'discrete electrical pulses called <i>spikes</i>. '
          'Models that respect this mechanism are called '
          '<b>Spiking Neural Networks (SNNs)</b>.'),
        p('SNNs are of great interest for neuromorphic computing, low-power '
          'embedded intelligence, and fundamental neuroscience research. '
          'However, simulating even small-scale networks on conventional '
          'computers is expensive. '
          'The key bottleneck is the learning rule that governs how synaptic '
          'connections are updated during simulation.'),
        p('In this work we focus on the dominant biological learning rule: '
          '<b>Spike-Timing Dependent Plasticity (STDP)</b>. '
          'We systematically benchmark its computational overhead using the '
          'widely used NEST simulator [2] and identify the scaling regimes '
          'where conventional CPUs become inadequate. '
          'Our ultimate goal is to offload STDP weight updates to an '
          'FPGA co-processor, enabling real-time simulation of large networks.'),
    ]

    # === Section 2: STDP Background ===
    story += [
        sec('2. Background: What is STDP?'),
        subsec('2.1 Synaptic Plasticity'),
        p('Learning in biological neural circuits is implemented by changing '
          'the <i>strength</i> (weight) of synaptic connections. '
          'A synapse connects a <i>pre-synaptic</i> neuron (the sender) '
          'to a <i>post-synaptic</i> neuron (the receiver). '
          'The key insight from Hebbian theory is: '
          '<i>neurons that fire together wire together</i> [3].'),
        subsec('2.2 The STDP Rule'),
        p('STDP refines this idea by making the weight change depend on the '
          '<i>relative timing</i> of pre- and post-synaptic spikes, first '
          'demonstrated experimentally by Markram et al. [4] and '
          'subsequently characterised in hippocampal neurons by Bi and Poo [5]. '
          'Define dt = t<sub>post</sub> - t<sub>pre</sub>. '
          'The weight update follows an exponential learning window '
          '(Fig. 1):'),
        p('dW = A+ x exp(-dt/tau+)   if dt >= 0   [LTP]',
          body_italic),
        p('dW = -A- x exp(+dt/tau-)   if dt &lt; 0   [LTD]',
          body_italic),
        p('When the pre-synaptic neuron fires <i>before</i> the post-synaptic '
          'one (dt > 0), the synapse is <b>strengthened</b> '
          '(Long-Term Potentiation, LTP). '
          'In the opposite case it is <b>weakened</b> '
          '(Long-Term Depression, LTD). '
          'Time constants tau± approx. 20 ms govern the time window '
          'over which coincidences are detected.'),
    ]

    # Figure 1 — STDP
    story += [
        fig_col('paper_fig1_stdp_intro.png',
                '<b>Figure 1.</b> STDP learning window. '
                '(A) Weight change dW vs inter-spike interval dt. '
                'Pre fires before post (dt>0): LTP (green). '
                'Post fires before pre (dt<0): LTD (red). '
                '(B) Spike-pair schematic showing the synapse W '
                'between two neurons.'),
    ]

    story += [
        subsec('2.3 Computational Cost'),
        p('For a network of N neurons, every spike triggers a weight update '
          'at each connected synapse. The total number of STDP operations '
          'per second scales as:'),
        p('Updates/s  =  f · N · k_in', body_italic),
        p('where f is the mean firing rate and k_in is the mean in-degree '
          '(average number of incoming synapses per neuron). '
          'With N = 2,000, f = 20 Hz, and connection probability 10%, '
          'this yields approx. 8 × 10<super>6</super> updates/second — all '
          'serialised on a CPU, incurring memory-bandwidth pressure '
          'that grows super-linearly with N, as shown below.'),
    ]

    # === Section 3: Methods ===
    story += [
        sec('3. Experimental Setup'),
        p('We used <b>NEST 3.6</b> [2] to simulate recurrent networks of '
          'integrate-and-fire neurons (iaf_psc_alpha) driven by Poisson '
          'external input. Each configuration was simulated for '
          '20 s of biological time. The parameter space covered:'),
        bullet('Network size N: 10 to 16,000 neurons (10 values, log-spaced)'),
        bullet('CPU cores: 1, 2, 4, 8'),
        bullet('Connection probability p: 5%, 10%, 20%'),
        bullet('External firing rate: 10, 20, 40 Hz'),
        bullet('Learning rule: static synapses vs additive STDP '
               '(tau± = 20 ms, lambda = 0.01)'),
        sp(4),
        p('In total, <b>2,160 simulations</b> were run (3 trials each for '
          'statistical error bars), collected over 2.9 days on a standard '
          'desktop workstation. Timing was measured with '
          'time.perf_counter(), separating the network setup phase from '
          'the simulation phase.'),
    ]

    # === Section 4: Results ===
    story += [
        sec('4. Results'),
        subsec('4.1 Overhead is Negligible Below 1,000 Neurons, Then Critical'),
        p('Fig. 2 compares simulation time for static and STDP networks '
          '(1 core, p = 0.1, 20 Hz). For small networks (N &lt;= 500) '
          'both rules complete in under 5 s and bars are nearly '
          'identical (Fig. 2A). The linear-scale plot (Fig. 2B) reveals '
          'a sharp transition around N ~ 1,000 neurons, above which '
          'the two curves diverge dramatically: '
          '<b>1.9× at N = 2,000</b> (53 s vs. 28 s), '
          '<b>2.3× at N = 4,000</b> (177 s vs. 76 s), and '
          '<b>2.4× at N = 8,000</b> (808 s vs. 330 s) — '
          'an absolute overhead of nearly <b>8 minutes per simulation</b>.'),
    ]

    story += [
        fig_col('paper_fig2_simple_timing.png',
                '<b>Figure 2.</b> Static vs STDP simulation time. '
                '(A) Bar chart for small networks (N &lt;= 500). '
                '(B) Full log-scale comparison; shaded region marks STDP '
                'overhead. 1 core, p = 0.1, 20 Hz Poisson input. '
                'Points = mean of 3 trials; error bars = SEM.'),
    ]

    story += [
        subsec('4.2 Why Overhead Peaks, Then Drops — and What the Per-Synapse Cost Reveals'),
        p('<b>Fig. 3A — Overhead percentage.</b> '
          'For all connection densities, STDP overhead is negligible '
          '(below 5%) for very small networks (N &lt; 100) and grows '
          'steeply beyond N ~ 500, peaking at <b>183%</b> for dense '
          'networks around N = 4,000. A 183% overhead means STDP takes '
          '<b>2.83× as long</b> as the static baseline (total = static + 183% of static).'),
        p('The <b>drop for p = 0.20 at N = 8,000</b> (red curve, Fig. 3A) '
          'is not a measurement error — it is a real network effect. '
          'At that size and density, neurons excite each other in a '
          'runaway chain reaction, driving the firing rate to 162 Hz — '
          'eight times the 20 Hz input signal. '
          'Think of it as a feedback loop: more spikes trigger more spikes. '
          'The <i>static</i> simulation must track all of these extra '
          'spikes (its runtime jumps 7.8× from N = 4,000 to 8,000), '
          'while STDP weight updates only grow 1.9× over the same step '
          'because the weight arithmetic is bounded. '
          'The overhead <i>percentage</i> therefore shrinks — not because '
          'STDP became cheaper (it still adds 434 s of absolute extra cost), '
          'but because the baseline exploded even faster. '
          'In biologically realistic models, networks are kept below such '
          'runaway regimes, placing them in the 100–170% overhead zone.'),
        p('<b>Fig. 3B — Cost per synapse.</b> '
          'If every synapse update took the same time regardless of '
          'network size, the curve would be flat (dashed reference). '
          'Instead, three regimes appear (Fig. 3B), explained by a '
          'fundamental property of modern processors:'),
        p('A CPU does not read data directly from RAM each time it needs it '
          '— that would be too slow. Instead, it keeps a small '
          '<b>cache</b> of recently used data in fast memory built '
          'right next to the processor chip. Think of it like a '
          'desk: the cache is the small desk surface where you keep '
          'the papers you are actively using, and RAM is the filing '
          'cabinet across the room. Working from the desk is fast; '
          'getting up to fetch a file from the cabinet takes much longer. '
          'A <b>cache miss</b> is exactly that: the processor needs a '
          'piece of data, finds it is not on the desk, and must '
          'walk to the cabinet — wasting time waiting instead of computing.'),
        bullet('<b>Zone 1 — Very small networks (N &lt; 100, blue):</b> '
               'Cost is ~60 µs per synapse, even though the table is tiny. '
               'The reason is fixed overhead per update event: every STDP '
               'calculation involves a function call, a timestamp lookup, '
               'and housekeeping that costs roughly the same regardless '
               'of how many synapses exist. With only ~10 synapses, '
               'this overhead is never amortised.'),
        bullet('<b>Zone 2 — Medium networks (N = 100–800, green):</b> '
               'Cost drops to ~36 µs — the cheapest regime. '
               'The synapse table is small enough (under ~1 MB) to '
               'fit entirely in the processor fast cache (the desk). '
               'Every weight lookup is served instantly from cache, '
               'no waiting for RAM.'),
        bullet('<b>Zone 3 — Large networks (N &gt;= 800, red):</b> '
               'Cost doubles to ~63 µs. The synapse table now exceeds '
               'the cache size (100k synapses × ~32 bytes = ~3 MB). '
               'The processor must fetch most weights from RAM — a '
               'cache miss on almost every update. It spends more time '
               'waiting for data than actually computing, which is why '
               'the cost per synapse is nearly twice as high as in Zone 2.'),
        p('This boundary at N ~ 800 explains the sharp rise in overhead '
          'seen in Fig. 3A. An FPGA eliminates this problem entirely: '
          'it stores the synapse table in dedicated on-chip memory '
          '(called Block RAM) that is as fast as a cache regardless '
          'of table size — there is no filing cabinet to walk to. '
          'The ~2x cost penalty of Zone 3 would vanish.'),
    ]

    story += [
        fig_col('paper_fig3_overhead_scaling.png',
                '<b>Figure 3.</b> '
                '(A) STDP overhead (%) vs. network size for three '
                'connection densities (1 core, 20 Hz). '
                'The annotated drop for p = 0.20 at N = 8,000 reflects '
                'network runaway excitation, not a reduction in STDP cost. '
                '(B) STDP cost per synapse. '
                'Three regimes are highlighted: fixed per-event overhead (blue), '
                'in-cache access (green, fast zone), '
                'and cache-miss zone (red, slow zone). '
                'The dashed line is the ideal constant cost.'),
    ]

    story += [
        subsec('4.3 Multi-Core Parallelism Does Not Close the STDP Gap'),
        p('A natural first response to the STDP overhead is to simply use '
          'more CPU cores: if STDP doubles the computation, perhaps running '
          'on two cores restores the original speed. Fig. 4 tests this '
          'intuition directly.'),
        p('<b>Speedup with more cores (Panel A).</b> For static synapses, '
          'parallelism is beneficial only for large networks. Below N = 500 '
          'neurons, adding cores barely helps or even hurts: coordinating '
          'multiple threads requires constant communication, and for small '
          'workloads this cost exceeds the time saved by splitting the work. '
          'At 10 neurons, 8 cores are actually 1.8x <i>slower</i> than a '
          'single core. For large networks (N >= 2,000), multi-core scaling '
          'is substantial, reaching close to or above ideal linear scaling, '
          'with a practical ceiling around 10x speedup at 8 cores.'),
        p('<b>The STDP overhead gap persists at every core count (Panel B).</b> '
          'Panel B shows the raw wall-clock simulation time for both static '
          'and STDP synapses at N = 4,000 neurons -- the most informative '
          'way to compare the two learning rules under parallelism. Both '
          'curves decrease steeply as cores are added, confirming that '
          'parallelism is effective for both rule types. Going from 1 to 8 '
          'cores reduces static time from 76 s to 7.8 s (9.7x faster) and '
          'STDP time from 177 s to 16 s (11.0x faster).'),
        p('Yet at every single core count, STDP remains approximately '
          '2.1-2.3x slower than static, as shown by the bracket at 8 cores. '
          'The two curves run in parallel: they drop at the same rate, '
          'keeping the same proportional distance. '
          '<b>Adding more CPU cores cannot eliminate the STDP overhead.</b> '
          'Each core still has to execute the same weight-update calculation '
          'for every synapse; parallelism simply divides this work among more '
          'processors, but does not make the calculation itself any cheaper.'),
        p('Think of it this way: if STDP requires ten steps where static '
          'requires four, then splitting across eight cores gives you eight '
          'workers each doing ten steps vs. four steps. Every worker is still '
          '2.5x slower on their share of the work. The ratio never changes.'),
        p('This finding has an important practical implication. Scaling from '
          '1 to 8 cores on a typical server is already a significant hardware '
          'investment. Yet even with this investment, a biologically realistic '
          'network with STDP synapses will always simulate 2x slower than the '
          'same network with fixed weights. For real-time or near-real-time '
          'brain-computer interface applications, this gap is unacceptable. '
          'A fundamentally different solution is needed -- one that reduces '
          'the cost of the STDP computation itself rather than distributing '
          'it more efficiently across more processors.'),
    ]

    story += [
        fig_col('paper_fig4_multicore.png',
                '<b>Figure 4.</b> Multi-core scaling. '
                '(A) Speedup relative to 1 core for different network sizes '
                '(static synapses, p = 0.1, 20 Hz); dashed line = ideal '
                'linear scaling. Small networks gain little or nothing from '
                'additional cores. '
                '(B) Absolute wall-clock simulation time for static and STDP '
                'at N = 4,000. Both rules scale similarly with core count '
                '(faint dashed = ideal from each 1-core baseline), but the '
                'STDP overhead of ~2.1x persists regardless of core count. '
                'More cores do not close the gap.'),
    ]

    # === Section 5: FPGA Motivation ===
    story += [
        sec('5. The Case for FPGA Acceleration'),
        p('The results above identify a clear bottleneck: STDP weight '
          'updates incur a super-linearly growing cost that neither '
          'multi-core CPUs nor software optimisation can fully address. '
          'The root cause is architectural: a general-purpose CPU must '
          '(1) <i>fetch</i> the synapse record from DRAM (cache-miss prone), '
          '(2) <i>compute</i> the exponential decay, and '
          '(3) <i>write back</i> the updated weight — millions of times '
          'per second, on a single sequential pipeline.'),
        p('<b>FPGAs</b> (Field-Programmable Gate Arrays) offer a '
          'compelling alternative:'),
        bullet('<b>Massive parallelism</b>: process hundreds of synaptic '
               'updates simultaneously in dedicated hardware lanes.'),
        bullet('<b>On-chip BRAM</b>: synapse tables for moderate networks '
               'fit in fast on-chip memory, eliminating DRAM latency.'),
        bullet('<b>Fixed-point exponentials</b>: the STDP kernel maps '
               'efficiently to lookup tables (LUTs).'),
        bullet('<b>Energy efficiency</b>: FPGA neural simulation consumes '
               '10–100× less power than GPU or CPU equivalents [5,6].'),
        bullet('<b>Deterministic latency</b>: cycle-accurate pipelines '
               'simplify spike-timestamp bookkeeping required by STDP.'),
        sp(4),
        p('Several FPGA-based SNN accelerators have been proposed [5,6,7], '
          'but none directly targets the STDP update pipeline as a '
          'co-processor to a CPU-based simulator such as NEST. '
          'Our planned architecture places the FPGA alongside the host CPU: '
          'the CPU manages network topology and spike routing, while the '
          'FPGA handles all weight-update arithmetic. '
          'The benchmarks presented here will guide the '
          'memory-bandwidth and arithmetic-throughput requirements '
          'for that co-processor design.'),
    ]

    # === Section 6: Conclusion ===
    story += [
        sec('6. Conclusion'),
        p('We have characterised the computational overhead of STDP in '
          'the NEST simulator across a wide range of network sizes, '
          'densities, and hardware configurations. Key findings:'),
        bullet('<b>Below ~1,000 neurons</b>: overhead is negligible (<10%); '
               'standard CPUs are sufficient.'),
        bullet('<b>Above ~1,000 neurons</b>: sharp transition — overhead '
               'exceeds 50% at 500 neurons and peaks at <b>183%</b> '
               '(2.83× as long) around 4,000 neurons.'),
        bullet('<b>Absolute cost</b>: ~8 minutes of extra runtime per '
               'simulation at 8,000 neurons — prohibitive for sweeps.'),
        bullet('<b>CPU parallelism limited</b>: synchronisation and '
               'irregular memory access prevent effective scaling.'),
        bullet('<b>Root bottleneck is memory access</b>: the synapse table '
               'outgrows the CPU cache, causing costly RAM fetches. '
               'FPGA on-chip memory eliminates this penalty.'),
        sp(4),
        p('These results establish a quantitative baseline for the design '
          'and evaluation of an FPGA-based STDP accelerator, which is '
          'the next step of this work.'),
    ]

    # === Data and Code Availability ===
    story += [
        Paragraph('Data and Code Availability', subsection_style),
        p('All simulation scripts, processed data (CSV), and '
          'figure-generation code are publicly available at:'),
        Paragraph(
            '<b>https://github.com/MaximeCarriere/nest-stdp-benchmark-</b>',
            ParagraphStyle('repo', parent=body_style,
                           alignment=1, textColor=colors.HexColor('#2E86AB'),
                           spaceAfter=4)),
        p('The repository includes step-by-step instructions to reproduce '
          'every figure in this paper from scratch using the NEST simulator, '
          'or to regenerate the figures directly from the pre-computed CSV '
          'data without running new simulations.'),
    ]

    # === References ===
    story += [
        hr(),
        Paragraph('References', subsection_style),
        Paragraph('[1] F. A. C. Azevedo et al., "Equal numbers of neuronal and '
                  'nonneuronal cells make the human brain an isometrically '
                  'scaled-up primate brain," J. Comp. Neurol., 2009.',
                  ref_style),
        Paragraph('[2] M.-O. Gewaltig and M. Diesmann, "NEST (NEural Simulation '
                  'Tool)," Scholarpedia, vol. 2, no. 4, p. 1430, 2007.',
                  ref_style),
        Paragraph('[3] D. O. Hebb, The Organization of Behavior. Wiley, 1949.',
                  ref_style),
        Paragraph('[4] H. Markram, J. Lübke, M. Frotscher, B. Sakmann, '
                  '"Regulation of synaptic efficacy by coincidence of '
                  'postsynaptic APs and EPSPs," Science, vol. 275, '
                  'pp. 213–215, 1997.',
                  ref_style),
        Paragraph('[5] G.-Q. Bi and M.-M. Poo, "Synaptic modifications in '
                  'cultured hippocampal neurons," J. Neurosci., 1998.',
                  ref_style),
        Paragraph('[6] K. Cheung, S. Schultz, W. Luk, "A large-scale spiking '
                  'neural network accelerator for FPGA systems," ICANN, 2012.',
                  ref_style),
        Paragraph('[7] S. Han et al., "Hardware acceleration of LSTM neural '
                  'networks and their applications," ACM TECS, 2020.',
                  ref_style),
        Paragraph('[8] A. Sripad, D. Sylvester, D. Blaauw, "FPGA-based spiking '
                  'neural network: design and challenges," DATE, 2020.',
                  ref_style),
    ]

    return story


# ---------------------------------------------------------------------------
# Custom two-column document template
# ---------------------------------------------------------------------------

def header_footer(canvas, doc):
    canvas.saveState()
    page = doc.page
    # Header line
    canvas.setStrokeColor(colors.HexColor('#888888'))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_OUTER, PAGE_H - MARGIN_OUTER + 2*mm,
                PAGE_W - MARGIN_OUTER, PAGE_H - MARGIN_OUTER + 2*mm)
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(MARGIN_OUTER, PAGE_H - MARGIN_OUTER + 4*mm,
                      'Carriere — STDP Benchmark Study Towards FPGA Acceleration')
    canvas.drawRightString(PAGE_W - MARGIN_OUTER, PAGE_H - MARGIN_OUTER + 4*mm,
                           f'Page {page}')
    # Footer
    canvas.line(MARGIN_OUTER, MARGIN_OUTER - 4*mm,
                PAGE_W - MARGIN_OUTER, MARGIN_OUTER - 4*mm)
    canvas.drawCentredString(PAGE_W / 2, MARGIN_OUTER - 8*mm,
                             'Preliminary — March 2026')
    canvas.restoreState()


class TwoColumnDoc(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        template = PageTemplate(
            id='TwoCol',
            frames=[left_frame, right_frame],
            onPage=header_footer,
        )
        self.addPageTemplates([template])


def main():
    doc = TwoColumnDoc(
        str(OUT),
        pagesize=A4,
        leftMargin=MARGIN_OUTER,
        rightMargin=MARGIN_OUTER,
        topMargin=MARGIN_OUTER + 1.0*cm,
        bottomMargin=MARGIN_OUTER + 0.5*cm,
    )
    story = build_story()
    doc.build(story)
    print(f"Saved: {OUT}")


if __name__ == '__main__':
    main()
