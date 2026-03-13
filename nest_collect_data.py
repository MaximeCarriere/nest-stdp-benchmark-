"""
NEST STDP Data Collection Script
================================
Comprehensive timing data collection for NEST STDP analysis.

Features:
- Core scaling (1, 2, 4, 8 cores)
- Learning rules (static, STDP)
- Spike divergence tracking
- Crash-proof checkpointing
- Memory safety checks
- Progress tracking

Estimated runtime: 2-3 hours on M1 Mac
Output: ~50MB pickle file with ~1500 simulation results
"""

import nest
import numpy as np
import time
import pickle
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Output directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Parameter grid (optimized for 2-3 hour runtime)
PARAM_GRID = {
    'n_neurons': [10, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000],
    'n_cores': [1, 2, 4, 8],
    'conn_prob': [0.05, 0.1, 0.2],
    'firing_rate': [10.0, 20.0, 40.0],
    'learning_rule': ['static', 'stdp'],
    'duration': 20000.0,  
    'n_trials': 3,
}

# STDP parameters
STDP_PARAMS = {
    "synapse_model": "stdp_synapse",
    "weight": 0.5,
    "Wmax": 1.0,
    "lambda": 0.01,
    "alpha": 1.2,
    "tau_plus": 20.0,
    "mu_plus": 0.0,
    "mu_minus": 0.0,
    "delay": 0.1
}

# Safety limits for M1 Mac (16GB RAM)
MAX_MEMORY_MB = 4000  # Conservative: use max 4GB for NEST
MAX_SIMULATION_TIME = 1200  # seconds timeout per simulation
CHECKPOINT_INTERVAL = 50  # Save every N configs

# ============================================================================
# Helper Functions
# ============================================================================

def estimate_memory_mb(n_neurons, conn_prob):
    """Estimate memory usage for a network configuration"""
    n_synapses = n_neurons * n_neurons * conn_prob
    # Rough estimate: 200 bytes per synapse
    memory_mb = n_synapses * 200 / 1e6
    return memory_mb

def should_skip_config(config):
    """
    Pre-flight check to avoid crashes
    Returns: (should_skip, reason)
    """
    # Memory check
    mem_estimate = estimate_memory_mb(config['n_neurons'], config['conn_prob'])
    if mem_estimate > MAX_MEMORY_MB:
        return True, f"memory_estimate_{mem_estimate:.0f}MB"
    
    # Known problematic configs
    if config['n_neurons'] > 1000 and config['conn_prob'] > 0.2:
        return True, "dense_large_network"
    
    return False, None

def run_nest_simulation(config, timeout=MAX_SIMULATION_TIME):
    """
    Run a single NEST simulation with timeout protection
    
    Returns: dict with timing, network, and activity data
    """
    try:
        # Start timing
        t_start_total = time.perf_counter()
        
        # Setup NEST
        nest.ResetKernel()
        nest.local_num_threads = config['n_cores']
        nest.resolution = 0.1
        nest.rng_seed = config['random_seed']
        
        # Create network
        t_start_setup = time.perf_counter()
        
        neurons = nest.Create("iaf_psc_alpha", config['n_neurons'], 
                             params={"I_e": 400.0})
        
        # External input
        pg = nest.Create("poisson_generator", params={"rate": config['firing_rate']})
        nest.Connect(pg, neurons, syn_spec={"weight": 50.0})
        
        # Recurrent connections
        conn_spec = {"rule": "pairwise_bernoulli", "p": config['conn_prob']}
        
        if config['learning_rule'] == 'stdp':
            nest.Connect(neurons, neurons, conn_spec=conn_spec, syn_spec=STDP_PARAMS)
        else:  # static
            nest.Connect(neurons, neurons, conn_spec=conn_spec,
                        syn_spec={"synapse_model": "static_synapse", 
                                 "weight": 0.5, "delay": 0.1})
        
        connections = nest.GetConnections(source=neurons, target=neurons)
        n_connections = len(connections)
        
        # Spike recorder
        sr = nest.Create("spike_recorder")
        nest.Connect(neurons, sr)
        
        t_setup = time.perf_counter() - t_start_setup
        
        # Run simulation
        t_start_sim = time.perf_counter()
        nest.Simulate(config['duration'])
        t_simulate = time.perf_counter() - t_start_sim
        
        # Extract results
        events = nest.GetStatus(sr, "events")[0]
        n_spikes = len(events["times"])
        
        # Compute spike rate (Hz per neuron)
        spike_rate = n_spikes / (config['duration'] / 1000.0) / config['n_neurons']
        
        t_total = time.perf_counter() - t_start_total
        
        # Return results
        return {
            'status': 'success',
            'T_simulate': t_simulate,
            'T_setup': t_setup,
            'T_total': t_total,
            'n_connections': n_connections,
            'n_spikes': n_spikes,
            'spike_rate': spike_rate,
            'error': None,
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'T_simulate': None,
            'T_setup': None,
            'T_total': None,
            'n_connections': None,
            'n_spikes': None,
            'spike_rate': None,
            'error': str(e),
        }

def generate_config_list():
    """Generate all configurations to test"""
    configs = []
    
    # For spike divergence tracking, we need to pair static and STDP runs
    # Strategy: For each (neurons, cores, conn_prob, rate), run both rules
    
    for n_neurons in PARAM_GRID['n_neurons']:
        for n_cores in PARAM_GRID['n_cores']:
            for conn_prob in PARAM_GRID['conn_prob']:
                for firing_rate in PARAM_GRID['firing_rate']:
                    for trial in range(1, PARAM_GRID['n_trials'] + 1):
                        # Base config
                        base_config = {
                            'n_neurons': n_neurons,
                            'n_cores': n_cores,
                            'conn_prob': conn_prob,
                            'firing_rate': firing_rate,
                            'duration': PARAM_GRID['duration'],
                            'trial': trial,
                        }
                        
                        # Add both learning rules
                        for learning_rule in PARAM_GRID['learning_rule']:
                            config = base_config.copy()
                            config['learning_rule'] = learning_rule
                            config['random_seed'] = np.random.randint(1, 1000000)
                            configs.append(config)
    
    return configs

def save_checkpoint(results, config_idx, session_metadata):
    """Save intermediate results"""
    checkpoint_file = DATA_DIR / f"checkpoint_{config_idx:04d}.pkl"
    
    data = {
        'metadata': session_metadata,
        'results': results,
        'checkpoint_idx': config_idx,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)
    
    return checkpoint_file

def load_checkpoint(checkpoint_file):
    """Load checkpoint to resume"""
    with open(checkpoint_file, 'rb') as f:
        data = pickle.load(f)
    return data

def find_latest_checkpoint():
    """Find most recent checkpoint file"""
    checkpoints = sorted(DATA_DIR.glob("checkpoint_*.pkl"))
    if checkpoints:
        return checkpoints[-1]
    return None

def compute_spike_divergence(results):
    """
    Compute spike divergence for STDP runs using paired static baselines
    Modifies results in-place
    """
    # Group results by (n_neurons, n_cores, conn_prob, firing_rate, trial)
    static_baseline = {}
    
    # First pass: collect static baselines
    for result in results:
        if result['learning_rule'] == 'static' and result['status'] == 'success':
            key = (
                result['n_neurons'],
                result['n_cores'],
                result['conn_prob'],
                result['firing_rate'],
                result['trial']
            )
            static_baseline[key] = result['n_spikes']
    
    # Second pass: add divergence to STDP results
    for result in results:
        if result['learning_rule'] == 'stdp' and result['status'] == 'success':
            key = (
                result['n_neurons'],
                result['n_cores'],
                result['conn_prob'],
                result['firing_rate'],
                result['trial']
            )
            
            if key in static_baseline:
                n_spikes_baseline = static_baseline[key]
                n_spikes_stdp = result['n_spikes']
                
                if n_spikes_baseline > 0:
                    divergence_pct = ((n_spikes_stdp - n_spikes_baseline) / 
                                     n_spikes_baseline * 100)
                else:
                    divergence_pct = None
                
                result['n_spikes_static_baseline'] = n_spikes_baseline
                result['spike_divergence_pct'] = divergence_pct
            else:
                result['n_spikes_static_baseline'] = None
                result['spike_divergence_pct'] = None
        else:
            result['n_spikes_static_baseline'] = None
            result['spike_divergence_pct'] = None

# ============================================================================
# Main Data Collection
# ============================================================================

def main():
    """Main data collection loop"""
    
    print("="*80)
    print("NEST STDP Data Collection")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Machine: {platform.node()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NEST: {nest.__version__}")
    
    # Session metadata
    session_metadata = {
        'created': datetime.now().isoformat(),
        'machine': platform.node(),
        'python_version': sys.version,
        'nest_version': nest.__version__,
        'param_grid': PARAM_GRID,
        'max_memory_mb': MAX_MEMORY_MB,
        'checkpoint_interval': CHECKPOINT_INTERVAL,
    }
    
    # Generate configuration list
    all_configs = generate_config_list()
    total_configs = len(all_configs)
    
    print(f"\nTotal configurations: {total_configs}")
    print(f"Estimated time: {total_configs * 6 / 3600:.1f} hours (@ 6 sec/sim avg)")
    
    # Check for existing checkpoint
    checkpoint_file = find_latest_checkpoint()
    if checkpoint_file:
        print(f"\nFound checkpoint: {checkpoint_file.name}")
        resume = input("Resume from checkpoint? (y/n): ").lower().strip()
        
        if resume == 'y':
            checkpoint_data = load_checkpoint(checkpoint_file)
            results = checkpoint_data['results']
            start_idx = checkpoint_data['checkpoint_idx']
            session_metadata = checkpoint_data['metadata']
            print(f"Resuming from config {start_idx}/{total_configs}")
        else:
            results = []
            start_idx = 0
    else:
        results = []
        start_idx = 0
    
    # Statistics
    stats = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
    }
    
    # Main loop
    print(f"\n{'='*80}")
    print("Running simulations...")
    print("="*80)
    
    with tqdm(total=total_configs, initial=start_idx, 
              desc="Progress", unit="sim") as pbar:
        
        for idx in range(start_idx, total_configs):
            config = all_configs[idx]
            
            # Update progress bar description
            desc = (f"n={config['n_neurons']}, cores={config['n_cores']}, "
                   f"p={config['conn_prob']}, {config['learning_rule']}")
            pbar.set_description(desc[:50])
            
            # Pre-flight check
            should_skip, skip_reason = should_skip_config(config)
            
            if should_skip:
                result = config.copy()
                result.update({
                    'status': 'skipped',
                    'skip_reason': skip_reason,
                    'timestamp': datetime.now().isoformat(),
                })
                results.append(result)
                stats['skipped'] += 1
                pbar.update(1)
                continue
            
            # Run simulation
            sim_result = run_nest_simulation(config)
            
            # Combine config and results
            result = config.copy()
            result.update(sim_result)
            result['timestamp'] = datetime.now().isoformat()
            
            results.append(result)
            
            # Update statistics
            if sim_result['status'] == 'success':
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            # Update progress bar postfix
            pbar.set_postfix({
                'success': stats['success'],
                'failed': stats['failed'],
                'skipped': stats['skipped'],
            })
            
            pbar.update(1)
            
            # Checkpoint
            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(results, idx + 1, session_metadata)
                pbar.write(f"✓ Checkpoint saved at {idx + 1}/{total_configs}")
    
    # Compute spike divergence
    print("\nComputing spike divergence...")
    compute_spike_divergence(results)
    
    # Final save
    print("\nSaving final data...")
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_file = DATA_DIR / f"nest_raw_data_{timestamp_str}.pkl"
    
    session_metadata['completed'] = datetime.now().isoformat()
    session_metadata['total_configs'] = total_configs
    session_metadata['statistics'] = stats
    
    final_data = {
        'metadata': session_metadata,
        'results': results,
    }
    
    with open(final_file, 'wb') as f:
        pickle.dump(final_data, f)
    
    print(f"✓ Saved: {final_file}")
    
    # Save human-readable metadata
    meta_file = DATA_DIR / f"nest_raw_data_{timestamp_str}_meta.json"
    with open(meta_file, 'w') as f:
        # Convert metadata to JSON-serializable format
        meta_json = {
            'created': session_metadata['created'],
            'completed': session_metadata['completed'],
            'machine': session_metadata['machine'],
            'nest_version': session_metadata['nest_version'],
            'total_configs': session_metadata['total_configs'],
            'statistics': session_metadata['statistics'],
            'param_grid': {k: v if isinstance(v, (int, float, str)) 
                          else list(v) if isinstance(v, (list, tuple))
                          else str(v) for k, v in PARAM_GRID.items()},
        }
        json.dump(meta_json, f, indent=2)
    
    print(f"✓ Saved metadata: {meta_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("DATA COLLECTION SUMMARY")
    print("="*80)
    print(f"Total configurations: {total_configs}")
    print(f"Successful: {stats['success']} ({stats['success']/total_configs*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/total_configs*100:.1f}%)")
    print(f"Skipped: {stats['skipped']} ({stats['skipped']/total_configs*100:.1f}%)")
    
    # File sizes
    file_size_mb = final_file.stat().st_size / 1e6
    print(f"\nData file size: {file_size_mb:.1f} MB")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
