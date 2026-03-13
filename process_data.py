"""
NEST Data Processing Script
============================
Processes raw NEST timing data to compute aggregate statistics and derived metrics.

Input:  data/nest_raw_data_*.pkl (raw simulation results)
Output: processed_data/ (aggregated statistics, derived metrics)
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# Configuration
# ============================================================================

RAW_DATA_DIR = Path("data")
PROCESSED_DATA_DIR = Path("processed_data")
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def load_raw_data(filepath):
    """Load raw pickle data"""
    print(f"Loading: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def aggregate_trials(df):
    """
    Aggregate multiple trials for each configuration.
    Returns DataFrame with mean ± SEM for each metric.
    """
    # Group by all config parameters (excluding trial number)
    group_cols = ['n_neurons', 'n_cores', 'conn_prob', 'firing_rate', 'learning_rule']
    
    # Metrics to aggregate
    agg_dict = {
        'T_simulate': ['mean', 'sem', 'count'],
        'T_setup': ['mean', 'sem'],
        'T_total': ['mean', 'sem'],
        'n_connections': ['mean', 'sem'],
        'n_spikes': ['mean', 'sem'],
        'spike_rate': ['mean', 'sem'],
    }
    
    # Add spike divergence if present
    if 'spike_divergence_pct' in df.columns:
        agg_dict['spike_divergence_pct'] = ['mean', 'sem']
    
    # Aggregate
    grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                       for col in grouped.columns.values]
    
    return grouped

def compute_overhead(df_agg):
    """
    Compute STDP overhead by matching static and STDP runs.
    For each (n_neurons, n_cores, conn_prob, firing_rate):
        overhead = T_stdp - T_static
    """
    # Separate static and STDP
    static_df = df_agg[df_agg['learning_rule'] == 'static'].copy()
    stdp_df = df_agg[df_agg['learning_rule'] == 'stdp'].copy()
    
    # Merge on matching parameters
    merge_cols = ['n_neurons', 'n_cores', 'conn_prob', 'firing_rate']
    
    merged = pd.merge(
        stdp_df, static_df,
        on=merge_cols,
        suffixes=('_stdp', '_static')
    )
    
    # Compute overhead metrics
    overhead_df = merged[merge_cols].copy()
    
    overhead_df['T_overhead'] = merged['T_simulate_mean_stdp'] - merged['T_simulate_mean_static']
    overhead_df['T_overhead_sem'] = np.sqrt(
        merged['T_simulate_sem_stdp']**2 + merged['T_simulate_sem_static']**2
    )
    
    overhead_df['overhead_pct'] = (overhead_df['T_overhead'] / merged['T_simulate_mean_static']) * 100
    
    # Error propagation for percentage
    rel_err_overhead = merged['T_simulate_sem_stdp'] / merged['T_simulate_mean_stdp'].abs()
    rel_err_static = merged['T_simulate_sem_static'] / merged['T_simulate_mean_static']
    overhead_df['overhead_pct_sem'] = overhead_df['overhead_pct'].abs() * np.sqrt(
        rel_err_overhead**2 + rel_err_static**2
    )
    
    # Include timing values for context
    overhead_df['T_static'] = merged['T_simulate_mean_static']
    overhead_df['T_static_sem'] = merged['T_simulate_sem_static']
    overhead_df['T_stdp'] = merged['T_simulate_mean_stdp']
    overhead_df['T_stdp_sem'] = merged['T_simulate_sem_stdp']
    
    # Activity metrics
    overhead_df['n_spikes_static'] = merged['n_spikes_mean_static']
    overhead_df['n_spikes_stdp'] = merged['n_spikes_mean_stdp']
    overhead_df['n_connections'] = merged['n_connections_mean_stdp']
    
    # Spike divergence
    if 'spike_divergence_pct_mean_stdp' in merged.columns:
        overhead_df['spike_divergence'] = merged['spike_divergence_pct_mean_stdp']
        overhead_df['spike_divergence_sem'] = merged['spike_divergence_pct_sem_stdp']
    
    return overhead_df

def compute_core_scaling(df_agg):
    """
    Compute speedup and efficiency for multi-core scaling.
    For each (n_neurons, conn_prob, firing_rate, learning_rule):
        speedup = T_1core / T_Ncores
        efficiency = speedup / n_cores
    """
    results = []
    
    # Group by parameters that vary with cores
    group_cols = ['n_neurons', 'conn_prob', 'firing_rate', 'learning_rule']
    
    for name, group in df_agg.groupby(group_cols):
        # Get 1-core baseline
        baseline = group[group['n_cores'] == 1]
        
        if len(baseline) == 0:
            continue
        
        T_1core = baseline['T_simulate_mean'].iloc[0]
        T_1core_sem = baseline['T_simulate_sem'].iloc[0]
        
        # Compute speedup for all core counts
        for _, row in group.iterrows():
            n_cores = row['n_cores']
            T_Ncores = row['T_simulate_mean']
            T_Ncores_sem = row['T_simulate_sem']
            
            speedup = T_1core / T_Ncores if T_Ncores > 0 else 0
            efficiency = speedup / n_cores if n_cores > 0 else 0
            
            # Error propagation for speedup
            if T_1core > 0 and T_Ncores > 0:
                speedup_sem = speedup * np.sqrt(
                    (T_1core_sem / T_1core)**2 + (T_Ncores_sem / T_Ncores)**2
                )
            else:
                speedup_sem = 0
            
            efficiency_sem = speedup_sem / n_cores if n_cores > 0 else 0
            
            result = {
                'n_neurons': row['n_neurons'],
                'n_cores': n_cores,
                'conn_prob': row['conn_prob'],
                'firing_rate': row['firing_rate'],
                'learning_rule': row['learning_rule'],
                'T_simulate': T_Ncores,
                'T_simulate_sem': T_Ncores_sem,
                'T_1core': T_1core,
                'speedup': speedup,
                'speedup_sem': speedup_sem,
                'efficiency': efficiency,
                'efficiency_sem': efficiency_sem,
            }
            results.append(result)
    
    return pd.DataFrame(results)

def compute_update_metrics(df_agg):
    """
    Compute per-update timing metrics.
    - Total updates = n_spikes * (n_connections / n_neurons) * 2
    - Updates per second
    - Microseconds per update
    """
    df = df_agg.copy()
    
    # Total synaptic updates (pre->post and post->pre)
    df['total_updates'] = df['n_spikes_mean'] * (df['n_connections_mean'] / df['n_neurons']) * 2
    
    # Updates per second
    df['updates_per_sec'] = df['total_updates'] / df['T_simulate_mean']
    
    # Microseconds per update (only for STDP, using overhead)
    # We'll need to merge with overhead data for this
    
    return df

def generate_summary_statistics(df_raw, df_agg, overhead_df, core_scaling_df):
    """Generate summary statistics for the processed data"""
    
    summary = {
        'data_collection': {
            'total_simulations': len(df_raw),
            'unique_configs': len(df_agg),
            'success_rate': (df_raw['status'] == 'success').sum() / len(df_raw) * 100,
            'trials_per_config': df_raw.groupby(['n_neurons', 'n_cores', 'conn_prob', 
                                                  'firing_rate', 'learning_rule']).size().mean(),
        },
        
        'parameter_ranges': {
            'n_neurons': sorted(df_agg['n_neurons'].unique().tolist()),
            'n_cores': sorted(df_agg['n_cores'].unique().tolist()),
            'conn_prob': sorted(df_agg['conn_prob'].unique().tolist()),
            'firing_rate': sorted(df_agg['firing_rate'].unique().tolist()),
            'learning_rules': sorted(df_agg['learning_rule'].unique().tolist()),
        },
        
        'timing_ranges': {
            'T_simulate_min': float(df_agg['T_simulate_mean'].min()),
            'T_simulate_max': float(df_agg['T_simulate_mean'].max()),
            'T_simulate_mean': float(df_agg['T_simulate_mean'].mean()),
        },
        
        'overhead_summary': {
            'mean_overhead_pct': float(overhead_df['overhead_pct'].mean()),
            'max_overhead_pct': float(overhead_df['overhead_pct'].max()),
            'min_overhead_pct': float(overhead_df['overhead_pct'].min()),
        },
        
        'core_scaling_summary': {
            'max_speedup': float(core_scaling_df['speedup'].max()),
            'max_speedup_config': core_scaling_df.loc[core_scaling_df['speedup'].idxmax()].to_dict(),
            'max_efficiency': float(core_scaling_df['efficiency'].max()),
        },
        
        'spike_divergence': {
            'mean': float(df_agg['spike_divergence_pct_mean'].mean()) if 'spike_divergence_pct_mean' in df_agg.columns else None,
            'std': float(df_agg['spike_divergence_pct_mean'].std()) if 'spike_divergence_pct_mean' in df_agg.columns else None,
        }
    }
    
    return summary

# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("="*80)
    print("NEST DATA PROCESSING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find most recent raw data file
    raw_files = sorted(RAW_DATA_DIR.glob("nest_raw_data_*.pkl"))
    if not raw_files:
        print("Error: No raw data files found in data/")
        return
    
    raw_file = raw_files[-1]  # Most recent
    print(f"\nProcessing: {raw_file.name}")
    
    # Load raw data
    raw_data = load_raw_data(raw_file)
    df_raw = pd.DataFrame(raw_data['results'])
    
    print(f"  Total simulations: {len(df_raw)}")
    print(f"  Successful: {(df_raw['status'] == 'success').sum()}")
    
    # Filter to successful runs only
    df_success = df_raw[df_raw['status'] == 'success'].copy()
    
    print(f"\n{'='*80}")
    print("Step 1: Aggregating trials...")
    print("="*80)
    
    df_agg = aggregate_trials(df_success)
    print(f"  Aggregated to {len(df_agg)} unique configurations")
    
    # Save aggregated data
    agg_file = PROCESSED_DATA_DIR / "aggregated_trials.pkl"
    df_agg.to_pickle(agg_file)
    print(f"  ✓ Saved: {agg_file}")
    
    # Also save as CSV for easy inspection
    csv_file = PROCESSED_DATA_DIR / "aggregated_trials.csv"
    df_agg.to_csv(csv_file, index=False)
    print(f"  ✓ Saved: {csv_file}")
    
    print(f"\n{'='*80}")
    print("Step 2: Computing STDP overhead...")
    print("="*80)
    
    overhead_df = compute_overhead(df_agg)
    print(f"  Computed overhead for {len(overhead_df)} configurations")
    print(f"  Mean overhead: {overhead_df['overhead_pct'].mean():.2f}%")
    print(f"  Max overhead: {overhead_df['overhead_pct'].max():.2f}%")
    
    # Save overhead data
    overhead_file = PROCESSED_DATA_DIR / "stdp_overhead.pkl"
    overhead_df.to_pickle(overhead_file)
    print(f"  ✓ Saved: {overhead_file}")
    
    csv_file = PROCESSED_DATA_DIR / "stdp_overhead.csv"
    overhead_df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved: {csv_file}")
    
    print(f"\n{'='*80}")
    print("Step 3: Computing core scaling metrics...")
    print("="*80)
    
    core_scaling_df = compute_core_scaling(df_agg)
    print(f"  Computed scaling for {len(core_scaling_df)} configurations")
    print(f"  Max speedup: {core_scaling_df['speedup'].max():.2f}x")
    max_speedup_row = core_scaling_df.loc[core_scaling_df['speedup'].idxmax()]
    print(f"    at {int(max_speedup_row['n_neurons'])} neurons, "
          f"{int(max_speedup_row['n_cores'])} cores, "
          f"{max_speedup_row['learning_rule']}")
    
    # Save core scaling data
    scaling_file = PROCESSED_DATA_DIR / "core_scaling.pkl"
    core_scaling_df.to_pickle(scaling_file)
    print(f"  ✓ Saved: {scaling_file}")
    
    csv_file = PROCESSED_DATA_DIR / "core_scaling.csv"
    core_scaling_df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved: {csv_file}")
    
    print(f"\n{'='*80}")
    print("Step 4: Computing update metrics...")
    print("="*80)
    
    update_metrics_df = compute_update_metrics(df_agg)
    
    # Save update metrics
    metrics_file = PROCESSED_DATA_DIR / "update_metrics.pkl"
    update_metrics_df.to_pickle(metrics_file)
    print(f"  ✓ Saved: {metrics_file}")
    
    csv_file = PROCESSED_DATA_DIR / "update_metrics.csv"
    update_metrics_df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved: {csv_file}")
    
    print(f"\n{'='*80}")
    print("Step 5: Generating summary statistics...")
    print("="*80)
    
    summary = generate_summary_statistics(df_raw, df_agg, overhead_df, core_scaling_df)
    
    # Save summary as JSON
    summary_file = PROCESSED_DATA_DIR / "summary_statistics.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: {summary_file}")
    
    # Print key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print("="*80)
    
    print(f"\nData Quality:")
    print(f"  Success rate: {summary['data_collection']['success_rate']:.1f}%")
    print(f"  Trials per config: {summary['data_collection']['trials_per_config']:.1f}")
    
    print(f"\nTiming Range:")
    print(f"  {summary['timing_ranges']['T_simulate_min']:.4f}s - "
          f"{summary['timing_ranges']['T_simulate_max']:.4f}s")
    
    print(f"\nSTDP Overhead:")
    print(f"  Mean: {summary['overhead_summary']['mean_overhead_pct']:.1f}%")
    print(f"  Range: {summary['overhead_summary']['min_overhead_pct']:.1f}% - "
          f"{summary['overhead_summary']['max_overhead_pct']:.1f}%")
    
    print(f"\nCore Scaling:")
    print(f"  Max speedup: {summary['core_scaling_summary']['max_speedup']:.2f}x")
    max_cfg = summary['core_scaling_summary']['max_speedup_config']
    print(f"    at {int(max_cfg['n_neurons'])} neurons, "
          f"{int(max_cfg['n_cores'])} cores")
    print(f"  Max efficiency: {summary['core_scaling_summary']['max_efficiency']:.2f}")
    
    if summary['spike_divergence']['mean'] is not None:
        print(f"\nSpike Divergence:")
        print(f"  Mean: {summary['spike_divergence']['mean']:.2f}%")
        print(f"  Std: {summary['spike_divergence']['std']:.2f}%")
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}/")
    print("\nFiles created:")
    print("  - aggregated_trials.pkl/.csv      (mean ± SEM for all configs)")
    print("  - stdp_overhead.pkl/.csv          (STDP vs static comparison)")
    print("  - core_scaling.pkl/.csv           (speedup & efficiency)")
    print("  - update_metrics.pkl/.csv         (per-update timing)")
    print("  - summary_statistics.json         (key findings)")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
