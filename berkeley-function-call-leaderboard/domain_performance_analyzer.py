#!/usr/bin/env python3
"""
Domain-Level Performance Analyzer for BFCL Evaluation

This script analyzes the performance of function calling models across different domains
by pairing data files with their corresponding score files.
"""

import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import argparse


def load_json_file(file_path: str) -> List[Dict]:
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_data_domains(data_item: Dict) -> Set[str]:
    """Extract domain information from a data item's involved_classes."""
    return set(data_item.get('involved_classes', []))


def analyze_domain_performance(data_dir: str, score_dir: str, model_name: str = None) -> Dict:
    """
    Analyze domain-level performance by matching data files with score files.
    
    Args:
        data_dir: Path to directory containing data JSON files
        score_dir: Path to directory containing score files 
        model_name: Specific model name to analyze (optional)
        
    Returns:
        Dictionary containing domain performance analysis
    """
    results = {
        'domain_stats_by_model_dataset': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            'total_instances': 0,
            'failed_instances': 0,
            'success_rate': 0.0,
            'error_types': defaultdict(int)
        }))),
        'dataset_stats': {},
        'overall_stats': {
            'total_instances': 0,
            'total_failed': 0,
            'overall_success_rate': 0.0
        }
    }
    
    # Find all data files that have involved_classes
    data_files = []
    for pattern in ['BFCL_v3_multi_turn_*.json']:
        data_files.extend(glob.glob(os.path.join(data_dir, pattern)))
    
    # First pass: count total instances per domain from data files
    domain_totals = defaultdict(int)
    dataset_totals = {}
    
    for data_file in data_files:
        dataset_name = os.path.basename(data_file).replace('.json', '')
        
        try:
            data_items = load_json_file(data_file)
            dataset_totals[dataset_name] = len(data_items)
            
            # Count domain instances
            for item in data_items:
                domains = get_data_domains(item)
                for domain in domains:
                    domain_totals[domain] += 1
                    
        except Exception as e:
            print(f"Error loading {data_file}: {e}")
            continue
    
    # Initialize stratified domain stats with total counts
    # Get all unique models
    all_models = set()
    for data_file in data_files:
        dataset_name = os.path.basename(data_file).replace('.json', '')
        score_pattern = f"{dataset_name}_score.json"
        if model_name:
            score_files = glob.glob(os.path.join(score_dir, f"*{model_name}*", score_pattern))
        else:
            score_files = glob.glob(os.path.join(score_dir, "*", score_pattern))
        for score_file in score_files:
            model_dir = os.path.basename(os.path.dirname(score_file))
            all_models.add(model_dir)
    
    # Initialize counts for each stratification
    for data_file in data_files:
        dataset_name = os.path.basename(data_file).replace('.json', '')
        try:
            data_items = load_json_file(data_file)
            # Count domain instances per dataset
            dataset_domain_counts = defaultdict(int)
            for item in data_items:
                domains = get_data_domains(item)
                for domain in domains:
                    dataset_domain_counts[domain] += 1
            
            # Initialize stratified stats
            for model_dir in all_models:
                for domain, count in dataset_domain_counts.items():
                    # By model and dataset only
                    results['domain_stats_by_model_dataset'][model_dir][dataset_name][domain]['total_instances'] = count
                    
        except Exception as e:
            print(f"Error loading {data_file}: {e}")
            continue
    
    # Second pass: count failures from score files
    for data_file in data_files:
        dataset_name = os.path.basename(data_file).replace('.json', '')
        print(f"Processing dataset: {dataset_name}")
        
        # Load data for ID mapping
        try:
            data_items = load_json_file(data_file)
        except Exception as e:
            print(f"Error loading {data_file}: {e}")
            continue
            
        # Find corresponding score files
        score_pattern = f"{dataset_name}_score.json"
        score_files = []
        
        if model_name:
            # Look for specific model
            score_files = glob.glob(os.path.join(score_dir, f"*{model_name}*", score_pattern))
        else:
            # Find all model score files for this dataset
            score_files = glob.glob(os.path.join(score_dir, "*", score_pattern))
        
        if not score_files:
            print(f"No score files found for {dataset_name}")
            continue
            
        # Process each score file
        for score_file in score_files:
            model_dir = os.path.basename(os.path.dirname(score_file))
            print(f"  Processing model: {model_dir}")
            
            try:
                score_data = load_json_file(score_file)
            except Exception as e:
                print(f"Error loading {score_file}: {e}")
                continue
                
            # Create mapping of data items by ID
            data_by_id = {item['id']: item for item in data_items}
            
            # Initialize dataset stats
            dataset_key = f"{dataset_name}_{model_dir}"
            if dataset_key not in results['dataset_stats']:
                results['dataset_stats'][dataset_key] = {
                    'total_instances': dataset_totals.get(dataset_name, 0),
                    'failed_instances': 0,
                    'success_rate': 0.0
                }
            
            # Get summary from first line
            summary = score_data[0] if score_data else {}
            dataset_failed = summary.get('total_count', 0) - summary.get('correct_count', 0)
            results['dataset_stats'][dataset_key]['failed_instances'] = dataset_failed
            
            # Process failed entries (skip first summary line - all remaining are failures)
            failed_entries = score_data[1:] if len(score_data) > 1 else []
            
            for failed_entry in failed_entries:
                item_id = failed_entry.get('id')
                if not item_id or item_id not in data_by_id:
                    continue
                    
                data_item = data_by_id[item_id]
                domains = get_data_domains(data_item)
                
                # Update domain failure statistics (model and dataset only)
                for domain in domains:
                    # By model and dataset only
                    results['domain_stats_by_model_dataset'][model_dir][dataset_name][domain]['failed_instances'] += 1
                    
                    # Track error types
                    error_info = failed_entry.get('error', {})
                    error_type = error_info.get('error_type', 'unknown')
                    results['domain_stats_by_model_dataset'][model_dir][dataset_name][domain]['error_types'][error_type] += 1
    
    # Calculate success rates (by model and dataset only)
    for model_dir, model_data in results['domain_stats_by_model_dataset'].items():
        for dataset_name, dataset_domains in model_data.items():
            for domain, stats in dataset_domains.items():
                if stats['total_instances'] > 0:
                    success_count = stats['total_instances'] - stats['failed_instances']
                    stats['success_rate'] = success_count / stats['total_instances']
    
    for dataset_key, stats in results['dataset_stats'].items():
        if stats['total_instances'] > 0:
            success_count = stats['total_instances'] - stats['failed_instances']
            stats['success_rate'] = success_count / stats['total_instances']
    
    # Update overall stats (use dataset stats to avoid double counting)
    results['overall_stats']['total_instances'] = sum(stats['total_instances'] for stats in results['dataset_stats'].values())
    results['overall_stats']['total_failed'] = sum(stats['failed_instances'] for stats in results['dataset_stats'].values())
    
    if results['overall_stats']['total_instances'] > 0:
        total_success = results['overall_stats']['total_instances'] - results['overall_stats']['total_failed']
        results['overall_stats']['overall_success_rate'] = total_success / results['overall_stats']['total_instances']
    
    return results


def print_analysis_results(results: Dict):
    """Print formatted analysis results."""
    print("\n" + "="*100)
    print("DOMAIN-LEVEL PERFORMANCE ANALYSIS (BY MODEL AND DATASET)")
    print("="*100)
    
    # Overall statistics
    overall = results['overall_stats']
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Instances: {overall['total_instances']}")
    print(f"  Total Failed: {overall['total_failed']}")
    print(f"  Overall Success Rate: {overall['overall_success_rate']:.2%}")
    
    # Domain performance by model and dataset
    print(f"\n" + "="*100)
    print("DOMAIN PERFORMANCE BY MODEL AND DATASET")
    print("="*100)
    
    for model_dir, model_data in results['domain_stats_by_model_dataset'].items():
        print(f"\n=== MODEL: {model_dir.upper()} ===")
        
        for dataset_name, dataset_domains in model_data.items():
            print(f"\n--- Dataset: {dataset_name} ---")
            print(f"{'Domain':<20} {'Total':<8} {'Failed':<8} {'Success Rate':<12} {'Top Error Types'}")
            print("-" * 80)
            
            # Sort domains by success rate
            sorted_domains = sorted(dataset_domains.items(), 
                                  key=lambda x: x[1]['success_rate'], reverse=True)
            
            for domain, stats in sorted_domains:
                if stats['total_instances'] > 0:  # Only show domains with instances
                    # Get top 3 error types
                    top_errors = sorted(stats['error_types'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]
                    error_str = ", ".join([f"{err.split(':')[-1]}({count})" for err, count in top_errors])
                    
                    print(f"{domain:<20} {stats['total_instances']:<8} {stats['failed_instances']:<8} "
                          f"{stats['success_rate']:<12.2%} {error_str}")
    
    # Dataset-level statistics
    print(f"\n" + "="*100)
    print("DATASET-LEVEL PERFORMANCE (INDIVIDUAL RUNS)")
    print("="*100)
    print(f"{'Dataset_Model':<60} {'Total':<8} {'Failed':<8} {'Success Rate'}")
    print("-" * 100)
    
    sorted_datasets = sorted(results['dataset_stats'].items(),
                           key=lambda x: x[1]['success_rate'])
    
    for dataset_key, stats in sorted_datasets:
        print(f"{dataset_key:<60} {stats['total_instances']:<8} {stats['failed_instances']:<8} "
              f"{stats['success_rate']:.2%}")
    
    # Summary table: Domain performance across all conditions
    print(f"\n" + "="*100)
    print("DOMAIN PERFORMANCE SUMMARY ACROSS ALL CONDITIONS")
    print("="*100)
    
    # Aggregate domain stats from model_dataset results
    domain_summary = defaultdict(lambda: {'total': 0, 'failed': 0})
    for model_data in results['domain_stats_by_model_dataset'].values():
        for dataset_domains in model_data.values():
            for domain, stats in dataset_domains.items():
                if stats['total_instances'] > 0:
                    domain_summary[domain]['total'] += stats['total_instances']
                    domain_summary[domain]['failed'] += stats['failed_instances']
    
    print(f"{'Domain':<20} {'Total':<8} {'Failed':<8} {'Success Rate':<12}")
    print("-" * 60)
    
    sorted_summary = sorted(domain_summary.items(), 
                          key=lambda x: (x[1]['total'] - x[1]['failed']) / x[1]['total'] if x[1]['total'] > 0 else 0, 
                          reverse=True)
    
    for domain, stats in sorted_summary:
        success_rate = (stats['total'] - stats['failed']) / stats['total'] if stats['total'] > 0 else 0
        print(f"{domain:<20} {stats['total']:<8} {stats['failed']:<8} {success_rate:<12.2%}")


def main():
    """Main function to run the domain performance analysis."""
    parser = argparse.ArgumentParser(description='Analyze domain-level performance for BFCL evaluation')
    parser.add_argument('--data-dir', required=True, 
                       help='Path to directory containing data JSON files')
    parser.add_argument('--score-dir', required=True,
                       help='Path to directory containing score files')
    parser.add_argument('--model', 
                       help='Specific model name to analyze (optional)')
    parser.add_argument('--output', 
                       help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found")
        return
        
    if not os.path.isdir(args.score_dir):
        print(f"Error: Score directory '{args.score_dir}' not found")
        return
    
    # Run analysis
    print("Starting domain-level performance analysis...")
    results = analyze_domain_performance(args.data_dir, args.score_dir, args.model)
    
    # Print results
    print_analysis_results(results)
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()