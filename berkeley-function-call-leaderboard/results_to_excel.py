#!/usr/bin/env python3
"""
Convert Domain Performance Analysis Results to Excel

This script takes the JSON output from domain_performance_analyzer.py and converts
the "domain_stats_by_model_dataset" results into an Excel worksheet.
"""

import json
import argparse
import pandas as pd
from typing import Dict, List


def load_analysis_results(json_file: str) -> Dict:
    """Load the JSON results from domain performance analyzer."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_model_dataset_results(results: Dict) -> List[Dict]:
    """Extract domain_stats_by_model_dataset into a flat list of records."""
    records = []
    
    # Access the domain_stats_by_model_dataset structure
    model_dataset_stats = results.get('domain_stats_by_model_dataset', {})
    
    for model_name, model_data in model_dataset_stats.items():
        for dataset_name, dataset_domains in model_data.items():
            for domain_name, domain_stats in dataset_domains.items():
                # Only include domains that have instances (avoid empty entries)
                if domain_stats.get('total_instances', 0) > 0:
                    record = {
                        'Domain': domain_name,
                        'Model': model_name,
                        'Dataset': dataset_name,
                        'Success_Rate': domain_stats.get('success_rate', 0.0)
                    }
                    records.append(record)
    
    return records


def create_excel_from_results(json_file: str, output_excel: str):
    """
    Convert domain performance analysis results to Excel format.
    
    Args:
        json_file: Path to JSON file from domain_performance_analyzer.py
        output_excel: Path for output Excel file
    """
    print(f"Loading results from: {json_file}")
    
    # Load the analysis results
    try:
        results = load_analysis_results(json_file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Extract the domain_stats_by_model_dataset data
    records = extract_model_dataset_results(results)
    
    if not records:
        print("No domain_stats_by_model_dataset data found in the results file.")
        return
    
    print(f"Extracted {len(records)} domain-model-dataset combinations")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by Model, Dataset, then Domain for better organization
    df = df.sort_values(['Model', 'Dataset', 'Domain']).reset_index(drop=True)
    
    # Convert success rate to percentage format for display
    df['Success_Rate_Percent'] = df['Success_Rate'] * 100
    
    # Create final DataFrame with desired columns
    final_df = df[['Domain', 'Model', 'Dataset', 'Success_Rate_Percent']].copy()
    final_df = final_df.rename(columns={'Success_Rate_Percent': 'Success_Rate_%'})
    
    # Save to Excel
    try:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            final_df.to_excel(writer, sheet_name='Domain_Performance', index=False)
            
            # Get the workbook and worksheet to apply formatting
            workbook = writer.book
            worksheet = writer.sheets['Domain_Performance']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Format the success rate column to 2 decimal places
            for row in range(2, len(final_df) + 2):  # Starting from row 2 (after header)
                cell = worksheet[f'D{row}']  # Success_Rate_% column
                cell.number_format = '0.00'
                
        print(f"Excel file created successfully: {output_excel}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"  Total records: {len(final_df)}")
        print(f"  Unique domains: {final_df['Domain'].nunique()}")
        print(f"  Unique models: {final_df['Model'].nunique()}")
        print(f"  Unique datasets: {final_df['Dataset'].nunique()}")
        
        # Show sample of the data
        print(f"\nFirst 10 records:")
        print(final_df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Error creating Excel file: {e}")
        return


def create_pivot_analysis(json_file: str, output_excel: str):
    """
    Create additional pivot table analysis in separate sheets.
    
    Args:
        json_file: Path to JSON file from domain_performance_analyzer.py
        output_excel: Path for output Excel file (will be updated)
    """
    results = load_analysis_results(json_file)
    records = extract_model_dataset_results(results)
    
    if not records:
        return
    
    df = pd.DataFrame(records)
    df['Success_Rate_Percent'] = df['Success_Rate'] * 100
    
    try:
        with pd.ExcelWriter(output_excel, mode='a', engine='openpyxl') as writer:
            # Domain by Model pivot
            domain_model_pivot = df.pivot_table(
                index='Domain',
                columns='Model', 
                values='Success_Rate_Percent',
                aggfunc='mean'
            ).round(2)
            domain_model_pivot.to_excel(writer, sheet_name='Domain_by_Model')
            
            # # Domain by Dataset pivot
            # domain_dataset_pivot = df.pivot_table(
            #     index='Domain',
            #     columns='Dataset',
            #     values='Success_Rate_Percent', 
            #     aggfunc='mean'
            # ).round(2)
            # domain_dataset_pivot.to_excel(writer, sheet_name='Domain_by_Dataset')
            
            # # Model by Dataset pivot (average across domains)
            # model_dataset_pivot = df.pivot_table(
            #     index='Model',
            #     columns='Dataset',
            #     values='Success_Rate_Percent',
            #     aggfunc='mean'
            # ).round(2)
            # model_dataset_pivot.to_excel(writer, sheet_name='Model_by_Dataset')
            
        print("Added pivot table analysis sheets to Excel file")
        
    except Exception as e:
        print(f"Error creating pivot tables: {e}")


def main():
    """Main function to convert domain analysis results to Excel."""
    parser = argparse.ArgumentParser(description='Convert domain performance analysis results to Excel')
    parser.add_argument('--input', '-i', required=True,
                       help='Input JSON file from domain_performance_analyzer.py')
    parser.add_argument('--output', '-o', required=True,
                       help='Output Excel file path (.xlsx)')
    parser.add_argument('--pivot', action='store_true',
                       help='Include additional pivot table analysis sheets')
    
    args = parser.parse_args()
    
    # Validate input file
    try:
        with open(args.input, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        return
    except Exception as e:
        print(f"Error accessing input file: {e}")
        return
    
    # Ensure output file has .xlsx extension
    output_file = args.output
    if not output_file.endswith('.xlsx'):
        output_file += '.xlsx'
    
    # Create the main Excel file
    create_excel_from_results(args.input, output_file)
    
    # Add pivot tables if requested
    if args.pivot:
        create_pivot_analysis(args.input, output_file)


if __name__ == "__main__":
    main()