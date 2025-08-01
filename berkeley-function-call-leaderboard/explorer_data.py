"""
Data loading and processing functions for the BFCL Score Explorer.

This module handles all data loading operations including CSV summaries,
JSON detailed scores, and data filtering/pagination.
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, List, Any


@st.cache_data
def load_csv_data() -> Dict[str, pd.DataFrame]:
    """Load all CSV summary files"""
    base_path = "score"
    csv_files = {
        "Multi-Turn": "data_multi_turn.csv",
        "Overall": "data_overall.csv",
        "Live": "data_live.csv",
        "Non-Live": "data_non_live.csv"
    }
    
    data = {}
    for name, filename in csv_files.items():
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            data[name] = pd.read_csv(file_path)
    
    return data


@st.cache_data
def load_model_directories() -> List[str]:
    """Get list of available model directories"""
    base_path = "score"
    directories = []
    
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                directories.append(item)
    
    return sorted(directories)


@st.cache_data
def load_detailed_scores(model_name: str) -> Dict[str, Dict[str, Any]]:
    """Load detailed JSON scores for a specific model"""
    base_path = f"score/{model_name}"
    scores = {}
    
    if os.path.exists(base_path):
        for filename in os.listdir(base_path):
            if filename.endswith('.json'):
                file_path = os.path.join(base_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        # Read first line for summary
                        first_line = f.readline().strip()
                        if first_line:
                            summary = json.loads(first_line)
                            test_category = filename.replace('.json', '').replace('BFCL_v3_', '')
                            scores[test_category] = {
                                'summary': summary,
                                'file_path': file_path
                            }
                except Exception as e:
                    st.error(f"Error loading {filename}: {e}")
    
    return scores


def load_full_json_data(file_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Load detailed JSON data with pagination"""
    data = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:limit+1]:  # Skip summary line
                if line.strip():
                    data.append(json.loads(line.strip()))
    except Exception as e:
        st.error(f"Error loading detailed data: {e}")
    
    return data


def extract_error_types_from_data(data_list: List[Dict[str, Any]]) -> List[str]:
    """Extract unique error types from a list of data items"""
    error_types = set()
    for item in data_list:
        if 'error' in item and item['error']:
            error_type = item['error'].get('error_type', 'Unknown')
            error_types.add(error_type)
    return sorted(list(error_types))


def apply_filters(data: List[Dict[str, Any]], search_term: str = "", 
                 error_type: str = "All", range_filter: str = "") -> List[Dict[str, Any]]:
    """Apply various filters to the data"""
    from utils import extract_id_number_global
    
    filtered_data = data.copy()
    
    # Apply search filter (searches full JSON content)
    if search_term:
        filtered_data = [
            item for item in filtered_data 
            if search_term.lower() in json.dumps(item, default=str).lower()
        ]
    
    # Apply error type filter
    if error_type and error_type != "All":
        filtered_data = [
            item for item in filtered_data
            if 'error' in item and item['error'] 
            and item['error'].get('error_type', 'Unknown') == error_type
        ]
    
    # Apply range filter
    if range_filter:
        try:
            # Parse range format like "45-90"
            if '-' in range_filter and len(range_filter.split('-')) == 2:
                start_str, end_str = range_filter.split('-')
                start_num = int(start_str.strip())
                end_num = int(end_str.strip())
                
                if start_num <= end_num:
                    filtered_data = [
                        item for item in filtered_data
                        if extract_id_number_global(item.get('id', '')) is not None
                        and start_num <= extract_id_number_global(item.get('id', '')) <= end_num
                    ]
                else:
                    st.error("Range start must be less than or equal to range end")
            else:
                st.error("Range format should be 'start-end' (e.g., '45-90')")
        except ValueError:
            st.error("Range values must be numbers (e.g., '45-90')")
        except Exception as e:
            st.error(f"Error parsing range: {e}")
    
    return filtered_data


def paginate_data(data: List[Dict[str, Any]], page: int, page_size: int) -> List[Dict[str, Any]]:
    """Paginate data based on current page and page size"""
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(data))
    return data[start_idx:end_idx]