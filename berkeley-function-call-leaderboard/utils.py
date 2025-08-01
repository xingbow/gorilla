"""
Utility functions for the BFCL Score Explorer.

This module contains helper functions and utilities used across the application.
"""

import streamlit as st
import re
from typing import Optional


def extract_id_number_global(item_id: Optional[str]) -> Optional[int]:
    """Extract numeric part from ID strings like 'multi_turn_base_45' -> 45"""
    if not item_id:
        return None
    
    # Try to extract number from the end of the ID
    match = re.search(r'(\d+)$', str(item_id))
    if match:
        return int(match.group(1))
    
    # If no trailing number, try to find any number in the ID
    numbers = re.findall(r'\d+', str(item_id))
    if numbers:
        return int(numbers[-1])  # Return the last number found
    
    return None


def initialize_session_state() -> None:
    """Initialize session state variables"""
    if 'expand_all' not in st.session_state:
        st.session_state.expand_all = False
    if 'expander_states' not in st.session_state:
        st.session_state.expander_states = {}
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'page_size' not in st.session_state:
        st.session_state.page_size = 25


def get_custom_css() -> str:
    """Return custom CSS styles for the application"""
    return """
    <style>
    .question-card {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .model-result {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 8px;
        margin: 3px 0;
        border-radius: 3px;
    }
    .ground-truth {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 8px;
        margin: 3px 0;
        border-radius: 3px;
    }
    .turn-separator {
        height: 2px;
        background: linear-gradient(to right, #e9ecef, #6c757d, #e9ecef);
        margin: 15px 0;
        border-radius: 1px;
    }
    </style>
    """


def get_default_category_index(available_categories: list) -> int:
    """Get the default index for category selection"""
    default_idx = 0
    if 'multi_turn_base_score' in available_categories:
        default_idx = available_categories.index('multi_turn_base_score')
    return default_idx