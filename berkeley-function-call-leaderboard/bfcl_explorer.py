"""
BFCL Score Explorer - Main Application

A Streamlit application for exploring BFCL (Berkeley Function Calling Leaderboard) 
model performance data with enhanced visualization and analysis capabilities.

This refactored version uses modular architecture for better code organization,
maintainability, and debugging.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any

# Import our modules
from explorer_data import (
    load_csv_data, 
    load_model_directories, 
    load_detailed_scores,
    load_full_json_data,
    apply_filters
)
from ui_components import (
    render_prompt_and_results,
    render_error_analysis,
    render_model_result_raw,
    render_summary_dashboard_tab,
    render_model_metrics,
    render_filter_controls,
    render_pagination_controls,
    render_expand_collapse_controls
)
from utils import (
    initialize_session_state,
    get_custom_css,
    get_default_category_index
)

# Configure Streamlit page
st.set_page_config(
    page_title="BFCL Score Explorer",
    page_icon="📊",
    layout="wide"
)


def render_summary_dashboard(csv_data: Dict[str, pd.DataFrame]) -> None:
    """Render the Summary Dashboard view"""
    st.header("Summary Dashboard")
    
    # Display CSV summaries
    if csv_data:
        tab_names = list(csv_data.keys())
        tabs = st.tabs(tab_names)
        
        for i, (name, df) in enumerate(csv_data.items()):
            with tabs[i]:
                render_summary_dashboard_tab(name, df)


def render_model_details(model_dirs: List[str]) -> None:
    """Render the Model Details view"""
    st.header("Model Details")
    
    if not model_dirs:
        st.warning("No model directories found in data/bfcl/score/")
        return
    
    # Model selection
    selected_model = st.selectbox("Select Model", model_dirs)
    
    if selected_model:
        # Load detailed scores
        detailed_scores = load_detailed_scores(selected_model)
        
        if detailed_scores:
            st.subheader(f"Detailed Results for {selected_model}")
            
            # Create metrics display
            render_model_metrics(detailed_scores)
            
            # Category selection
            available_categories = list(detailed_scores.keys())
            default_idx = get_default_category_index(available_categories)
            
            selected_category = st.selectbox(
                "Select Category for Detailed View",
                available_categories,
                index=default_idx
            )
            
            if selected_category:
                render_category_details(detailed_scores[selected_category])


def render_category_details(category_data: Dict[str, Any]) -> None:
    """Render detailed view for a selected category"""
    file_path = category_data['file_path']
    
    # Load all data first
    all_data = load_full_json_data(file_path, 1000)  # Load more data for pagination
    
    if all_data:
        # Render filter controls
        search_term, selected_error_type, range_filter = render_filter_controls(all_data)
        
        # Apply filters to all data first
        filtered_all_data = apply_filters(all_data, search_term, selected_error_type, range_filter)
        total_filtered_items = len(filtered_all_data)
        
        # Render pagination controls
        page_size, _ = render_pagination_controls(total_filtered_items)
        
        # Calculate data for current page from filtered data
        start_idx = (st.session_state.current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_filtered_items)
        filtered_data = filtered_all_data[start_idx:end_idx]
        
        # Render expand/collapse controls
        render_expand_collapse_controls(filtered_all_data)
        
        # Render individual items
        render_data_items(filtered_data)


def render_data_items(filtered_data: List[Dict[str, Any]]) -> None:
    """Render individual data items with expandable details"""
    for item in filtered_data:
        item_id = item.get('id', 'N/A')
        is_valid = item.get('valid', 'N/A')
        status_icon = "✅" if is_valid else "❌"
        
        # Determine expanded state
        expanded = st.session_state.expander_states.get(item_id, False)
        
        with st.expander(f"{status_icon} ID: {item_id} - Valid: {is_valid}", expanded=expanded):
            # Update expander state when manually toggled
            st.session_state.expander_states[item_id] = True
            
            # First, show the prompt and results comparison
            render_prompt_and_results(item)
            
            # Then show error analysis if there's an error
            if 'error' in item and item['error']:
                st.markdown("---")
                render_error_analysis(item['error'])
            else:
                st.success("✅ No errors detected")
            
            # Show model_result_raw field with enhanced visualization
            render_model_result_raw(item)
            
            # Show full data toggle
            with st.expander("🔍 Show Full JSON Data", expanded=False):
                st.json(item)


def render_raw_data_explorer(csv_data: Dict[str, pd.DataFrame], model_dirs: List[str]) -> None:
    """Render the Raw Data Explorer view"""
    st.header("Raw Data Explorer")
    
    # File browser
    data_type = st.selectbox(
        "Select Data Type",
        ["CSV Summaries", "JSON Details"]
    )
    
    if data_type == "CSV Summaries":
        render_csv_explorer(csv_data)
    elif data_type == "JSON Details":
        render_json_explorer(model_dirs)


def render_csv_explorer(csv_data: Dict[str, pd.DataFrame]) -> None:
    """Render CSV data explorer"""
    if csv_data:
        selected_csv = st.selectbox("Select CSV File", list(csv_data.keys()))
        if selected_csv:
            df = csv_data[selected_csv]
            
            # Search functionality
            search = st.text_input("Search in data")
            if search:
                mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
                df = df[mask]
            
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_csv.lower().replace(' ', '_')}_filtered.csv",
                mime="text/csv"
            )


def render_json_explorer(model_dirs: List[str]) -> None:
    """Render JSON data explorer"""
    if model_dirs:
        selected_model = st.selectbox("Select Model", model_dirs)
        if selected_model:
            detailed_scores = load_detailed_scores(selected_model)
            if detailed_scores:
                selected_category = st.selectbox(
                    "Select Category", 
                    list(detailed_scores.keys())
                )
                if selected_category:
                    file_path = detailed_scores[selected_category]['file_path']
                    
                    # Load and display raw JSON
                    limit = st.slider("Number of records to load", 10, 500, 50)
                    raw_data = load_full_json_data(file_path, limit)
                    
                    if raw_data:
                        st.subheader(f"Raw JSON Data ({len(raw_data)} records)")
                        
                        # Search in JSON
                        json_search = st.text_input("Search in JSON data")
                        if json_search:
                            raw_data = [
                                item for item in raw_data 
                                if json_search.lower() in str(item).lower()
                            ]
                        
                        # Display as expandable JSON
                        for i, item in enumerate(raw_data):
                            with st.expander(f"Record {i+1}: {item.get('id', 'N/A')}"):
                                st.json(item)


def main() -> None:
    """Main application function"""
    st.title("📊 BFCL Score Explorer")
    st.markdown("Explore BFCL (Berkeley Function Calling Leaderboard) model performance data")
    
    # Add custom CSS for better visual appearance
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Load data
    csv_data = load_csv_data()
    model_dirs = load_model_directories()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    view_mode = st.sidebar.radio(
        "Select View",
        ["📈 Summary Dashboard", "🔍 Model Details", "📋 Raw Data Explorer"]
    )
    
    # Route to appropriate view
    if view_mode == "📈 Summary Dashboard":
        render_summary_dashboard(csv_data)
    elif view_mode == "🔍 Model Details":
        render_model_details(model_dirs)
    elif view_mode == "📋 Raw Data Explorer":
        render_raw_data_explorer(csv_data, model_dirs)


if __name__ == "__main__":
    main()