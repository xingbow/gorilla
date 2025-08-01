"""
UI components and rendering functions for the BFCL Score Explorer.

This module contains all the Streamlit UI components, visualization functions,
and rendering logic for displaying data in the explorer.
"""

import streamlit as st
import plotly.express as px
import difflib
from typing import Dict, List, Any


def render_prompt_and_results(item: Dict[str, Any]) -> None:
    """Render user prompt questions, model results, and ground truth in an intuitive layout"""
    prompt_data = item.get('prompt', {})
    model_result_decoded = item.get('model_result_decoded', [])
    possible_answer = item.get('possible_answer', [])
    
    if prompt_data or model_result_decoded or possible_answer:
        st.write("**💬 Question & Answer Analysis**")
        
        # Extract questions from prompt
        questions = []
        if 'question' in prompt_data:
            for turn_idx, turn in enumerate(prompt_data['question']):
                if isinstance(turn, list) and turn:
                    for msg in turn:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            questions.append(f"Turn {turn_idx + 1}: {msg.get('content', '')}")
        
        # Display questions with better formatting
        if questions:
            with st.expander("📝 User Prompt Questions", expanded=True):
                for i, question in enumerate(questions):
                    st.markdown(f'<div class="question-card"><strong>{question}</strong></div>', unsafe_allow_html=True)
                    if i < len(questions) - 1:
                        st.markdown('<div class="turn-separator"></div>', unsafe_allow_html=True)
        
        # Compare model results vs ground truth
        if model_result_decoded or possible_answer:
            with st.expander("🔄 Model vs Ground Truth Comparison", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="model-result"><strong>🤖 Model Result (Decoded)</strong></div>', unsafe_allow_html=True)
                    if model_result_decoded:
                        for turn_idx, turn_result in enumerate(model_result_decoded):
                            if turn_result:  # Only show non-empty turns
                                st.markdown(f"<div style='color: #856404; font-weight: bold; margin: 10px 0;'>Turn {turn_idx + 1}:</div>", unsafe_allow_html=True)
                                if isinstance(turn_result, list):
                                    for step in turn_result:
                                        if isinstance(step, list):
                                            for func_call in step:
                                                st.code(func_call, language='python')
                                        else:
                                            st.code(str(step), language='python')
                                else:
                                    st.code(str(turn_result), language='python')
                                if turn_idx < len(model_result_decoded) - 1:
                                    st.markdown('<div class="turn-separator"></div>', unsafe_allow_html=True)
                    else:
                        st.info("No decoded model results available")
                
                with col2:
                    st.markdown('<div class="ground-truth"><strong>🎯 Ground Truth (Expected)</strong></div>', unsafe_allow_html=True)
                    if possible_answer:
                        for turn_idx, turn_answer in enumerate(possible_answer):
                            if turn_answer:  # Only show non-empty turns
                                st.markdown(f"<div style='color: #155724; font-weight: bold; margin: 10px 0;'>Turn {turn_idx + 1}:</div>", unsafe_allow_html=True)
                                if isinstance(turn_answer, list):
                                    for func_call in turn_answer:
                                        st.code(func_call, language='python')
                                else:
                                    st.code(str(turn_answer), language='python')
                                if turn_idx < len(possible_answer) - 1:
                                    st.markdown('<div class="turn-separator"></div>', unsafe_allow_html=True)
                    else:
                        st.info("No ground truth answers available")


def render_error_analysis(error_data: Dict[str, Any]) -> None:
    """Render error analysis in a human-friendly format"""
    if not error_data:
        return
    
    # Error type only (remove message as requested)
    error_type = error_data.get('error_type', 'Unknown')
    st.write(f"**🔍 Error Type**: `{error_type}`")
    
    # Show all error details
    if 'details' in error_data:
        details = error_data['details']
        
        # Handle differences if they exist
        if 'differences' in details:
            differences = details['differences']
            formatted_diff = format_difference_display(differences)
            
            if formatted_diff:
                st.write("**🔄 Detailed Differences:**")
                
                for field, diff_info in formatted_diff.items():
                    with st.expander(f"Field: {field}", expanded=False):
                        if isinstance(diff_info, dict) and 'ground_truth' in diff_info:
                            if diff_info.get('diff') and diff_info['diff'] != 'Values are identical':
                                # Show colored side-by-side diff for different values
                                st.write("**🎨 Side-by-Side Comparison with Highlighted Differences:**")
                                
                                # Use container to control width
                                with st.container():
                                    render_side_by_side_diff(
                                        diff_info.get('gt_lines', []), 
                                        diff_info.get('model_lines', [])
                                    )
                                
                                # Also show raw unified diff in a collapsible section
                                with st.expander("🔄 Raw Unified Diff (Technical)", expanded=False):
                                    st.code(diff_info['diff'], language='diff', width=1000)
                            else:
                                st.success("✅ Values are identical")
                                # Still show the content for reference
                                with st.expander("📝 View Content", expanded=False):
                                    st.write("**Content:**")
                                    st.text_area(
                                        "Content", 
                                        value=diff_info['ground_truth'], 
                                        height=100, 
                                        key=f"identical_{field}_{hash(diff_info['ground_truth'])}",
                                        disabled=True
                                    )
                        else:
                            st.write(f"**Value**: {diff_info}")
        
        # # Show other details that are not differences
        # other_details = {k: v for k, v in details.items() if k != 'differences'}
        # if other_details:
        #     st.write("**📝 Additional Details:**")
        #     for key, value in other_details.items():
        #         st.write(f"**{key.replace('_', ' ').title()}:**")
        #         if isinstance(value, (dict, list)):
        #             st.json(value)
        #         else:
        #             st.text(str(value))
    
    # Handle cases where there are no details but other error info
    elif error_data:
        st.write("**📝 Error Information:**")
        with st.expander("🗺️ Show All Error Data", expanded=False):
            # Show all error data except type (already shown above)
            error_info = {k: v for k, v in error_data.items() if k not in ['error_type', 'details']}
            if error_info:
                for key, value in error_info.items():
                    st.write(f"**{key.replace('_', ' ').title()}:**")
                    if isinstance(value, (dict, list)):
                        st.json(value)
                    else:
                        st.text(str(value))
            else:
                st.info("No additional error details available.")


def format_difference_display(differences: Dict[str, Any]) -> Dict[str, Any]:
    """Format difference data in a human-readable way"""
    formatted_diff = {}
    
    for key, diff_data in differences.items():
        if isinstance(diff_data, dict) and 'model' in diff_data and 'ground_truth' in diff_data:
            model_val = str(diff_data['model'])
            gt_val = str(diff_data['ground_truth'])
            
            # Create a diff display
            if model_val != gt_val:
                diff_lines = list(difflib.unified_diff(
                    gt_val.splitlines(keepends=True),
                    model_val.splitlines(keepends=True),
                    fromfile='Ground Truth',
                    tofile='Model Output',
                    lineterm=''
                ))
                
                # Create side-by-side comparison data
                gt_lines = gt_val.splitlines()
                model_lines = model_val.splitlines()
                
                formatted_diff[key] = {
                    'ground_truth': gt_val,
                    'model': model_val,
                    'diff': '\n'.join(diff_lines) if diff_lines else 'No differences found',
                    'gt_lines': gt_lines,
                    'model_lines': model_lines
                }
            else:
                formatted_diff[key] = {
                    'ground_truth': gt_val,
                    'model': model_val,
                    'diff': 'Values are identical',
                    'gt_lines': gt_val.splitlines(),
                    'model_lines': model_val.splitlines()
                }
        else:
            formatted_diff[key] = diff_data
    
    return formatted_diff


def create_colored_diff_html(gt_lines: List[str], model_lines: List[str]) -> tuple:
    """Create HTML with colored differences for side-by-side comparison"""
    # Use difflib to get detailed differences
    matcher = difflib.SequenceMatcher(None, gt_lines, model_lines)
    
    gt_html_lines = []
    model_html_lines = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Lines are the same
            for i in range(i1, i2):
                gt_html_lines.append(f'<div class="diff-equal">{_escape_html(gt_lines[i])}</div>')
            for j in range(j1, j2):
                model_html_lines.append(f'<div class="diff-equal">{_escape_html(model_lines[j])}</div>')
        elif tag == 'delete':
            # Lines only in ground truth (deleted in model)
            for i in range(i1, i2):
                gt_html_lines.append(f'<div class="diff-delete">- {_escape_html(gt_lines[i])}</div>')
            # Add empty lines to model side for alignment
            for i in range(i1, i2):
                model_html_lines.append(f'<div class="diff-empty"></div>')
        elif tag == 'insert':
            # Lines only in model (inserted)
            # Add empty lines to ground truth side for alignment
            for j in range(j1, j2):
                gt_html_lines.append(f'<div class="diff-empty"></div>')
            for j in range(j1, j2):
                model_html_lines.append(f'<div class="diff-insert">+ {_escape_html(model_lines[j])}</div>')
        elif tag == 'replace':
            # Lines are different
            max_lines = max(i2 - i1, j2 - j1)
            for k in range(max_lines):
                if k < (i2 - i1):
                    gt_html_lines.append(f'<div class="diff-delete">- {_escape_html(gt_lines[i1 + k])}</div>')
                else:
                    gt_html_lines.append(f'<div class="diff-empty"></div>')
                
                if k < (j2 - j1):
                    model_html_lines.append(f'<div class="diff-insert">+ {_escape_html(model_lines[j1 + k])}</div>')
                else:
                    model_html_lines.append(f'<div class="diff-empty"></div>')
    
    return gt_html_lines, model_html_lines


def _escape_html(text: str) -> str:
    """Escape HTML special characters"""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))


def render_model_result_raw(item: Dict[str, Any]) -> None:
    """Render model_result_raw field with compact visual representation"""
    model_result_raw = item.get('model_result_raw')
    
    if not model_result_raw:
        return
    
    # More compact header
    st.write("**📝 Additional Details:**")
    with st.expander("🔧 Raw Model Output", expanded=False):
        # Handle different types of model_result_raw data
        if isinstance(model_result_raw, str):
            render_raw_string_data_compact(model_result_raw)
        elif isinstance(model_result_raw, list):
            render_raw_list_data_compact(model_result_raw)
        elif isinstance(model_result_raw, dict):
            render_raw_dict_data_compact(model_result_raw)
        else:
            st.code(str(model_result_raw), language='text')


def render_raw_string_data_compact(raw_data: str) -> None:
    """Render raw string data with compact formatting"""
    # Detect format and show compact info
    raw_data_lower = raw_data.lower().strip()
    
    # Compact stats in single line
    stats = f"📊 {len(raw_data)} chars, {len(raw_data.splitlines())} lines"
    
    if raw_data.strip().startswith(('{', '[')):
        try:
            import json
            parsed = json.loads(raw_data)
            st.caption(f"JSON format • {stats}")
            st.json(parsed)
        except json.JSONDecodeError:
            st.caption(f"Invalid JSON • {stats}")
            st.code(raw_data, language='json', line_numbers=False)
    elif any(keyword in raw_data_lower for keyword in ['def ', 'import ', 'class ', 'return ', 'print(']):
        st.caption(f"Python code • {stats}")
        st.code(raw_data, language='python', line_numbers=False)
    elif '(' in raw_data and ')' in raw_data:
        st.caption(f"Function calls • {stats}")
        st.code(raw_data, language='python', line_numbers=False)
    else:
        st.caption(f"Plain text • {stats}")
        st.code(raw_data, language='text', line_numbers=False)


def get_turn_summary(turn_data) -> str:
    """Generate a summary of the turn data for the header"""
    if isinstance(turn_data, str):
        if len(turn_data) > 100:
            return f"- `{turn_data[:50]}...` ({len(turn_data)} chars)"
        return f"- `{turn_data}`"
    elif isinstance(turn_data, list):
        if not turn_data:
            return "- Empty list"
        # Analyze list content
        if len(turn_data) == 1:
            item = turn_data[0]
            if isinstance(item, dict) and 'function' in str(item).lower():
                return f"- Single function call"
            return f"- 1 item ({type(item).__name__})"
        else:
            return f"- {len(turn_data)} items"
    elif isinstance(turn_data, dict):
        keys = list(turn_data.keys())
        if len(keys) <= 3:
            return f"- {{{', '.join(keys)}}}"
        return f"- {{{', '.join(keys[:2])}, ...}} ({len(keys)} keys)"
    else:
        return f"- {type(turn_data).__name__}"


def render_turn_structure(turn_data, turn_num: int = None) -> None:
    """Render the structure of a single turn with direct content display"""
    if isinstance(turn_data, str):
        # For string data, detect and format appropriately
        if turn_data.strip().startswith(('{', '[')):
            try:
                import json
                parsed = json.loads(turn_data)
                render_compact_data(parsed)
            except json.JSONDecodeError:
                st.code(turn_data, language='json', line_numbers=False)
        else:
            st.code(turn_data, language='text', line_numbers=False)
    
    elif isinstance(turn_data, list):
        # For list data, directly display all items neck-to-neck
        if not turn_data:
            st.info("Empty turn")
            return
        
        for item in turn_data:
            render_compact_data(item)
    
    elif isinstance(turn_data, dict):
        # For dict data, directly show key-value pairs
        render_compact_data(turn_data)
    
    else:
        # Fallback for other types
        st.code(str(turn_data), language='text', line_numbers=False)


def render_compact_data(data, indent: str = "") -> None:
    """Render data in a compact function call format"""
    if isinstance(data, dict):
        # Check if this looks like a function call (single key-value pair)
        if len(data) == 1:
            key, value = next(iter(data.items()))
            if isinstance(value, dict):
                # Format as function call: function_name: {params}
                import json
                params_str = json.dumps(value, separators=(',', ': '))
                st.write(f"`{key}: {params_str}`")
            elif isinstance(value, str):
                st.write(f"`{key}: \"{value}\"`")
            else:
                st.write(f"`{key}: {value}`")
        else:
            # Multiple keys - show each key-value pair
            for key, value in data.items():
                if isinstance(value, str):
                    display_value = value if len(value) <= 100 else f"{value[:100]}..."
                    st.write(f"{indent}**{key}**: `{display_value}`")
                elif isinstance(value, dict):
                    import json
                    params_str = json.dumps(value, separators=(',', ': '))
                    st.write(f"{indent}**{key}**: `{params_str}`")
                elif isinstance(value, list):
                    st.write(f"{indent}**{key}**: `{value}`")
                else:
                    st.write(f"{indent}**{key}**: `{value}`")
    
    elif isinstance(data, list):
        for item in data:
            render_compact_data(item, indent)
    
    elif isinstance(data, str):
        st.write(f"{indent}`{data}`")
    
    else:
        st.write(f"{indent}`{data}`")


def render_raw_list_data_compact(raw_data: List) -> None:
    """Render raw list data with optimized structure for each turn"""
    st.caption(f"📋 {len(raw_data)} turns")
    
    # Display turns vertically for better readability of structure
    for i, turn_data in enumerate(raw_data):
        # Create a more informative turn header
        turn_summary = get_turn_summary(turn_data)
        
        with st.expander(f"**Turn {i + 1}** {turn_summary}", expanded=True):
            render_turn_structure(turn_data, i + 1)


def render_raw_dict_data_compact(raw_data: Dict) -> None:
    """Render raw dict data with compact structured view"""
    st.caption(f"🗂️ Dict with {len(raw_data)} keys: {', '.join(list(raw_data.keys())[:3])}{'...' if len(raw_data) > 3 else ''}")
    
    # Compact key-value pairs
    for key, value in raw_data.items():
        with st.expander(f"{key} ({type(value).__name__})", expanded=False):
            if isinstance(value, str):
                st.code(value, language='text', line_numbers=False)
            elif isinstance(value, (dict, list)):
                st.json(value)
            else:
                st.text(str(value))


def render_side_by_side_diff(gt_lines: List[str], model_lines: List[str]) -> None:
    """Render side-by-side diff with colors and horizontal scrolling"""
    gt_html_lines, model_html_lines = create_colored_diff_html(gt_lines, model_lines)
    
    # CSS for styling
    diff_css = """
    <style>
    .diff-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
    }
    .diff-row {
        display: flex;
        gap: 10px;
        width: 100%;
        max-width: 100%;
    }
    .diff-panel {
        flex: 1;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 8px;
        background-color: #f8f9fa;
        overflow-x: auto;
        overflow-y: auto;
        max-height: 200px;
        max-width: 100%;
        width: 100%;
        box-sizing: border-box;
        word-break: break-all;
        word-wrap: break-word;
    }
    .diff-header {
        font-weight: bold;
        margin-bottom: 8px;
        color: #333;
        background-color: #e9ecef;
        padding: 4px 8px;
        border-radius: 3px;
        word-break: break-word;
        white-space: normal;
    }
    .diff-equal {
        color: #333;
        padding: 2px 0;
        white-space: pre-wrap;
        word-break: break-all;
        overflow-wrap: break-word;
    }
    .diff-delete {
        color: #d73027;
        background-color: #ffebee;
        padding: 2px 4px;
        margin: 1px 0;
        border-radius: 2px;
        white-space: pre-wrap;
        word-break: break-all;
        overflow-wrap: break-word;
    }
    .diff-insert {
        color: #4caf50;
        background-color: #e8f5e8;
        padding: 2px 4px;
        margin: 1px 0;
        border-radius: 2px;
        white-space: pre-wrap;
        word-break: break-all;
        overflow-wrap: break-word;
    }
    .diff-empty {
        height: 20px;
        padding: 2px 0;
    }
    </style>
    """
    
    # Create HTML content
    gt_content = '\n'.join(gt_html_lines)
    model_content = '\n'.join(model_html_lines)
    
    html_content = f"""
    {diff_css}
    <div class="diff-container">
        <div class="diff-row">
            <div class="diff-panel">
                <div class="diff-header">🎯 Ground Truth</div>
                {gt_content}
            </div>
        </div>
        <div class="diff-row">
            <div class="diff-panel">
                <div class="diff-header">🤖 Model Output</div>
                {model_content}
            </div>
        </div>
    </div>
    """
    
    st.markdown(html_content, unsafe_allow_html=True)


def render_summary_dashboard_tab(name: str, df) -> None:
    """Render a single tab in the summary dashboard"""
    st.subheader(f"{name} Results")
    
    # Display metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df, use_container_width=True)
    
    with col2:
        if name == "Multi-Turn" and "Multi Turn Overall Acc" in df.columns:
            # Bar chart for multi-turn accuracy
            fig = px.bar(
                df, 
                x="Model", 
                y="Multi Turn Overall Acc",
                title="Multi-Turn Overall Accuracy"
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        elif "Overall Acc" in df.columns:
            # Bar chart for overall accuracy
            fig = px.bar(
                df, 
                x="Model", 
                y="Overall Acc",
                title="Overall Accuracy"
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


def render_model_metrics(detailed_scores: Dict[str, Dict[str, Any]]) -> None:
    """Render model performance metrics"""
    cols = st.columns(len(detailed_scores))
    for i, (category, data) in enumerate(detailed_scores.items()):
        with cols[i % len(cols)]:
            summary = data['summary']
            accuracy = summary.get('accuracy', 0)
            correct = summary.get('correct_count', 0)
            total = summary.get('total_count', 0)
            
            st.metric(
                label=category.replace('_', ' ').title(),
                value=f"{accuracy:.1%}",
                delta=f"{correct}/{total}"
            )


def render_filter_controls(all_data: List[Dict[str, Any]]) -> tuple:
    """Render filter controls and return filter values"""
    from explorer_data import extract_error_types_from_data
    
    st.subheader("🔍 Filters")
    
    search_term = st.text_input("Search in full JSON content (searches all fields)")
    
    # Second row for range filter and error type filter
    col3, col4, col5 = st.columns(3)
    with col3:
        range_filter = st.text_input(
            "Filter by ID range (e.g., '45-90' or '10-20')",
            placeholder="e.g., 45-90"
        )
    with col4:
        # Extract error types from all data for filter options
        available_error_types = extract_error_types_from_data(all_data)
        error_type_options = ["All"] + available_error_types
        selected_error_type = st.selectbox(
            "Filter by Error Type",
            error_type_options,
            index=0
        )
    with col5:
        st.write("**Filter Help:**")
        st.caption("Range: '45-90' includes IDs 45 through 90")
        st.caption(f"Error types: {len(available_error_types)} unique types found")
    
    return search_term, selected_error_type, range_filter


def render_pagination_controls(total_filtered_items: int) -> tuple:
    """Render pagination controls and return page settings"""
    st.subheader("📄 Pagination Settings")
    col1, col2 = st.columns([1, 1])
    with col1:
        page_size = st.selectbox(
            "Items per page", 
            [10, 25, 50, 100], 
            index=[10, 25, 50, 100].index(st.session_state.page_size)
        )
        st.session_state.page_size = page_size
    
    with col2:
        total_pages = max(1, (total_filtered_items + page_size - 1) // page_size)
        st.write(f"**Filtered Items**: {total_filtered_items} | **Total Pages**: {total_pages}")
    
    # Reset page if current page is beyond total pages
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = 1
    
    # Enhanced pagination controls with + and - buttons
    st.write("**Navigate Pages:**")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 1])
    
    with nav_col1:
        if st.button("⏮️ First", disabled=st.session_state.current_page <= 1):
            st.session_state.current_page = 1
    
    with nav_col2:
        if st.button("◀️ Prev", disabled=st.session_state.current_page <= 1):
            st.session_state.current_page = max(1, st.session_state.current_page - 1)
    
    with nav_col3:
        if st.button("▶️ Next", disabled=st.session_state.current_page >= total_pages):
            st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)
    
    with nav_col4:
        if st.button("⏭️ Last", disabled=st.session_state.current_page >= total_pages):
            st.session_state.current_page = total_pages
    
    # Display current page info
    st.info(f"📍 Showing page {st.session_state.current_page} of {total_pages}")
    
    return page_size, total_pages


def render_expand_collapse_controls(filtered_data: List[Dict[str, Any]]) -> None:
    """Render expand/collapse all controls"""
    results_col1, results_col2 = st.columns([3, 1])
    with results_col1:
        total_filtered_items = len(filtered_data)
        start_idx = (st.session_state.current_page - 1) * st.session_state.page_size
        end_idx = min(start_idx + st.session_state.page_size, total_filtered_items)
        st.subheader(f"📋 Results (Page {st.session_state.current_page}: {start_idx + 1}-{end_idx} of {total_filtered_items} items)")
    with results_col2:
        col_expand, col_collapse = st.columns(2)
        with col_expand:
            if st.button("🔼 Expand All", key="expand_all_btn"):
                st.session_state.expand_all = True
                for item in filtered_data:
                    st.session_state.expander_states[item.get('id', 'N/A')] = True
        with col_collapse:
            if st.button("🔽 Collapse All", key="collapse_all_btn"):
                st.session_state.expand_all = False
                for item in filtered_data:
                    st.session_state.expander_states[item.get('id', 'N/A')] = False