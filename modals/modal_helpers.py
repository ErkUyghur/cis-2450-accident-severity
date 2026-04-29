"""Shared helper functions for modal components."""

from dash import html, dcc, dash_table
import numpy as np


def create_modal_title_bar(title: str, close_button_id: str):
    """Create standardized modal title bar with gradient background."""
    return html.Div([
        html.H2(title, style={'margin': 0, 'color': 'white', 'flex': 1}),
        html.Button(
            '✕',
            id=close_button_id,
            n_clicks=0,
            style={
                'background': 'none',
                'border': 'none',
                'color': 'white',
                'fontSize': '24px',
                'cursor': 'pointer',
                'padding': '0 10px',
                'lineHeight': '1'
            }
        )
    ], style={
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': 'white',
        'padding': '20px',
        'borderRadius': '12px 12px 0 0',
        'marginBottom': '20px'
    })


def create_modal_backdrop(backdrop_id: str):
    """Create standardized modal backdrop."""
    return html.Div(
        id=backdrop_id,
        className='modal-backdrop',
        style={'display': 'none'}
    )


def create_modal_container(modal_id: str, children, width='80%', max_width='1000px'):
    """Create standardized modal container."""
    return html.Div(
        children,
        id=modal_id,
        className='modal-container',
        style={
            'display': 'none',
            'width': width,
            'maxWidth': max_width,
            'backgroundColor': 'white',
            'padding': '0'
        }
    )


def get_standard_table_styles():
    """Get standardized table styling configuration."""
    return {
        'style_table': {
            'overflowX': 'auto',
            'border': '1px solid #e0e0e0',
            'borderRadius': '8px'
        },
        'style_cell': {
            'textAlign': 'left',
            'padding': '12px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px'
        },
        'style_header': {
            'backgroundColor': '#f5f5f5',
            'fontWeight': 'bold',
            'borderBottom': '2px solid #667eea',
            'color': '#333'
        },
        'style_data': {
            'borderBottom': '1px solid #f0f0f0'
        },
        'style_data_conditional': [
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#fafafa'
            }
        ]
    }


def table_records_and_columns(pandas_df, list_columns=None, column_name_map=None):
    """Convert a pandas DataFrame to (records, columns) for Dash DataTable."""
    if list_columns:
        pandas_df = pandas_df.copy()
        for col in list_columns:
            if col in pandas_df.columns:
                pandas_df[col] = pandas_df[col].apply(
                    lambda x: ', '.join(str(i) for i in x)
                    if isinstance(x, (list, tuple, set, np.ndarray))
                    else str(x)
                )
    records = pandas_df.to_dict('records')
    if column_name_map:
        columns = [
            {'name': column_name_map.get(col, col), 'id': col}
            for col in pandas_df.columns
        ]
    else:
        columns = [{'name': col, 'id': col} for col in pandas_df.columns]
    return records, columns
