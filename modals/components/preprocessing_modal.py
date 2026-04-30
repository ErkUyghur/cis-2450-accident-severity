"""Data Pre-processing & Feature Engineering modal."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, dash_table

from modals.modal_helpers import (
    create_modal_title_bar,
    create_modal_backdrop,
    create_modal_container,
    get_standard_table_styles,
)

_CARD = {
    'backgroundColor': '#f9f9f9',
    'borderRadius': '8px',
    'padding': '16px',
    'marginBottom': '16px',
    'border': '1px solid #e0e0e0',
}
_HALF = {**_CARD, 'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}
_HALF_RIGHT = {**_HALF, 'marginLeft': '2%'}
_NOTE = {'color': '#666', 'fontSize': '13px', 'marginTop': '8px', 'marginBottom': '0'}
_SECTION_TITLE = {
    'fontSize': '16px', 'fontWeight': 'bold', 'color': '#444',
    'borderBottom': '2px solid #667eea', 'paddingBottom': '6px',
    'marginBottom': '12px', 'marginTop': '0',
}


# ── Summary table data ────────────────────────────────────────────────────────

_STEPS = [
    {
        'Step': 'Null handling',
        'Action': 'Drop rows with nulls only in model features',
        'EDA Section': '2.1',
        'Justification': 'High-null columns (End_Lat, Wind_Chill, Precipitation) would cause 90%+ data loss if all-null rows were dropped.',
    },
    {
        'Step': 'Outlier treatment',
        'Action': 'Cap Wind_Speed at 99th percentile',
        'EDA Section': '2.2',
        'Justification': 'Raw max = 822 mph — physically impossible, confirmed sensor error. Capping preserves all rows.',
    },
    {
        'Step': 'Feature engineering',
        'Action': 'Extract Hour from Start_Time',
        'EDA Section': '3.3',
        'Justification': 'Accident counts and severity rates vary strongly by hour; severe accidents peak in the evening.',
    },
    {
        'Step': 'Feature engineering',
        'Action': 'Extract DayOfWeek from Start_Time',
        'EDA Section': '3.6',
        'Justification': 'Weekdays have far more accidents than weekends — commuting is the primary driver.',
    },
    {
        'Step': 'Multicollinearity',
        'Action': 'Exclude Wind_Chill from model features',
        'EDA Section': '3.5',
        'Justification': 'r = 0.98 with Temperature. Including both confuses LR — model cannot attribute effect to either.',
    },
    {
        'Step': 'Road feature selection',
        'Action': 'Keep only Traffic_Signal, Junction, Crossing',
        'EDA Section': '3.7',
        'Justification': 'All other boolean road features are too sparse (<5% presence) to provide reliable signal.',
    },
    {
        'Step': 'Class imbalance',
        'Action': 'Downsample to ~67k rows per severity class',
        'EDA Section': '3.2',
        'Justification': 'Raw data is ~90% Severity 2. Without balancing, models cheat by always predicting Severity 2.',
    },
    {
        'Step': 'Categorical encoding',
        'Action': 'OneHotEncoder (handle_unknown=ignore)',
        'EDA Section': 'Pipeline',
        'Justification': 'Required for sklearn. handle_unknown=ignore safely handles unseen values at prediction time.',
    },
    {
        'Step': 'Feature scaling',
        'Action': 'StandardScaler on all numerical features',
        'EDA Section': 'Pipeline',
        'Justification': 'LR and distance-based models are sensitive to scale; unscaled features bias toward high-magnitude columns.',
    },
    {
        'Step': 'Feature exclusion',
        'Action': 'Exclude State as a model feature',
        'EDA Section': '3.9',
        'Justification': (
            'Kruskal-Wallis confirmed State IS associated with severity (p ≈ 0), '
            'but Spearman ρ between raw and per-capita rates was low — the signal is '
            'confounded by population density, not generalizable weather/road risk. '
            '50 state categories also risk overfitting to geographic artifacts.'
        ),
    },
]


# ── Chart builders ────────────────────────────────────────────────────────────

def _wind_speed_chart(pdf: pd.DataFrame):
    """Histogram of Wind_Speed after capping — shows clean distribution."""
    ws = pdf['Wind_Speed(mph)'].dropna()
    fig = px.histogram(
        ws, nbins=50,
        labels={'value': 'Wind Speed (mph)', 'count': 'Number of Accidents'},
        color_discrete_sequence=['#667eea'],
        height=300,
    )
    fig.update_layout(
        title='Wind Speed Distribution (after 99th-percentile cap)',
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        bargap=0.05,
    )
    fig.update_xaxes(title='Wind Speed (mph)')
    fig.update_yaxes(title='Number of Accidents')
    return fig


def _corr_chart(pdf: pd.DataFrame):
    """Correlation heatmap framed as justification for dropping Wind_Chill."""
    cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
            'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
    present = [c for c in cols if c in pdf.columns]
    corr = pdf[present].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True,
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        height=320,
    )
    fig.update_layout(
        title='Feature Correlation — Wind_Chill vs Temperature (r = 0.98)',
        margin=dict(t=40, b=20, l=20, r=20),
        coloraxis_showscale=False,
    )
    return fig


def _null_rate_chart(pdf: pd.DataFrame):
    """Bar chart of null rates across all original numerical columns."""
    # These are the columns with known high null rates from the raw dataset
    null_rates = {
        'Wind_Chill(F)':      79.0,
        'Precipitation(in)':  51.0,
        'End_Lat':            48.0,
        'End_Lng':            48.0,
        'Number':             18.0,
        'Wind_Speed(mph)':     8.0,
        'Visibility(mi)':      5.0,
        'Humidity(%)':         5.0,
        'Temperature(F)':      5.0,
    }
    df_nulls = pd.DataFrame(
        list(null_rates.items()), columns=['Column', 'Null Rate (%)']
    ).sort_values('Null Rate (%)', ascending=True)

    fig = px.bar(
        df_nulls, x='Null Rate (%)', y='Column',
        orientation='h',
        color='Null Rate (%)', color_continuous_scale='Oranges',
        labels={'Column': ''},
        height=320,
    )
    fig.update_layout(
        title='Approximate Null Rates in Raw Dataset',
        coloraxis_showscale=False,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis_range=[0, 100],
    )
    fig.add_vline(x=50, line_dash='dash', line_color='red',
                  annotation_text='50%', annotation_position='top right')
    return fig


def _severity_balance_chart(pdf: pd.DataFrame):
    """Side-by-side bar: raw imbalance (approximate) vs balanced sample."""
    raw_approx = pd.DataFrame({
        'Severity': ['1', '2', '3', '4'],
        'Count':    [67_366, 2_243_939, 389_847, 69_068],
        'Dataset':  ['Raw'] * 4,
    })
    balanced = pdf['Severity'].astype(str).value_counts().reset_index()
    balanced.columns = ['Severity', 'Count']
    balanced['Dataset'] = 'Balanced'

    combined = pd.concat([raw_approx, balanced], ignore_index=True)
    fig = px.bar(
        combined, x='Severity', y='Count', color='Dataset',
        barmode='group',
        color_discrete_map={'Raw': '#e57373', 'Balanced': '#667eea'},
        labels={'Count': 'Row Count', 'Severity': 'Severity Class'},
        height=320,
    )
    fig.update_layout(
        title='Class Distribution — Raw vs Balanced Sample',
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig


# ── Factory ───────────────────────────────────────────────────────────────────

def create_preprocessing_modal(pdf: pd.DataFrame):
    """Factory: return (backdrop, modal) for the preprocessing section."""
    fig_wind   = _wind_speed_chart(pdf)
    fig_corr   = _corr_chart(pdf)
    fig_nulls  = _null_rate_chart(pdf)
    fig_bal    = _severity_balance_chart(pdf)

    tbl_styles = get_standard_table_styles()
    # Merge the override before unpacking — avoids duplicate keyword argument error.
    tbl_styles['style_cell'] = {
        **tbl_styles['style_cell'],
        'whiteSpace': 'normal',
        'height': 'auto',
    }
    cfg = {'displayModeBar': False}

    content = html.Div([
        create_modal_title_bar(
            'Data Pre-processing & Feature Engineering',
            'close-preprocessing-modal'
        ),
        html.Div([
            html.P(
                'Every preprocessing step below is an EDA-informed decision, not an arbitrary default. '
                'Steps marked with EDA § are backed by visual patterns; those from §3.9–3.10 '
                'are additionally validated by formal statistical tests (see EDA → Hypothesis Testing).',
                style={'color': '#555', 'marginBottom': '20px', 'fontSize': '14px'}
            ),

            # ── Section 1: Summary table ─────────────────────────
            html.Div([
                html.H3('All Preprocessing & Feature Engineering Steps', style=_SECTION_TITLE),
                dash_table.DataTable(
                    data=_STEPS,
                    columns=[
                        {'name': 'Step',           'id': 'Step'},
                        {'name': 'Action',         'id': 'Action'},
                        {'name': 'EDA §',          'id': 'EDA Section'},
                        {'name': 'Justification',  'id': 'Justification'},
                    ],
                    **tbl_styles,
                    style_cell_conditional=[
                        {'if': {'column_id': 'Step'},         'width': '130px', 'fontWeight': 'bold'},
                        {'if': {'column_id': 'EDA Section'},  'width': '60px',  'textAlign': 'center'},
                        {'if': {'column_id': 'Action'},       'width': '240px'},
                    ],
                    page_size=10,
                ),
            ], style=_CARD),

            # ── Section 2: Null rates | Imbalance ────────────────
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_nulls, config=cfg),
                    html.P(
                        'Columns above 50% null rate (red line) were excluded entirely from features. '
                        'Dropping all rows with any null would have removed nearly the entire dataset.',
                        style=_NOTE,
                    ),
                ], style=_HALF),
                html.Div([
                    dcc.Graph(figure=fig_bal, config=cfg),
                    html.P(
                        'The raw dataset has 2.2M Severity 2 records vs. 67k Severity 1. '
                        'Our balancing script downsamples each class to 67k so models cannot '
                        'cheat by always predicting Severity 2.',
                        style=_NOTE,
                    ),
                ], style=_HALF_RIGHT),
            ]),

            # ── Section 3: Outlier | Correlation ─────────────────
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_wind, config=cfg),
                    html.P(
                        'After capping at the 99th percentile, Wind_Speed shows a clean right-skewed '
                        'distribution. Before capping, a handful of rows with 400–822 mph values '
                        'would have dominated the feature range.',
                        style=_NOTE,
                    ),
                ], style=_HALF),
                html.Div([
                    dcc.Graph(figure=fig_corr, config=cfg),
                    html.P(
                        'Temperature and Wind_Chill are nearly perfectly correlated (r = 0.98). '
                        'Wind_Chill was excluded from all models — keeping both would split '
                        'the same signal across two features, confusing LR coefficients.',
                        style=_NOTE,
                    ),
                ], style=_HALF_RIGHT),
            ]),

        ], style={'padding': '0 20px 20px 20px'}),
    ])

    backdrop = create_modal_backdrop('preprocessing-modal-backdrop')
    modal = create_modal_container(
        'preprocessing-modal', content, width='92%', max_width='1400px'
    )
    return backdrop, modal
