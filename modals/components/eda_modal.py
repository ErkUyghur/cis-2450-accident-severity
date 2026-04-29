"""EDA modal — six static Plotly charts matching the notebook's section 3."""

import pandas as pd
import plotly.express as px
from dash import html, dcc

from modals.modal_helpers import (
    create_modal_title_bar,
    create_modal_backdrop,
    create_modal_container,
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


def _severity_chart(pdf: pd.DataFrame):
    counts = pdf['Severity'].value_counts().sort_index().reset_index()
    counts.columns = ['Severity', 'Count']
    fig = px.bar(
        counts, x='Severity', y='Count',
        color='Severity', color_continuous_scale='Viridis',
        labels={'Count': 'Number of Accidents'},
        height=320,
    )
    fig.update_layout(
        title='Severity Distribution (Balanced Sample)',
        showlegend=False, coloraxis_showscale=False,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    fig.update_yaxes(range=[0, counts['Count'].max() * 1.1])
    return fig


def _hour_chart(pdf: pd.DataFrame):
    hour_counts = pdf.groupby('Hour').size().reset_index(name='Count')
    fig = px.bar(
        hour_counts, x='Hour', y='Count',
        color_discrete_sequence=['steelblue'],
        labels={'Hour': 'Hour of Day (0–23)', 'Count': 'Number of Accidents'},
        height=320,
    )
    fig.update_layout(
        title='Accidents by Hour of Day',
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig


def _dow_chart(pdf: pd.DataFrame):
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = pdf['DayOfWeek'].value_counts().reindex(days_order).reset_index()
    counts.columns = ['DayOfWeek', 'Count']
    fig = px.bar(
        counts, x='DayOfWeek', y='Count',
        color='DayOfWeek', color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={'DayOfWeek': '', 'Count': 'Number of Accidents'},
        height=320,
    )
    fig.update_layout(
        title='Accidents by Day of Week',
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig


def _corr_chart(pdf: pd.DataFrame):
    num_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
    corr = pdf[num_cols].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True,
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        height=340,
    )
    fig.update_layout(
        title='Feature Correlation Heatmap',
        margin=dict(t=40, b=20, l=20, r=20),
        coloraxis_showscale=False,
    )
    return fig


def _weather_chart(pdf: pd.DataFrame):
    weather_counts = pdf['Weather_Condition'].value_counts()
    valid = weather_counts[weather_counts >= 100].index
    wdf = pdf[pdf['Weather_Condition'].isin(valid)].copy()
    wdf['Is_Severe'] = (wdf['Severity'] >= 3).astype(int)
    rate = (
        wdf.groupby('Weather_Condition')['Is_Severe']
        .mean()
        .sort_values()
        .tail(20)
        .reset_index()
    )
    rate.columns = ['Weather_Condition', 'Pct_Severe']
    rate['Pct_Severe'] = (rate['Pct_Severe'] * 100).round(1)
    fig = px.bar(
        rate, x='Pct_Severe', y='Weather_Condition',
        orientation='h',
        color='Pct_Severe', color_continuous_scale='Reds',
        labels={'Pct_Severe': '% Severe (Sev 3+4)', 'Weather_Condition': ''},
        height=420,
    )
    fig.update_layout(
        title='Severe Accident Rate by Weather Condition (top 20)',
        coloraxis_showscale=False,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig


def _road_chart(pdf: pd.DataFrame):
    bool_cols = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
        'Traffic_Signal', 'Turning_Loop',
    ]
    present = [c for c in bool_cols if c in pdf.columns]
    counts = pdf[present].sum().sort_values().reset_index()
    counts.columns = ['Feature', 'Count']
    fig = px.bar(
        counts, x='Count', y='Feature',
        orientation='h',
        color='Count', color_continuous_scale='Magma',
        labels={'Count': 'Accidents where feature = True', 'Feature': ''},
        height=380,
    )
    fig.update_layout(
        title='Road Feature Presence During Accidents',
        coloraxis_showscale=False,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig


def create_eda_modal(pdf: pd.DataFrame):
    """Factory: return (backdrop, modal) for the EDA section."""
    fig_sev     = _severity_chart(pdf)
    fig_hour    = _hour_chart(pdf)
    fig_dow     = _dow_chart(pdf)
    fig_corr    = _corr_chart(pdf)
    fig_weather = _weather_chart(pdf)
    fig_road    = _road_chart(pdf)

    cfg = {'displayModeBar': False}

    content = html.Div([
        create_modal_title_bar('Exploratory Data Analysis', 'close-eda-modal'),
        html.Div([
            html.P(
                'Six charts covering the target variable, temporal patterns, weather effects, '
                'feature correlations, and road context. Each finding directly informed our '
                'modeling decisions.',
                style={'color': '#555', 'marginBottom': '20px', 'fontSize': '14px'}
            ),

            # Row 1: severity | hour
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_sev, config=cfg),
                    html.P(
                        'The balanced sample has ~67k rows per severity class. Without balancing, '
                        'Severity 2 makes up ~90% of the raw dataset — models would default to '
                        'always predicting Severity 2.',
                        style=_NOTE,
                    ),
                ], style=_HALF),
                html.Div([
                    dcc.Graph(figure=fig_hour, config=cfg),
                    html.P(
                        'Rush-hour spikes at 7–9 AM and 3–6 PM. Severe accidents (3+4) peak '
                        'later in the evening — less commute traffic, more fatigue and reduced '
                        'visibility.',
                        style=_NOTE,
                    ),
                ], style=_HALF_RIGHT),
            ]),

            # Row 2: day of week | correlation
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_dow, config=cfg),
                    html.P(
                        'Weekdays have significantly more accidents than weekends, confirming '
                        'that commuting is the primary driver of accident frequency.',
                        style=_NOTE,
                    ),
                ], style=_HALF),
                html.Div([
                    dcc.Graph(figure=fig_corr, config=cfg),
                    html.P(
                        'Temperature and Wind_Chill are nearly perfectly correlated (r ≈ 0.98). '
                        'We drop Wind_Chill from model features to avoid multicollinearity in '
                        'Logistic Regression.',
                        style=_NOTE,
                    ),
                ], style=_HALF_RIGHT),
            ]),

            # Row 3: weather severity rate | road features
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_weather, config=cfg),
                    html.P(
                        'Severity rate (% of accidents that were Sev 3+4) is more informative '
                        'than raw counts. Adverse conditions like heavy fog and freezing rain '
                        'produce disproportionately severe accidents.',
                        style=_NOTE,
                    ),
                ], style=_HALF),
                html.Div([
                    dcc.Graph(figure=fig_road, config=cfg),
                    html.P(
                        'Most road feature flags are very sparse. We keep only Traffic_Signal, '
                        'Junction, and Crossing as model features — the three with sufficient '
                        'data to contribute signal.',
                        style=_NOTE,
                    ),
                ], style=_HALF_RIGHT),
            ]),

        ], style={'padding': '0 20px 20px 20px'}),
    ])

    backdrop = create_modal_backdrop('eda-modal-backdrop')
    modal = create_modal_container('eda-modal', content, width='92%', max_width='1400px')
    return backdrop, modal
