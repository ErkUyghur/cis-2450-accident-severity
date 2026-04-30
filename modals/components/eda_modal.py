"""EDA modal — full correlation heatmap, dual-series hour chart, and hypothesis testing."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
_HIGHLIGHT_CARD = {
    **_CARD,
    'backgroundColor': '#fff8e1',
    'border': '1px solid #ffe082',
}


# ── Chart builders ────────────────────────────────────────────────────────────

def _imbalance_comparison_chart(pdf: pd.DataFrame):
    """Side-by-side: approximate raw distribution (log scale) vs balanced working dataset."""
    raw_counts = [67_366, 2_600_000, 310_000, 60_000]
    raw_total  = sum(raw_counts)
    sev_labels = ['Sev 1', 'Sev 2', 'Sev 3', 'Sev 4']
    colors     = ['#2196F3', '#F44336', '#FF9800', '#9C27B0']

    bal = pdf['Severity'].value_counts().sort_index().reset_index()
    bal.columns = ['Severity', 'Count']
    bal_total = bal['Count'].sum()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Raw Kaggle Dataset (approximate, log scale)',
            'Balanced Working Dataset',
        ],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=sev_labels,
            y=raw_counts,
            marker_color=colors,
            text=[f'~{c:,}<br>({c/raw_total*100:.0f}%)' for c in raw_counts],
            textposition='outside',
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=sev_labels,
            y=bal['Count'].tolist(),
            marker_color=colors,
            text=[f'{c:,}<br>({c/bal_total*100:.0f}%)' for c in bal['Count']],
            textposition='outside',
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_yaxes(type='log', title_text='Count (log scale)', row=1, col=1)
    fig.update_yaxes(title_text='Count', range=[0, 80_000], tickformat=',', row=1, col=2)
    fig.update_layout(
        title='Class Imbalance: Original Raw Distribution vs Balanced Working Dataset',
        height=380,
        margin=dict(t=80, b=20, l=60, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig


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


def _hour_dual_chart(pdf: pd.DataFrame):
    """Bar (total accident count) with severe rate % line on a secondary y-axis.

    Shows that rush-hour peaks are volume-driven, while evening hours carry
    a disproportionately high severity rate — two distinct patterns in one chart.
    """
    tmp = pdf.copy()
    tmp['Is_Severe'] = (tmp['Severity'] >= 3).astype(int)
    hour_total = tmp.groupby('Hour').size().reset_index(name='Total')
    hour_rate  = (
        tmp.groupby('Hour')['Is_Severe'].mean()
        .mul(100).round(1).reset_index()
    )
    hour_rate.columns = ['Hour', 'Severe_Rate']

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(
        go.Bar(
            x=hour_total['Hour'], y=hour_total['Total'],
            name='Total Accidents', marker_color='steelblue', opacity=0.8,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hour_rate['Hour'], y=hour_rate['Severe_Rate'],
            name='Severe Rate (%)', mode='lines+markers',
            line=dict(color='darkred', width=2),
            marker=dict(size=5),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title='Accidents by Hour: Total Count vs Severe Rate (%)',
        height=340,
        margin=dict(t=50, b=20, l=20, r=60),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
    )
    fig.update_xaxes(title_text='Hour of Day (0–23)', dtick=2)
    fig.update_yaxes(title_text='Total Accidents', secondary_y=False,
                     gridcolor='#eeeeee')
    fig.update_yaxes(title_text='Severe Rate (%)', secondary_y=True,
                     showgrid=False)
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


def _corr_chart_full(pdf: pd.DataFrame):
    """Full 7-feature correlation heatmap including Wind_Chill and Precipitation.

    Explicitly exposes the Temperature ↔ Wind_Chill (r = 0.98) collinearity
    that motivated dropping Wind_Chill from all models.
    """
    cols = [
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
        'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
    ]
    present = [c for c in cols if c in pdf.columns]
    corr = pdf[present].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        height=420,
        aspect='auto',
    )
    # Annotate the high-collinearity pair with an arrow
    if 'Wind_Chill(F)' in present and 'Temperature(F)' in present:
        wc_idx = present.index('Wind_Chill(F)')
        t_idx  = present.index('Temperature(F)')
        fig.add_annotation(
            x=wc_idx, y=t_idx,
            text='r = 0.98<br>Wind_Chill removed',
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor='#c0392b',
            font=dict(size=10, color='#c0392b', family='Arial'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#c0392b', borderwidth=1,
            ax=80, ay=-50,
        )

    fig.update_layout(
        title='Full Feature Correlation Heatmap (all 7 numerical features)',
        margin=dict(t=50, b=20, l=20, r=20),
        coloraxis_colorbar=dict(thickness=14, len=0.8),
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
        title='Severe Accident Rate by Weather Condition (top 20, min 100 accidents)',
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


def _temp_dist_chart(pdf: pd.DataFrame):
    """Box plot comparing Temperature(F) for Severe vs Not Severe accidents."""
    tmp = pdf[['Severity', 'Temperature(F)']].dropna().copy()
    tmp['Severity Class'] = tmp['Severity'].apply(
        lambda x: 'Severe (3–4)' if x >= 3 else 'Not Severe (1–2)'
    )
    fig = px.box(
        tmp, x='Severity Class', y='Temperature(F)',
        color='Severity Class',
        color_discrete_map={
            'Not Severe (1–2)': '#667eea',
            'Severe (3–4)':     '#e74c3c',
        },
        points=False,
        labels={'Temperature(F)': 'Temperature (°F)', 'Severity Class': ''},
        height=300,
    )
    fig.update_layout(
        title='Temperature Distribution by Severity Class',
        showlegend=False,
        margin=dict(t=40, b=30, l=50, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(gridcolor='#eeeeee'),
    )
    return fig


def _hourly_rate_chart(pdf: pd.DataFrame):
    """Line chart of severe accident rate (%) by hour, with mean reference line."""
    tmp = pdf.copy()
    tmp['Is_Severe'] = (tmp['Severity'] >= 3).astype(int)
    hourly = (
        tmp.groupby('Hour')['Is_Severe']
        .mean().mul(100).round(2).reset_index()
    )
    hourly.columns = ['Hour', 'Severe_Rate']
    overall_mean = hourly['Severe_Rate'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly['Hour'], y=hourly['Severe_Rate'],
        mode='lines+markers',
        line=dict(color='darkred', width=2),
        marker=dict(size=5),
        name='Severe Rate (%)',
    ))
    fig.add_hline(
        y=overall_mean, line_dash='dash', line_color='gray',
        annotation_text=f'Mean {overall_mean:.1f}%',
        annotation_position='bottom right',
    )
    fig.update_layout(
        title='Severe Accident Rate (%) by Hour of Day',
        height=300,
        margin=dict(t=40, b=30, l=50, r=20),
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False,
        xaxis=dict(title='Hour of Day (0–23)', dtick=2, gridcolor='#eeeeee'),
        yaxis=dict(title='% Severe', gridcolor='#eeeeee'),
    )
    return fig


def _compute_hypothesis_tests(pdf: pd.DataFrame) -> list[dict]:
    """Run 3 non-parametric tests. Falls back to hardcoded values if scipy missing."""
    try:
        from scipy import stats
        tmp = pdf.copy()
        tmp['Is_Severe'] = (tmp['Severity'] >= 3).astype(int)

        # Test 1 — Chi-square: Weather Condition vs Severity
        contingency = pd.crosstab(tmp['Weather_Condition'], tmp['Is_Severe'])
        chi2, p1, dof, _ = stats.chi2_contingency(contingency)

        # Test 2 — Mann-Whitney U: Temperature vs Severity
        severe_t     = tmp[tmp['Is_Severe'] == 1]['Temperature(F)'].dropna()
        not_severe_t = tmp[tmp['Is_Severe'] == 0]['Temperature(F)'].dropna()
        u_stat, p2 = stats.mannwhitneyu(severe_t, not_severe_t, alternative='two-sided')
        mean_diff = severe_t.mean() - not_severe_t.mean()

        # Test 3 — Kruskal-Wallis: Hour vs Severity
        hour_groups = [tmp[tmp['Hour'] == h]['Is_Severe'].values for h in range(24)]
        h_stat, p3 = stats.kruskal(*hour_groups)

        def _fmt_p(p):
            return '< 0.001' if p < 0.001 else f'{p:.4f}'

        return [
            {
                'Test': 'Chi-square',
                'Feature': 'Weather_Condition',
                'Null Hypothesis (H₀)': 'Weather condition and severity are independent',
                'Statistic': f'χ² = {chi2:,.0f}, dof = {dof}',
                'p-value': _fmt_p(p1),
                'Decision': 'Reject H₀ — include Weather_Condition',
            },
            {
                'Test': 'Mann-Whitney U',
                'Feature': 'Temperature(F)',
                'Null Hypothesis (H₀)': 'Temperature distribution identical for Severe vs Not Severe',
                'Statistic': f'Δmean = {mean_diff:+.1f}°F (Severe lower)',
                'p-value': _fmt_p(p2),
                'Decision': 'Reject H₀ — include Temperature(F)',
            },
            {
                'Test': 'Kruskal-Wallis',
                'Feature': 'Hour',
                'Null Hypothesis (H₀)': 'Severe-accident probability equal across all 24 hours',
                'Statistic': f'H = {h_stat:,.1f}',
                'p-value': _fmt_p(p3),
                'Decision': 'Reject H₀ — extract Hour as a feature',
            },
        ]
    except Exception:
        # Hardcoded fallback from notebook §3.10
        return [
            {
                'Test': 'Chi-square',
                'Feature': 'Weather_Condition',
                'Null Hypothesis (H₀)': 'Weather condition and severity are independent',
                'Statistic': 'χ² = 20,747, dof = 92',
                'p-value': '< 0.001',
                'Decision': 'Reject H₀ — include Weather_Condition',
            },
            {
                'Test': 'Mann-Whitney U',
                'Feature': 'Temperature(F)',
                'Null Hypothesis (H₀)': 'Temperature distribution identical for Severe vs Not Severe',
                'Statistic': 'Δmean = −6.8°F (Severe lower)',
                'p-value': '< 0.001',
                'Decision': 'Reject H₀ — include Temperature(F)',
            },
            {
                'Test': 'Kruskal-Wallis',
                'Feature': 'Hour',
                'Null Hypothesis (H₀)': 'Severe-accident probability equal across all 24 hours',
                'Statistic': 'H = 6,769',
                'p-value': '< 0.001',
                'Decision': 'Reject H₀ — extract Hour as a feature',
            },
        ]


# ── Factory ───────────────────────────────────────────────────────────────────

def create_eda_modal(pdf: pd.DataFrame):
    """Factory: return (backdrop, modal) for the EDA section."""
    fig_imbalance = _imbalance_comparison_chart(pdf)
    fig_sev      = _severity_chart(pdf)
    fig_hour     = _hour_dual_chart(pdf)
    fig_dow      = _dow_chart(pdf)
    fig_corr     = _corr_chart_full(pdf)
    fig_weather  = _weather_chart(pdf)
    fig_road     = _road_chart(pdf)
    fig_temp     = _temp_dist_chart(pdf)
    fig_hr_rate  = _hourly_rate_chart(pdf)
    hyp_results  = _compute_hypothesis_tests(pdf)

    tbl_styles = get_standard_table_styles()
    tbl_styles['style_cell'] = {
        **tbl_styles['style_cell'],
        'whiteSpace': 'normal',
        'height': 'auto',
    }

    cfg = {'displayModeBar': False}

    content = html.Div([
        create_modal_title_bar('Exploratory Data Analysis', 'close-eda-modal'),
        html.Div([
            html.P(
                'Charts covering the target variable, temporal patterns, weather effects, '
                'full feature correlations, road context, and statistical hypothesis tests. '
                'Every finding directly informed a modeling decision.',
                style={'color': '#555', 'marginBottom': '20px', 'fontSize': '14px'},
            ),

            # ── Row 0: Class imbalance — before vs after ─────────────────────
            html.Div([
                html.H3('Why We Balanced the Dataset', style=_SECTION_TITLE),
                dcc.Graph(figure=fig_imbalance, config=cfg),
                html.Div([
                    html.Strong('The raw dataset is ~87% Severity 2. ', style={'color': '#333'}),
                    html.Span(
                        'A model trained on the raw distribution would learn to always predict '
                        '"Severity 2" and achieve ~87% accuracy — while completely failing to '
                        'flag high-risk conditions. '
                        'The left panel shows the approximate original counts (log scale — '
                        'note how Severity 2 dwarfs all other classes). '
                        'The right panel shows our balanced 269k working dataset, where each '
                        'severity level has ~67k records, forcing models to learn '
                        'genuinely distinguishing patterns.',
                        style={'fontSize': '13px', 'color': '#555'},
                    ),
                ], style={**_HIGHLIGHT_CARD, 'marginTop': '12px', 'padding': '12px 16px'}),
            ], style=_CARD),

            # ── Row 1: Balanced severity | Dual-series hour ──────────────────
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_sev, config=cfg),
                    html.P(
                        'After null-dropping on model features, each class retains ~65–67k '
                        'rows. Minor height differences reflect differing null rates across '
                        'severity levels — expected and acceptable.',
                        style=_NOTE,
                    ),
                ], style=_HALF),
                html.Div([
                    dcc.Graph(figure=fig_hour, config=cfg),
                    html.P(
                        'Blue bars show rush-hour volume peaks at 7–9 AM and 3–6 PM. '
                        'The red line tells a different story: severe accident rate peaks '
                        'in the early morning hours (midnight–5 AM) when traffic is light '
                        'but fatigue and reduced visibility are highest. '
                        'These are two independent signals, not one.',
                        style=_NOTE,
                    ),
                ], style=_HALF_RIGHT),
            ]),

            # ── Row 2: Day of week | Weather severity rate ───────────────────
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_dow, config=cfg),
                    html.P(
                        'Weekdays have significantly more accidents than weekends, '
                        'confirming that commuting is the primary driver of accident '
                        'frequency. Saturday and Sunday drop ~40% vs weekday average.',
                        style=_NOTE,
                    ),
                ], style=_HALF),
                html.Div([
                    dcc.Graph(figure=fig_weather, config=cfg),
                    html.P(
                        'Severity rate (% of accidents that were Sev 3+4) is more '
                        'informative than raw counts. Adverse conditions like heavy fog '
                        'and freezing rain produce disproportionately severe accidents. '
                        'This justifies including Weather_Condition as a model feature.',
                        style=_NOTE,
                    ),
                ], style=_HALF_RIGHT),
            ]),

            # ── Row 3: Full correlation heatmap (full width) ─────────────────
            html.Div([
                html.H3('Feature Correlation — All 7 Numerical Features', style=_SECTION_TITLE),
                dcc.Graph(figure=fig_corr, config=cfg),
                html.Div([
                    html.Strong('Key finding (annotated above): ', style={'color': '#333'}),
                    html.Span(
                        'Temperature(F) and Wind_Chill(F) are nearly perfectly correlated '
                        '(r = 0.98). Including both in a Logistic Regression model splits '
                        'the same signal across two coefficients — the model cannot attribute '
                        'the effect to either, inflating their standard errors and destabilizing '
                        'the solution. Wind_Chill was excluded from all models.',
                        style={'fontSize': '13px', 'color': '#555'},
                    ),
                ], style={**_HIGHLIGHT_CARD, 'marginTop': '12px', 'padding': '12px 16px'}),
                html.P(
                    'Note: Wind_Chill(F) and Precipitation(in) were not in the model-feature '
                    'null-drop list, so correlations for those pairs are computed on their '
                    'available rows only (21% and 49% of dataset respectively). '
                    'The Temperature–Wind_Chill r = 0.98 is physically expected and robust.',
                    style={**_NOTE, 'marginTop': '8px'},
                ),
            ], style=_CARD),

            # ── Row 4: Road features ─────────────────────────────────────────
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_road, config=cfg),
                    html.P(
                        'Most road feature flags are very sparse (<5% of accidents). '
                        'We keep only Traffic_Signal, Junction, and Crossing — the three '
                        'with sufficient data to provide reliable signal to models.',
                        style=_NOTE,
                    ),
                ], style=_HALF),
            ]),

            # ── Row 5: Hypothesis Testing ────────────────────────────────────
            html.Div([
                html.H3('Hypothesis Testing — Statistical Validation of Feature Choices',
                        style=_SECTION_TITLE),
                html.P(
                    'Visual patterns can arise by chance. We apply non-parametric tests '
                    'throughout because our target (Is_Severe) is binary and most numerical '
                    'features are not normally distributed — assumptions that would make '
                    't-tests and ANOVA incorrect. All three tests below confirm that the '
                    'observed patterns are statistically significant and justify including '
                    'each feature in the model.',
                    style={'color': '#555', 'marginBottom': '14px', 'fontSize': '13px'},
                ),

                # Summary table
                html.Div(
                    dash_table.DataTable(
                        data=hyp_results,
                        columns=[
                            {'name': 'Test',     'id': 'Test'},
                            {'name': 'Feature',  'id': 'Feature'},
                            {'name': 'H₀',       'id': 'Null Hypothesis (H₀)'},
                            {'name': 'Statistic','id': 'Statistic'},
                            {'name': 'p-value',  'id': 'p-value'},
                            {'name': 'Decision', 'id': 'Decision'},
                        ],
                        **tbl_styles,
                        style_cell_conditional=[
                            {'if': {'column_id': 'Test'},
                             'width': '120px', 'fontWeight': 'bold'},
                            {'if': {'column_id': 'Feature'},
                             'width': '130px', 'fontFamily': 'monospace'},
                            {'if': {'column_id': 'p-value'},
                             'width': '110px', 'color': '#c0392b', 'fontWeight': 'bold'},
                            {'if': {'column_id': 'Decision'},
                             'color': '#27ae60'},
                        ],
                    ),
                    style={'marginBottom': '16px'},
                ),

                # Temperature box + hourly rate charts side by side
                html.Div([
                    html.Div([
                        dcc.Graph(figure=fig_temp, config=cfg),
                        html.P(
                            'Mann-Whitney U confirms a significant temperature difference '
                            'between severity classes. Severe accidents occur at lower '
                            'temperatures on average (60.5°F vs 67.3°F) — cold or wet '
                            'conditions reduce traction and visibility. '
                            'The non-parametric test is correct here because temperature '
                            'is not normally distributed.',
                            style=_NOTE,
                        ),
                    ], style=_HALF),
                    html.Div([
                        dcc.Graph(figure=fig_hr_rate, config=cfg),
                        html.P(
                            'Kruskal-Wallis confirms severe-accident probability varies '
                            'significantly across hours (H = 6,769, p ≈ 0). '
                            'The rate peaks overnight (25–30%) and dips during rush hours '
                            '(~20%), opposite to the volume pattern. This validates '
                            'extracting Hour as a standalone feature.',
                            style=_NOTE,
                        ),
                    ], style=_HALF_RIGHT),
                ]),
            ], style=_CARD),

        ], style={'padding': '0 20px 20px 20px'}),
    ])

    backdrop = create_modal_backdrop('eda-modal-backdrop')
    modal = create_modal_container('eda-modal', content, width='92%', max_width='1400px')
    return backdrop, modal
