"""Main Dash application — US Traffic Accident Severity Predictor."""

from dash import Dash, html, Input, Output, ctx

from data.data_loader import load_accident_data, get_weather_options, get_slider_bounds
from modals.modal_config import MODAL_CONFIGS, get_modal_ids, get_modal_buttons
from modals.modal_loader import load_modals, build_layout_from_modals

_ACCENT   = '#667eea'
_ACCENT2  = '#764ba2'
_TEXT     = '#333333'
_MUTED    = '#666666'
_BORDER   = '#e0e0e0'


def _stat_tile(value: str, label: str, sublabel: str = '') -> html.Div:
    return html.Div([
        html.Div(value, style={
            'fontSize': '28px', 'fontWeight': 'bold',
            'color': _ACCENT, 'lineHeight': '1.1',
        }),
        html.Div(label, style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': _TEXT, 'marginTop': '4px',
        }),
        html.Div(sublabel, style={
            'fontSize': '11px', 'color': _MUTED, 'marginTop': '2px',
        }) if sublabel else html.Span(),
    ], style={
        'flex': '1',
        'minWidth': '110px',
        'backgroundColor': 'white',
        'border': f'1px solid {_BORDER}',
        'borderTop': f'3px solid {_ACCENT}',
        'borderRadius': '6px',
        'padding': '14px 16px',
        'textAlign': 'center',
    })


def _section_card(number: str, title: str, description: str,
                  button_id: str, button_label: str) -> html.Div:
    return html.Div([
        html.Div([
            html.Span(number, style={
                'display': 'inline-block',
                'width': '28px', 'height': '28px',
                'borderRadius': '50%',
                'backgroundColor': _ACCENT,
                'color': 'white',
                'fontSize': '13px', 'fontWeight': 'bold',
                'lineHeight': '28px', 'textAlign': 'center',
                'marginRight': '10px', 'flexShrink': '0',
            }),
            html.Strong(title, style={'color': _TEXT, 'fontSize': '14px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
        html.P(description, style={
            'fontSize': '13px', 'color': _MUTED,
            'margin': '0 0 0 38px', 'lineHeight': '1.5',
            'flex': '1',
        }),
        html.Button(
            button_label,
            id=button_id,
            n_clicks=0,
            className='modal-button',
            style={'width': '100%', 'marginTop': '14px'},
        ),
    ], style={
        'flex': '1',
        'minWidth': '220px',
        'backgroundColor': 'white',
        'border': f'1px solid {_BORDER}',
        'borderRadius': '8px',
        'padding': '16px',
        'display': 'flex',
        'flexDirection': 'column',
    })


def _build_intro(modal_buttons: list) -> html.Div:
    """Hero / introduction section displayed above the navigation tabs.

    modal_buttons — list of (button_id, label) tuples from get_modal_buttons().
    """
    btn = {label: bid for bid, label in modal_buttons}

    return html.Div([

        # ── Row 1: Problem + Data ──────────────────────────────────────────
        html.Div([

            # Left: The Problem & Impact
            html.Div([
                html.H3('The Problem', style={
                    'margin': '0 0 10px',
                    'fontSize': '15px', 'fontWeight': 'bold', 'color': _TEXT,
                    'borderBottom': f'2px solid {_ACCENT}',
                    'paddingBottom': '6px',
                }),
                html.P([
                    'Given that a traffic accident has occurred, ',
                    html.Strong('will it be severe'),
                    ' based on weather conditions, road features, and time of day? '
                    'We frame this as binary classification — ',
                    html.Strong('Severe (Severity 3–4) vs. Not Severe (Severity 1–2)'),
                    ' — because adjacent severity classes share nearly identical feature '
                    'distributions. Their distinction is driven by emergency response times '
                    'and injury counts that fall outside the dataset. '
                    'Binary classification gives our models a learnable signal and directly '
                    'answers the operational question.',
                ], style={'fontSize': '13px', 'color': _MUTED, 'lineHeight': '1.6',
                          'margin': '0 0 14px'}),

                html.H3('Potential Impact', style={
                    'margin': '0 0 8px',
                    'fontSize': '15px', 'fontWeight': 'bold', 'color': _TEXT,
                    'borderBottom': f'2px solid {_ACCENT}',
                    'paddingBottom': '6px',
                }),
                html.Ul([
                    html.Li([
                        html.Strong('City planners: '),
                        'Identify high-risk intersections and time windows for targeted '
                        'infrastructure improvements.',
                    ], style={'fontSize': '13px', 'color': _MUTED,
                              'lineHeight': '1.6', 'marginBottom': '6px'}),
                    html.Li([
                        html.Strong('Emergency dispatch: '),
                        'Triage response resources by pre-flagging conditions likely to '
                        'produce severe outcomes before units arrive.',
                    ], style={'fontSize': '13px', 'color': _MUTED,
                              'lineHeight': '1.6', 'marginBottom': '6px'}),
                    html.Li([
                        html.Strong('Traffic safety researchers: '),
                        'Quantify which weather and road-feature combinations are the '
                        'strongest predictors of injury severity.',
                    ], style={'fontSize': '13px', 'color': _MUTED,
                              'lineHeight': '1.6'}),
                ], style={'margin': '0', 'paddingLeft': '18px'}),
            ], style={
                'flex': '1.4',
                'minWidth': '280px',
                'backgroundColor': 'white',
                'border': f'1px solid {_BORDER}',
                'borderRadius': '8px',
                'padding': '20px',
                'marginRight': '16px',
            }),

            # Right: The Data
            html.Div([
                html.H3('The Dataset', style={
                    'margin': '0 0 10px',
                    'fontSize': '15px', 'fontWeight': 'bold', 'color': _TEXT,
                    'borderBottom': f'2px solid {_ACCENT}',
                    'paddingBottom': '6px',
                }),
                html.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td('Source', style={'fontWeight': '600', 'fontSize': '13px',
                                                     'color': _TEXT, 'paddingBottom': '8px',
                                                     'paddingRight': '12px', 'whiteSpace': 'nowrap'}),
                            html.Td('Kaggle — US Accidents (Sobhan Moosavi)',
                                    style={'fontSize': '13px', 'color': _MUTED, 'paddingBottom': '8px'}),
                        ]),
                        html.Tr([
                            html.Td('Coverage', style={'fontWeight': '600', 'fontSize': '13px',
                                                        'color': _TEXT, 'paddingBottom': '8px',
                                                        'paddingRight': '12px', 'whiteSpace': 'nowrap'}),
                            html.Td('Feb 2016 – Mar 2021, 49 US states',
                                    style={'fontSize': '13px', 'color': _MUTED, 'paddingBottom': '8px'}),
                        ]),
                        html.Tr([
                            html.Td('Raw size', style={'fontWeight': '600', 'fontSize': '13px',
                                                        'color': _TEXT, 'paddingBottom': '8px',
                                                        'paddingRight': '12px', 'whiteSpace': 'nowrap'}),
                            html.Td('269,464 records, 46 columns',
                                    style={'fontSize': '13px', 'color': _MUTED, 'paddingBottom': '8px'}),
                        ]),
                        html.Tr([
                            html.Td('After cleaning', style={'fontWeight': '600', 'fontSize': '13px',
                                                              'color': _TEXT, 'paddingBottom': '8px',
                                                              'paddingRight': '12px', 'whiteSpace': 'nowrap'}),
                            html.Td('246,347 records (null handling on model features only)',
                                    style={'fontSize': '13px', 'color': _MUTED, 'paddingBottom': '8px'}),
                        ]),
                        html.Tr([
                            html.Td('Model features', style={'fontWeight': '600', 'fontSize': '13px',
                                                              'color': _TEXT, 'paddingBottom': '8px',
                                                              'paddingRight': '12px', 'whiteSpace': 'nowrap'}),
                            html.Td([
                                html.Span('Weather: ', style={'fontWeight': '600'}),
                                'Temperature, Humidity, Pressure, Visibility, Wind Speed',
                                html.Br(),
                                html.Span('Time: ', style={'fontWeight': '600'}),
                                'Hour of day, Day of week',
                                html.Br(),
                                html.Span('Road: ', style={'fontWeight': '600'}),
                                'Traffic Signal, Junction, Crossing',
                            ], style={'fontSize': '13px', 'color': _MUTED, 'paddingBottom': '8px',
                                      'lineHeight': '1.7'}),
                        ]),
                        html.Tr([
                            html.Td('Train / Test', style={'fontWeight': '600', 'fontSize': '13px',
                                                            'color': _TEXT, 'paddingRight': '12px',
                                                            'whiteSpace': 'nowrap'}),
                            html.Td('170,868 training (balanced) / 46,532 test (stratified 80/20)',
                                    style={'fontSize': '13px', 'color': _MUTED}),
                        ]),
                    ]),
                ], style={'width': '100%', 'borderCollapse': 'collapse'}),
            ], style={
                'flex': '1',
                'minWidth': '260px',
                'backgroundColor': 'white',
                'border': f'1px solid {_BORDER}',
                'borderRadius': '8px',
                'padding': '20px',
            }),

        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '0',
            'marginBottom': '16px',
        }),

        # ── Row 2: Key stats ───────────────────────────────────────────────
        html.Div([
            _stat_tile('269,464', 'Total Records', 'raw dataset'),
            _stat_tile('11', 'Model Features', 'weather, road, time'),
            _stat_tile('3', 'Models Tested', 'LR → RF → GBM'),
            _stat_tile('72.2%', 'Best Macro F1', 'Random Forest'),
            _stat_tile('0.75', 'Best Severe Recall', 'Random Forest'),
            _stat_tile('+4.4 pp', 'F1 Gain vs Baseline', 'LR → RF improvement'),
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '10px',
            'marginBottom': '16px',
        }),

        # ── Row 3: Section guide ───────────────────────────────────────────
        html.Div([
            html.P('What to explore:', style={
                'fontSize': '13px', 'fontWeight': '600',
                'color': _TEXT, 'margin': '0 0 10px',
            }),
            html.Div([
                _section_card(
                    '1', 'Exploratory Data Analysis',
                    'Temporal patterns, weather severity rates, feature correlations, '
                    'and road feature sparsity — the findings that drove every '
                    'preprocessing and modeling decision.',
                    btn['Exploratory Data Analysis'],
                    'Exploratory Data Analysis',
                ),
                _section_card(
                    '2', 'Pre-processing & Feature Engineering',
                    '9 EDA-informed preprocessing steps with justifications: null handling, '
                    'outlier capping, feature engineering, multicollinearity removal, '
                    'class balancing, and pipeline construction.',
                    btn['Pre-processing & Feature Engineering'],
                    'Pre-processing & Feature Engineering',
                ),
                _section_card(
                    '3', 'Modeling & Results',
                    '3-model progression from Logistic Regression to Random Forest to '
                    'Gradient Boosting, with confusion matrices, feature importances, '
                    'and GridSearchCV hyperparameter tuning results.',
                    btn['Modeling & Results'],
                    'Modeling & Results',
                ),
            ], style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '12px',
            }),
        ]),

    ], style={
        'padding': '24px 32px',
        'backgroundColor': '#f8f9fc',
        'borderBottom': f'1px solid {_BORDER}',
    })


def _register_toggle(app: Dash, modal_key: str, width: str, max_width: str) -> None:
    """Register open/close callbacks for one modal."""
    ids = get_modal_ids(modal_key)
    open_style = {
        'display': 'block',
        'width': width,
        'maxWidth': max_width,
        'backgroundColor': 'white',
        'padding': '0',
    }
    hidden = {'display': 'none'}

    @app.callback(
        [Output(ids['modal_id'], 'style'),
         Output(ids['backdrop_id'], 'style')],
        [Input(ids['button_id'], 'n_clicks'),
         Input(ids['close_button_id'], 'n_clicks')],
        prevent_initial_call=True,
    )
    def toggle(_open, _close):
        if ctx.triggered_id == ids['button_id']:
            return open_style, {'display': 'block'}
        return hidden, hidden


def create_app() -> Dash:
    # ── Load data ────────────────────────────────────────────
    pdf = load_accident_data()

    data_context = {
        'pdf': pdf,
        'weather_options': get_weather_options(pdf),
        'slider_bounds': get_slider_bounds(pdf),
    }

    # ── Create app and load modals ───────────────────────────
    app = Dash(__name__, suppress_callback_exceptions=True)
    modals = load_modals(MODAL_CONFIGS, data_context)

    # ── Header ───────────────────────────────────────────────
    header = html.Div([
        html.H1(
            'US Traffic Accident Severity',
            style={'margin': '0', 'color': 'white', 'fontSize': '26px'},
        ),
        html.P(
            'Predicting accident severity from weather, road features, and time of day',
            style={'margin': '6px 0 0', 'color': 'rgba(255,255,255,0.85)', 'fontSize': '14px'},
        ),
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'padding': '24px 32px',
    })

    # ── Intro (contains the section-open buttons) ────────────
    intro = _build_intro(get_modal_buttons())

    # ── Layout ───────────────────────────────────────────────
    layout_children = build_layout_from_modals([header, intro], modals)
    app.layout = html.Div(layout_children, style={'fontFamily': 'Arial, sans-serif'})

    # ── Register callbacks ───────────────────────────────────
    for key, config in MODAL_CONFIGS.items():
        _register_toggle(app, key, config['width'], config['max_width'])

    return app


if __name__ == '__main__':
    create_app().run(debug=True, port=8050, dev_tools_ui=False, dev_tools_props_check=False)
