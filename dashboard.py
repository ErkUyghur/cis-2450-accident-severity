"""Main Dash application — US Traffic Accident Severity Predictor."""

from dash import Dash, html, Input, Output, ctx

from data.data_loader import load_accident_data, get_weather_options, get_slider_bounds
from modals.modal_config import MODAL_CONFIGS, get_modal_ids, get_modal_buttons
from modals.modal_loader import load_modals, build_layout_from_modals


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

    # ── Button bar ───────────────────────────────────────────
    button_bar = html.Div(
        [
            html.Button(label, id=btn_id, n_clicks=0, className='modal-button')
            for btn_id, label in get_modal_buttons()
        ],
        style={
            'padding': '16px 32px',
            'backgroundColor': '#f5f5f5',
            'borderBottom': '1px solid #e0e0e0',
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '8px',
        },
    )

    # ── Layout ───────────────────────────────────────────────
    layout_children = build_layout_from_modals([header, button_bar], modals)
    app.layout = html.Div(layout_children, style={'fontFamily': 'Arial, sans-serif'})

    # ── Register callbacks ───────────────────────────────────
    for key, config in MODAL_CONFIGS.items():
        _register_toggle(app, key, config['width'], config['max_width'])

    return app


if __name__ == '__main__':
    create_app().run(debug=True, port=8050, dev_tools_ui=False, dev_tools_props_check=False)
