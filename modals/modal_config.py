"""Single source of truth for all modal configuration."""

MODAL_CONFIGS = {
    'eda': {
        'title': 'Exploratory Data Analysis',
        'factory': 'create_eda_modal',
        'module': 'components.eda_modal',
        'dependencies': ['pdf'],
        'width': '92%',
        'max_width': '1400px',
        'button_label': 'Exploratory Data Analysis',
    },
}


def get_modal_ids(modal_key: str) -> dict:
    """Generate standard IDs for a modal."""
    key_dashes = modal_key.replace('_', '-')
    return {
        'modal_id':       f'{key_dashes}-modal',
        'backdrop_id':    f'{key_dashes}-modal-backdrop',
        'button_id':      f'open-{key_dashes}-modal',
        'close_button_id': f'close-{key_dashes}-modal',
    }


def get_modal_buttons() -> list[tuple[str, str]]:
    """Return (button_id, label) pairs for every modal."""
    return [
        (f'open-{key.replace("_", "-")}-modal', config['button_label'])
        for key, config in MODAL_CONFIGS.items()
    ]
