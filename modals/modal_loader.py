"""Dynamic modal loader for configuration-driven modal creation."""

import importlib
from typing import Dict, Tuple
from dash import html


def load_modals(modal_configs: Dict, data_context: Dict) -> Dict[str, Tuple[html.Div, html.Div]]:
    """Dynamically load and create all modals from configuration."""
    loaded_modals = {}
    for key, config in modal_configs.items():
        try:
            module = importlib.import_module(f'modals.{config["module"]}')
            factory_fn = getattr(module, config['factory'])
            deps = [data_context[dep] for dep in config['dependencies']]
            backdrop, modal = factory_fn(*deps)
            loaded_modals[key] = (backdrop, modal)
        except Exception as e:
            raise RuntimeError(f"Failed to load modal '{key}': {e}") from e
    return loaded_modals


def build_layout_from_modals(base_components: list, modals: Dict[str, Tuple]) -> list:
    """Build layout components list including all modals."""
    layout_children = list(base_components)
    for backdrop, modal in modals.values():
        layout_children.extend([backdrop, modal])
    return layout_children
