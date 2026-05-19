"""Job-monitoring agent package.

Splits cleanly so the same building blocks back both the interactive notebook
flow and the MLflow-logged chain that gets deployed to Model Serving.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"]
