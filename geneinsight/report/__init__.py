"""
Report generation modules for the TopicGenes package.
"""

from .geneplotter import generate_heatmaps
from .circleplot import generate_circle_plot
from .summary import generate_json_summary
from .rst_generator import generate_rst_files, generate_download_rst
from .sphinx_builder import build_sphinx_docs  # Changed from setup_sphinx_project

__all__ = [
    "generate_heatmaps",
    "generate_circle_plot",
    "generate_json_summary",
    "generate_rst_files",
    "generate_download_rst",
    "build_sphinx_docs"  # Changed from setup_sphinx_project
]