"""
Module for building Sphinx documentation.
"""

import os
import glob
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

def append_line_to_file(file_path: str, line: str) -> None:
    """
    Append a line to the end of the specified file.

    Args:
        file_path: The path to the file
        line: The line to append
    """
    with open(file_path, 'a') as file:
        file.write(f"{line}\n")

def setup_sphinx_project(
    project_dir: str,
    external_docs_dir: str,
    image_dir: str,
    html_embedding_file: str,
    logo_path: str,
    log_file: Optional[str] = None,
    project_title: str = "GenesetInsight"
) -> None:
    """
    Set up a Sphinx project by creating necessary files and directories,
    copying external documentation, generating an index.rst, and building
    the HTML documentation.

    Args:
        project_dir: The directory path where the Sphinx project will be created
        external_docs_dir: The directory path where the external documentation files are located
        image_dir: The directory path where the image files are located
        html_embedding_file: The path to the HTML file to be embedded
        logo_path: The path to the logo image that will be copied into _static
        log_file: Path to the log file (optional)
        project_title: The title of the project

    Raises:
        subprocess.CalledProcessError: If any of the subprocess commands fail
    """
    # Set up logging if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    logger.info(f"Ensuring the project directory exists: {project_dir}")
    os.makedirs(project_dir, exist_ok=True)

    logger.info("Creating a new Sphinx project in quiet mode...")
    subprocess.run(
        [
            "sphinx-quickstart",
            project_dir,
            "--quiet",
            "--project",
            project_title,
            "--author",
            "",
            "--sep",
        ],
        check=True,
    )

    logger.info(f"Copying documentation files from {external_docs_dir} to {project_dir}/source")
    source_docs_dir = os.path.join(project_dir, "source")
    for rst_file in glob.glob(os.path.join(external_docs_dir, "*.*")):
        shutil.copy(rst_file, source_docs_dir)

    logger.info(f"Copying image files from {image_dir} to {source_docs_dir}")
    for img_file in glob.glob(os.path.join(image_dir, "*.*")):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            shutil.copy(img_file, source_docs_dir)

    # Copy the HTML embedding file to the source directory
    logger.info(f"Copying HTML embedding file from {html_embedding_file} to {source_docs_dir}")
    shutil.copy(html_embedding_file, source_docs_dir)

    # Copy the logo image to the _static directory
    static_dir = os.path.join(source_docs_dir, "_static")
    os.makedirs(static_dir, exist_ok=True)
    shutil.copy(logo_path, static_dir)

    # ------------------------------------------------------------
    # Generate the index.rst file INSIDE the source directory
    # ------------------------------------------------------------
    logger.info("Identifying doc*.rst files for potential index population...")
    theme_files = sorted(glob.glob(os.path.join(source_docs_dir, "Theme_*.rst")))
    theme_files = [os.path.basename(doc).replace(".rst", "") for doc in theme_files]
    
    # Sort theme files by theme number
    theme_files.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 99999)

    logger.info(f"Found {len(theme_files)} theme files")

    index_content = [
        f"{project_title} Topic Descriptions\n",
        "==================================\n\n",
        ".. toctree::\n",
        "   :maxdepth: 1\n",
        "   :caption: Major themes:\n",
        "\n",
    ]

    # Add each theme file to the toctree
    index_content.extend(f"   {doc}\n" for doc in theme_files)

    # Optionally include additional sections (example):
    metadata_content = """
.. toctree::
   :maxdepth: 1
   :caption: Metadata:

   summary
   download
"""
    index_content.append(metadata_content)

    # Write the index.rst file to the source directory so Sphinx will read it
    index_path = os.path.join(source_docs_dir, "index.rst")
    logger.info(f"Writing index.rst to {index_path}")
    with open(index_path, "w") as file:
        file.writelines(index_content)

    # ------------------------------------------------------------
    # Update conf.py with theme/extension settings
    # ------------------------------------------------------------
    logger.info("Updating Sphinx configuration...")
    
    # Try to use the sphinx_rtd_theme if available, otherwise use the default theme
    try:
        import sphinx_rtd_theme
        theme_available = True
    except ImportError:
        theme_available = False
        
    conf_file_path = os.path.join(source_docs_dir, 'conf.py')
    append_line_to_file(conf_file_path, "\n# Theme configuration")
    
    if theme_available:
        append_line_to_file(conf_file_path, "import sphinx_rtd_theme")
        append_line_to_file(conf_file_path, "html_theme = 'sphinx_rtd_theme'")
        append_line_to_file(conf_file_path, "html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]")
    else:
        # Use a basic theme that comes with Sphinx
        append_line_to_file(conf_file_path, "html_theme = 'nature'")
    
    # Add extensions
    append_line_to_file(conf_file_path, "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode']")
    
    # Customize theme
    append_line_to_file(conf_file_path, "html_theme_options = {")
    append_line_to_file(conf_file_path, f"    'logo': '{os.path.basename(logo_path)}',")
    append_line_to_file(conf_file_path, "    'logo_only': True,")
    append_line_to_file(conf_file_path, "    'display_version': False,")
    append_line_to_file(conf_file_path, "}")
    
    # Hide attribution
    append_line_to_file(conf_file_path, "html_show_copyright = False")
    append_line_to_file(conf_file_path, "html_show_sphinx = False")
    
    # Add static files
    append_line_to_file(conf_file_path, "html_static_path = ['_static']")
    append_line_to_file(conf_file_path, "html_css_files = ['custom.css']")
    
    # Create a custom CSS file with basic styling
    custom_css_file = os.path.join(static_dir, "custom.css")
    with open(custom_css_file, "w") as css_file:
        css_file.write("""
/* Custom CSS */
.wy-side-nav-search {
    background-color: #2980B9;
}
.wy-nav-content {
    max-width: 1200px;
}
""")

    # ------------------------------------------------------------
    # Build the HTML documentation
    # ------------------------------------------------------------
    logger.info("Building HTML documentation with sphinx-build...")
    build_html_dir = os.path.join(project_dir, "build", "html")
    subprocess.run(
        [
            "sphinx-build",
            "-b", "html",
            source_docs_dir,
            build_html_dir,
        ],
        check=True,
    )

    logger.info("HTML documentation built successfully; check build/html/index.html.")