#!/usr/bin/env python3
"""
Sphinx Builder Module for GeneInsight

This module handles building Sphinx documentation from the generated RST files.
It can be imported as a library (using build_sphinx_docs) or run as a standalone script.
"""

import os
import glob
import shutil
import subprocess
import argparse
import logging

def append_line_to_file(file_path, line):
    """
    Append a line to the end of the specified file.
    
    Args:
        file_path (str): The path to the file.
        line (str): The line to append.
    """
    with open(file_path, 'a') as file:
        file.write(f"{line}\n")

def generate_index_rst(source_dir):
    """
    Generate the index.rst file for the Sphinx documentation.
    
    This version uses the logic from the second script: it collects all .rst files
    in the source directory (excluding index.rst, summary.rst, and download.rst) and
    builds a toctree.
    
    Args:
        source_dir (str): Source directory of the Sphinx project.
    """
    logging.info("Identifying .rst files for index.rst generation...")
    # Find all .rst files in the source directory.
    rst_files = sorted(glob.glob(os.path.join(source_dir, "*.rst")))
    # Exclude specific files.
    rst_files = [f for f in rst_files if os.path.basename(f) not in ["index.rst", "summary.rst", "download.rst"]]
    # Get file names without the .rst extension.
    doc_files = [os.path.basename(f).replace(".rst", "") for f in rst_files]
    logging.info(f"Found the following document files: {doc_files}")

    index_content = [
        "GeneInsight Topic Descriptions\n",
        "==================================\n\n",
        ".. toctree::\n",
        "   :maxdepth: 1\n",
        "   :caption: Major themes:\n",
        "\n",
    ]
    for doc in doc_files:
        index_content.append(f"   {doc}\n")
    
    # Append a metadata section.
    metadata_content = """
.. toctree::
   :maxdepth: 1
   :caption: Metadata:

   summary
   download
"""
    index_content.append(metadata_content)
    
    index_path = os.path.join(source_dir, "index.rst")
    with open(index_path, "w") as f:
        f.writelines(index_content)
    
    logging.info(f"Generated index.rst at {index_path}")

def update_conf_py(source_dir, logo_path):
    """
    Update the conf.py file by appending Sphinx theme and extension settings.
    
    This version uses the line-appending method from the second script.
    
    Args:
        source_dir (str): Source directory of the Sphinx project.
        logo_path (str): Full path to the logo image.
    """
    logging.info("Appending theme and extension settings to conf.py...")
    conf_file_path = os.path.join(source_dir, "conf.py")
    
    # Append new configuration settings.
    append_line_to_file(conf_file_path, "\n")
    append_line_to_file(conf_file_path, "html_theme = 'sphinx_wagtail_theme'")
    append_line_to_file(conf_file_path, "extensions = ['sphinx_wagtail_theme', 'sphinx_togglebutton']")
    append_line_to_file(conf_file_path, "html_theme_options = dict(")
    append_line_to_file(conf_file_path, "    project_name = \"GeneInsight\",")
    append_line_to_file(conf_file_path, "    footer_links = \",\".join([")
    append_line_to_file(conf_file_path, "        \"Contact|http://example.com/contact\",")
    append_line_to_file(conf_file_path, "        \"Developers|http://example.com/dev/null\",")
    append_line_to_file(conf_file_path, "    ]),")
    append_line_to_file(conf_file_path, f"    logo = \"{os.path.basename(logo_path)}\",")
    append_line_to_file(conf_file_path, "    logo_alt = \"GeneInsight\",")
    append_line_to_file(conf_file_path, "    logo_height = 59,")
    append_line_to_file(conf_file_path, "    logo_url = \"/\",")
    append_line_to_file(conf_file_path, "    logo_width = 45,")
    append_line_to_file(conf_file_path, ")")
    append_line_to_file(conf_file_path, "html_show_copyright = False")
    append_line_to_file(conf_file_path, "html_show_sphinx = False")
    
    logging.info(f"Updated conf.py at {conf_file_path}")

def build_sphinx_docs(rst_folder, image_folder, output_folder, log_path, html_embedding_path, logo_path):
    """
    Build Sphinx documentation from RST files.
    
    This function creates a new Sphinx project, copies over RST and image files,
    embeds the given HTML file, generates an index.rst, updates conf.py, and builds
    the HTML documentation.
    
    Args:
        rst_folder (str): Folder containing RST files.
        image_folder (str): Folder containing image files.
        output_folder (str): Folder to save the Sphinx build (project directory).
        log_path (str): Path to save the log file.
        html_embedding_path (str): Path to the HTML file to embed.
        logo_path (str): Path to the logo image.
    
    Returns:
        bool: True if build succeeded, False otherwise.
    """
    logging.info("Starting Sphinx documentation build process...")
    
    try:
        # Ensure the output (project) folder exists.
        os.makedirs(output_folder, exist_ok=True)
        
        # Create a new Sphinx project in quiet mode.
        subprocess.run([
            "sphinx-quickstart",
            output_folder,
            "--quiet",
            "--project", "GeneInsight",
            "--author", "",
            "--sep",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Created Sphinx project in {output_folder}")
        
        # Define the source directory.
        source_docs_dir = os.path.join(output_folder, "source")
        
        # Copy RST files.
        for rst_file in glob.glob(os.path.join(rst_folder, "*.*")):
            shutil.copy(rst_file, source_docs_dir)
        logging.info(f"Copied RST files from {rst_folder} to {source_docs_dir}")
        
        # Copy image files.
        for img_file in glob.glob(os.path.join(image_folder, "*.*")):
            shutil.copy(img_file, source_docs_dir)
        logging.info(f"Copied image files from {image_folder} to {source_docs_dir}")
        
        # Copy the HTML embedding file.
        shutil.copy(html_embedding_path, source_docs_dir)
        logging.info(f"Copied HTML embedding file from {html_embedding_path} to {source_docs_dir}")
        
        # Copy the logo image into the _static folder.
        static_dir = os.path.join(source_docs_dir, "_static")
        os.makedirs(static_dir, exist_ok=True)
        shutil.copy(logo_path, static_dir)
        logging.info(f"Copied logo from {logo_path} to {static_dir}")
        
        # Generate index.rst with the updated logic.
        generate_index_rst(source_docs_dir)
        
        # Update conf.py with additional configuration.
        update_conf_py(source_docs_dir, logo_path)
        
        # Build the HTML documentation.
        build_dir = os.path.join(output_folder, "build", "html")
        result = subprocess.run([
            "sphinx-build",
            "-b", "html",
            source_docs_dir,
            build_dir,
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Write the build output to the log file.
        with open(log_path, 'w') as f:
            f.write(result.stdout.decode())
            f.write("\n\n")
            f.write(result.stderr.decode())
        
        logging.info(f"Built Sphinx documentation at {build_dir}")
        logging.info(f"Log saved to {log_path}")
        
        return True
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error building Sphinx documentation: {e}")
        with open(log_path, 'w') as f:
            f.write(f"Error: {str(e)}\n")
            if e.stdout:
                f.write("\nSTDOUT:\n")
                f.write(e.stdout.decode())
            if e.stderr:
                f.write("\nSTDERR:\n")
                f.write(e.stderr.decode())
        return False
    
    except Exception as e:
        logging.error(f"Error building Sphinx documentation: {e}")
        with open(log_path, 'w') as f:
            f.write(f"Error: {str(e)}\n")
        return False

def main():
    """
    Parse command-line arguments and run the Sphinx documentation build.
    """
    parser = argparse.ArgumentParser(
        description="Create and build a Sphinx documentation project for GeneInsight."
    )
    parser.add_argument("--rst_folder", required=True, help="Folder containing RST files")
    parser.add_argument("--image_folder", required=True, help="Folder containing image files")
    parser.add_argument("--output_folder", required=True, help="Folder to create the Sphinx project")
    parser.add_argument("--log_path", default="sphinx_build.log", help="Path to the log file (default: sphinx_build.log)")
    parser.add_argument("--html_embedding_path", required=True, help="Path to the HTML file to embed")
    parser.add_argument("--logo_path", required=True, help="Path to the logo image")
    args = parser.parse_args()
    
    # Configure logging to print timestamps to both console and log file.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_path),
            logging.StreamHandler()
        ]
    )
    
    success = build_sphinx_docs(
        args.rst_folder,
        args.image_folder,
        args.output_folder,
        args.log_path,
        args.html_embedding_path,
        args.logo_path
    )
    
    if not success:
        logging.error("Sphinx documentation build failed.")
        exit(1)
    else:
        logging.info("Sphinx documentation build completed successfully.")

if __name__ == "__main__":
    main()