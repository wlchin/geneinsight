"""
Sphinx Builder Module for GeneInsight

This module handles building Sphinx documentation from the generated RST files.
"""

import os
import glob
import shutil
import subprocess
import logging

def build_sphinx_docs(rst_folder, image_folder, output_folder, log_path, 
                      html_embedding_path, logo_path):
    """
    Build Sphinx documentation from RST files.
    
    Args:
        rst_folder (str): Folder containing RST files
        image_folder (str): Folder containing image files
        output_folder (str): Folder to save the Sphinx build
        log_path (str): Path to save the log file
        html_embedding_path (str): Path to the HTML file to embed
        logo_path (str): Path to the logo image
    """
    logging.info(f"Building Sphinx documentation")
    
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Create a new Sphinx project
        result = subprocess.run([
            "sphinx-quickstart",
            output_folder,
            "--quiet",
            "--project", "GeneInsight",
            "--author", "",
            "--sep",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logging.info(f"Created Sphinx project in {output_folder}")
        
        # Copy RST files to the source directory
        source_docs_dir = os.path.join(output_folder, "source")
        for rst_file in glob.glob(os.path.join(rst_folder, "*.*")):
            shutil.copy2(rst_file, source_docs_dir)
        
        logging.info(f"Copied RST files to {source_docs_dir}")
        
        # Copy image files
        for img_file in glob.glob(os.path.join(image_folder, "*.*")):
            shutil.copy2(img_file, source_docs_dir)
        
        logging.info(f"Copied image files to {source_docs_dir}")
        
        # Copy the HTML embedding file
        shutil.copy2(html_embedding_path, source_docs_dir)
        
        logging.info(f"Copied HTML embedding file to {source_docs_dir}")
        
        # Create _static directory and copy logo
        static_dir = os.path.join(source_docs_dir, "_static")
        os.makedirs(static_dir, exist_ok=True)
        shutil.copy2(logo_path, static_dir)
        
        logging.info(f"Copied logo to {static_dir}")
        
        # Generate the index.rst file
        generate_index_rst(source_docs_dir)
        
        # Update conf.py
        update_conf_py(source_docs_dir, os.path.basename(logo_path))
        
        # Build the HTML documentation
        build_dir = os.path.join(output_folder, "build", "html")
        result = subprocess.run([
            "sphinx-build",
            "-b", "html",
            source_docs_dir,
            build_dir,
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Save the output to the log file
        with open(log_path, 'w') as f:
            f.write(result.stdout.decode())
            f.write("\n\n")
            f.write(result.stderr.decode())
        
        logging.info(f"Built Sphinx documentation at {build_dir}")
        logging.info(f"Log saved to {log_path}")
        
        return True
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error building Sphinx documentation: {e}")
        
        # Save the error to the log file
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
        
        # Save the error to the log file
        with open(log_path, 'w') as f:
            f.write(f"Error: {str(e)}\n")
        
        return False

def generate_index_rst(source_dir):
    """
    Generate the index.rst file for the Sphinx documentation.
    
    Args:
        source_dir (str): Source directory of the Sphinx project
    """
    logging.info(f"Generating index.rst")
    
    # Identify doc*.rst files
    doc_files = sorted(glob.glob(os.path.join(source_dir, "cluster_*.rst")))
    doc_files = [os.path.basename(doc).replace(".rst", "") for doc in doc_files]
    
    # Create index content
    index_content = [
        "GeneInsight Topic Descriptions\n",
        "==================================\n\n",
        ".. toctree::\n",
        "   :maxdepth: 1\n",
        "   :caption: Major themes:\n",
        "\n",
    ]
    
    # Add each cluster file to the toctree
    for doc in doc_files:
        index_content.append(f"   {doc}\n")
    
    # Add metadata section
    metadata_content = """
.. toctree::
   :maxdepth: 1
   :caption: Metadata:

   summary
   download
"""
    index_content.append(metadata_content)
    
    # Write the index.rst file
    index_path = os.path.join(source_dir, "index.rst")
    with open(index_path, "w") as f:
        f.writelines(index_content)
    
    logging.info(f"Generated index.rst at {index_path}")

def update_conf_py(source_dir, logo_filename):
    """
    Update the conf.py file with theme and extension settings.
    
    Args:
        source_dir (str): Source directory of the Sphinx project
        logo_filename (str): Filename of the logo image
    """
    logging.info(f"Updating conf.py")
    
    conf_path = os.path.join(source_dir, "conf.py")
    
    # Read the original content
    with open(conf_path, "r") as f:
        content = f.read()
    
    # Add theme and extensions
    additional_content = f"""
# Theme configuration
html_theme = 'sphinx_rtd_theme'  # Using ReadTheDocs theme as fallback if wagtail not available
try:
    import sphinx_wagtail_theme
    html_theme = 'sphinx_wagtail_theme'
except ImportError:
    pass

# Extensions
extensions = []
try:
    import sphinx_wagtail_theme
    extensions.append('sphinx_wagtail_theme')
except ImportError:
    pass

try:
    import sphinx_togglebutton
    extensions.append('sphinx_togglebutton')
except ImportError:
    pass

# Theme options
html_theme_options = dict(
    project_name = "GenesetInsight",
    footer_links = ",".join([
        "Contact|http://example.com/contact",
        "Developers|http://example.com/dev/null",
    ]),
    logo = "{logo_filename}",
    logo_alt = "GenesetInsight",
    logo_height = 59,
    logo_url = "/",
    logo_width = 45,
)

# Other settings
html_show_copyright = False
html_show_sphinx = False
"""
    
    # Write the updated content
    with open(conf_path, "w") as f:
        f.write(content)
        f.write(additional_content)
    
    logging.info(f"Updated conf.py at {conf_path}")