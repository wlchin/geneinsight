#!/usr/bin/env python3

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
    with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
        file.write(f"{line}\n")


def setup_sphinx_project(project_dir, external_docs_dir, image_dir, html_embedding_file, logo_path):
    """
    Set up a Sphinx project by creating necessary files and directories,
    copying external documentation, generating an index.rst, and building
    the HTML documentation so that Sphinx actually uses the newly generated
    index.rst file.

    Args:
        project_dir (str): The directory path where the Sphinx project will be created.
        external_docs_dir (str): The directory path where the external documentation files are located.
        image_dir (str): The directory path where the image files are located.
        html_embedding_file (str): The path to the HTML file to be embedded.
        logo_path (str): The path to the logo image that will be copied into _static.

    Raises:
        subprocess.CalledProcessError: If any of the subprocess commands fail.
    """

    logging.info(f"Ensuring the project directory exists: {project_dir}")
    os.makedirs(project_dir, exist_ok=True)

    logging.info("Creating a new Sphinx project in quiet mode...")
    subprocess.run(
        [
            "sphinx-quickstart",
            project_dir,
            "--quiet",
            "--project",
            "GeneInsight",
            "--author",
            "",
            "--sep",
        ],
        check=True,
    )

    logging.info(f"Copying documentation files from {external_docs_dir} to {project_dir}/source")
    source_docs_dir = os.path.join(project_dir, "source")
    for rst_file in glob.glob(os.path.join(external_docs_dir, "*.*")):
        shutil.copy(rst_file, source_docs_dir)

    logging.info(f"Copying image files from {image_dir} to {source_docs_dir}")
    for img_file in glob.glob(os.path.join(image_dir, "*.*")):
        shutil.copy(img_file, source_docs_dir)

    # Copy the HTML embedding file to the source directory
    logging.info(f"Copying HTML embedding file from {html_embedding_file} to {source_docs_dir}")
    shutil.copy(html_embedding_file, source_docs_dir)

    # Copy the logo image to the _static directory
    static_dir = os.path.join(source_docs_dir, "_static")
    os.makedirs(static_dir, exist_ok=True)
    shutil.copy(logo_path, static_dir)

    # ------------------------------------------------------------
    # Generate the index.rst file INSIDE the source directory
    # ------------------------------------------------------------
    logging.info("Identifying doc*.rst files for potential index population...")
    doc_files = sorted(glob.glob(os.path.join(source_docs_dir, "*.rst")))
    doc_files = [doc for doc in doc_files if os.path.basename(doc) not in ["index.rst", "summary.rst", "download.rst"]]
    print("here are the doc files")
    print(doc_files)
    doc_files = [os.path.basename(doc).replace(".rst", "") for doc in doc_files]

    logging.info("Found the following doc*.rst files:")
    print(doc_files)

    index_content = [
        "GenesetInsight Topic Descriptions\n",
        "==================================\n\n",
        ".. toctree::\n",
        "   :maxdepth: 1\n",
        "   :caption: Major themes:\n",
        "\n",
    ]

    # Add each doc*.rst file to the toctree
    index_content.extend(f"   {doc}\n" for doc in doc_files)

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
    logging.info(f"Writing index.rst to {index_path}")
    with open(index_path, "w") as file:
        file.writelines(index_content)

    # ------------------------------------------------------------
    # Update conf.py with theme/extension settings
    # ------------------------------------------------------------
    logging.info("Appending Sphinx theme and extensions to conf.py...")
    conf_file_path = os.path.join(source_docs_dir, 'conf.py')
    append_line_to_file(conf_file_path, "\n")
    append_line_to_file(conf_file_path, "html_theme = 'sphinx_wagtail_theme'")
    append_line_to_file(
        conf_file_path,
        "extensions = ['sphinx_wagtail_theme', 'sphinx_togglebutton']"
    )
    append_line_to_file(conf_file_path, "html_theme_options = dict(")
    append_line_to_file(conf_file_path, "    project_name = \"GenesetInsight\",")
    append_line_to_file(conf_file_path, "    footer_links = \",\".join([")
    append_line_to_file(conf_file_path, "        \"Contact|http://example.com/contact\",")
    append_line_to_file(conf_file_path, "        \"Lassmann Group|http://example.com/dev/null\",")
    append_line_to_file(conf_file_path, "    ]),")
    append_line_to_file(conf_file_path, f"    logo = \"{os.path.basename(logo_path)}\",")
    append_line_to_file(conf_file_path, "    logo_alt = \"GenesetInsight\",")
    append_line_to_file(conf_file_path, "    logo_height = 59,")
    append_line_to_file(conf_file_path, "    logo_url = \"/\",")
    append_line_to_file(conf_file_path, "    logo_width = 45,")
    append_line_to_file(conf_file_path, ")")
    append_line_to_file(conf_file_path, "html_show_copyright = False")
    append_line_to_file(conf_file_path, "html_show_sphinx = False")
    # append_line_to_file(conf_file_path, "html_sidebars = {\"**\": [")
    # append_line_to_file(conf_file_path, "    \"searchbox.html\",")
    # append_line_to_file(conf_file_path, "    \"globaltoc.html\",")
    # append_line_to_file(conf_file_path, "]}")

    # ------------------------------------------------------------
    # Build the HTML documentation
    # ------------------------------------------------------------
    logging.info("Building HTML documentation with sphinx-build...")
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

    logging.info("HTML documentation built successfully; check build/html/index.html.")


def main():
    """
    Parse command-line arguments and run the Sphinx project setup.
    """
    parser = argparse.ArgumentParser(
        description="Create and build a Sphinx documentation project."
    )
    parser.add_argument(
        "--project_dir",
        required=True,
        help="Path to the directory where the Sphinx project will be created."
    )
    parser.add_argument(
        "--external_docs_dir",
        required=True,
        help="Path to the directory containing external documentation (.rst) files."
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Path to the directory containing image files."
    )
    parser.add_argument(
        "--html_embedding_file",
        required=True,
        help="Path to the HTML file to be embedded."
    )
    parser.add_argument(
        "--logo_path",
        required=True,
        help="Path to the logo image that will be copied into _static."
    )
    parser.add_argument(
        "--log_file",
        default="sphinx_build.log",
        help="Path to the log file. Defaults to 'sphinx_build.log'."
    )
    args = parser.parse_args()

    # Configure logging with timestamps and log to file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )

    try:
        setup_sphinx_project(
            args.project_dir,
            args.external_docs_dir,
            args.image_dir,
            args.html_embedding_file,
            logo_path=args.logo_path
        )
        logging.info("Sphinx project setup and build completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during Sphinx setup/build process: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()