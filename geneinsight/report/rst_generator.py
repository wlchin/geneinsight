"""
Module for generating reStructuredText (RST) files for Sphinx documentation.
"""

import os
import json
import logging
import shutil
import base64
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def generate_rst_files(
    headings_csv: str,
    merged_csv: str,
    output_dir: str,
    log_file: Optional[str] = None,
    filtered_genesets_csv: Optional[str] = None,
    csv_folder: Optional[str] = None
) -> None:
    """
    Generate RST files for each theme from the headings and merged data.
    
    Args:
        headings_csv: Path to CSV with theme headings
        merged_csv: Path to CSV with merged data (context and ontology)
        output_dir: Directory to write RST files
        log_file: Path to log file (optional)
        filtered_genesets_csv: Path to filtered gene sets CSV (optional)
        csv_folder: Folder to save theme CSV files (optional)
    """
    try:
        # Set up logging if log_file is provided
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(file_handler)
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Also create CSV folder if specified
        if csv_folder:
            os.makedirs(csv_folder, exist_ok=True)
            
        # Load headings data
        logger.info(f"Loading theme headings from {headings_csv}")
        headings_df = pd.read_csv(headings_csv)
        
        # Load merged data
        logger.info(f"Loading merged data from {merged_csv}")
        merged_df = pd.read_csv(merged_csv)
        
        # Load filtered gene sets if provided
        filtered_df = None
        if filtered_genesets_csv and os.path.exists(filtered_genesets_csv):
            logger.info(f"Loading filtered gene sets from {filtered_genesets_csv}")
            filtered_df = pd.read_csv(filtered_genesets_csv)
            
        # Process each theme (cluster)
        for _, row in headings_df.iterrows():
            cluster_id = row.get('cluster')
            heading = row.get('heading')
            
            if cluster_id is None or heading is None:
                logger.warning("Missing cluster ID or heading, skipping")
                continue
                
            # Filter merged_df for this cluster
            cluster_data = merged_df[merged_df['query'].apply(
                lambda q: q.startswith(f"Topic_{cluster_id}_")
            )]
            
            # Skip if no data for this cluster
            if len(cluster_data) == 0:
                logger.warning(f"No data found for cluster {cluster_id}, skipping")
                continue
                
            # Create RST file
            rst_file = os.path.join(output_dir, f"Theme_{cluster_id}_{heading.replace(' ', '_')}.rst")
            logger.info(f"Generating RST file: {rst_file}")
            
            # Generate RST content
            rst_content = generate_theme_rst(cluster_id, heading, cluster_data, filtered_df)
            
            # Write RST file
            with open(rst_file, "w") as f:
                f.write(rst_content)
                
            # Save cluster data to CSV if csv_folder is provided
            if csv_folder:
                csv_file = os.path.join(csv_folder, f"Theme_{cluster_id}_{heading.replace(' ', '_')}.csv")
                cluster_data.to_csv(csv_file, index=False)
                logger.info(f"Saved cluster data to {csv_file}")
                
        logger.info(f"Generated RST files for {len(headings_df)} themes")
        
    except Exception as e:
        logger.error(f"Error generating RST files: {e}")
        import traceback
        traceback.print_exc()

def generate_theme_rst(
    cluster_id: int,
    heading: str,
    cluster_data: pd.DataFrame,
    filtered_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate RST content for a single theme.
    
    Args:
        cluster_id: Cluster ID
        heading: Theme heading
        cluster_data: DataFrame with data for this cluster
        filtered_df: DataFrame with filtered gene sets (optional)
        
    Returns:
        RST content as string
    """
    # Create heading with proper RST underline
    title = f"Theme {cluster_id}: {heading}"
    title_underline = "=" * len(title)
    
    content = []
    content.append(title)
    content.append(title_underline)
    content.append("")
    
    # Add theme description
    content.append("Theme Description")
    content.append("-----------------")
    content.append("")
    content.append(f"{heading} is a theme identified through topic modeling and enrichment analysis.")
    content.append("")
    
    # Add subtopics section if we have data
    if len(cluster_data) > 0:
        content.append("Subtopics")
        content.append("---------")
        content.append("")
        content.append(".. list-table::")
        content.append("   :header-rows: 1")
        content.append("   :widths: 70 30")
        content.append("")
        content.append("   * - Subtopic")
        content.append("     - Genes")
        
        # Add each subtopic
        for _, row in cluster_data.iterrows():
            subtopic = row.get('generated_result', 'No title available')
            genes_dict = row.get('unique_genes', '{}')
            
            # Try to parse genes dictionary
            try:
                genes_dict = eval(genes_dict)
                genes_str = ", ".join(list(genes_dict.keys())[:5])
                if len(genes_dict) > 5:
                    genes_str += f" and {len(genes_dict) - 5} more"
            except:
                genes_str = "Error parsing genes"
                
            content.append(f"   * - {subtopic}")
            content.append(f"     - {genes_str}")
            
        content.append("")
    
    # Add ontology enrichment section if we have ontology data
    if 'ontology_dict' in cluster_data.columns:
        content.append("Ontology Enrichment")
        content.append("-------------------")
        content.append("")
        content.append("Ontologies that are enriched in this theme:")
        content.append("")
        
        # Get the first row's ontology dict (they should all be the same for a theme)
        ontology_str = cluster_data.iloc[0].get('ontology_dict', '{}')
        
        # Try to parse ontology dictionary
        try:
            ontology_dict = eval(ontology_str)
            
            content.append(".. list-table::")
            content.append("   :header-rows: 1")
            content.append("")
            content.append("   * - Ontology Term")
            content.append("     - Genes")
            
            # Add each ontology term
            for term, genes in ontology_dict.items():
                genes_list = genes.split(";")
                genes_display = ", ".join(genes_list[:3])
                if len(genes_list) > 3:
                    genes_display += f" and {len(genes_list) - 3} more"
                    
                content.append(f"   * - {term}")
                content.append(f"     - {genes_display}")
                
        except:
            content.append("Error parsing ontology data")
            
        content.append("")
    
    # Add gene set visualization if we have an image
    content.append("Gene Set Visualization")
    content.append("----------------------")
    content.append("")
    content.append(f".. image:: Theme_{cluster_id}_{heading.replace(' ', '_')}.png")
    content.append("   :width: 800px")
    content.append("   :alt: Gene set visualization")
    content.append("")
    
    return "\n".join(content)

def generate_download_rst(csv_folder: str, output_file: str) -> None:
    """
    Generate a download page with interactive download functionality.
    
    Args:
        csv_folder: Folder containing CSV files to make available for download
        output_file: Path to the output RST file
    """
    try:
        # Create the directory for the output file if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Parse CSV folder and encode files
        theme_dict = {}
        if os.path.exists(csv_folder):
            for filename in os.listdir(csv_folder):
                if filename.lower().endswith('.csv'):
                    theme_name = os.path.splitext(filename)[0]  # derive theme name
                    csv_file_path = os.path.join(csv_folder, filename)
                    with open(csv_file_path, 'rb') as f:
                        encoded_contents = base64.b64encode(f.read()).decode('utf-8')
                        theme_dict[theme_name] = encoded_contents
        
        # Build a list that captures (numericTheme, originalName, displayLabel, base64Data)
        theme_list = []
        for name, encoded in theme_dict.items():
            # Expect filename like "Theme_1_Something_Else"
            parts = name.split('_', 2)  # ["Theme", "1", "Something_Else"]
            
            if len(parts) < 2:
                continue
                
            num_str = parts[1] if len(parts) > 1 else "0"
            try:
                theme_num = int(num_str)
            except ValueError:
                theme_num = 0

            remainder = parts[2] if len(parts) > 2 else ""
            # Remove trailing underscore if present
            if remainder.endswith("_"):
                remainder = remainder[:-1]
            # Replace underscores for nice display
            remainder_for_label = remainder.replace("_", " ")
            display_label = f"Theme {theme_num}: {remainder_for_label}"
            theme_list.append((theme_num, name, display_label, encoded))

        # Sort by numeric theme order
        theme_list.sort(key=lambda x: x[0])

        # Create a dict to hold base64 data keyed by original name
        all_files_dict = {t[1]: t[3] for t in theme_list}
        
        # Generate RST content
        rst_content = """
Downloads
=========

Select the themes you want to download by checking the boxes. Click "Download Selected" to get a ZIP file of the selected themes.

.. raw:: html

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">

    <script>
    var allFiles = """ + json.dumps(all_files_dict) + """;

    function updateCounter() {
        var checkboxes = document.querySelectorAll('input[type="checkbox"]');
        var counter = document.getElementById('file-counter');
        var basketContainer = document.querySelector('.basket-container');
        var count = 0;

        checkboxes.forEach(cb => {
            var label = cb.parentElement;
            if (cb.checked) {
                label.style.fontWeight = 'bold';
                count++;
            } else {
                label.style.fontWeight = 'normal';
            }
        });

        if (count > 0) {
            basketContainer.style.display = "flex";
            counter.innerHTML = '<i class="fas fa-shopping-basket"></i> ' + count;
        } else {
            basketContainer.style.display = "none";
        }
    }

    async function downloadSelected() {
        var checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
        if (checkboxes.length === 0) return;

        var zip = new JSZip();
        var folder = zip.folder("selected_files");

        for (let cb of checkboxes) {
            // "name" is the original filename (minus `.csv`)
            let name = cb.value;
            let base64Data = allFiles[name];
            let csvData = atob(base64Data);

            // Remove trailing underscore before ".csv"
            let finalName = name.endsWith("_") ? name.slice(0, -1) + ".csv" : name + ".csv";
            let blob = new Blob([csvData], {type: 'text/csv'});
            folder.file(finalName, blob);
        }

        zip.generateAsync({ type: "blob" }).then(function (content) {
            let link = document.createElement("a");
            link.href = URL.createObjectURL(content);
            link.download = "selected_files.zip";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    function toggleSelectAll() {
        var checkboxes = document.querySelectorAll('input[type="checkbox"]');
        var selectAllBtn = document.getElementById('select-all-btn');
        var allSelected = Array.from(checkboxes).every(cb => cb.checked);

        checkboxes.forEach(cb => cb.checked = !allSelected);
        selectAllBtn.textContent = allSelected ? "Select All Themes" : "Deselect All Themes";
        updateCounter();
    }
    </script>

    <style>
        .checkbox-group {
            margin: 10px 0;
        }
        .checkbox-list {
            border: 1px solid #ccc;
            padding: 10px;
            margin: 10px 0;
        }
        .download-btn {
            margin-top: 20px;
            padding: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        .basket-container {
            display: none; /* Initially hidden */
            align-items: center;
            position: fixed;
            bottom: 10px;
            right: 10px;
        }
        .basket-icon {
            background-color: transparent;
            color: black;
            padding: 8px 12px;
            border-radius: 50%;
            font-size: 14px;
            cursor: pointer;
            text-align: center;
            font-weight: bold;
        }
        .basket-icon i {
            margin-right: 0px;
        }
        .basket-text {
            margin-left: 0px; /* Move closer to the basket */
            font-size: 14px;
            color: black;
        }
        .select-all-btn {
            margin-top: 20px;
            padding: 10px;
            background-color: #28a745;
            color: white;
            border: none;
            cursor: pointer;
        }
    </style>

    <div class="basket-container">
        <div id="file-counter" class="basket-icon" onclick="downloadSelected()">
            <i class="fas fa-shopping-basket"></i> 0
        </div>
        <div class="basket-text">Themes selected</div>
    </div>

    <div class="checkbox-list">
    """

        # Generate checkboxes in sorted order
        for _, original_name, display_label, _ in theme_list:
            rst_content += f"""
    <div class="checkbox-group">
        <input type="checkbox" value="{original_name}" onclick="updateCounter()"> {display_label.split(':')[0]} - {display_label.split(':')[1]}
    </div>
    """

        rst_content += """
    </div>
    <button id="select-all-btn" class="select-all-btn" onclick="toggleSelectAll()">Select All Themes</button>
    <button class="download-btn" onclick="downloadSelected()">Download Selected</button>
    """
        
        # Write RST file
        with open(output_file, "w") as file:
            file.write(rst_content)
            
        logger.info(f"Generated download RST file: {output_file}")
    
    except Exception as e:
        logger.error(f"Error generating download RST: {e}")
        import traceback
        traceback.print_exc()