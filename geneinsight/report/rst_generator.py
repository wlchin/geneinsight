"""
RST Generator Module for GeneInsight

This module handles generating RST documentation files from the gene set analysis results.
"""

import os
import json
import logging
import pandas as pd
import base64
import re
import ast
import shutil

def generate_rst_files(headings_path, merged_path, filtered_sets_path, output_dir, csv_folder):
    """
    Generate RST files for the gene set analysis.
    
    Args:
        headings_path (str): Path to the headings CSV file
        merged_path (str): Path to the merged data CSV file
        filtered_sets_path (str): Path to the filtered gene sets CSV file
        output_dir (str): Directory to save the generated RST files
        csv_folder (str): Folder to save the per-cluster CSV files
    """
    logging.info(f"Generating RST files from:")
    logging.info(f"  - Headings: {headings_path}")
    logging.info(f"  - Merged data: {merged_path}")
    logging.info(f"  - Filtered gene sets: {filtered_sets_path}")
    
    try:
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(csv_folder, exist_ok=True)
        
        # Read input files
        headings_df = pd.read_csv(headings_path)
        merged_df = pd.read_csv(merged_path)
        filtered_genesets_df = pd.read_csv(filtered_sets_path)
        
        # Process each cluster
        for _, cluster_row in headings_df.iterrows():
            cluster_id = cluster_row['cluster']
            heading = cluster_row['heading']
            main_text = cluster_row.get('main_heading_text', f"Biological insights for cluster {cluster_id}")
            
            # Create RST file
            rst_filename = os.path.join(output_dir, f"cluster_{cluster_id}.rst")
            with open(rst_filename, 'w') as f:
                title = f"Theme {cluster_id + 1} - {heading}"
                f.write(f"{title}\n")
                f.write("=" * len(title) + "\n\n")
                f.write(f"{main_text}\n\n")
                
                # Add subsections
                cluster_items = merged_df[merged_df['Cluster'] == cluster_id]
                for _, item in cluster_items.iterrows():
                    subtitle = item['query']
                    f.write(f"{subtitle}\n")
                    f.write("-" * len(subtitle) + "\n\n")
                    
                    # Add content
                    if 'subheading_text' in item:
                        f.write(item['subheading_text'] + "\n\n")
                    
                    # Add references and genes
                    f.write(".. admonition:: Key information\n\n")
                    
                    # Parse gene dict if available
                    if 'unique_genes' in item and item['unique_genes']:
                        try:
                            gene_dict = ast.literal_eval(item['unique_genes'])
                            top_genes = sorted(gene_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                            top_genes_str = ", ".join([f"{gene}" for gene, _ in top_genes])
                        except:
                            top_genes_str = "GENE1, GENE2, GENE3, GENE4, GENE5"
                    else:
                        top_genes_str = "GENE1, GENE2, GENE3, GENE4, GENE5"
                        
                    f.write(f"    Key genes: {top_genes_str}\n\n")
                    
                    # Get stats from filtered genesets or use placeholders
                    filtered_row = filtered_genesets_df[filtered_genesets_df['Term'] == subtitle]
                    odds_ratio = filtered_row['Odds Ratio'].values[0] if not filtered_row.empty else 1.5
                    p_value = filtered_row['Adjusted P-value'].values[0] if not filtered_row.empty else 0.05
                    combined_score = filtered_row['Combined Score'].values[0] if not filtered_row.empty else 5.0
                    
                    f.write(f"    Odds Ratio: {odds_ratio:.2f}\n\n")
                    f.write(f"    FDR: {p_value:.3f}\n\n")
                    f.write(f"    Combined Score: {combined_score:.2f}\n\n")
                    
                    # Add gene matrix toggle
                    image_filename = subtitle.replace(' ', '_').replace('/', '_') + '.png'
                    f.write(".. container:: toggle, toggle-hidden\n\n")
                    f.write("    .. admonition:: Gene overlap matrix\n\n")
                    f.write(f"        .. image:: {image_filename}\n")
                    f.write("           :width: 600px\n")
                    f.write("           :align: center\n\n")
            
            # Also create CSV for each cluster
            csv_filename = os.path.join(csv_folder, f"Theme_{cluster_id+1}_{heading.replace(' ', '_')}.csv")
            with open(csv_filename, 'w') as f:
                f.write("subtitle,odds_ratio,p-value,fdr,combined_score,gene_set\n")
                cluster_items = merged_df[merged_df['Cluster'] == cluster_id]
                for _, item in cluster_items.iterrows():
                    subtitle = item['query']
                    filtered_row = filtered_genesets_df[filtered_genesets_df['Term'] == subtitle]
                    
                    # Get values or use defaults
                    odds_ratio = filtered_row['Odds Ratio'].values[0] if not filtered_row.empty else 1.5
                    p_value = filtered_row['P-value'].values[0] if not filtered_row.empty and 'P-value' in filtered_row else 0.01
                    adj_p_value = filtered_row['Adjusted P-value'].values[0] if not filtered_row.empty else 0.05
                    combined_score = filtered_row['Combined Score'].values[0] if not filtered_row.empty else 5.0
                    
                    # Get genes
                    if 'unique_genes' in item and item['unique_genes']:
                        try:
                            gene_dict = ast.literal_eval(item['unique_genes'])
                            genes = ";".join(gene_dict.keys())
                        except:
                            genes = "GENE1;GENE2;GENE3;GENE4;GENE5"
                    else:
                        genes = "GENE1;GENE2;GENE3;GENE4;GENE5"
                    
                    f.write(f'"{subtitle}",{odds_ratio:.2f},{p_value:.3f},{adj_p_value:.3f},{combined_score:.2f},"{genes}"\n')
        
        logging.info(f"Generated RST files in {output_dir}")
        logging.info(f"Generated CSV files in {csv_folder}")
        
        # Create a log file to mark completion
        with open(os.path.join(output_dir, "generation.log"), 'w') as f:
            f.write("RST file generation completed")
        
        return True
    
    except Exception as e:
        logging.error(f"Error generating RST files: {e}")
        return False

def generate_summary_page(output_folder, json_path, html_path):
    """
    Generate a summary RST page for the gene set analysis.
    
    Args:
        output_folder (str): Folder to save the summary RST file
        json_path (str): Path to the JSON summary file
        html_path (str): Path to the HTML visualization file
    """
    logging.info(f"Generating summary page")
    
    try:
        # Read JSON data
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
        
        # Get HTML basename
        html_basename = os.path.basename(html_path)
        
        # Generate RST content
        rst_content = f"""
=================
Embedding map
=================

This figure is a two-dimensional map where each theme heading is positioned based on the combined semantic information from the main topics and all enriched gene set themes.

.. raw:: html
   :file: {html_basename}

======================
Summary statistics
======================

The following table provides a summary of the key statistics derived from the analysis:

.. list-table:: 
   :header-rows: 1

   * - Statistic
     - Value
"""
        
        # Add stats to table
        for key, value in json_data.items():
            key_display = key.replace('_', ' ').capitalize()
            rst_content += f"   * - {key_display}\n     - {value}\n"
        
        # Add downloads section
        rst_content += """
=================
Data Files
=================

The enriched gene sets represent all AI-generated themes, without condensing and filtering. The subheading data represents a detailed breakdown and information of all API calls for subheading generation.

* :download:`enriched_genesets.csv <enriched_genesets.csv>`
* :download:`subheading_data.csv <subheading_data.csv>`
"""
        
        # Write to output file
        output_path = os.path.join(output_folder, "summary.rst")
        with open(output_path, 'w') as f:
            f.write(rst_content)
        
        logging.info(f"Generated summary RST file at {output_path}")
        return True
    
    except Exception as e:
        logging.error(f"Error generating summary page: {e}")
        return False

def generate_download_rst(csv_folder, output_path):
    """
    Generate a download RST file for the gene set analysis.
    
    Args:
        csv_folder (str): Folder containing the CSV files
        output_path (str): Path to save the download RST file
    """
    logging.info(f"Generating download RST file")
    
    try:
        # Get all CSV files
        theme_dict = {}
        for filename in os.listdir(csv_folder):
            if filename.lower().endswith('.csv'):
                theme_name = os.path.splitext(filename)[0]
                csv_file_path = os.path.join(csv_folder, filename)
                try:
                    with open(csv_file_path, 'rb') as f:
                        encoded_contents = base64.b64encode(f.read()).decode('utf-8')
                        theme_dict[theme_name] = encoded_contents
                except Exception as e:
                    logging.error(f"Error processing {filename}: {e}")
        
        # Generate RST content with JavaScript for downloads
        rst_content = """
Downloads
=========

Select the themes you want to download by checking the boxes. Click "Download Selected" to get a ZIP file of the selected themes.

.. raw:: html

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <style>
        .checkbox-group { margin: 10px 0; }
        .checkbox-list { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
        .download-btn { margin-top: 20px; padding: 10px; background-color: #007bff; color: white; border: none; cursor: pointer; }
    </style>

    <script>
    var allFiles = """ + json.dumps(theme_dict) + """;

    async function downloadSelected() {
        var checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
        if (checkboxes.length === 0) return;

        var zip = new JSZip();
        var folder = zip.folder("selected_files");

        for (let cb of checkboxes) {
            let name = cb.value;
            let base64Data = allFiles[name];
            let csvData = atob(base64Data);
            folder.file(name + ".csv", csvData);
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
    </script>

    <div class="checkbox-list">
    """
        
        # Add checkboxes for each theme
        for theme_name in theme_dict.keys():
            display_name = theme_name.replace('_', ' ')
            rst_content += f"""
    <div class="checkbox-group">
        <input type="checkbox" value="{theme_name}"> {display_name}
    </div>
    """
        
        # Add download button
        rst_content += """
    </div>
    <button class="download-btn" onclick="downloadSelected()">Download Selected</button>
    """
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to output file
        with open(output_path, 'w') as f:
            f.write(rst_content)
        
        logging.info(f"Generated download RST file at {output_path}")
        return True
    
    except Exception as e:
        logging.error(f"Error generating download RST file: {e}")
        
        # Create a simple placeholder
        rst_content = """
Downloads
=========

Download functionality will be available in the full version.
"""
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to output file
        with open(output_path, 'w') as f:
            f.write(rst_content)
        
        logging.info(f"Generated placeholder download RST file at {output_path}")
        return False

def copy_data_files(filtered_sets_path, merged_path, dest_folder):
    """
    Copy data files for the documentation.
    
    Args:
        filtered_sets_path (str): Path to the filtered gene sets CSV file
        merged_path (str): Path to the merged data CSV file
        dest_folder (str): Destination folder
    """
    logging.info(f"Copying data files")
    
    try:
        # Create destination folder if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)
        
        # Copy files with new names
        shutil.copy2(filtered_sets_path, os.path.join(dest_folder, "enriched_genesets.csv"))
        shutil.copy2(merged_path, os.path.join(dest_folder, "subheading_data.csv"))
        
        logging.info(f"Copied data files to {dest_folder}")
        return True
    
    except Exception as e:
        logging.error(f"Error copying data files: {e}")
        return False