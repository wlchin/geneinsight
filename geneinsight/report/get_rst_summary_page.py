import json
import argparse
import os

def generate_rst(file_name, json_data, html_basename):
    """
    Generate an RST file with three sections: Embedding map, Summary statistics, and Data Files.

    Args:
        file_name (str): Name of the output .rst file.
        json_data (dict): Dictionary containing summary statistics.
        html_basename (str): Basename of the HTML file to be embedded.
    """
    # Generate table from JSON data
    list_table_header = """
.. list-table:: 
   :header-rows: 1

   * - Statistic
     - Value
"""
    glossary = {
        "number_of_genes_considered": "The number of genes from the query gene set that have a relevant String API return response.",
        "documents_considered": "The total number of documents across all StringDB references returned by the API calls.",
        "average_topics": "The number of potential themes elicited through the StringDB references in one BERT topic run.",
        "range_of_max_topics": "The average number of topics across multiple samples of BERT topic runs.",
        "api_calls_made": "The number of API calls to either the OpenAI or TogetherAI API.",
        "number_of_themes_after_filtering": "The number of large language model or AI-induced gene sets after enrichment analysis.",
        "number_of_clusters": "The automatically optimized number of clusters for the filtered themes.",
        "compression_ratio": "The amount of compression of biological themes as a ratio of documents considered versus the number of themes.",
        "time_of_analysis": "The time of the analysis in Linux time, representing when the analysis occurred."
    }

    table_rows = []
    for key, value in json_data.items():
        row = f"   * - :abbr:`{key.replace('_', ' ').capitalize()} ({glossary.get(key, '')})`\n     - {value}"
        table_rows.append(row)

    joined_rows = "\n".join(table_rows)

    # Define content for the RST file
    content = f"""
=================
Embedding map
=================

This figure is a two-dimensional map where each theme heading is positioned based on the combined semantic information from the main topics and all enriched gene set themes, allowing us to see how similar or different the themes are in meaning. Headings that appear closer together share more common content, helping to visually identify clusters of related biological themes.

.. raw:: html
   :file: {html_basename}

======================
Summary statistics
======================

The following table provides a summary of the key statistics derived from the analysis:

{list_table_header}
{joined_rows}

=================
Data Files
=================

The enriched gene sets represent all AI-generated themes, without condensing and filtering. The subheading data represents a detailed breakdown and information of all API calls for subheading generation.

* :download:`enriched_genesets.csv <enriched_genesets.csv>`
* :download:`subheading_data.csv <subheading_data.csv>`

"""
    # Write content to the specified file
    with open(file_name, 'w') as file:
        file.write(content)

def main():
    parser = argparse.ArgumentParser(description="Generate an RST file for Sphinx documentation.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--file_name", required=True, help="Name of the output .rst file.")
    parser.add_argument("--json", required=True, help="Path to the JSON file containing summary statistics.")
    parser.add_argument("--html", required=True, help="Path to the HTML file to embed in the iframe.")

    args = parser.parse_args()

    # Load JSON data from the specified file
    with open(args.json, 'r') as json_file:
        json_data = json.load(json_file)

    # Get the basename of the HTML file
    html_basename = os.path.basename(args.html)

    # Generate the RST file
    output_path = os.path.join(args.output_folder, args.file_name)
    generate_rst(output_path, json_data, html_basename)

    print(f"RST file '{output_path}' has been generated successfully.")

if __name__ == "__main__":
    main()
