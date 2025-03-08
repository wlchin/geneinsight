import os
import json
import argparse
import base64

def parse_csv_folder(folder_path):
    theme_dict = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            theme_name = os.path.splitext(filename)[0]  # derive theme name
            csv_file_path = os.path.join(folder_path, filename)
            with open(csv_file_path, 'rb') as f:
                encoded_contents = base64.b64encode(f.read()).decode('utf-8')
                theme_dict[theme_name] = encoded_contents
    return theme_dict

def generate_rst(folder_path):
    theme_dict = parse_csv_folder(folder_path)

    # Build a list that captures (numericTheme, originalName, displayLabel, base64Data)
    theme_list = []
    for name, encoded in theme_dict.items():
        # Expect filename like "Theme_1_Something_Else_"
        parts = name.split('_', 2)  # ["Theme", "1", "Something_Else_"]
        num_str = parts[1] if len(parts) > 1 else "0"
        try:
            theme_num = int(num_str)
        except ValueError:
            theme_num = 0

        remainder = parts[2] if len(parts) > 2 else ""
        # Remove trailing underscore if present
        if (remainder.endswith("_")):
            remainder = remainder[:-1]
        # Replace underscores for nice display
        remainder_for_label = remainder.replace("_", " ")
        display_label = f"Theme {theme_num}: {remainder_for_label}"
        theme_list.append((theme_num, name, display_label, encoded))

    # Sort by numeric theme order
    theme_list.sort(key=lambda x: x[0])

    # Create a dict to hold base64 data keyed by original name
    all_files_dict = {t[1]: t[3] for t in theme_list}

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
    return rst_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RST file with download links for CSV themes.")
    parser.add_argument('--target_folder', required=True, help="The path to the folder containing CSV files.")
    parser.add_argument('--output_file_rst', required=True, help="The path to the output RST file.")
    args = parser.parse_args()

    rst_content = generate_rst(args.target_folder)
    with open(args.output_file_rst, "w") as file:
        file.write(rst_content)
