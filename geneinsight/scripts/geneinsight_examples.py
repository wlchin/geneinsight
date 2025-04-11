#!/usr/bin/env python
import os
import shutil
import argparse
from pathlib import Path
from importlib import resources
from rich.console import Console

console = Console()

def create(dest_path):
    """Create an examples folder and copy sample files from package data."""
    if dest_path is None:
        dest_path = Path.cwd()
    else:
        dest_path = Path(dest_path).resolve()
    
    # Create examples folder
    examples_path = dest_path / "examples"
    examples_path.mkdir(exist_ok=True)
    console.print(f"[green]Created examples folder at {examples_path}[/green]")
    
    # Define files to copy
    sample_files = [
        "sample.txt",
        "sample_background.txt",
        "README.md"
    ]
    
    # Copy sample files using importlib.resources
    try:
        # This approach works for Python 3.9+
        with resources.path("geneinsight.examples", ".") as data_path:
            for file_name in sample_files:
                src_file = data_path / file_name
                if src_file.exists():
                    shutil.copy2(src_file, examples_path / file_name)
                    console.print(f"[green]Copied {file_name} to examples folder[/green]")
                else:
                    console.print(f"[yellow]Warning: {file_name} not found in package data[/yellow]")
    except (ModuleNotFoundError, FileNotFoundError) as e:
        console.print(f"[red]Error accessing package data: {str(e)}[/red]")
        console.print("[yellow]Make sure you have a 'data' directory in your package with sample files.[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Create examples folder with sample files")
    parser.add_argument(
        "--path", "-p", 
        dest="dest_path",
        type=Path,
        help="Path where to create examples folder (default: current directory)"
    )
    
    args = parser.parse_args()
    create(args.dest_path)

if __name__ == "__main__":
    main()
