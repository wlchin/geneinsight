import os
import zipfile
import tempfile
import pytest

from geneinsight.utils.zip_helper import zip_directory

@pytest.fixture
def sample_dir_structure():
    """
    Create a temporary directory structure with files and subdirectories.
    Some files are named to match the default ignore patterns.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a directory to be zipped, e.g. "test_dir"
        test_dir = os.path.join(temp_dir, "test_dir")
        os.makedirs(test_dir)
        
        # Create regular files
        file1 = os.path.join(test_dir, "file1.txt")
        file2 = os.path.join(test_dir, "file2.txt")
        with open(file1, "w") as f:
            f.write("Content of file1")
        with open(file2, "w") as f:
            f.write("Content of file2")
        
        # Create a subdirectory with a file inside
        sub_dir = os.path.join(test_dir, "subdir")
        os.makedirs(sub_dir)
        file3 = os.path.join(sub_dir, "file3.txt")
        with open(file3, "w") as f:
            f.write("Content of file3")
        
        # Create files that match default ignore patterns
        ds_store = os.path.join(test_dir, ".DS_Store")
        thumbs_db = os.path.join(test_dir, "Thumbs.db")
        with open(ds_store, "w") as f:
            f.write("Should be ignored")
        with open(thumbs_db, "w") as f:
            f.write("Should be ignored")
        
        yield test_dir

def test_zip_directory_creates_zip(sample_dir_structure):
    """
    Test that zip_directory creates a zip file with the expected content,
    ignoring files that match the default ignore patterns.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_zip = os.path.join(temp_dir, "output.zip")
        zip_path = zip_directory(sample_dir_structure, output_zip)
        
        # Check that the zip file was created
        assert os.path.exists(zip_path)
        
        # Open the created zip file and verify its contents
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            archived_files = set(zipf.namelist())
            
            # Given the implementation, the zip stores files with their path relative
            # to the parent directory of sample_dir_structure. For example, if sample_dir_structure
            # is "/tmp/.../test_dir", then the archived files will have a prefix "test_dir/"
            base_dir = os.path.basename(sample_dir_structure)
            expected_files = {
                os.path.join(base_dir, "file1.txt"),
                os.path.join(base_dir, "file2.txt"),
                os.path.join(base_dir, "subdir", "file3.txt"),
            }
            
            # Assert that only the expected files are in the archive
            assert archived_files == expected_files

def test_custom_ignore_patterns(sample_dir_structure):
    """
    Test that providing a custom ignore pattern works.
    Here, we add an extra file that should be ignored when a custom ignore pattern is provided.
    """
    # Create an extra file that we intend to ignore using a custom pattern.
    extra_file = os.path.join(sample_dir_structure, "ignore_me.txt")
    with open(extra_file, "w") as f:
        f.write("This file should be ignored")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_zip = os.path.join(temp_dir, "custom_output.zip")
        # Pass a custom ignore pattern that will ignore files starting with "ignore_me"
        zip_path = zip_directory(sample_dir_structure, output_zip, ignore_patterns=["ignore_me"])
        
        # Open the zip file and check its contents
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            archived_files = set(zipf.namelist())
            base_dir = os.path.basename(sample_dir_structure)
            expected_files = {
                os.path.join(base_dir, "file1.txt"),
                os.path.join(base_dir, "file2.txt"),
                os.path.join(base_dir, "subdir", "file3.txt"),
            }
            # The extra file "ignore_me.txt" should not be included
            assert archived_files == expected_files
