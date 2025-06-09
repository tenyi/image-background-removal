import subprocess
import os
import pytest
from PIL import Image
import numpy as np

# Define test file names
TEST_INPUT_IMAGE = "input.png"  # Assuming this file exists from previous steps or is added
TEST_OUTPUT_IMAGE = "test_output.png"
NON_EXISTENT_INPUT_IMAGE = "non_existent_input.png"

@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """
    Ensure the input image is available for tests.
    In a real CI, this image would be part of the repo or downloaded.
    For this environment, we assume 'input.png' (Lena) was downloaded in a previous task.
    If not, this is a good place to download it.
    """
    if not os.path.exists(TEST_INPUT_IMAGE):
        # Attempt to download if not present (useful for standalone test runs)
        try:
            print(f"Test input image {TEST_INPUT_IMAGE} not found. Attempting download...")
            subprocess.run(
                ["wget", "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png", "-O", TEST_INPUT_IMAGE],
                check=True, capture_output=True, text=True
            )
            print(f"Downloaded {TEST_INPUT_IMAGE} successfully.")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Failed to download test input image {TEST_INPUT_IMAGE}: {e.stderr}")
        except FileNotFoundError:
            pytest.fail("wget command not found. Please ensure wget is installed if input.png is not present.")

    yield

    # Teardown: Clean up generated files
    if os.path.exists(TEST_OUTPUT_IMAGE):
        os.remove(TEST_OUTPUT_IMAGE)
    # if os.path.exists(TEST_INPUT_IMAGE_COPY_FOR_ERROR_TEST): # If we were making copies
    #     os.remove(TEST_INPUT_IMAGE_COPY_FOR_ERROR_TEST)


def run_script(args_list):
    """Helper function to run the background_remover.py script."""
    return subprocess.run(["python", "background_remover.py"] + args_list, capture_output=True, text=True)

def test_basic_execution_and_output_creation():
    """Test if the script runs and creates an output file."""
    result = run_script([TEST_INPUT_IMAGE, TEST_OUTPUT_IMAGE])

    assert result.returncode == 0, f"Script execution failed: {result.stderr}"
    assert os.path.exists(TEST_OUTPUT_IMAGE), "Output file was not created."

def test_output_image_properties():
    """Test the output image for transparency and content."""
    # Ensure the script runs first to generate the output
    result = run_script([TEST_INPUT_IMAGE, TEST_OUTPUT_IMAGE])
    if result.returncode != 0 or not os.path.exists(TEST_OUTPUT_IMAGE):
        pytest.fail(f"Prerequisite script run failed or output missing: {result.stderr}")

    output_image = Image.open(TEST_OUTPUT_IMAGE)
    assert output_image.mode == 'RGBA', "Output image is not in RGBA mode (no alpha channel)."

    alpha = output_image.getchannel('A')
    alpha_data = np.array(alpha)

    assert np.any(alpha_data == 0), "Output image has no fully transparent pixels. Background might not be removed."
    assert np.any(alpha_data == 255), "Output image has no fully opaque pixels. Foreground might be missing."
    # Check if there's a mix of transparency, not all one or the other
    assert not np.all(alpha_data == 0), "Output image is entirely transparent."
    assert not np.all(alpha_data == 255), "Output image is entirely opaque (no transparency)."


def test_error_handling_non_existent_input():
    """Test error handling for a non-existent input file."""
    # Ensure the output file from a previous test doesn't exist
    if os.path.exists(TEST_OUTPUT_IMAGE):
        os.remove(TEST_OUTPUT_IMAGE)

    result = run_script([NON_EXISTENT_INPUT_IMAGE, TEST_OUTPUT_IMAGE])

    assert result.returncode != 0, "Script should exit with a non-zero status code for missing input."
    # Check for the specific error message from background_remover.py
    assert "Error: Input file not found" in result.stdout or "Error: Input file not found" in result.stderr
    assert not os.path.exists(TEST_OUTPUT_IMAGE), "Output file should not be created on input error."

if __name__ == "__main__":
    pytest.main()
