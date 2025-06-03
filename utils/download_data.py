import os
import zipfile
from pathlib import Path

import gdown


def download_data():
    """
    Downloads the dataset from Google Drive, extracts it, and removes the zip file.
    """
    current_dir = Path.cwd()
    target_dir = current_dir / "data"

    file_id = "0B8-rUzbwVRk0c054eEozWG9COHM"
    resource_key = "0-8nyl7K9_x37HlQm34MmrYQ"
    destination = target_dir / "dataset.zip"

    url = f"https://drive.google.com/uc?id={file_id}&resourcekey={resource_key}"

    print(f"Downloading dataset to {target_dir}...")
    gdown.download(url, destination, quiet=False)

    if not destination.exists():
        print("Error: Failed to download the dataset.")
        return False

    print(f"Dataset downloaded successfully to {destination}")

    # Extract the zip file
    try:
        with zipfile.ZipFile(destination, "r") as zip_ref:
            print(f"Extracting dataset to {target_dir}...")
            zip_ref.extractall(target_dir)
        print("Dataset extracted successfully.")

        # Remove the zip file after extraction
        os.remove(destination)
        print("Removed the zip file.")
    except zipfile.BadZipFile:
        print("The downloaded file is not a valid zip file.")
        return False
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False

    return True


def download_additional_files():
    """
    Downloads the additional files from Google Drive, extracts it, and removes the zip file.
    """
    current_dir = Path.cwd()
    target_dir = current_dir / "data"

    destination = target_dir / "additional.zip"

    url = "https://drive.google.com/uc?export=download&id=1Yb5bl01yUKtI7v4F1Q3AiNSL68rtqHry"

    print(f"Downloading additional files to {target_dir}...")
    gdown.download(url, destination, quiet=False)

    if not destination.exists():
        print("Error: Failed to download the additional files.")
        return False

    print(f"Additional files downloaded successfully to {destination}")

    # Extract the zip file
    try:
        with zipfile.ZipFile(destination, "r") as zip_ref:
            print(f"Extracting additional files to {target_dir}...")
            zip_ref.extractall(target_dir)
        print("Dataset extracted successfully.")

        # Remove the zip file after extraction
        os.remove(destination)
        print("Removed the zip file.")
    except zipfile.BadZipFile:
        print("The downloaded file is not a valid zip file.")
        return False
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False

    return True


if __name__ == "__main__":
    download_data()
    download_additional_files()
