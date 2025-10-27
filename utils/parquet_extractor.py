import json
import shutil
from pathlib import Path

import pandas as pd
from iotools import mkdir_if_missing, write_json
from tqdm import tqdm


def extract_cuhk_pedes_dataset(
    parquet_dir="data/CUHK-PEDES",
    output_dir="data/CUHK-PEDES",
    images_dir="images",
    metadata_file="metadata.json",
):
    """
    Extract images and text from CUHK-PEDES parquet files to create a normal dataset.

    Args:
        parquet_dir (str): Directory containing the parquet files
        output_dir (str): Directory to save the extracted dataset
        images_dir (str): Name of the subdirectory to save images
        metadata_file (str): Name of the metadata file

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        current_dir = Path.cwd()
        output_dir = current_dir / output_dir
        parquet_dir = current_dir / parquet_dir
        images_output_dir = output_dir / images_dir

        # Create output directories
        mkdir_if_missing(images_output_dir)

        print(f"Extracting CUHK-PEDES dataset from {parquet_dir} to {output_dir}")

        # Get all parquet files
        parquet_files = sorted([f for f in parquet_dir.glob("*.parquet")])

        print(f"Found {len(parquet_files)} parquet files")

        # Initialize metadata dictionary
        metadata = {
            "dataset": "CUHK-PEDES",
            "num_files": len(parquet_files),
            "samples": [],
        }

        # Process each parquet file
        total_samples = 0
        for parquet_file in parquet_files:
            print(f"Processing {parquet_file.name}")

            # Read parquet file
            df = pd.read_parquet(parquet_file)
            file_samples = len(df)
            total_samples += file_samples

            print(f"  - Contains {file_samples} samples")

            # Extract and save each image, and collect metadata
            for idx, row in tqdm(
                df.iterrows(), total=file_samples, desc="Extracting samples"
            ):
                # Get image path and binary data
                image_path = row["image"]["path"]
                image_bytes = row["image"]["bytes"]
                text = row["text"]

                # Save image to disk
                image_file_path = images_output_dir / image_path
                with open(image_file_path, "wb") as f:
                    f.write(image_bytes)

                # Add to metadata
                metadata["samples"].append(
                    {"image_path": f"{images_dir}/{image_path}", "text": text}
                )

        # Update metadata with total count
        metadata["total_samples"] = total_samples

        # Save metadata
        metadata_path = output_dir / metadata_file
        write_json(metadata, metadata_path)

        print(f"Extraction complete. Saved {total_samples} samples.")
        print(f"Images saved to: {images_output_dir}")
        print(f"Metadata saved to: {metadata_path}")

        return True

    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False


def filter_images_from_reid_json(
    reid_json_path="data/CUHK-PEDES/reid_raw.json",
    images_dir="data/CUHK-PEDES/images",
    output_dir="data/CUHK-PEDES/images_filtered",
    output_json_path="data/CUHK-PEDES/filtered_captions.json",
    output_id_mapping_path="data/CUHK-PEDES/image_id_mapping.json",
):
    """
    Filter images based on reid_raw.json and save them in a single directory.

    Args:
        reid_json_path (str): Path to the reid_raw.json file
        images_dir (str): Directory containing the source images
        output_dir (str): Directory to save all filtered images
        output_json_path (str): Path to save the captions mapping JSON
        output_id_mapping_path (str): Path to save the image name to ID mapping JSON

    Returns:
        bool: True if filtering was successful, False otherwise
    """
    try:
        current_dir = Path.cwd()
        reid_json_path = current_dir / reid_json_path
        images_dir = current_dir / images_dir
        output_dir = current_dir / output_dir
        output_json_path = current_dir / output_json_path
        output_id_mapping_path = current_dir / output_id_mapping_path

        print(f"Loading JSON metadata from {reid_json_path}")

        # Load the reid_raw.json file
        with open(reid_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} entries from JSON")

        # Create output directory
        mkdir_if_missing(output_dir)
        print(f"Created directory: {output_dir}")

        # Get all available images in the source directory
        print("Scanning available images...")
        available_images = {}
        for img_file in images_dir.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
            ]:
                available_images[img_file.name] = img_file

        print(f"Found {len(available_images)} images in source directory")

        # Process each entry in the JSON
        filtered_captions = {}
        image_id_mapping = {}
        copied_count = 0
        missing_count = 0

        for entry in tqdm(data, desc="Processing entries"):
            # Extract filename from file_path (e.g., "CUHK01/0363004.png" -> "0363004.png")
            file_path = entry["file_path"]
            filename = Path(file_path).name

            # Check if the image exists in the source directory
            if filename in available_images:
                # Define source and destination paths (keep original filename)
                source_path = available_images[filename]
                dest_path = output_dir / filename

                # Copy the image with original name directly to images_filtered
                shutil.copy2(source_path, dest_path)
                copied_count += 1

                # Add to the captions mapping (using original filename)
                filtered_captions[filename] = entry["captions"]

                # Add to the image ID mapping
                image_id_mapping[filename] = entry["id"]
            else:
                missing_count += 1
                print(f"Warning: Image not found: {filename}")

        # Save the captions mapping JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(filtered_captions, f, indent=2, ensure_ascii=False)

        # Save the image ID mapping JSON
        with open(output_id_mapping_path, "w", encoding="utf-8") as f:
            json.dump(image_id_mapping, f, indent=2, ensure_ascii=False)

        print(f"\nFiltering complete!")
        print(f"Images copied: {copied_count}")
        print(f"Images missing: {missing_count}")
        print(f"Images saved to: {output_dir}")
        print(f"Captions mapping saved to: {output_json_path}")
        print(f"Image ID mapping saved to: {output_id_mapping_path}")
        print(f"Total filtered images: {len(list(output_dir.glob('*')))}")

        return True

    except Exception as e:
        print(f"Error filtering images: {e}")
        return False


if __name__ == "__main__":
    # extract_cuhk_pedes_dataset()
    filter_images_from_reid_json()
