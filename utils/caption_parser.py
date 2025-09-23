import json
from pathlib import Path
from tqdm import tqdm


def parse_cuhk_captions(
    input_json_path="data/CUHK-PEDES/caption_all.json",
    output_captions_path="data/CUHK-PEDES/captions_parsed.json",
    output_ids_path="data/CUHK-PEDES/imgs_id.json"
):
    """
    Parse CUHK-PEDES caption_all.json and create structured output files.
    
    Args:
        input_json_path (str): Path to the input caption_all.json file
        output_captions_path (str): Path to save captions in img_name: [captions] format
        output_ids_path (str): Path to save image IDs in img_name: id format
    
    Returns:
        bool: True if parsing was successful, False otherwise
    """
    try:
        current_dir = Path.cwd()
        input_json_path = current_dir / input_json_path
        output_captions_path = current_dir / output_captions_path
        output_ids_path = current_dir / output_ids_path
        
        print(f"Loading captions from {input_json_path}")
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} entries from JSON")
        
        captions_parsed = {}
        imgs_id = {}
        
        # Process each entry
        for entry in tqdm(data, desc="Processing captions"):
            file_path = entry['file_path']
            img_name = Path(file_path).name

            captions_parsed[img_name] = entry['captions']
            imgs_id[img_name] = entry['id']
        
        # Save the captions mapping
        with open(output_captions_path, 'w', encoding='utf-8') as f:
            json.dump(captions_parsed, f, indent=2, ensure_ascii=False)
        
        # Save the ID mapping
        with open(output_ids_path, 'w', encoding='utf-8') as f:
            json.dump(imgs_id, f, indent=2, ensure_ascii=False)
        
        print(f"\nParsing complete!")
        print(f"Processed {len(captions_parsed)} images")
        print(f"Captions saved to: {output_captions_path}")
        print(f"Image IDs saved to: {output_ids_path}")
        
        total_captions = sum(len(caps) for caps in captions_parsed.values())
        avg_captions_per_image = total_captions / len(captions_parsed)
        print(f"Total captions: {total_captions}")
        print(f"Average captions per image: {avg_captions_per_image:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error parsing captions: {e}")
        return False


if __name__ == "__main__":
    parse_cuhk_captions()
