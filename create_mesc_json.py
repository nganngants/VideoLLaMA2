import os
import json
import argparse
from glob import glob


def create_mesc_json(video_dir, output_path):
    """Create JSON file for MESC dataset.
    
    Args:
        video_dir: Directory containing .mp4 files
        output_path: Path to save the JSON file
    """
    # Get all mp4 files
    video_files = glob(os.path.join(video_dir, "*.mp4"))
    
    # Create list of dictionaries with video paths
    data = []
    for video_path in sorted(video_files):
        data.append({
            "video": video_path,
        })
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created JSON file with {len(data)} videos at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str, required=True,
                        help='Directory containing MESC mp4 files')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the output JSON file')
    args = parser.parse_args()
    
    create_mesc_json(args.video_dir, args.output_path)
