import argparse
import numpy as np
import os
import pandas as pd
from PIL import Image
from pyts.image import GramianAngularField
from pathlib import Path
import shutil

def get_args():
    parser = argparse.ArgumentParser(
        'Script to convert segmented ECG signals from CSV to GAF images', add_help=False)
    parser.add_argument(
        '--source-path', default='ecg_segment/', type=str,
        help='Directory containing the CSV files with segmented heartbeats.')
    parser.add_argument(
        '--dest-path', default='data/', type=str,
        help='Root directory to save the training images.')
    parser.add_argument(
        '--dataset-name', default='ecgid', type=str,
        help='The name of the dataset (e.g., ecgid, ptb), used to create the destination folder.')
    parser.add_argument(
        '--image-size', default=224, type=int,
        help='The size of the output square image (height and width).')
    return parser.parse_args()

def convert_and_save(args):
    """
    Finds all CSV files in the source path, converts each heartbeat signal (column)
    into a Gramian Angular Field image, and saves them to the destination path
    following an ImageFolder structure.
    """
    source_root = Path(args.source_path)
    dest_root = Path(args.dest_path) / args.dataset_name / 'train'
    
    # Clean up the destination directory before generation
    if dest_root.exists():
        print(f"Cleaning up existing directory: {dest_root}")
        shutil.rmtree(dest_root)
        
    print(f"Source directory: {source_root}")
    print(f"Destination directory: {dest_root}")

    # Initialize the GAF transformer
    gaf = GramianAngularField(image_size=args.image_size)

    # Walk through the source directory, which should contain subdirs for each person
    person_dirs = [d for d in source_root.iterdir() if d.is_dir()]
    if not person_dirs:
        print(f"Error: No person-specific subdirectories found in {source_root}. Please run preproc.py first.")
        return

    for person_dir in person_dirs:
        person_name = person_dir.name
        target_person_dir = dest_root / person_name
        
        # Create the target directory for the person
        target_person_dir.mkdir(parents=True, exist_ok=True)
        
        beat_counter = 0
        
        # Find all CSV files for that person
        csv_files = list(person_dir.glob('*.csv'))
        
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, index_col='Time')
            except Exception as e:
                print(f"Warning: Could not read {csv_path}. Error: {e}")
                continue

            # Each column (except the index) is a heartbeat
            for beat_label in df.columns:
                signal = df[beat_label].dropna().to_numpy()
                
                if signal.ndim == 1:
                    # Add a batch dimension for the transformer
                    signal = signal.reshape(1, -1)
                
                # Transform the signal to a GAF image
                gaf_image = gaf.fit_transform(signal)
                
                # Squeeze the batch dimension, resulting in a 2D array
                gaf_image = gaf_image[0]
                
                # Scale image from [-1, 1] to [0, 255] and convert to uint8
                scaled_image = np.uint8((gaf_image + 1) / 2 * 255)
                
                # Create a PIL Image from the numpy array
                pil_image = Image.fromarray(scaled_image, mode='L') # 'L' for grayscale
                
                # Convert to RGB as expected by the dataset loader
                pil_image_rgb = pil_image.convert('RGB')
                
                # Save the image
                image_path = target_person_dir / f'beat_{beat_counter}.png'
                pil_image_rgb.save(image_path)
                
                beat_counter += 1
        
        print(f"Processed {person_name}: Found {len(csv_files)} CSV files, generated {beat_counter} images.")

def main(args):
    convert_and_save(args)
    print("\nImage generation complete.")
    print(f"You can now try running the main training script. Point it to the '{args.dataset_name}' dataset.")

if __name__ == '__main__':
    args = get_args()
    main(args)
