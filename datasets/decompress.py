"""
Script to decompress all .gz files in a directory.
"""

import gzip
import shutil
import sys
import glob
import os

def decompress_gz(input_file, output_file=None):
    """Decompress a .gz file."""
    if output_file is None:
        output_file = input_file.rstrip('.gz')
        
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed: {input_file} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)
    directory_path = sys.argv[1]
    
    # Use glob to get all .gz files in the directory
    gz_files = glob.glob(os.path.join(directory_path, '*.gz'))
    
    for gz_file in gz_files:
        decompress_gz(gz_file)
