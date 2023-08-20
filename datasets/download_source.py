"""
Script to download the source files of an arXiv paper.
"""

import requests
import shutil
import os
import tarfile


def get_source(arxiv_id):
    source_url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(source_url, stream=True)
    if response.status_code == 200:
        with open(f"{arxiv_id}.tar.gz", "wb") as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
    else:
        print(f"Error: received status code {response.status_code} from arXiv.")
    with tarfile.open(f"{arxiv_id}.tar.gz", "r:gz") as f:
        f.extractall(path=f"{arxiv_id}_source_files")
   # Delete the tar.gz file
    os.remove(f"{arxiv_id}.tar.gz")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <arxiv_id>")
        sys.exit(1)
    arxiv_id = sys.argv[1]
    get_source(arxiv_id)
