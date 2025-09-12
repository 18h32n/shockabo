#!/usr/bin/env python3
"""
Download real ARC tasks for performance testing
"""

import json
import time
import urllib.request
from pathlib import Path


def download_real_arc_tasks(num_tasks: int = 200) -> int:
    """Download real ARC tasks from the official repository"""

    # Create data directory
    data_dir = Path("data/tasks/training")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get list of training files from GitHub API
    api_url = "https://api.github.com/repos/fchollet/ARC-AGI/contents/data/training"

    try:
        with urllib.request.urlopen(api_url) as response:
            files_data = json.load(response)
    except Exception as e:
        print(f"Error getting file list: {e}")
        return 0

    # Filter for JSON files and get download URLs
    json_files = [f for f in files_data if f["name"].endswith(".json")][:num_tasks]

    print(f"Found {len(json_files)} JSON files, downloading first {len(json_files)}...")

    downloaded = 0
    for i, file_info in enumerate(json_files):
        filename = file_info["name"]
        download_url = file_info["download_url"]
        local_path = data_dir / filename

        if local_path.exists():
            print(f"[{i+1}/{len(json_files)}] Skip existing: {filename}")
            downloaded += 1
            continue

        try:
            urllib.request.urlretrieve(download_url, local_path)
            print(f"[{i+1}/{len(json_files)}] Downloaded: {filename}")
            downloaded += 1

            # Add small delay to be respectful to GitHub API
            if i % 10 == 9:
                time.sleep(0.5)

        except Exception as e:
            print(f"[{i+1}/{len(json_files)}] Error downloading {filename}: {e}")

    print(f"\nDownloaded {downloaded} ARC training tasks to {data_dir}")
    return downloaded


if __name__ == "__main__":
    count = download_real_arc_tasks(200)  # Download 200 tasks for testing
    if count > 0:
        print(f"Successfully downloaded {count} real ARC tasks for performance testing")
    else:
        print("Failed to download ARC tasks")
