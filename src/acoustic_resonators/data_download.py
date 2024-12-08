import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout
from urllib3.util.retry import Retry
from tqdm import tqdm


def download_file(url, dest_path):
    retries = Retry(
        total=5, 
        backoff_factor=0.5, 
        status_forcelist=[500, 502, 503, 504]
    )
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))

    existing_file_size = dest_path.stat().st_size if dest_path.exists() else 0
    headers = {'Range': f'bytes={existing_file_size}-'}

    try:
        response = session.get(url, headers=headers, stream=True, timeout=(10, 30))
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0)) + existing_file_size
        block_size = 1024 * 1024  # 1MB chunks

        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name, initial=existing_file_size) as pbar:
            with open(dest_path, "ab") as f:  # Append if the file exists
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)

    except ConnectionError as e:
        print(f"Connection error: {e}")
    except HTTPError as e:
        print(f"HTTP error: {e}")
    except Timeout as e:
        print(f"Timeout error: {e}")
    except RequestException as e:
        print(f"General error: {e}")


def extract_zip(file_path, dest_path):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        for f in zip_ref.namelist():
            if not any(x in f for x in ["__MACOSX", ".DS_Store"]):
                zip_ref.extract(f, dest_path)

def main():
    record_id = "13220717"
    response = requests.get(f"https://zenodo.org/api/records/{record_id}")
    response.raise_for_status()

    data = response.json()
    download_urls = [f["links"]["self"] for f in data["files"]]
    filenames = [f["key"] for f in data["files"]]

    store_dir = Path("_downloads")
    store_dir.mkdir(exist_ok=True)

    for i, (filename, url) in enumerate(zip(filenames, download_urls), 1):
        file_path = store_dir / filename
        if file_path.exists():
            print(
                f"Skipping ({i}/{len(download_urls)}) - already downloaded: {filename}"
            )
            continue

        print(f"Downloading ({i}/{len(download_urls)}) {filename}")
        try:
            download_file(url, file_path)
        except requests.RequestException as e:
            print(f"Failed to download {filename}: {e}")
            continue

    root = Path("src/acoustic_resonators/")
    destinations = {
        "asops_data.zip": root / "asops",
        "stroboscopy_data.zip": root / "stroboscopy",
        "xray_data.zip": root / "xrays",
    }

    for filename, dest in destinations.items():
        file_path = store_dir / filename
        if file_path.exists():
            print(f"Extracting {filename} to {dest}")
            dest.mkdir(parents=True, exist_ok=True)
            try:
                extract_zip(file_path, dest)
            except zipfile.BadZipFile as e:
                print(f"Failed to extract {filename}: {e}")


if __name__ == "__main__":
    main()
