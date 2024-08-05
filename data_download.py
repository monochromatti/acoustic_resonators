import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
        with open(dest_path, "wb") as f:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)


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

    root = Path("src/experiment/acoustic_resonators/")
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
