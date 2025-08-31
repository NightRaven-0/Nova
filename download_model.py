# download_model.py
import os
import requests
import tarfile

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_DIR = "model"
MODEL_ZIP = "model.zip"

def download_file(url, filename):
    """Download file from URL with streaming to avoid memory issues."""
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as file, \
            requests.get(url, stream=True) as r:
        downloaded = 0
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                done = int(50 * downloaded / total)
                print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded/1024/1024:.2f}MB/{total/1024/1024:.2f}MB", end="")

    print("\nDownload completed.")

def extract_zip(zip_path, extract_to):
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_ZIP):
        print("Downloading Vosk model...")
        download_file(MODEL_URL, MODEL_ZIP)
    else:
        print("Model archive already exists, skipping download.")

    print(" Extracting model...")
    extract_zip(MODEL_ZIP, MODEL_DIR)

if __name__ == "__main__":
    main()
