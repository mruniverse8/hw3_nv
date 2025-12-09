import requests
import argparse
import os
import zipfile
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Download and unzip a ZIP file from a URL.")
    parser.add_argument("url", help="The URL to download the ZIP file from")
    parser.add_argument("-o", "--output_dir", default=".", help="Output directory to extract to (default: current directory)")
    args = parser.parse_args()
    url = args.url
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_zip_path = temp_file.name
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.unlink(temp_zip_path)
        print(f"Downloaded and extracted ZIP file to {output_dir} successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error unzipping file: {e}")

if __name__ == "__main__":
    main()  