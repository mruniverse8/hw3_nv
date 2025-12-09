import requests
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download a file from a URL.")
    parser.add_argument("url", default="https://drive.usercontent.google.com/download?id=1wTa6dAc5UIbWqTgXXS7x6W-bfxh1ZBWZ&export=download&authuser=0&confirm=t&uuid=275caf1d-6efa-40e5-85b7-b8a1787de2e4&at=ALWLOp5p5VzJolU-WzRKsTEPoWEP:1765315652384")
    parser.add_argument("-o", "--output", default="last_check.pth")
    
    args = parser.parse_args()
    
    url = args.url
    output_file = args.output
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {output_file} successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

if __name__ == "__main__":
    main()
