import requests
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download a file from a URL.")
    parser.add_argument("url")
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
