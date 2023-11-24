import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()
import requests
import zipfile


OUTPUT_PATH_MODELS = os.path.join(script_path, "models/")

def download_url(url: str, output_path: str) -> None:
    """
    Download file by url.


    """
    response = requests.get(url)
    content = response.content
    with open(output_path, "wb") as output_file:
        output_file.write(content)
    


if __name__ == "__main__":
    # Download model for symbol detection
    print("DOWNLOADING DETECTION MODEL...")
    detector_path = os.path.join(OUTPUT_PATH_MODELS, "comictextdetector.pt")
    download_url("https://github.com/Nagim123/DeepImageInpainting/releases/download/Models/comictextdetector.pt", detector_path)
    
    # Download model for inpainting
    print("DOWNLOADING INPAINT MODEL...")
    inpainter_path = os.path.join(OUTPUT_PATH_MODELS, "pretrained_models.zip")
    download_url("https://github.com/Nagim123/DeepImageInpainting/releases/download/Models/pretrained_models.zip", inpainter_path)    
    with zipfile.ZipFile(inpainter_path, 'r') as zip_ref:
        zip_ref.extractall(OUTPUT_PATH_MODELS)
    os.remove(inpainter_path)