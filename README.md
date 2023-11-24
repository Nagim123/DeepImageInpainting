# Manga Inpainting Application
We used the best model - LAMA for manga inpainting problem.
## Installation
1. Clone repository
```console
git clone --recurse-submodules https://github.com/Nagim123/DeepImageInpainting.git -b InpaintingPipelineLama
```
2. Create and activate virtual environment
```console
cd DeepImageInpainting
```
```console
python -m env .env
```
```console
.venv\Scripts\activate
```
3. Install dependencies
```console
pip install -r requirements.txt
```
4. Download models
```console
python src/model_downloader.py
```
## How to use
Run the following to apply manga cleaning in all images from **example** directory
```console
python src/manga_cleaner.py --manga_dir example/
```