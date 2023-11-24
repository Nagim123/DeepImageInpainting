# Navigator
[Deep Image Inpainting research](#deepimageinpainting)

[Manga symbol cleaning](#manga-inpainting-application)
# DeepImageInpainting
A project to solve the Image Inpainting problem using generative deep neural networks. Essentially, we are trying to predict a masked part of the image based on the entire context of the image.
## Examaple
You can try our old models on collab.
1. [Vanilla AE](https://colab.research.google.com/drive/1ApWMUxXFhksdLbjA5b-s3QTS4_3HkgM8?usp=sharing)
2. [Pixel CNN](https://colab.research.google.com/drive/1LzLzK4YSU2tJrdFndGbIIYSAx8qYjUWI?usp=sharing)
3. [Unet](https://colab.research.google.com/drive/1lJDoIicTpp9nR_8kCOgljNYpAiKmqnvP?usp=sharing)
4. [Diagonal BLSTM](https://colab.research.google.com/drive/1JPW-W1iz4bMrzoSki7o9YplzFAHS-0pz?usp=sharing)
## Results on CIFAR10 and STL10
* Vanilla AE <br/>
  ![img](https://lh7-us.googleusercontent.com/H2WlLnSjqd883zxG9cVAEpq6owVMtMrjbWVRCpP4Sji2Lgvl7tIS_8jqJBxK1ROZoI1JCCcy_8jkzzAcju82wHu2V9vKe6tWMeUaLtuTMZYrVRJ6luA4zwB5iHHn6bCmZIYb-3nQeZh8urcxKX1-Qw)

* Pixel CNN <br/>
  ![img](https://lh7-us.googleusercontent.com/lASsFK-NBf21hzrucQxMw_ouhfMtjFVHrRkiCKpvoaOjj4LH9-oVi9AWGNuZpEDVXF-WRZ3zhqS1lv2GqtyZWrUWV2nf6lVd3Tq7vP_kFgA8SQdMol0fHr60CfgUnkd__VpPVB_bQ72vfwij3hAEAQ)

* UNet <br/>
  ![img](https://lh7-us.googleusercontent.com/EU_ztXX8Lyrl0xGUVJd8gdlt71Sox6gfkpqVX43dbF11wl6r2PhDLky0RP3gq7T22EDLqFVPfRN1y7O_qRTKmPNTl24q9iS5ufR8z9sPvJQMYWgvhAGOGVUkvQQHSMiGo7zygG0lepYa1fr_CwrliA)
* We also tried some open-source models. The most successfull example found so far is [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything/tree/main). It combines an inpainting model and Meta's SAM model for segmenting objects <br/>
![img](https://gcdnb.pbrd.co/images/ORKe8k1DSuY2.png?o=1)


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
python -m venv .venv
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
Run the following command to apply manga cleaning on all images from **example** directory
```console
python src/manga_cleaner.py --manga_dir example/ --line_width 12
```
You can specify line width for Japanese character selector through **--line width** argument. Command results will be saved in *output* diirectory.
## GUI tutorial
After running the command to clean manga pages the manual mask editor will appear.

 GUI control:
* LMB - draw mask
* RMB - remove mask
* Mouse scroll - change brush size
* Enter - next page

Screenshot of mask editor:
![Alt text](pictures/image.png)
## Example of model inference
|original image|model's output|
|----|----|
|![Alt text](pictures/original.png)|![Alt text](pictures/model_res.png)|
