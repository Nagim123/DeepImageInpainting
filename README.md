# DeepImageInpainting
A project to solve the Image Inpainting problem using generative deep neural networks. Essentially, we are trying to predict a masked part of the image based on the entire context of the image.
## Examaple
You can try our current models on collab.
1. [Vanilla AE](https://colab.research.google.com/drive/1ApWMUxXFhksdLbjA5b-s3QTS4_3HkgM8?usp=sharing)
2. Pixel CNN (Not working right now)
3. [Unet](https://colab.research.google.com/drive/1lJDoIicTpp9nR_8kCOgljNYpAiKmqnvP?usp=sharing)
## Results so far
* Vanilla AE <br/>
  ![img](https://lh7-us.googleusercontent.com/H2WlLnSjqd883zxG9cVAEpq6owVMtMrjbWVRCpP4Sji2Lgvl7tIS_8jqJBxK1ROZoI1JCCcy_8jkzzAcju82wHu2V9vKe6tWMeUaLtuTMZYrVRJ6luA4zwB5iHHn6bCmZIYb-3nQeZh8urcxKX1-Qw)

* Pixel CNN <br/>
  ![img](https://lh7-us.googleusercontent.com/lASsFK-NBf21hzrucQxMw_ouhfMtjFVHrRkiCKpvoaOjj4LH9-oVi9AWGNuZpEDVXF-WRZ3zhqS1lv2GqtyZWrUWV2nf6lVd3Tq7vP_kFgA8SQdMol0fHr60CfgUnkd__VpPVB_bQ72vfwij3hAEAQ)

* UNet <br/>
  ![img](https://lh7-us.googleusercontent.com/EU_ztXX8Lyrl0xGUVJd8gdlt71Sox6gfkpqVX43dbF11wl6r2PhDLky0RP3gq7T22EDLqFVPfRN1y7O_qRTKmPNTl24q9iS5ufR8z9sPvJQMYWgvhAGOGVUkvQQHSMiGo7zygG0lepYa1fr_CwrliA)
* We also tried some open-source models. The most successfull example found so far is [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything/tree/main). It combines an inpainting model and Meta's SAM model for segmenting objects <br/>
![img](https://gcdnb.pbrd.co/images/ORKe8k1DSuY2.png?o=1)
