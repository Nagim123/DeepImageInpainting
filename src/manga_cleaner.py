import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()
import argparse

TEMP_FOLDER_PATH = os.path.join(script_path, "../temp/")
LAMA_CONFIG_PATH = os.path.join(script_path, "fixed-lama-inpainter/lama/configs/prediction/default.yaml")
LAMA_MODEL_PATH = os.path.join(script_path, "models/big-lama/")
MANGA_SEG_MODEL_PATH = os.path.join(script_path, "models/comictextdetector.pt")
SEGMENT_SCRIPT_PATH = os.path.join(script_path, "comic-text-detector/run_model.py")
PREDICT_SCRIPT_PATH = os.path.join(script_path, "fixed-lama-inpainter/lama_inpaint.py")
OUTPUTS_PATH = os.path.join(script_path, "../outputs/")

import cv2
import numpy as np

def dilate_rgb_mask(mask: np.array, kernel_size: int = 5) -> np.array:
    """
    
    """
    # Separate the channels
    b, g, r = cv2.split(mask)

    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate each channel independently
    dilated_b = cv2.dilate(b, kernel, iterations=1)
    dilated_g = cv2.dilate(g, kernel, iterations=1)
    dilated_r = cv2.dilate(r, kernel, iterations=1)

    # Merge the dilated channels back into an RGB image
    dilated_mask = cv2.merge([dilated_b, dilated_g, dilated_r])

    return dilated_mask[:,:,0]

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Script that clean manga from symbols")

    parser.add_argument("--manga_dir", type=str)
    args = parser.parse_args()

    # Check if directory with manga specified
    if args.manga_dir is None:
        raise Exception("You should specify directory with all manga pages in --manga_dir")

    # Get all filenames from manga directory
    image_names = []
    for (dirpath, dirnames, filenames) in os.walk(args.manga_dir):
        image_names.extend(filenames)
        break

    # Run mask extractor
    os.system(f"python {SEGMENT_SCRIPT_PATH} --image_dir {args.manga_dir} --output_dir {TEMP_FOLDER_PATH} --model_path {MANGA_SEG_MODEL_PATH}")

    for image_name in image_names:
        # Prepare paths
        path_to_img = os.path.join(TEMP_FOLDER_PATH, image_name)
        path_to_mask = os.path.join(TEMP_FOLDER_PATH, f"mask-{image_name}")
        #output_path = os.path.join(OUTPUTS_PATH, image_name)

        masked_img = cv2.imread(path_to_mask)
        masked_img = dilate_rgb_mask(masked_img, kernel_size=7)
        cv2.imwrite(path_to_mask, masked_img)

        # Run inpainting model        
        os.system(f"python {PREDICT_SCRIPT_PATH} --input_img {path_to_img} --input_mask_glob {path_to_mask} --lama_config {LAMA_CONFIG_PATH} --lama_ckpt {LAMA_MODEL_PATH} --output_dir {OUTPUTS_PATH}")
        
        # Remove temporary files
        # os.remove(path_to_img)
        # os.remove(path_to_mask)
        # clean_name = image_name.split(".")[0]
        # os.remove(os.path.join(TEMP_FOLDER_PATH, f"{clean_name}.txt"))
        # os.remove(os.path.join(TEMP_FOLDER_PATH, f"line-{clean_name}.txt"))


