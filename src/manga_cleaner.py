import pathlib
import os
import shutil
script_path = pathlib.Path(__file__).parent.resolve()
import argparse
import cv2
import numpy as np

from display_images import appl

TEMP_FOLDER_PATH = os.path.join(script_path, "../temp/")
LAMA_CONFIG_PATH = os.path.join(script_path, "fixed-lama-inpainter/lama/configs/prediction/default.yaml")
LAMA_MODEL_PATH = os.path.join(script_path, "models/big-lama/")
MANGA_SEG_MODEL_PATH = os.path.join(script_path, "models/comictextdetector.pt")
SEGMENT_SCRIPT_PATH = os.path.join(script_path, "comic-text-detector/run_model.py")
PREDICT_SCRIPT_PATH = os.path.join(script_path, "fixed-lama-inpainter/lama_inpaint.py")
OUTPUTS_PATH = os.path.join(script_path, "../outputs/")


def dilate_rgb_mask(mask: np.array, kernel_size: int = 7) -> np.array:
    """
    Dilate mask lines width in image for better performance.

    Parameters:
        mask (np.array): Link for download
        kernel_size (int): The value of dilate. Default is 7.
    
    Return:
        (np.array): Image with increased lines mask width.
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
    parser.add_argument("--line_width", type=int, default=7)
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
    
    
    # Mask extension
    for image_name in image_names:
        path_to_mask = os.path.join(TEMP_FOLDER_PATH, f"mask-{image_name}")
        masked_img = cv2.imread(path_to_mask)
        masked_img = dilate_rgb_mask(masked_img, kernel_size=args.line_width)
        cv2.imwrite(path_to_mask, masked_img)

    appl(args.manga_dir, TEMP_FOLDER_PATH)

    for image_name in image_names:
        clean_name = image_name.split(".")[0]

        # Prepare paths
        path_to_img = os.path.join(TEMP_FOLDER_PATH, image_name)
        path_to_mask = os.path.join(TEMP_FOLDER_PATH, f"mask-{image_name}")
        model_result_folder = os.path.join(OUTPUTS_PATH, clean_name)
        model_result_file_path = os.path.join(model_result_folder, f"inpainted_with_mask-{image_name}")
        output_file_path = os.path.join(OUTPUTS_PATH, image_name)

        # Run inpainting model        
        os.system(f"python {PREDICT_SCRIPT_PATH} --input_img {path_to_img} --input_mask_glob {path_to_mask} --lama_config {LAMA_CONFIG_PATH} --lama_ckpt {LAMA_MODEL_PATH} --output_dir {OUTPUTS_PATH}")
        # Move the resulted file to outputs folder
        shutil.move(model_result_file_path, output_file_path)
        shutil.rmtree(model_result_folder)

        # Remove temporary files
        os.remove(path_to_img)
        os.remove(path_to_mask)
        clean_name = image_name.split(".")[0]
        os.remove(os.path.join(TEMP_FOLDER_PATH, f"{clean_name}.txt"))
        os.remove(os.path.join(TEMP_FOLDER_PATH, f"line-{clean_name}.txt"))


