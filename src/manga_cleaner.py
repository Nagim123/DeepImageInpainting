import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()
import argparse
print(script_path)

TEMP_FOLDER_PATH = os.path.join(script_path, "../temp/")
UNET_MODEL_PATH = os.path.join(script_path, "models/")
MANGA_SEG_MODEL_PATH = os.path.join(script_path, "models/comictextdetector.pt")
SEGMENT_SCRIPT_PATH = os.path.join(script_path, "comic-text-detector/run_model.py")
PREDICT_SCRIPT_PATH = os.path.join(script_path, "predict_model.py")
OUTPUTS_PATH = os.path.join(script_path, "../outputs/")

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
        output_path = os.path.join(OUTPUTS_PATH, image_name)

        # Run inpainting model        
        os.system(f"python {PREDICT_SCRIPT_PATH} --image_path {path_to_img} --image_mask_path {path_to_mask} --model_path {UNET_MODEL_PATH} --output_file {output_path}")
        
        # Remove temporary files
        os.remove(path_to_img)
        os.remove(path_to_mask)
        clean_name = image_name.split(".")[0]
        os.remove(os.path.join(TEMP_FOLDER_PATH, f"{clean_name}.txt"))
        os.remove(os.path.join(TEMP_FOLDER_PATH, f"line-{clean_name}.txt"))


