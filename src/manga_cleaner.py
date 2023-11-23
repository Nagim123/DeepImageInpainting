import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()
import argparse
print(script_path)

TEMP_FOLDER_PATH = os.path.join(script_path, "../temp/")
UNET_MODEL_PATH = os.path.join(script_path, "/models")
MANGA_SEG_MODEL_PATH = os.path.join(script_path, "/models")
SEGMENT_SCRIPT_PATH = os.path.join(script_path, "comic-text-detector/run_model.py")
PREDICT_SCRIPT_PATH = os.path.join(script_path, "predict_model.py")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Script that clean manga from symbols")

    parser.add_argument("--manga_dir", type=str)
    args = parser.parse_args()

    # Check if model's path specified by user
    if args.manga_dir is None:
        raise Exception("You should specify directory with all manga pages in --manga_dir")

    os.system(f"python {SEGMENT_SCRIPT_PATH} --image_dir {args.manga_dir} --output_dir {TEMP_FOLDER_PATH} --model_path {MANGA_SEG_MODEL_PATH}")