from PIL import Image
from create_dataset_base import MaskImageDataset
import argparse
import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset", type=str)
    parser.add_argument("is_masked", type=bool)
    parser.add_argument("--image_index", type=int)
    args = parser.parse_args()
    dataset = MaskImageDataset(from_file=args.dataset)
    if args.image_index is None:
        pass
    else:
        if args.is_masked:
            image = Image.fromarray(dataset.get_masked_image(args.image_index).permute((1, 2, 0)))
        else:
            image = Image.fromarray(dataset.get_image(args.image_index).permute((1, 2, 0)))
        image.save(os.path.join(script_path, "temp/temp.png"))
    
    print("Saved to temp/temp.png!")
