from PIL import Image
from create_dataset_base import MaskImageDataset
import argparse
import pathlib
import os
import numpy as np
script_path = pathlib.Path(__file__).parent.resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--masked", action='store_true')
    parser.add_argument("--image_index", type=int)
    args = parser.parse_args()
    dataset = MaskImageDataset(from_file=args.dataset)
    
    if args.image_index is None:
        # Ten images by default
        # First row
        row_matrix1 = []
        row_matrix2 = []
        for i in range(10):
            matrix = dataset.get_masked_image(i) if args.masked else dataset.get_image(i)
            matrix = matrix.permute((1, 2, 0)).numpy()
            if i < 5:
                row_matrix1.append(matrix)
            else:
                row_matrix2.append(matrix)
        row_matrix1 = np.concatenate(row_matrix1, axis=1)
        row_matrix2 = np.concatenate(row_matrix2, axis=1)
        full_matrix = np.concatenate((row_matrix1, row_matrix2), axis=0)
        image = Image.fromarray((full_matrix*255).astype(np.uint8))
        image.save(os.path.join(script_path, "temp/temp.png"))
    else:
        matrix = dataset.get_masked_image(args.image_index) if args.masked else dataset.get_image(args.image_index)
        matrix = matrix.permute((1, 2, 0)).numpy()
        image = Image.fromarray((matrix*255).astype(np.uint8))
        image.save(os.path.join(script_path, "temp/temp.png"))
    
    print("Saved to temp/temp.png!")
