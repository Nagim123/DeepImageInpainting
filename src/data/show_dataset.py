import matplotlib.pyplot as plt
from create_dataset_base import MaskImageDataset
import argparse

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
            plt.imshow(dataset.get_masked_image(args.image_index).permute((1, 2, 0)))
        else:
            plt.imshow(dataset.get_image(args.image_index).permute((1, 2, 0)))
