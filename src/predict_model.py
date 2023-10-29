import numpy as np
import argparse
import torch
from data.create_dataset_base import MaskImageDataset
from utils.model_loader import load_model
from PIL import Image

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    parser.add_argument("weights", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--image_index", type=int)
    args = parser.parse_args()

    # Check if user specify dataset or single image path
    if args.image_path is None and args.dataset is None:
        raise Exception("You must specify --image_path <path/to/image.png> or --dataset <name> --image_index <index>")

    # Get input depend on user arguments
    if args.image_path is None:
        dataset = MaskImageDataset(from_file=args.dataset)
        input = dataset.get_masked_image(args.image_index).unsqueeze(0)
    else:
        input = torch.tensor((np.asarray(Image.open(args.image_path).convert('RGB'))/255).astype(np.float32))
        input = input.permute((2,0,1)).unsqueeze(0)
    
    # Do inference
    model = load_model(args.model_name, args.weights)
    model.eval()
    with torch.no_grad():
        output = model(input, do_inference=True)
        matrix = output[0].permute((1, 2, 0)).numpy()
        image = Image.fromarray((matrix*255).astype(np.uint8))
        image.save("prediction.png")