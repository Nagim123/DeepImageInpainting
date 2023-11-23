import numpy as np
import argparse
import torch
from utils.model_loader import load_model
from PIL import Image

def apply_mask_to_img(orig_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Make pixels on image black if they are masked.

    Parameters:
        orig_image (Tensor): Original image to apply mask on.
        mask (Tensor): Image mask.

    Returns:
        Tensor: Masked image.
    """
    # Get locations of mask pixels
    masked_images = orig_image.detach().clone()
    locations = (mask == 1).nonzero()
    masked_images[:, locations[:, 0], locations[:, 1]] = 0
    return masked_images

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Script that run model's inference.")

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--image_mask_path", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    # Check if model's path specified by user
    if args.model_path is None:
        raise Exception("You must specify --model_path <path/to/model/directory>")

    # Check if user specify dataset or single image path
    if args.image_path is None or args.image_mask_path is None:
        raise Exception("You must specify --image_path <path/to/image.png> --image_mask_path <path/to/image_mask.png>")

    # Get input depend on user arguments
    if args.image_path is None:
        raise Exception("You must provide image for inference!")

    # Read image
    image = torch.tensor((np.asarray(Image.open(args.image_path).convert('RGB'))/255).astype(np.float32))
    image = image.permute((2,0,1))

    # Read mask
    mask = torch.tensor((np.asarray(Image.open(args.image_mask_path).convert('L'))/255).astype(np.float32))

    # Apply mask on image
    masked_img = apply_mask_to_img(image, mask).unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Concatinate image with mask
    input = torch.cat([masked_img, mask], dim=1)

    # Do inference
    model = load_model(f"{args.model_path}/unet_model.pt", f"{args.model_path}/unet_weights.pt")
    model.eval()
    with torch.no_grad():
        output = model(input)
        matrix = output[0].permute((1, 2, 0)).numpy()
        image = Image.fromarray((matrix*255).astype(np.uint8))
        if args.output_file is None:
            image.save("prediction.png")
        else:
            image.save(args.output_file)
