import numpy as np
from PIL import Image
import argparse
import torch
from data.create_dataset_base import MaskImageDataset
import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--image_index", type=int)
    args = parser.parse_args()

    path_to_model = os.path.join(script_path, f"models/checkpoints/{args.model_name}.pt")
    path_to_weights = os.path.join(script_path, f"models/checkpoints/{args.model_name}.pth")
    
    if args.image_path is None and args.dataset is None:
        raise Exception("You must specify --image_path <path/to/image.png> or --dataset <name> --image_index <index>")

    if args.image_path is None:
        dataset = MaskImageDataset(from_file=args.dataset)
        input = dataset.get_masked_image(args.image_index).unsqueeze(0)
    else:
        input = torch.tensor((np.asarray(Image.open(args.image_path).convert('RGB'))/255).astype(np.float32))
        input = input.permute((2,0,1)).unsqueeze(0)
    model = torch.jit.load(path_to_model)
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load(path_to_weights))
        
        output = model(input)
        matrix = output[0].permute((1, 2, 0)).numpy()
        image = Image.fromarray((matrix*255).astype(np.uint8))
        image.save(os.path.join(script_path, "data/temp/prediction.png"))