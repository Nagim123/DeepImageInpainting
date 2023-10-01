import argparse
import torch
from models.model import CurrentModel
import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("image_index", type=int)
    args = parser.parse_args()

    current_model = CurrentModel()
    model_scripted = torch.jit.script(current_model)
    model_scripted.save(os.path.join(script_path, f"models/checkpoints/{args.model_name}.pt"))
