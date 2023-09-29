import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()

class MaskImageDataset:
    """
    Basic class to download clean images and generate masks.
    """

    output_dir = os.path.join(script_path, "..\\..\\datasets")

    def __init__(self, from_file=None):
        """
        Load dataset from file if specified.
        Otherwise automaticly download and create from scratch.
        """

        if not from_file is None:
            self.load_dataset(from_file)
            return
        # Load some images
        self.images = self.load_images()
        # Add masks to images
        self.masked_images = torch.cat(
            [self.mask_image(image).unsqueeze(0) for image in self.images]
            , dim=0)

    def load_images(self) -> torch.Tensor:
        """
        Load image from the internet.
        """
        return torch.tensor([])

    def mask_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for image.
        """
        return image

    def get_image(self, index):
        """
        Give image by index.
        """
        return self.images[index]

    def get_masked_image(self, index):
        """
        Give image with mask by index.
        """
        return self.masked_images[index]

    def save_dataset(self, filename="dataset.pt"):
        """
        Save any dataset to a file.
        """
        torch.save({
            "images": self.images,
            "masked_images": self.masked_images
        }, os.path.join(MaskImageDataset.output_dir, filename))

    def load_dataset(self, filename="dataset.pt"):
        """
        Load a dataset from a file.
        """
        dataset = torch.load(os.path.join(MaskImageDataset.output_dir, filename))
        self.images = dataset["images"]
        self.masked_images = dataset["masked_images"]

    def pack_to_dataloaders(self, batch_size=32, train_fraction=0.7) -> tuple[DataLoader, DataLoader]:
        """
        Pack image data into dataloaders for model training.
        """
        images_dataset = TensorDataset(self.masked_images, self.images)
        images_train, images_val = random_split(images_dataset, [train_fraction, 1-train_fraction])
        return (DataLoader(images_train, batch_size=batch_size, shuffle=False),
                DataLoader(images_val, batch_size=batch_size, shuffle=True))

    def __len__(self):
        return self.images.shape[0]

if __name__ == "__main__":
    raise Exception("Do not call this script! It's base class!")