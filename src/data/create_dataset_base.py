import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pathlib
import os
script_path = pathlib.Path(__file__).parent.resolve()

class MaskImageDataset:
    """
    Basic class to download clean images and generate masks.
    """

    output_dir = os.path.join(script_path, "../../datasets")

    def __init__(self, from_file:str = None) -> None:
        """
        Load dataset from file if specified.
        Otherwise automaticly download and create from scratch.
        
        Parameters:
            from_file (str): Path to dataset file.
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

        Returns:
            tensor: Images.
        """
        return torch.tensor([])

    def mask_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for image.

        Parameters:
            image (Tensor): Image to apply mask.

        Returns:
            tensor: Images.
        """
        return image

    def get_image(self, index: int) -> torch.Tensor:
        """
        Give image by index.

        Parameters:
            index (int): Index of image.

        Returns:
            tensor: Images.
        """
        return self.images[index]

    def get_masked_image(self, index: int):
        """
        Give image with mask by index.

        Parameters:
            index (int): Index of image.

        Returns:
            tensor: Images.
        """
        return self.masked_images[index]

    def save_dataset(self, filename: str = "dataset.pt") -> None:
        """
        Save any dataset to a file.

        Parameters:
            filename (int): Name of saved dataset file.
        """
        torch.save({
            "images": self.images,
            "masked_images": self.masked_images
        }, os.path.join(MaskImageDataset.output_dir, filename))

    def load_dataset(self, filename="dataset.pt") -> None:
        """
        Load a dataset from a file.

        Parameters:
            filename (int): Name of saved dataset file.
        """
        dataset = torch.load(os.path.join(MaskImageDataset.output_dir, filename))
        self.images = dataset["images"]
        self.masked_images = dataset["masked_images"]

    def pack_to_dataloaders(self, batch_size: int = 32, train_fraction:float = 0.7) -> tuple[DataLoader, DataLoader]:
        """
        Pack image data into dataloaders for model training.

        Parameters:
            batch_size (int): Batch size for dataloaders.
            train_fraction (float): How much fraction of data will be in train dataloader.
        Returns:
            tuple[DataLoader, DataLoader]: Train and validation dataloaders.
        """
        images_dataset = TensorDataset(self.masked_images, self.images)
        images_train, images_val = random_split(images_dataset, [train_fraction, 1-train_fraction])
        return (DataLoader(images_train, batch_size=batch_size, shuffle=False),
                DataLoader(images_val, batch_size=batch_size, shuffle=True))

    def __len__(self) -> int:
        return self.images.shape[0]

if __name__ == "__main__":
    raise Exception("Do not call this script! It's base class!")