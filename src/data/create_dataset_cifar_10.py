import torch
import numpy as np
from create_dataset_base import MaskImageDataset
import tensorflow as tf

class CIFAR10Masked(MaskImageDataset):
    """
    CIFAR10 dataset with 32x32 RGB images.
    """

    def load_images(self) -> np.array:
        # Load images using keras
        (images1, _), (images2, _) = tf.keras.datasets.cifar10.load_data()
        torch_data = torch.cat((torch.tensor(images1, dtype=torch.float32)/255, torch.tensor(images2, dtype=torch.float32)/255), dim=0)
        return torch_data.permute((0, 3, 1, 2))

    def mask_image(self, image: torch.Tensor) -> np.array:
        image = image.permute((1, 2, 0))
        # Generate random top-left coordinates.
        x1, y1 = np.random.randint(0, 26, size=2)
        # Generate random bottom-right coordinates
        min_w, min_h, max_w, max_h = 5, 5, 9, 9
        x2 = np.random.randint(min(x1+min_w,32), min(x1+max_w,32))
        y2 = np.random.randint(min(y1+min_h,32), min(y1+max_h,32))
        # Copy original image
        masked_image = image.clone().detach()
        # Fill black rectange on generated coordinates
        mask = torch.tensor(np.zeros((y2-y1)*(x2-x1)*3).reshape(y2-y1,x2-x1,3))
        masked_image[y1:y2,x1:x2] = mask
        # Return result
        return masked_image.permute((2, 0, 1))

if __name__ == "__main__":
    cifar10_test = CIFAR10Masked()
    cifar10_test.save_dataset(filename="cifar10.pt")
    print("CIFAR10 dataset is created and saved in datasets\cifar10.pt")