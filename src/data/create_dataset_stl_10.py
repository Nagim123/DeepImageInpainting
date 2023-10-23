from create_dataset_base import MaskImageDataset
import tensorflow_datasets as tfds
import torch
import numpy as np

class STL10Masked(MaskImageDataset):
    """
    STL10 dataset with 96x96 RGB images.
    """

    def load_images(self) -> np.array:
        # Load images using tensorflow
        images = tfds.load('stl10',
                        split='train',
                        #split=['train', 'test'],
                        data_dir="/content/dataset")
        torch_data = torch.tensor([img['image'].numpy()/255 for img in images], dtype=torch.float32)
        # torch_data = torch.tensor(images, dtype=torch.float32)/255
        return torch_data.permute((0, 3, 1, 2))

    def mask_image(self, image: torch.Tensor) -> np.array:
        image = image.permute((1, 2, 0))
        # Generate random top-left coordinates.
        x1, y1 = np.random.randint(0, 26, size=2)
        # Generate random bottom-right coordinates
        min_w, min_h, max_w, max_h = 10, 10, 27, 27
        x2 = np.random.randint(min(x1+min_w,96), min(x1+max_w,96))
        y2 = np.random.randint(min(y1+min_h,96), min(y1+max_h,96))
        # Copy original image
        masked_image = image.clone().detach()
        # Fill black rectange on generated coordinates
        mask = torch.tensor(np.zeros((y2-y1)*(x2-x1)*3).reshape(y2-y1,x2-x1,3))
        masked_image[y1:y2,x1:x2] = mask
        # Return result
        return masked_image.permute((2, 0, 1))

if __name__ == "__main__":
    stl10_dataset = STL10Masked()
    stl10_dataset.save_dataset(filename="stl10.pt")
    print("CIFAR10 dataset is created and saved in datasets\cifar10.pt")