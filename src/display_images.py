import os
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageDisplayApp:
    def __init__(self, root: tk.Tk, image_directory: str, mask_directory: str):
        """
       Initialize the ImageDisplayApp.

       Parameters:
           root (tkinter.Tk): The root Tkinter window.
           image_directory (str): The directory containing image files.
           mask_directory (str): The directory to store mask files.
       """
        self.root = root
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.image_list = self.load_images()
        self.current_image_index = 0

        self.photo_label = tk.Label(root)
        self.photo_label.pack()

        self.root.bind('<Return>', self.change_image)
        self.root.bind('<Button-1>', self.start_drawing)  # Left mouse button
        self.root.bind('<B1-Motion>', lambda event: self.draw(event, 255))
        self.root.bind('<ButtonRelease-1>', self.stop_drawing)
        self.root.bind('<Button-3>', self.start_drawing)  # Right mouse button
        self.root.bind('<B3-Motion>', lambda event: self.draw(event, 0))
        self.root.bind('<ButtonRelease-3>', self.stop_drawing)
        self.root.bind('<Control-Return>', self.save_mask)
        self.root.bind('<MouseWheel>', self.change_drawing_size)

        self.drawing = False
        self.mask = None
        self.pointer_size = 8

        self.show_image()

    def load_images(self) -> list:
        """
        Load a list of image files from the specified directory.

        Returns:
            list: List of image file names.
        """
        image_list = [f for f in os.listdir(self.image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return image_list

    def load_mask(self, image_name: str) -> np.array:
        """
        Load a mask associated with the given image name.

        Parameters:
            image_name (str): Name of the image file.

        Returns:
            numpy.ndarray: Loaded mask or None if no mask found.
        """
        mask_name = f"mask-{image_name}"
        mask_path = os.path.join(self.mask_directory, mask_name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return mask
        else:
            return None

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.array:
        """
        Apply the loaded mask to the given image.

        Parameters:
            image (numpy.ndarray): Input image.
            mask (numpy.ndarray): Mask to be applied.

        Returns:
            numpy.ndarray: Image with applied mask.
        """
        red_mask = np.zeros_like(image)
        red_mask[:, :, 0] = mask  # Set red channel to the mask
        return cv2.addWeighted(image, 0.75, red_mask, 1, 0)  # Combine image and red mask

    def start_drawing(self, event: tk.Event) -> None:
        """
        Set the drawing flag to True when the mouse button is pressed.

        Parameters:
            event (tkinter.Event): Mouse event.
        """
        self.drawing = True

    def draw(self, event: tk.Event, value: int) -> None:
        """
        Draw on the mask when the mouse is moved during drawing.

        Parameters:
            event (tkinter.Event): Mouse event.
            value (int): Value to draw on the mask.
        """
        if self.drawing and 0 <= event.x < self.current_image.shape[1] and 0 <= event.y < self.current_image.shape[0]:
            # Convert Tkinter coordinates to OpenCV coordinates
            x_cv = event.x
            y_cv = event.y + self.pointer_size
            d = self.pointer_size
            cv2.rectangle(self.mask, (x_cv - d, y_cv - d), (x_cv + d, y_cv + d), value, -1)
            self.update_display()

    def change_drawing_size(self, event: tk.Event) -> None:
        """
        Change the drawing size based on mouse wheel movement.

        Parameters:
            event (tkinter.Event): Mouse wheel event.
        """
        delta = event.delta
        # Increase or decrease the drawing size based on mouse wheel movement
        self.pointer_size += int(delta / 120)
        self.pointer_size = max(1, self.pointer_size)

    def stop_drawing(self, event: tk.Event) -> None:
        """
        Set the drawing flag to False when the mouse button is released.

        Parameters:
            event (tkinter.Event): Mouse event.
        """
        self.drawing = False

    def update_display(self) -> None:
        """
        Update the displayed image with the applied mask.
        """
        if self.mask is not None:
            image_with_mask = self.apply_mask(self.current_image, self.mask)
            photo = ImageTk.PhotoImage(Image.fromarray(image_with_mask))
            self.photo_label.config(image=photo)
            self.photo_label.image = photo

    def show_image(self) -> None:
        """
        Show the current image with the associated mask.
        """
        if self.current_image_index < len(self.image_list):
            image_name = self.image_list[self.current_image_index]
            image_path = os.path.join(self.image_directory, image_name)
            self.current_image = cv2.imread(image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            self.shape = self.current_image.shape[:2]

            self.mask = self.load_mask(image_name)
            if self.mask is None:
                self.mask = np.zeros((self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8)

            # Resize the image to the window size
            ratio_h = self.current_image.shape[0] / self.root.winfo_screenheight()
            ratio_w = self.current_image.shape[1] / self.root.winfo_screenwidth()
            ratio = max(ratio_h, ratio_w)
            h = int(self.current_image.shape[0] / ratio)
            w = int(self.current_image.shape[1] / ratio)
            self.current_image = cv2.resize(self.current_image, (w, h))
            self.mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)

            self.update_display()

        else:
            # No more images, close the app
            self.root.destroy()

    def change_image(self, event: tk.Event) -> None:
        """
        Change to the next image and save the current mask.

        Parameters:
            event (tkinter.Event): Key press event.
        """
        self.save_mask(event)  # Save mask before changing image
        self.current_image_index += 1
        self.show_image()

    def save_mask(self, event: tk.Event) -> None:
        """
        Save the current mask associated with the current image.

        Parameters:
            event (tkinter.Event): Key press event.
        """
        if self.mask is not None:
            self.mask = cv2.resize(self.mask, self.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            image_name = self.image_list[self.current_image_index]
            mask_name = f"mask-{image_name}"
            mask_path = os.path.join(self.mask_directory, mask_name)
            cv2.imwrite(mask_path, self.mask)


def appl(image_folder: str, mask_folder: str) -> None:
    """
    Run the application with the specified image and mask folders.

    Parameters:
        image_folder (str): The directory containing image files.
        mask_folder (str): The directory to store mask files.
    """
    root = tk.Tk()
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
    app = ImageDisplayApp(root, image_folder, mask_folder)
    root.mainloop()
