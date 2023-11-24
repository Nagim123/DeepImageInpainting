import os
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageDisplayApp:
    def __init__(self, root, image_directory, mask_directory):
        self.root = root
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.image_list = self.load_images()
        self.current_image_index = 0

        self.photo_label = tk.Label(root)
        self.photo_label.pack()

        self.root.bind('<Return>', self.change_image)
        self.root.bind('<Button-1>', self.start_drawing)
        self.root.bind('<B1-Motion>', self.draw)
        self.root.bind('<ButtonRelease-1>', self.stop_drawing)
        self.root.bind('<Control-Return>', self.save_mask)

        self.drawing = False
        self.mask = None
        self.start_x = None
        self.start_y = None

        self.show_image()

    def load_images(self):
        image_list = [f for f in os.listdir(self.image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return image_list

    def load_mask(self, image_name):
        mask_name = f"mask-{image_name}"
        mask_path = os.path.join(self.mask_directory, mask_name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return mask
        else:
            return None

    def apply_mask(self, image, mask):
        red_mask = np.zeros_like(image)
        red_mask[:, :, 0] = mask  # Set red channel to the mask
        return cv2.addWeighted(image, 0.75, red_mask, 1, 0)  # Combine image and red mask

    def start_drawing(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y

    def draw(self, event):
        if self.drawing and 0 <= event.x < self.current_image.shape[1] and 0 <= event.y < self.current_image.shape[0]:
            # Convert Tkinter coordinates to OpenCV coordinates
            x_cv = event.x + self.offsetw
            y_cv = event.y + self.offseth + 8

            cv2.rectangle(self.mask, (x_cv - 8, y_cv - 8), (x_cv + 8, y_cv + 8), 255, -1)
            self.update_display()

    def stop_drawing(self, event):
        self.drawing = False

    def update_display(self):
        if self.mask is not None:
            image_with_mask = self.apply_mask(self.current_image, self.mask)
            photo = ImageTk.PhotoImage(Image.fromarray(image_with_mask))
            self.photo_label.config(image=photo)
            self.photo_label.image = photo

    def show_image(self):
        if self.current_image_index < len(self.image_list):
            image_name = self.image_list[self.current_image_index]
            image_path = os.path.join(self.image_directory, image_name)
            self.current_image = cv2.imread(image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            self.offseth = max((self.current_image.shape[0] - self.root.winfo_screenheight()) // 2, 0)
            self.offsetw = max((self.current_image.shape[1] - self.root.winfo_screenwidth()) // 2, 0)
            print(self.offseth, self.offsetw)

            self.mask = self.load_mask(image_name)
            if self.mask is None:
                self.mask = np.zeros((self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8)

            self.update_display()

        else:
            # No more images, close the app
            self.root.destroy()

    def change_image(self, event):
        self.save_mask(event)  # Save mask before changing image
        self.current_image_index += 1
        self.show_image()

    def save_mask(self, event):
        if self.mask is not None:
            image_name = self.image_list[self.current_image_index]
            mask_name = f"mask-{image_name}"
            mask_path = os.path.join(self.mask_directory, mask_name)
            cv2.imwrite(mask_path, self.mask)


def appl(image_folder, mask_folder):
    root = tk.Tk()
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
    app = ImageDisplayApp(root, image_folder, mask_folder)
    root.mainloop()


if __name__ == "__main__":
    appl("C:/Users/user/PycharmProjects/DeepImageInpainting/example",
         "C:/Users/user/PycharmProjects/DeepImageInpainting/temp")
