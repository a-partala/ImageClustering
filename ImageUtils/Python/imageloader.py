import tkinter as tk
from tkinter import filedialog
from PIL import Image


def load_image():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image_rgb = image.convert("RGB")
        image_rgb.show()
        return image_rgb
    else:
        print("File hasn't been chosen")
        return None

