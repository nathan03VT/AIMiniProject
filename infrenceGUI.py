import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from inference import ImageToWordModel
import numpy as np


app = tk.Tk()
app.title('Image Recognition AI Interface')
canvas = tk.Canvas(app, cursor="cross")
canvas.pack(fill=tk.BOTH, expand=True)

img = None
img_tk = None
rect = None
cropping = False
image_for_inference: Image
image_for_inference = None
top_left = None
bottom_right = None
tl_set = False
br_set = False
inference = ImageToWordModel(model_path="model.onnx")

def open_image():
    global img, img_tk, canvas
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor='nw', image=img_tk)
        canvas.config(scrollregion=canvas.bbox(tk.ALL), height=img.height, width=img.width)

def start_cropping():
    global tl_set, br_set, rect
    tl_set = False
    br_set = False
    canvas.delete('rect')

def set_coordinates(event):
    global tl_set, br_set, top_left, bottom_right, rect
    if not tl_set:
        top_left = (event.x, event.y)
        tl_set = True
        canvas.create_rectangle(top_left[0], top_left[1], top_left[0] + 1, top_left[1] + 1, outline='red', tag='rect')
    elif not br_set:
        bottom_right = (event.x, event.y)
        br_set = True
        canvas.coords('rect', top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        crop_image()

def edit_rectangle(event):
    global rect, tl_set, br_set
    if tl_set and not br_set:
        canvas.coords('rect', top_left[0], top_left[1], event.x, event.y)

def crop_image():
    global image_for_inference, img_tk, canvas, img
    image_for_inference = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

def run_model():
    global image_for_inference, inference
    if image_for_inference == None:
        image_for_inference = img
    image_for_inference = image_for_inference.convert('RGB')
    model_output = inference.predict(np.array(image_for_inference))
    output_text.set(model_output)
    
    
def grayscale_image():
    global img, img_tk, canvas
    img = img.convert("L")
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor='nw', image=img_tk)
    canvas.config(scrollregion=canvas.bbox(tk.ALL), height=img.height, width=img.width)


open_button = tk.Button(app, text='Open Image', command=open_image)
open_button.pack()

image_label = tk.Label(app)
image_label.pack()

crop_button = tk.Button(app, text='Grey Scale', command=grayscale_image)
crop_button.pack()

crop_button = tk.Button(app, text='Crop Image', command=start_cropping)
crop_button.pack()

run_button = tk.Button(app, text='Run Model', command=run_model)
run_button.pack()

output_text = tk.StringVar()
output_label = tk.Label(app, textvariable=output_text)
output_label.pack()

canvas.bind("<Button-1>", set_coordinates)
canvas.bind("<Motion>", edit_rectangle)


app.mainloop()
