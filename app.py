import gradio as gr
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def convert_to_grayscale(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image


iface = gr.Interface(fn=convert_to_grayscale, inputs=gr.Image(type="filepath"), outputs=gr.Image())
iface.launch(debug=True)
