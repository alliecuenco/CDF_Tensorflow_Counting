import sys
import os
import cv2
import tensorflow as tf
import vehicle_detection_main

from tkinter import *
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfilename

from utils import backbone

window=Tk()

window.title("Running Python Script")
window.geometry('550x200')

if tf.__version__ < '1.0.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')

def output_file():
    window.fileName = askopenfilename(parent=window, title='Choose a file', initialdir='C:\\')
    input_video = window.fileName
    vid = cv2.VideoCapture(input_video)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = vid.get(cv2.CAP_PROP_FPS)
    roi = 500
    print(input_video, height, width, fps)
    vehicle_detection_main.object_detection_function(input_video, detection_graph, category_index, width, roi)

btn1 = Button(window, text="CLICK ME", bg="black", fg="white",command=output_file)
btn1.grid(column=0, row=0)

window.mainloop()
