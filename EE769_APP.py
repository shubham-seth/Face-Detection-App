import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras import backend as K
import os
import random
import io

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from FaceConfig import FaceConfig

from frcnn import utils as frcnn_utils
from frcnn import visualize as frcnn_visualize
from frcnn.visualize import display_images
import frcnn.model as frcnn_modellib
from fasterFaceConfig import FaceConfig as faster_FaceConfig


import yolo as yolo_utils

class ImageNetwork:
    def __init__(self, file, network, session):
        self.file = file
        self.net = network
        self.sess = session

def hash_image(ImgNet):
    return ImgNet.file

def main():
    st.title("Face Detector")
    st.sidebar.title("Choose the app mode")

    app_mode = st.sidebar.selectbox("what to do?", ["Introduction", "FasterRCNN", "MaskRCNN",  "YOLO v3"])

    if(app_mode == "Introduction"):
        Introduction()
    elif(app_mode == "MaskRCNN"):
        MaskRCNN()
    elif(app_mode == "FasterRCNN"):
        FasterRCNN()
    else:
        YOLOv3()

################# Introduction #################################################
def Introduction():
    f = open("Introduction.txt", 'r')
    intro = f.read().replace('\n','')
    intro = intro.replace("<br>", "")
    st.markdown(intro,unsafe_allow_html=True)

################################################################################

################## MaskRCNN ####################################################
def MaskRCNN():
    st.title("MaskRCNN")
    min_confidence = st.sidebar.slider("Select Confidence Threshold", min_value=0.1, max_value=0.95, value=0.9, step=0.05)
    
    f = open("MaskRCNN.txt","r")
    st.markdown(f.readline(), unsafe_allow_html=True)
    st.image("masked_show.png", use_column_width=True)
    st.markdown(f.readline(), unsafe_allow_html=True)
    MRCNN, session = mrcnn("mask_rcnn_face_0010.h5")
    class_names = ["BG", "Face"]
    file = st.file_uploader("Upload image")
    
    # try:
    if file:
        ImgNet = ImageNetwork(file, MRCNN, session)
        r, image = predict(ImgNet)
        
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], title="Predictions", 
                                    confidence_threshold=min_confidence)

        st.image(file, caption = "Uploaded Image", use_column_width = True)
        st.image("MaskRCNN_output.jpg", caption = "Faces Detected", use_column_width = True)
    # except:
    #     st.write("File Uploaded is not an image. Please upload am Image")

################################################################################

################## FasterRCNN ##################################################
def FasterRCNN():
    st.title("FasterRCNN")
    f = open("FasterRCNN.txt","r")
    st.markdown(f.readline(), unsafe_allow_html=True)
    st.image("faster_show.png", use_column_width=True)
    st.markdown(f.readline(), unsafe_allow_html=True)

    FRCNN, session = frcnn("faster_rcnn_face_0010.h5")
    class_names = ["BG", "Face"]
    file = st.file_uploader("Upload image")
    
    if file:
        ImgNet = ImageNetwork(file, FRCNN, session)
        r, image = fast_predict(ImgNet)
        
        frcnn_visualize.display_instances(image, r['rois'], None, r['class_ids'], 
                        class_names, r['scores'],
                        title="", show_mask=False)

        st.image(file, caption = "Uploaded Image", use_column_width = True)
        st.image("FasterRCNN_output.jpg", caption = "Faces Detected", use_column_width = True)
    # except:
    #     st.write("File Uploaded is not an image. Please upload am Image")
    return

################################################################################

################## YOLO v3 #####################################################
def YOLOv3():
    st.title("YOLO v3")

    f = open("YOLO v3.txt","r")
    st.markdown(f.readline(), unsafe_allow_html=True)
    st.image("yolo_display.png", use_column_width=True)
    st.markdown(f.readline(), unsafe_allow_html=True)
    file = st.file_uploader("Upload image")
    yolo, session = load_yolo()
    
    # try:
    if file:
        K.set_session(session)
        image = Image.open(file)
        detect_image = yolo.detect_image(image)
        st.image(file, caption="Uploaded Image", use_column_width=True)
        st.image(detect_image, caption="Predictions", use_column_width=True)
        # except:
        #    st.write("File uploaded is not an image, Please upload an image.")
    # except:
    #     st.write("File uploaded is not an image, Please upload an image.")
    
################################################################################

################## Results #####################################################
def Results():
    return

################################################################################

@st.cache(allow_output_mutation=True)
def mrcnn(weights_path):
    config = FaceConfig()
    MODEL_DIR = ""
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(weights_path, by_name=True)
    model.keras_model._make_predict_function()
    session = K.get_session()
    return model, session

@st.cache(allow_output_mutation=True, hash_funcs={ImageNetwork: hash_image})
def predict(ImgNet):
    K.set_session(ImgNet.sess)
    image = Image.open(ImgNet.file)
    image = np.array(image)

    result = ImgNet.net.detect([image], verbose=0)
    r = result[0]
    return r, image

@st.cache(allow_output_mutation=True, hash_funcs={ImageNetwork: hash_image})
def fast_predict(ImgNet):
    K.set_session(ImgNet.sess)
    image = Image.open(ImgNet.file)
    image = np.array(image)

    result = ImgNet.net.detect([image], verbose=0)
    r = result[0]
    return r, image

@st.cache(allow_output_mutation=True)
def frcnn(weights_path):
    config = faster_FaceConfig()
    MODEL_DIR = ""
    model = frcnn_modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(weights_path, by_name=True)
    model.keras_model._make_predict_function()
    session = K.get_session()
    return model, session

@st.cache(allow_output_mutation=True)
def load_yolo():
    yolo = yolo_utils.YOLO()
    # yolo._make_predict_function()
    session = K.get_session()
    return yolo, session

if __name__=="__main__":
    main()