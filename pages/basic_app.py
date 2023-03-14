# https://blog.streamlit.io/common-app-problems-resource-limits/

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pathlib import Path
import av
import cv2
import numpy as np
import tensorflow as tf
import queue
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img


HERE = Path(__file__).parent
# ROOT = HERE.parent


## Cache to prevent resource overload
@st.cache_resource    
def load_model():
    ## load model - local
    #  
    ## load model - streamlit
    # tf.keras.models.load_model(HERE / './saved_model/prettyfish_model')
    return tf.keras.models.load_model('saved_model/prettyfish_model')
 
model = load_model()


# class names
class_names = ['clownfish', 
               'longnose_butterflyfish',
               'oranda_goldfish',
               'powder_blue_tang',
               'queen_angelfish',
               'snakeskin_discus']

# inputs
def process_img (img):
    ## resize
    img = cv2.resize(img, (299,299))
    ## expand dims to 4D
    test_img = np.expand_dims(img, axis=0)
    ## log-outs of probabilities
    logits = model.predict(test_img)
    ## select index of highest logits
    predict_output = tf.argmax(logits, -1).numpy()
    ## run thru classifier to get name  ##
    pred_text = class_names[predict_output[0]]   
    return pred_text, logits  
    

st.title("Basic PrettyFish Setup")
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,40)
fontScale = 1
fontColor = (255,255,255)
lineType = 2



# NOTE: the callback will be called in another thread
# so use a queue here for thread-safety to pass the data
# from inside to outside the callback

## init queue, queue.put(value), queue.get(value)
result_queue: "queue.Queue[]" = queue.Queue()  


def videoFilter(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    # detect model here!
    pred_text, logits = process_img(img)
    text =  "HELLO! " + pred_text    
    cv2.putText(img, 
                text,
                bottomLeftCornerOfText,
                font, 
                fontScale, 
                fontColor, 
                lineType)
    result_queue.put(logits)
    return av.VideoFrame.from_ndarray(img, format="bgr24")




# webrtc_streamer(key="example")
# webrtc_streamer(key="example", video_frame_callback=videoFilter)


# NOTE: To deploy it to cloud
# config necessary to establish media streaming connection 
# when the server is on remote host
# streamlit_webrtc uses webRTC for its video and audio streaming
# it has to access a STUN server in the global network for
# the remote peers to establish WebRTC connections...
# this code uses a free STUN server provided by Google
# the value of the rtc_configuration argument will be passed to
# the RTCPeerConnection constructor on the frontend.
webrtc_ctx = webrtc_streamer (
    key="prettyfish",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback = videoFilter,
    rtc_configuration = { #Add this line
        "iceServers": [{"urls":["stun:stun.l.google.com:19302"]}]#   #192.168.68.101:3478
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


# NOTE: The video transformation with object detection and
# this loop displaying the result labels are running
# in different threads asynchronously.
# Then the rendered video frames and the labels displayed here
# are not strictly synchronized.
if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()     
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
            result_queue.task_done()