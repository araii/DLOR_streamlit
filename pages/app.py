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



# NOTE: Cache to prevent resource overload
@st.cache_resource    
def load_model():
    # --load model local--
    # 
    # --load model streamlit--
    # tf.keras.models.load_model(HERE / './saved_model/prettyfish_modelv1')
    return tf.keras.models.load_model('saved_model/prettyfish_modelv1')
 
model = load_model()




# class names
class_names = ['clownfish',
               'longnose_butterflyfish',
               'not_a_fish',
               'oranda_goldfish',
               'powder_blue_tang',
               'queen_angelfish',
               'snakeskin_discus']



# -------------- Basic -----------------
# def process_img (img):
#     ## resize
#     img = cv2.resize(img, (299,299))
#     ## expand dims to 4D
#     test_img = np.expand_dims(img, axis=0)
#     ## log-outs of probabilities
#     logits = model.predict(test_img)
#     ## select index of highest logits
#     predict_output = tf.argmax(logits, -1).numpy()
#     ## run thru classifier to get name  ##
#     pred_text = class_names[predict_output[0]]   
#     return pred_text, logits  


##---------------- Mars ------------------##
# NOTE: use tf.nn.softmax to convert logits to probabilities
# scaler: 0 Dimension tensor i.e ()
# vector: 1 Dimension tensor i.e (9, )
# matrix: 2 Dimension tensor i.e (1, 6)
# tf.nn.softmax(): converts logits to probabilities
# - output of tf.nn.softmax() is tf.EagerTensor 
# - numpy(): converts tf.EagerTensor into np.array
# - np.any(): any of the ele is true, return true 
def process_img (img):
    ## resize
    img = cv2.resize(img, (299,299))
    ## expand dims to 4D
    test_img = np.expand_dims(img, axis=0)
    ## log-outs of probabilities
    logits = model.predict(test_img)   
    proba = tf.nn.softmax(logits) 
    # --MOVE OUTSIDE OF THREAD--
    # threshold = 0.9 
    ## --any of the ele is true, return true--
    # if (proba > threshold).numpy().any():  
    #     predict_output = tf.argmax(logits, -1).numpy()
    # else:
    #     predict_output = [6]
    pred_text=""
    return pred_text, proba
    

    

st.title("Prettyfish classifier")
st.write("streamlit==1.19.0 | streamlit_webrtc==0.45.0 | xirsys/STUN/TURN| modelv1")

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
fish = ""


# -------------- Basic -----------------
# def videoFilter(frame: av.VideoFrame) -> av.VideoFrame:
#     img = frame.to_ndarray(format="bgr24")
#     # --detect model here--
#     pred_text, logits = process_img(img)
#     text =  "HELLO! " + pred_text    
#     cv2.putText(img, 
#                 text,
#                 bottomLeftCornerOfText,
#                 font, 
#                 fontScale, 
#                 fontColor, 
#                 lineType)
#     result_queue.put(logits)
#     return av.VideoFrame.from_ndarray(img, format="bgr24")


def videoFilter(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    # --detect model here--
    pred_text, proba = process_img(img)
    text =  "HELLO! " + fish
    # --text to print--   
    cv2.putText(img, 
                text,
                bottomLeftCornerOfText,
                font, 
                fontScale, 
                fontColor, 
                lineType)
    result_queue.put(proba)
    return av.VideoFrame.from_ndarray(img, format="bgr24")



# webrtc_streamer(key="example")
# webrtc_streamer(key="example", video_frame_callback=videoFilter)


## NOTE: To deploy it to cloud
## config necessary to establish media streaming connection 
## when the server is on remote host
## streamlit_webrtc uses webRTC for its video and audio streaming
## it has to access a STUN server in the global network for
## the remote peers to establish WebRTC connections...
## this code uses a free STUN server provided by Google
## the value of the rtc_configuration argument will be passed to
## the RTCPeerConnection constructor on the frontend.
webrtc_ctx = webrtc_streamer (
    key="prettyfish2345",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback = videoFilter,
    rtc_configuration = { 
        # --Add this line--
        # "iceServers": [{"urls":["stun:103.6.151.242:3478"]}] 
        # "iceServers": [{"urls":["stun:stunserver.stunprotocol.org:3478",
        #                         "stun:stun2.l.google.com:19302",
        #                         "stun:stun.l.google.com:19302"]}]  
        "iceServers":[
            {
                "urls":[ #"stun:openrelay.metered.ca:80",
                         # "stun:stun.l.google.com:19302",
                         "stun:hk-turn1.xirsys.com"
                       ]               
            },
            {
                "username": "Iy1RO4E0429UUzReltaix02EB-QqhiQ9JT_6Es2ShxeYEusv-tIMoBg0bm18RvcIAAAAAGQhIKhsanJp",
                "credential": "f178fd36-cc5a-11ed-b629-0242ac120004",
                "urls": [
                    "turn:hk-turn1.xirsys.com:80?transport=tcp",
                    "turn:hk-turn1.xirsys.com:3478?transport=tcp",
                    "turn:hk-turn1.xirsys.com:443?transport=tcp",   
                    "turn:hk-turn1.xirsys.com:5349?transport=tcp",
                    "turn:hk-turn1.xirsys.com:80?transport=udp",
                    "turn:hk-turn1.xirsys.com:3478?transport=udp",
                ] 
            },
            
            # {
            #     "urls":["turn:openrelay.metered.ca:443?transport=tcp"],
            #     "username":"6b5543aef5255ed431083a10",
            #     "credential":"dDGwNclpzebJ+wpG"
            # },
            # {
            #     "urls":["turn:openrelay.metered.ca:443"],
            #     "username":"6b5543aef5255ed431083a10",
            #     "credential":"dDGwNclpzebJ+wpG"
            # },
            # {
            #     "urls": "turn:openrelay.metered.ca:443?transport=tcp",
            #     "username": "openrelayproject",
            #     "credential": "openrelayproject",
            # },
        ],
        "iceTransportPolicy":"all"
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)



## NOTE: The video transformation with object detection and
## this loop displaying the result labels are running
## in different threads asynchronously.
## Then the rendered video frames and the labels displayed here
## are not strictly synchronized.

# -------------- Basic -----------------
# if st.checkbox("Show the detected labels", value=True):
#     if webrtc_ctx.state.playing:
#         labels_placeholder = st.empty()     
#         while True:
#             result = result_queue.get()
#             labels_placeholder.table(result)
#             result_queue.task_done()



## -------------- Mars ---------------- ##
if st.checkbox("Show logits", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        probas = []
        threshold = 0.8
        while True:
            result = result_queue.get()
            probas.append(tf.reshape(result, [7]))
            # --keep last 5 frames--
            if len(probas) > 10:     
                probas = probas[1:]
            # --get average of last 5 probas--
            avg = np.array(probas).mean(axis=0) 
            # --display probas--
            # labels_placeholder.table(avg)     
            # --display class--
            # labels_placeholder.table(pd.DataFrame({"fish":[class_names[avg.argmax()]]}))  
            # --any avg of the ele is more than threshold, return true-- 
            if (avg>threshold).any():
                fish_class = class_names[avg.argmax()]
            # else:
            #     fish_class = "not a fish"
            # fish_class_queue.put(fish_class) 
            fish = fish_class
            labels_placeholder.table(pd.DataFrame({"fish":[fish_class]}))
            result_queue.task_done()
            