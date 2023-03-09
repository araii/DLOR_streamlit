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


## 'Path_to/my_model.h5'

@st.cache_resource
def load_model():
    ## load model local
    # tf.keras.models.load_model('saved_model/prettyfish_model')
    ## load model streamlit
    # 
    return tf.keras.models.load_model(HERE / './saved_model/prettyfish_model')
 
model = load_model()


# class names
class_names = ['clownfish', 
               'longnose_butterflyfish',
               'oranda_goldfish',
               'powder_blue_tang',
               'queen_angelfish',
               'snakeskin_discus',
               'cuddlefish'] # Mars

# inputs
def process_img (img):
    ## resiZe
    img = cv2.resize(img, (299,299))
    ## expand dims to 4D
    test_img = np.expand_dims(img, axis=0)
    ## log-outs of probabilities
    logits = model.predict(test_img)
    ## select index of highest logits
    ## Mars Start....
    # scaler - 0 Dimension tensor i.e ()
    # vector - 1 Dimension tensor i.e (9, )
    # matrix - 2 Dimension tensor i.e (1, 6)
    # output of tf.nn.softmax() is tf.EagerTensor
    proba = tf.nn.softmax(logits) 
    # threshold = 0.9
    ## numpy() - converts tf.EagerTensor into np.array
    ## np.any() - any of the ele is true, return true
    # if (proba > threshold).numpy().any():  
    #     predict_output = tf.argmax(logits, -1).numpy()
    # else:
    #     predict_output = [6]
    ## Mars End....
    ## run thru classifier to get name
    # pred_text = class_names[predict_output[0]]   # running every frame...
    ##---------------------------##
    ## resize image
    # img = tf.keras.utils.load_img(img, target_size=(299,299))
    ## normalize
    # img = tf.keras.utils.img_to_array(load_img).astype('float32')/255
    # y_pred = model.predict(test_img, verbose=1)[0]
    # y_pred_class = np.argmax(y_pred)
    # y_pred_prob = y_pred[y_pred_class]*100
    pred_text=""
    return pred_text, proba   # extract proba   
    

st.title("My first streamlit app")
st.write("Hello, world-6")


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,40)
fontScale = 1
fontColor = (255,255,255)
lineType = 2

# note: the callback will be called in another thread
# so use a queue here for thread-safety to pass the data
# from inside to outside the callback
result_queue: "queue.Queue[]" = queue.Queue()


def videoFilter(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    # detect model here!
    pred_text, proba = process_img(img)
    text =  "HELLO! "+ pred_text     # text to print 
    # cv2.putText(img, 
    #             text,
    #             bottomLeftCornerOfText,
    #             font, 
    #             fontScale, 
    #             fontColor, 
    #             lineType)
    result_queue.put(proba)
    return av.VideoFrame.from_ndarray(img, format="bgr24")



# webrtc_streamer(key="example")

# webrtc_streamer(key="example", video_frame_callback=videoFilter)


# to deploy it to cloud
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
        "iceServers": [{"urls":["stun:stun.l.google.com:19302"]}]   # stun2.l.google.com:19302
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)



if st.checkbox("Show logits", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # note: the video transformation with obj detection and
        # this loop displaying the result labels are running
        # in different threads asychronously
        #then the rendered video frames and labels displayed here
        # are not strictly synchronized
        probas = []
        threshold = 0.9
        while True:
            result = result_queue.get()
            #st.write(result)
            ## Mars start
            probas.append(tf.reshape(result, [6]))
            if len(probas) > 10:  #keep last 5 frames...
                probas = probas[1:]
            avg = np.array(probas).mean(axis=0) # avg of last 5 probas
            # labels_placeholder.table(avg)  # display probas
            # # labels_placeholder.table(pd.DataFrame({"fish":[class_names[avg.argmax()]]}))  # display class
            if (avg>threshold).any():
                fish_class = class_names[avg.argmax()]
            else:
                fish_class = "not a fish"
            
            labels_placeholder.table(pd.DataFrame({"fish":[fish_class]}))