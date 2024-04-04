import streamlit as st
#from PIL import Image
import numpy as np
#from ultralytics import YOLO
#from utils import YOLO_DETECT

#model = YOLO('yolov8n.pt')

st.title("YOLO Object Detection")
c1,c2=st.columns(2)


selected_option = st.radio("Select an option:", ["Uplode from device" , "Use Camera", ])
if(selected_option =="Uplode from device"):
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if(selected_option =="Use Camera"):
    uploaded_file = st.camera_input("Using camera...")





if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)


    im ,items , counts= YOLO_DETECT(img_array)

    
    st.image(im, caption='Detection Results', use_column_width=True)

    # st.markdown("**All Discovered Objects**")
    st.markdown("<h3><b>All Discovered Objects</b></h3>", unsafe_allow_html=True)
    st.markdown(items)

    # st.markdown("**Filtered Object Count**")
    st.markdown("<h3><b>Filtered Object Count</b></h3>", unsafe_allow_html=True)
    
    data = {'Object name': items, 'Object Count': counts}
    st.table(data)

    

    
    
