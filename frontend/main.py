import requests
import streamlit as st
from PIL import Image

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Retail ML web app")

# displays a file uploader widget
image = st.file_uploader("Choose an image")

# displays a button
if st.button("Get label"):
    if image is not None:
        st.image(image, width=300)
        files = {"file": image.getvalue()}
        res = requests.post("http://192.168.0.12:8090/probability", files=files)
        resolve = res.json()
        probability = resolve.get("probability")
        label = resolve.get("label")
        st.write(label, probability)