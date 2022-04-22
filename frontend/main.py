import requests
import streamlit as st
from PIL import Image

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Style transfer web app")

# displays a file uploader widget
image = st.file_uploader("Choose an image")

# displays a button
if st.button("Get label"):
    if image is not None:
        files = {"file": image.getvalue()}
        res = requests.post("http://127.0.0.1:8090/probability", files=files)
        probability = res.json()
        image = Image.open(probability.get("name"))
        label = probability.get("label")
        st.image(image, width=500)
        st.write(label)