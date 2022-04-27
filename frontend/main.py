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
        res = requests.post("http://62.148.235.120:8090/probability", files=files)

        resolve = res.json()
        label = resolve.get("label")
        probability = resolve.get("probability")
        # results = resolve.get("results")
        # st.image(results, width=300)
        st.write(label, probability)