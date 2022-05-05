import requests
import streamlit as st
from PIL import Image
import io
import base64
import time

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Retail ML web app")

# displays a file uploader widget
image = st.file_uploader("Choose an image")

# displays a button
if st.button("Get label"):
    if image is not None:
        t = time.time()
        files = {"file": image.getvalue()}
        res = requests.post("http://62.148.235.120:8090/probability", files=files)

        resolve = res.json()
        label = resolve.get("label")
        probability = resolve.get("probability")
        st.write(label, probability)

        result_image_byts = resolve.get("result")

        im = Image.open(io.BytesIO(base64.b64decode(result_image_byts)))

        st.image(im, width=300)

        price_res = requests.post("http://62.148.235.120:8090/get_price", files=files)
        price_resolve = price_res.json()
        price = resolve.get("price")
        st.write("Цена: " + price + " рублей")
        st.write("elapsed time : {}s".format(time.time() - t))

        requests.post("http://62.148.235.120:8090/write_db", label, probability, price, result_image_byts)







#TODO сделать раскрывающийся список, при нажатии на него - подгружать историю всех запросов

if st.button("View history"):

    history = requests.get("http://62.148.235.120:8090/get_history")
    history.json()
    st.write(history[0]['image'])
    container = st.container()


