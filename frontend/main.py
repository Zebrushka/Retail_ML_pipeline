import requests
import streamlit as st
from PIL import Image
import io
import base64
import time
import json

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
        # res = requests.post("http://62.148.235.120:8090/probability", files=files)
        res = requests.post("http://192.168.0.47:8090/probability", files=files)

        resolve = res.json()
        label = resolve.get("label")
        probability = resolve.get("probability")
        st.write(label, probability)

        result_image_byts = resolve.get("result")

        im = Image.open(io.BytesIO(base64.b64decode(result_image_byts)))

        st.image(im, width=300)
        st.write("Прошедшее время для предсказания класса товара: {}s".format(time.time() - t))
        # price_res = requests.post("http://62.148.235.120:8090/get_price", files=files)
        price_res = requests.post("http://192.168.0.47:8090/get_price", files=files)
        price_resolve = price_res.json()
        price = price_resolve.get("price")
        st.write("Цена: ", price, " рублей")
        st.write("Прошедшее время для распознования цены : {}s".format(time.time() - t))

        item = {'label': label, "probability": probability, "price": price, "result": result_image_byts}
        # item = [label, probability, price, result_image_byts]

        # requests.post("http://62.148.235.120:8090/write_db", label, probability, price, result_image_byts)
        requests.post("http://192.168.0.47:8090/write_db", json=item)







#TODO сделать раскрывающийся список, при нажатии на него - подгружать историю всех запросов

if st.button("View history"):

    history = requests.get("http://192.168.0.47:8090/get_history")
    dictData = json.loads(history)

    st.write(history)
    container = st.container()


