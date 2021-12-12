import streamlit as st
from PIL import Image


@st.cache()
def load_image(img):
    image = Image.open(img)
    return image

st.title('Fashion recommender system')

file_upload = st.file_uploader('Browse your photo', type=['jpg', 'png', 'jpeg'])

if file_upload is not None:
    image = load_image(file_upload)
    st.image(image, use_column_width=True)


