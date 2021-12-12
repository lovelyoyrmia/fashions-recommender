import os
import streamlit as st
import pickle
import numpy as np
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image

FILENAMES = np.array(pickle.load(open('models/filenames.pkl', 'rb')))
FEATURES_LIST = pickle.load(open('models/embeddings.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
  model,
  GlobalMaxPool2D()
])

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_image = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_image)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    _, indices = neighbors.kneighbors([features])

    return indices

def load_image(img):
    try:
        with open(os.path.join('uploads', img.name), 'wb') as f:
            f.write(img.getbuffer())
        return 1
    except:
        return 0

st.title('Fashion recommender system')

file_upload = st.file_uploader('Browse your photo', type=['jpg', 'png', 'jpeg'])

if file_upload is not None:
    if load_image(file_upload):
        display_img = Image.open(file_upload)
        st.image(display_img, use_column_width=True)
        features = feature_extraction(os.path.join('uploads', file_upload.name), model)

        indices = recommend(features, FEATURES_LIST)

        col1, col2, col3, col4 = st.beta_columns(4)

        with col1:
            st.image(FILENAMES[indices[0][0]])
        with col2:
            st.image(FILENAMES[indices[0][1]])
        with col3:
            st.image(FILENAMES[indices[0][2]])
        with col4:
            st.image(FILENAMES[indices[0][3]])
    else:
        st.error('Some errors occured in file upload')
