import tensorflow
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

filenames = np.array(pickle.load(open('filenames.pkl', 'rb')))
feature_list = pickle.load(open('embeddings.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
  model,
  GlobalMaxPool2D()
])

img = image.load_img('sample/1211.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expand_image = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expand_image)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distance, indices = neighbors.kneighbors([normalized_result])