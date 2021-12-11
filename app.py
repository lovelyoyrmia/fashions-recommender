import os
import tensorflow
import numpy as np
import pickle
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
  model,
  GlobalMaxPool2D()
])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_image = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_image)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

filenames = []

for imgfile in os.listdir('images'):
  filenames.append(os.path.join('images', imgfile))

feature_list = []

for imgfile in tqdm(filenames):
  feature_list.append(extract_features(imgfile, model))

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
