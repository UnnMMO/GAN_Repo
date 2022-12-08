import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras


@st.cache
def load_model():
    # El modelo se lee de la carpeta 'model'
    saved_model = keras.models.load_model('GAN_FashionMNIST.h5')
    return saved_model


saved_model = load_model()

col1, col2, col3 = st.columns([2, 5, 2])
btn = col2.button('Generar imágen aleatoria')
if btn:
    from matplotlib import pyplot as plt
    plt.imshow(saved_model.gen_image[11].reshape(28,28), interpolation='none')
    plt.show()

    

