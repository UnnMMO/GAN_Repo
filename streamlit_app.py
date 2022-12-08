import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np


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
    
    noise=np.random.normal(loc=0, scale=1, size=(100,100))

    gen_image = saved_model.predict(noise)
    
    #plt.imshow(gen_image[0].reshape(28,28), interpolation='none')
    plt.imshow(gen_image, interpolation='none')
    plt.show()

    

