import streamlit as st
from PIL import Image

@st.cache
def load_model():
    # El modelo se lee de la carpeta 'model'
    saved_model = keras.models.load_model('path_to_my_model.h5')
    return saved_model


saved_model = load_model()
    
btn = col2.button('Generar imágen aleatoria')
if btn:
    from matplotlib import pyplot as plt
    plt.imshow(saved_model.gen_image[11].reshape(28,28), interpolation='none')
    plt.show()

    

