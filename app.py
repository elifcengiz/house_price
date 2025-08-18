#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import sys

#Loading the Model
try:
    model = load_model("dog_breed.keras")
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error("An error occurred while loading the model.❌")
    st.error(str(e))
    sys.exit()  # Model yüklenemezse uygulamayı durdur

#Name of Classes
class_names = ["rhodesian_ridgeback","shih-tzu","eskimo_dog"]

#Setting Title of App
st.title("Dog Breed Finder")
st.markdown("Upload a photo of your dog")

#Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
submit = st.button('Predict')
#On predict button click
if submit:


    if dog_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)


        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (224,224))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,224,224,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        st.title(str"This dog breed :" + class_names[np.argmax(Y_pred)]))

 