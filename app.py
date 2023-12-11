import tensorflow as tf
import streamlit as st
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cifar10_model.keras')

# Streamlit app
st.title("Image Classification App")

# Upload an image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

try:
    if uploaded_file is not None:
        # Preprocess the image
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(32, 32))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)  # Create a batch

        # Normalize the image (assuming the model was trained with normalized input)
        image_array /= 255.0

        # Make predictions
        predictions = model.predict(image_array)
        top_classes = tf.argsort(predictions[0], direction='DESCENDING')[:3]  # Display top 3 predictions
        top_scores = tf.nn.softmax(predictions[0][top_classes])

        # Display the results
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Top Predictions:")
        for i, (class_idx, score) in enumerate(zip(top_classes, top_scores)):
            st.write(f"{i + 1}. Class: {class_idx}, Confidence: {100 * score:.2f}%")
except Exception as e:
    st.write("An error occurred:", str(e))
