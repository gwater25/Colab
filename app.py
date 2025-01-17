import tensorflow as tf
import streamlit as st
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cifar10_model.h5')



# Streamlit app
st.title("Image Classification App")

def label_class(class_number):
    switch = {
        0: "Airplane",
        1: "Automobile",
        2: "Bird",
        3: "Cat",
        4: "Deer",
        5: "Dog",
        6: "Frog",
        7: "Horse",
        8: "Ship",
        9: "Truck",
    }
    return switch.get(class_number.numpy(), "Invalid class number")

with st.sidebar:
    st.title("Image Classification App")
    # Tooltips also support markdown
    radio_markdown = '''
    Upload an image, There are **limitations**!
    '''.strip()
    limit_expander = st.expander("**NOTICE**", expanded=False)
    with limit_expander:
        st.caption('This **APP** is limited to classifying these images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.')
    # Upload an image through Streamlit
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], help=radio_markdown)

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
        # Display the highest confidence prediction
        highest_confidence_idx = top_classes[0]
        highest_confidence_label = label_class(highest_confidence_idx)
        highest_confidence_score = 100 * top_scores[0].numpy()
        st.write(f"Highest Confidence: Class {highest_confidence_label}, Confidence: {highest_confidence_score:.2f}%")
        
        st.write("Top Predictions:")
        for i, (class_idx, score) in enumerate(zip(top_classes, top_scores)):
            class_label = label_class(class_idx)
            st.write(f"{i + 1}. Class: {class_label}, Confidence: {100 * score:.2f}%")        
        st.image(image, caption="Uploaded Image", use_column_width=True, width=300)




except Exception as e:
    st.write("An error occurred:", str(e))
