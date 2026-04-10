import streamlit as st
from predict import predict_image
from PIL import Image

st.title("🧠 Brain Tumor Detector")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    image.save("temp.jpg")

    result, marked_img = predict_image("temp.jpg")

    st.subheader("Result:")
    st.success(result)
    
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    
    if marked_img is not None:
        with col2:
            st.image(marked_img, caption="Tumor Location Marked", use_container_width=True)