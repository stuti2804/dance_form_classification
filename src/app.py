import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Dance Form Classifier",
    page_icon="💃",
    layout="wide"
)

st.title("Indian Classical Dance Form Classifier")
st.markdown("Upload an image to classify the dance form!")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mobilenetv2_model.keras')
        return model
    except Exception as e:
        st.error("Model loading failed. Make sure 'dance_classifier_model.h5' is in the same directory.")
        return None

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

# Classes (update if your model uses different order)
dance_forms = {
    0: "Bharatanatyam",
    1: "Kathak",
    2: "Kathakali",
    3: "Kuchipudi",
    4: "Manipuri",
    5: "Mohiniyattam",
    6: "Odissi",
    7: "Sattriya"
}

def main():
    model = load_model()
    
    if model:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption='Uploaded Image', use_column_width=True)
            
            with st.spinner('Predicting...'):
                try:
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)
                    predicted_class = np.argmax(prediction[0])
                    confidence = float(prediction[0][predicted_class])
                    
                    with col2:
                        st.markdown(f"### Predicted Dance Form: **{dance_forms[predicted_class]}**")
                        st.markdown(f"Confidence: **{confidence:.2%}**")
                        st.markdown("#### Probability Distribution")
                        for i, prob in enumerate(prediction[0]):
                            st.markdown(f"{dance_forms[i]}: {prob:.2%}")
                            st.progress(float(prob))
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
