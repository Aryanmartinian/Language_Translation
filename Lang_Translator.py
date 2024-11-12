import streamlit as st
from googletrans import Translator
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense

language_mapping = {
   "Bengali": "bn",
   "French": "fr",
   "Tamil": "ta",
   "Russian": "ru"
}


# Streamlit Interface
st.title("Language Translator")
text_to_translate = st.text_area("Enter text to translate:")
selected_language_name = st.selectbox("Select target language:", list(language_mapping.keys()))

if st.button("Translate"):
    if text_to_translate:
        target_language = language_mapping[selected_language_name]
        translator = Translator()
        
        # Use Google Translate if model isn't suitable for the selected language
        if selected_language_name != "French" or model is None:
            try:
                translated_text = translator.translate(text_to_translate, dest=target_language).text
                st.write("**Translated Text:**")
                st.write(translated_text)
            except Exception as e:
                st.error(f"Translation failed: {e}")
        else:
            # Use neural model for English to French translation if available
            input_seq = preprocess_input(text_to_translate)
            translation = decode_sequence(input_seq)
            st.write("**Translated Text (Neural Model):**")
            st.write(translation)
    else:
        st.warning("Please enter text to translate.")




