import streamlit as st
from deep_translator import GoogleTranslator

# Language code mappings for deep-translator
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
        try:
            # Initialize Google Translator from deep-translator
            translator = GoogleTranslator(source='auto', target=target_language)
            translated_text = translator.translate(text_to_translate)
            st.write("**Translated Text:**")
            st.write(translated_text)
        except Exception as e:
            st.error(f"Translation failed: {e}")
    else:
        st.warning("Please enter text to translate.")




