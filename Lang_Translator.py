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


model_path = "s2s.h5"  
try:
    model = load_model(model_path)
    latent_dim = 256
    num_encoder_tokens = 70
    num_decoder_tokens = 91
    max_encoder_seq_length = 14
    max_decoder_seq_length = 59

    # Encoder and Decoder model setup based on `s2s.h5`
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    # Define character sets
    input_characters = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?"))
    target_characters = sorted(list("abcdefghijklmnopqrstuvwxyzéèàçûô\t\n"))
    input_token_index = {char: i for i, char in enumerate(input_characters)}
    target_token_index = {char: i for i, char in enumerate(target_characters)}
    reverse_target_char_index = {i: char for char, i in target_token_index.items()}

    def preprocess_input(text):
        encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
        for t, char in enumerate(text):
            if t < max_encoder_seq_length and char in input_token_index:
                encoder_input_data[0, t, input_token_index[char]] = 1.0
        return encoder_input_data

    def decode_sequence(input_seq):
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_token_index.get('\t', 0)] = 1.0
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index.get(sampled_token_index, '')
            decoded_sentence += sampled_char
            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0
            states_value = [h, c]
        return decoded_sentence
except Exception as e:
    st.warning(f"Neural translation model could not be loaded. Only Google Translate is available. Error: {e}")
    model = None

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




