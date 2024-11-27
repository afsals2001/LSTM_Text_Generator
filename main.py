import streamlit as st
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


st.set_page_config("LSTM Text Generator")

# Global variables for text and mappings
SEQUENCE_LENGTH = 40
STEP_SIZE = 3
text = ''
unique_characters = []
char_to_index = {}
index_to_char = {}

# Function to load and process the text data
def load_text_data(uploaded_file):
    global text, unique_characters, char_to_index, index_to_char

    try:
        # Read the text from the uploaded file and convert to lowercase
        text = uploaded_file.read().decode(encoding='utf-8').strip().lower()
        
        if len(text) == 0:
            raise ValueError("The uploaded file is empty or contains no readable text.")

        # Get unique characters in the text
        unique_characters = sorted(set(text))

        # Create dictionaries to map characters to indices and vice versa
        char_to_index = dict((char, i) for i, char in enumerate(unique_characters))
        index_to_char = dict((i, char) for i, char in enumerate(unique_characters))
    except Exception as e:
        raise ValueError(f"Error processing the uploaded file: {e}")

# Function to train the model
def train_model(uploaded_file, epochs=4):
    load_text_data(uploaded_file)

    if len(text) <= SEQUENCE_LENGTH:
        raise ValueError(f"Text length ({len(text)}) must be greater than the sequence length ({SEQUENCE_LENGTH}).")

    # Generate sequences and corresponding next characters
    sequences = []
    next_characters = []

    for i in range(0, len(text) - SEQUENCE_LENGTH, STEP_SIZE):
        sequences.append(text[i: i + SEQUENCE_LENGTH])
        next_characters.append(text[i + SEQUENCE_LENGTH])

    # Prepare input (x) and output (y) data arrays
    x = np.zeros((len(sequences), SEQUENCE_LENGTH, len(unique_characters)), dtype=np.bool_)
    y = np.zeros((len(sequences), len(unique_characters)), dtype=np.bool_)

    # Convert sequences and next characters into one-hot encoding
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            x[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_characters[i]]] = 1

    # Define model architecture
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(unique_characters))))
    model.add(Dense(len(unique_characters)))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

    # Train the model
    model.fit(x, y, batch_size=256, epochs=epochs)

    # Save the trained model
    model.save("text_generator_model.keras")

    return model

# Function to generate text
def generate_text(model, length=300, temperature=1.0):
    if len(text) <= SEQUENCE_LENGTH:
        raise ValueError(f"Text length ({len(text)}) must be greater than the sequence length ({SEQUENCE_LENGTH}).")

    # Start index for generating text
    start_index = random.randint(0, len(text) - SEQUENCE_LENGTH - 1)
    generated_text = ''
    current_sequence = text[start_index: start_index + SEQUENCE_LENGTH]
    generated_text += current_sequence

    for i in range(length):
        # Prepare input for prediction
        x = np.zeros((1, SEQUENCE_LENGTH, len(unique_characters)))
        for t, char in enumerate(current_sequence):
            x[0, t, char_to_index[char]] = 1

        # Predict next character
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated_text += next_character
        current_sequence = current_sequence[1:] + next_character

    return generated_text

# Function to sample from predicted probabilities
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Streamlit UI
st.title("Text Generation with LSTM")

# File upload section
uploaded_file = st.file_uploader("Upload Text File for Training", type=["txt"])

if uploaded_file:
    try:
        # Check text content length
        load_text_data(uploaded_file)

        if len(text) <= SEQUENCE_LENGTH:
            st.error(f"Uploaded text file is too short. Please upload a file with at least {SEQUENCE_LENGTH + 1} characters.")
        else:
            # Training section
            if st.button("Train Model"):
                st.write("Training the model... This might take a while.")
                model = train_model(uploaded_file)
                st.success("Model training complete!")

            # Text generation section
            if st.button("Generate Text"):
                try:
                    # Load model if not already trained
                    if 'model' not in locals():
                        model = tf.keras.models.load_model('text_generator_model.keras')

                    temperature = st.slider("Select Temperature", 0.2, 1.0, 0.5, 0.1)
                    length = st.slider("Select Text Length", 100, 1000, 300, 50)

                    generated_text = generate_text(model, length, temperature)
                    st.subheader("Generated Text:")
                    st.text_area("", generated_text, height=300)
                except Exception as e:
                    st.error(f"Error generating text: {e}")
    except Exception as e:
        st.error(f"Error: {e}")