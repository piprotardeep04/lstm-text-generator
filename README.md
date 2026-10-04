# Character-Level LSTM Text Generator

This project is a character-level text generation model built using Python, TensorFlow, and Keras. The model is trained on the Tiny Shakespeare dataset and learns character patterns to generate new text from a given starting phrase.

## Project Overview

The project uses an LSTM (Long Short-Term Memory) neural network to predict the next character based on the previous characters. The text is converted into numerical sequences before training.

The model uses:
- Embedding layer
- LSTM layer
- Dropout layer
- Dense layer with Softmax activation

## Features

- Downloads the Tiny Shakespeare dataset automatically
- Preprocesses and encodes the text
- Creates character-based training sequences
- Trains an LSTM text generation model
- Uses training and validation data
- Uses Early Stopping and Model Checkpointing
- Generates text from different starting phrases
- Tests different temperature values
- Saves the generated text and trained model

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- LSTM

## How to Run

Install the required dependencies:

pip install -r requirements.txt