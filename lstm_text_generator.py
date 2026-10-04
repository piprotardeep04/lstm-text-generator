import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load data
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

file_path = tf.keras.utils.get_file("shakespeare.txt", url)

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().lower()

text = text.replace("\n", " ")
text = re.sub(r"\s+", " ", text)

print("Text length:", len(text))


# Character encoding
chars = sorted(set(text))

char_to_num = {c: i for i, c in enumerate(chars)}
num_to_char = np.array(chars)

numbers = np.array(
    [char_to_num[c] for c in text],
    dtype=np.int32
)

print("Number of characters:", len(chars))


# Create sequences
sequence_length = 60
step = 3

inputs = []
targets = []

for i in range(0, len(numbers) - sequence_length, step):
    inputs.append(numbers[i:i + sequence_length])
    targets.append(numbers[i + 1:i + sequence_length + 1])

inputs = np.array(inputs)
targets = np.array(targets)

print("Number of sequences:", len(inputs))


# Train/validation split
split = int(len(inputs) * 0.9)

x_train = inputs[:split]
y_train = targets[:split]

x_val = inputs[split:]
y_val = targets[split:]

print("Training data:", len(x_train))
print("Validation data:", len(x_val))


# TensorFlow datasets
batch_size = 64

train_data = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .shuffle(5000)
    .batch(batch_size)
)

val_data = (
    tf.data.Dataset
    .from_tensor_slices((x_val, y_val))
    .batch(batch_size)
)


# LSTM model
model = tf.keras.Sequential([
    layers.Input(shape=(sequence_length,)),
    layers.Embedding(len(chars), 128),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.2),
    layers.Dense(len(chars), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Training callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_lstm_model.keras",
    monitor="val_loss",
    save_best_only=True
)

print("\nTraining started...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop, checkpoint]
)


# Text generation
def generate_text(seed, length=200, temperature=0.8):

    result = seed.lower()

    for _ in range(length):

        current = result[-sequence_length:]

        if len(current) < sequence_length:
            current = current.rjust(sequence_length)

        encoded = np.array([[
            char_to_num.get(c, 0)
            for c in current
        ]])

        prediction = model.predict(
            encoded,
            verbose=0
        )[0, -1]

        prediction = np.log(
            prediction + 1e-8
        ) / temperature

        probabilities = np.exp(prediction)
        probabilities /= probabilities.sum()

        next_char = np.random.choice(
            len(chars),
            p=probabilities
        )

        result += num_to_char[next_char]

    return result


# Generate some text
seeds = [
    "to be or not to be",
    "the king",
    "love is"
]

print("\n-----------------------------")
print("Generated Text")
print("-----------------------------")

with open("generated_output.txt", "w", encoding="utf-8") as f:

    for seed in seeds:

        output = generate_text(
            seed,
            200,
            0.8
        )

        print("\nSeed:", seed)
        print(output)

        f.write("\nSeed: " + seed + "\n")
        f.write(output + "\n")


# Temperature experiment
print("\n-----------------------------")
print("Temperature Experiment")
print("-----------------------------")

for temperature in [0.5, 0.8, 1.0]:

    print("\nTemperature:", temperature)

    output = generate_text(
        "the king",
        100,
        temperature
    )

    print(output)


# Save model
model.save("lstm_text_generator.keras")

print("\nModel saved successfully.")
print("Task completed successfully.")