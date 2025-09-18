import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence

# Parameters
max_features = 10000  # Vocabulary size
maxlen = 500          # Max review length
embedding_dim = 32

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Build Simple RNN model
model = Sequential([
    Embedding(max_features, embedding_dim, input_length=maxlen),
    SimpleRNN(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

# Save model
model.save('simple_rnn_imdb.h5')
