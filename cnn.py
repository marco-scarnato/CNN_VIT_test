import tensorflow as tf
from tensorflow import keras
from keras import layers

class CNN_model():
    
    def __init__(self, num_classes):
        super().__init__()
        self.model = keras.Sequential([
            # Block One
            layers.Conv2D(filters=32,
                        kernel_size=3, 
                        activation='relu', 
                        padding='same',
                        input_shape=[256, 256, 3]),
            layers.MaxPool2D(),

            # Block Two
            layers.Conv2D(filters=64,
                        kernel_size=3,
                        activation='relu',
                        padding='same'),
            layers.MaxPool2D(),

            # Block Three
            layers.Conv2D(filters=128, 
                        kernel_size=3, 
                        activation='relu', 
                        padding='same'),
            layers.Conv2D(filters=128, 
                        kernel_size=3, 
                        activation='relu', 
                        padding='same'),
            layers.MaxPool2D(),

            # Head
            layers.Flatten(),
            layers.Dense(256, activation='relu'),  # Layer fully connected aggiuntivo per migliorare la capacità di apprendimento
            layers.Dropout(0.5),  # Dropout per evitare overfitting
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid'),
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['categorical_accuracy' if num_classes > 2 else 'binary_accuracy'],
        )