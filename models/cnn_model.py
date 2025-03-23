import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import tensorflow.keras as keras
from kerastuner.tuners import RandomSearch

class CNN(keras.Model):
    def __init__(self, num_classes=8, in_channels=1, img_size=64, 
                 first_filters=32, filters_multiplier=2, dense_neurons=128, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.img_size = img_size

        # Calculate filters for each layer based on the multiplier
        second_filters = first_filters * filters_multiplier
        third_filters = second_filters * filters_multiplier

        # Feature extraction layers
        self.features = keras.Sequential([
            # First conv layer
            layers.Conv2D(first_filters, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            # Second conv layer
            layers.Conv2D(second_filters, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            # Third conv layer
            layers.Conv2D(third_filters, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2)
        ])

        # Calculate feature size after 3 pooling layers
        feature_size = img_size // 8

        # Classification layers
        self.classifier = keras.Sequential([
            layers.Flatten(),
            layers.Dense(dense_neurons, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, x, training=False):
        x = self.features(x, training=training)
        x = self.classifier(x, training=training)
        return x

# Keep the existing functions
def build_cnn_tunable(hp):
    """
    Build a CNN model with tunable hyperparameters
    
    Parameters:
    hp: HyperParameters instance from KerasTuner
    
    Returns:
    A compiled Keras model
    """
    # Input shape - assuming 28x28 for MNIST or other sizes
    input_shape = (None, None, 1)  # Height, width, channels
    
    # Create model
    model = models.Sequential()
    
    # First convolutional block
    model.add(layers.Conv2D(
        filters=hp.Int('conv1_filters', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('conv1_kernel', values=[3, 5]),
        activation='relu',
        padding='same',
        input_shape=input_shape
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional block
    model.add(layers.Conv2D(
        filters=hp.Int('conv2_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
        activation='relu',
        padding='same'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Third convolutional block (optional)
    if hp.Boolean('include_third_conv', default=True):
        model.add(layers.Conv2D(
            filters=hp.Int('conv3_filters', min_value=64, max_value=256, step=64),
            kernel_size=hp.Choice('conv3_kernel', values=[3, 5]),
            activation='relu',
            padding='same'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and dense layers
    model.add(layers.Flatten())
    
    # First dense layer
    model.add(layers.Dense(
        units=hp.Int('dense1_units', min_value=64, max_value=512, step=64),
        activation='relu'
    ))
    model.add(layers.Dropout(
        rate=hp.Float('dropout1_rate', min_value=0.0, max_value=0.5, step=0.1)
    ))
    
    # Second dense layer (optional)
    if hp.Boolean('include_second_dense', default=True):
        model.add(layers.Dense(
            units=hp.Int('dense2_units', min_value=32, max_value=256, step=32),
            activation='relu'
        ))
        model.add(layers.Dropout(
            rate=hp.Float('dropout2_rate', min_value=0.0, max_value=0.5, step=0.1)
        ))
    
    # Output layer - num_classes will be determined by the dataset
    model.add(layers.Dense(hp.Choice('num_classes', values=[8, 10]), activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def tune_cnn(train_dataset, val_dataset, num_classes, project_name='cnn_tuning'):
    """
    Tune CNN hyperparameters using KerasTuner
    
    Parameters:
    train_dataset: TF dataset for training
    val_dataset: TF dataset for validation
    num_classes: Number of output classes
    project_name: Name for the tuning project
    
    Returns:
    The best tuned model
    """
    # Define the hyperparameter search
    tuner = RandomSearch(
        lambda hp: build_cnn_tunable(hp),
        objective='val_accuracy',
        max_trials=10,
        directory='tuning',
        project_name=project_name
    )
    
    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Search for the best hyperparameters
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[early_stopping]
    )
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    
    # Print the best hyperparameters
    print("Best CNN hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
    
    return model