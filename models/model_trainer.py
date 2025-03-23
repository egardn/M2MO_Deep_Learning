import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

class ModelTrainer:
    def __init__(self, model, train_dataset, test_dataset, model_name="Model"):
        """
        Initialize the model trainer
        
        Parameters:
        model: The compiled TensorFlow model
        train_dataset: TF dataset for training
        test_dataset: TF dataset for testing
        model_name: Name of the model for display purposes
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_name = model_name
        self.history = None
        self.training_time = None
        
    def train(self, epochs=20, early_stopping=True, patience=5):
        """
        Train the model
        
        Parameters:
        epochs: Number of training epochs
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        
        Returns:
        Training history
        """
        callbacks = []
        
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
            callbacks.append(es_callback)
        
        # Time the training
        start_time = time.time()
        
        # Train the model
        self.history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.test_dataset,
            callbacks=callbacks
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        
        return self.history
    
    def evaluate(self):
        """
        Evaluate the model on the test dataset
        
        Returns:
        Test loss and accuracy
        """
        results = self.model.evaluate(self.test_dataset)
        
        print(f"\n{self.model_name} Test Loss: {results[0]:.4f}")
        print(f"{self.model_name} Test Accuracy: {results[1]:.4f}")
        
        return results
    
    def plot_training_history(self):
        """
        Plot the training history (accuracy and loss)
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        # Plot accuracy
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{self.model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.model_name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the model to disk
        
        Parameters:
        filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")