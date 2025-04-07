import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class ModelEvaluator:
    def __init__(self, model_trainers=None, test_dataset=None, class_names=None):
        """
        Initialize model evaluator
        
        Parameters:
        model_trainers: Dictionary of {name: ModelTrainer} or list of ModelTrainer objects
        test_dataset: Test dataset to evaluate on
        class_names: Names of classes (optional)
        """
        if model_trainers is None:
            self.model_trainers = {}
        elif isinstance(model_trainers, list):
            self.model_trainers = {trainer.model_name: trainer for trainer in model_trainers}
        else:
            self.model_trainers = model_trainers
            
        self.test_dataset = test_dataset
        self.class_names = class_names
        self.results = {}
        
    def add_model(self, name, trainer):
        """Add a model trainer to the evaluator"""
        self.model_trainers[name] = trainer
    
    def evaluate_all(self, verbose=1):
        """Evaluate all models and store the results"""
        for name, trainer in self.model_trainers.items():
            if verbose:
                print(f"\nEvaluating {name}...")
            
            # Use test dataset from trainer if none provided
            dataset = self.test_dataset if self.test_dataset is not None else trainer.test_dataset
            
            loss, accuracy, preds, labels = self.evaluate_model(trainer.model, dataset)
            
            if verbose:
                print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
            
            self.results[name] = {
                'loss': loss,
                'accuracy': accuracy,
                'predictions': preds,
                'true_labels': labels,
                'history': trainer.history.history if hasattr(trainer, 'history') else None
            }
        
        return self.results
    
    def evaluate_model(self, model, dataset):
        """Evaluate a model and return loss, accuracy, predictions and true labels"""
        # Get all batches from dataset
        all_images = []
        all_labels = []
        
        for images, labels in dataset:
            all_images.append(images)
            all_labels.append(labels)
        
        # Concatenate batches
        all_images = tf.concat(all_images, axis=0)
        all_labels = tf.concat(all_labels, axis=0)
        
        # Evaluate model
        loss, accuracy = model.evaluate(all_images, all_labels, verbose=0)
        
        # Get predictions
        predictions = model.predict(all_images)
        pred_classes = tf.argmax(predictions, axis=1).numpy()
        true_classes = all_labels.numpy()
        
        return loss, accuracy, pred_classes, true_classes
    
    def compare_training_history(self, figsize=(12, 5)):
        """Plot comparison of training histories for all models"""
        if not self.results:
            print("No evaluation results available. Run evaluate_all() first.")
            return
            
        plt.figure(figsize=figsize)
        
        # Accuracy subplot
        plt.subplot(1, 2, 1)
        for name, result in self.results.items():
            if result['history'] is not None:
                plt.plot(result['history']['accuracy'], label=f'{name} Train')
                plt.plot(result['history']['val_accuracy'], label=f'{name} Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss subplot
        plt.subplot(1, 2, 2)
        for name, result in self.results.items():
            if result['history'] is not None:
                plt.plot(result['history']['loss'], label=f'{name} Train')
                plt.plot(result['history']['val_loss'], label=f'{name} Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, figsize=(12, 5)):
        """Plot confusion matrices for all models"""
        if not self.results:
            print("No evaluation results available. Run evaluate_all() first.")
            return
            
        n_models = len(self.results)
        
        if n_models == 1:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_models, figsize=(figsize[0] * n_models // 2, figsize[1]))
        
        for i, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            
            current_ax = axes[i] if n_models > 1 else axes
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=current_ax)
            current_ax.set_title(f'{name} Confusion Matrix')
            current_ax.set_xlabel('Predicted')
            current_ax.set_ylabel('True')
        
        plt.tight_layout()
        plt.show()
    
    def print_classification_reports(self):
        """Print classification reports for all models"""
        if not self.results:
            print("No evaluation results available. Run evaluate_all() first.")
            return
            
        for name, result in self.results.items():
            print(f"\n{name} Classification Report:")
            print(classification_report(
                result['true_labels'], 
                result['predictions'], 
                target_names=self.class_names if self.class_names is not None else None
            ))
    
    def visualize_attention_maps(self, model_name, num_examples=9, rows=3, cols=3, figsize=(12, 12)):
        """
        Visualize attention maps for a ViT model
        
        Parameters:
        model_name: Name of the ViT model to visualize attention maps for
        num_examples: Number of examples to visualize
        rows, cols: Layout of the visualization grid
        figsize: Figure size
        """
        if model_name not in self.model_trainers:
            print(f"Model {model_name} not found in evaluator")
            return
        
        trainer = self.model_trainers[model_name]
        model = trainer.model
        
        # Check if model has a call method that accepts return_attention
        if not hasattr(model, 'call') or 'return_attention' not in model.call.__code__.co_varnames:
            print(f"Model {model_name} does not support attention visualization")
            return
        
        # Get a few test samples
        test_samples = []
        test_labels = []
        dataset = self.test_dataset if self.test_dataset is not None else trainer.test_dataset
        
        for images, labels in dataset.take(1):  # Get first batch
            test_samples = images.numpy()
            test_labels = labels.numpy()
            break
        
        # Determine patch size based on model configuration
        # This assumes the ViT model has a patch_size attribute or can be inferred
        if hasattr(model, 'patch_size'):
            patch_size = model.patch_size
        else:
            # Try to infer from the first layer weights
            patch_size = 7  # Default fallback for MNIST
            
        # Plot original images and attention maps
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        for i in range(rows):
            for j in range(cols):
                idx = i*cols + j
                if idx < min(num_examples, len(test_samples)):
                    image = test_samples[idx]
                    label = test_labels[idx]
                    
                    # Get attention map
                    attention_map = self.get_attention_maps(model, image, patch_size)
                    
                    # Plot attention overlay
                    ax = axes[i, j] if rows > 1 or cols > 1 else axes
                    # Plot original image
                    ax.imshow(np.squeeze(image), cmap='gray')
                    
                    # Resize attention map to match image size
                    h, w = image.shape[0], image.shape[1]
                    attention_size = int(np.sqrt(attention_map.size))
                    patch_size_inferred = h // attention_size
                    
                    # Upsample attention map to match image size
                    attention_upsampled = np.repeat(
                        np.repeat(attention_map, patch_size_inferred, axis=0),
                        patch_size_inferred, axis=1
                    )
                    
                    # Plot attention heatmap with transparency
                    ax.imshow(attention_upsampled, cmap='hot', alpha=0.5)
                    
                    # Add title with true label and predicted label
                    pred_label = tf.argmax(model.predict(tf.expand_dims(image, axis=0)), axis=1).numpy()[0]
                    if self.class_names is not None:
                        class_label = self.class_names[int(label)]
                        pred_class = self.class_names[int(pred_label)]
                        ax.set_title(f"True: {class_label}, Pred: {pred_class}")
                    else:
                        ax.set_title(f"True: {label}, Pred: {pred_label}")
                    
                    ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_attention_maps(self, model, image, patch_size):
        """Extract attention maps from a ViT model"""
        # Add batch dimension if missing
        if len(tf.shape(image)) == 3:
            image = tf.expand_dims(image, axis=0)

        # Get attention weights from the model
        _, attention_maps = model(image, return_attention=True, training=False)

        # Get image size
        img_size = image.shape[1]  # assuming square image
        grid_size = img_size // patch_size

        # Average across attention heads
        mean_attention = tf.reduce_mean(attention_maps, axis=1)[0]  # Shape: [seq_len, seq_len]
        
        # Average attention received by each patch
        patch_importance = tf.reduce_mean(mean_attention, axis=0)
        
        # Reshape to grid
        attention_grid = tf.reshape(patch_importance, [grid_size, grid_size])
        
        return attention_grid.numpy()