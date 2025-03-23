import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import rotate

class MNISTDataset:
    def __init__(self):
        """
        Load and prepare a transformed version of the MNIST dataset
        with rotations, translations, noise, and random inversions.
        """
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize the images to [0, 1]
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        
        # Combine train and test data
        self.original_data = np.concatenate([x_train, x_test])
        self.targets = np.concatenate([y_train, y_test])
        
        # Apply transformations to create the modified dataset
        self.data = []
        for i in range(len(self.original_data)):
            transformed_img = self._transform_image(self.original_data[i])
            self.data.append(transformed_img.squeeze())  # Remove the channel dimension for storage
    
    def _transform_image(self, image):
        """Apply various transformations to the image"""
        # Apply random rotation
        angle = np.random.uniform(-25, 25)  # Rotate between -25 and 25 degrees
        rotated = rotate(image, angle, reshape=False)

        # Apply random translation
        tx, ty = np.random.randint(-3, 3), np.random.randint(-3, 3)  # Shift by max 3 pixels
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(rotated, translation_matrix, (28, 28))

        # Add Gaussian noise
        noise = np.random.normal(0, 0.1, translated.shape)
        noisy = np.clip(translated + noise, 0, 1)

        # Invert colors randomly
        if np.random.rand() > 0.5:
            inverted = 1 - noisy
        else:
            inverted = noisy

        return inverted
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        # Convert to TensorFlow tensor and normalize (already normalized)
        img_transformed = tf.convert_to_tensor(img, dtype=tf.float32)
        # Add channel dimension
        img_transformed = tf.expand_dims(img_transformed, axis=-1)  # TF uses channels-last

        return img_transformed, target
    
    def visualize_samples(self, num_samples=5):
        """Visualize random samples from each class"""
        # Determine number of classes
        num_classes = len(self.class_names)
        
        # Create a figure
        fig, axs = plt.subplots(num_classes, num_samples, figsize=(15, 15))
        
        # For each class
        for class_idx in range(num_classes):
            # Find samples from this class
            class_indices = [i for i in range(len(self.targets)) if self.targets[i] == class_idx]
            
            # Select random samples
            if len(class_indices) >= num_samples:
                selected_indices = np.random.choice(class_indices, num_samples, replace=False)
            else:
                selected_indices = class_indices
                
            # Plot each sample
            for i, idx in enumerate(selected_indices):
                if i < num_samples:
                    img, _ = self[idx]
                    axs[class_idx, i].imshow(img.numpy().squeeze(), cmap='gray')
                    axs[class_idx, i].axis('off')
                    
        # Set class names
        for i, name in enumerate(self.class_names):
            axs[i, 0].set_ylabel(name)
            
        plt.tight_layout()
        plt.show()

    def plot_class_distribution(self):
        """Plot the distribution of classes in the dataset"""
        # Get class counts
        classes = np.array(self.targets)
        unique_classes, counts = np.unique(classes, return_counts=True)
        
        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(unique_classes, counts)
        plt.xticks(unique_classes, self.class_names, rotation=45, ha='right')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                     str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()