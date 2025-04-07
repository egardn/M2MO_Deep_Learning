import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_sample_images(dataset, num_examples=25, rows=5, cols=5, figsize=(10, 10), title=None):
    """
    Plot sample images from a dataset
    
    Parameters:
    dataset: TensorFlow dataset containing (image, label) tuples
    num_examples: Number of examples to show
    rows, cols: Grid dimensions
    figsize: Figure size
    title: Figure title
    """
    plt.figure(figsize=figsize)
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    # Get the first batch
    for images, labels in dataset.take(1):
        images = images.numpy()
        labels = labels.numpy()
        
        for i in range(min(num_examples, len(images))):
            plt.subplot(rows, cols, i + 1)
            
            # Handle different channel configurations
            img = images[i]
            if len(img.shape) == 2:
                # Grayscale image without channel dimension
                plt.imshow(img, cmap='gray')
            elif img.shape[-1] == 1:
                # Grayscale image with channel dimension
                plt.imshow(np.squeeze(img), cmap='gray')
            else:
                # Color image
                plt.imshow(img)
                
            plt.title(f'Label: {labels[i]}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_model_examples(model, dataset, class_names=None, num_examples=10, figsize=(15, 10)):
    """
    Plot examples with model predictions
    
    Parameters:
    model: Trained model
    dataset: Dataset to get examples from
    class_names: List of class names
    num_examples: Number of examples to show
    figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Get a batch of images
    for images, labels in dataset.take(1):
        # Get predictions
        preds = model.predict(images)
        pred_classes = tf.argmax(preds, axis=1).numpy()
        
        for i in range(min(num_examples, len(images))):
            plt.subplot(2, num_examples//2, i+1)
            
            # Handle different channel configurations
            img = images[i].numpy()
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            elif img.shape[-1] == 1:
                plt.imshow(np.squeeze(img), cmap='gray')
            else:
                plt.imshow(img)
            
            true_label = labels[i].numpy()
            pred_label = pred_classes[i]
            
            if class_names:
                true_name = class_names[true_label]
                pred_name = class_names[pred_label]
                plt.title(f'True: {true_name}\nPred: {pred_name}')
            else:
                plt.title(f'True: {true_label}\nPred: {pred_label}')
                
            # Highlight correct/incorrect predictions
            if true_label == pred_label:
                plt.gca().spines['bottom'].set_color('green')
                plt.gca().spines['top'].set_color('green') 
                plt.gca().spines['right'].set_color('green')
                plt.gca().spines['left'].set_color('green')
                plt.gca().spines['bottom'].set_linewidth(5)
                plt.gca().spines['top'].set_linewidth(5) 
                plt.gca().spines['right'].set_linewidth(5)
                plt.gca().spines['left'].set_linewidth(5)
            else:
                plt.gca().spines['bottom'].set_color('red')
                plt.gca().spines['top'].set_color('red') 
                plt.gca().spines['right'].set_color('red')
                plt.gca().spines['left'].set_color('red')
                plt.gca().spines['bottom'].set_linewidth(5)
                plt.gca().spines['top'].set_linewidth(5) 
                plt.gca().spines['right'].set_linewidth(5)
                plt.gca().spines['left'].set_linewidth(5)
            
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()