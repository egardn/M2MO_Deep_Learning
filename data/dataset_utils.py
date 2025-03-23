import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def create_tf_datasets(dataset, batch_size=64):
    """
    Convert a dataset (like RelationalDataset or MNISTDataset) to TensorFlow datasets
    
    Parameters:
    dataset: A dataset with __getitem__ and __len__ methods
    batch_size: Batch size for the TF datasets
    
    Returns:
    train_dataset, test_dataset: Training and test TF datasets
    """
    # Get all data and targets
    all_images = []
    all_targets = []

    for i in range(len(dataset)):
        img, target = dataset[i]
        all_images.append(img.numpy())  # Convert to numpy arrays
        all_targets.append(target)

    # Convert to TensorFlow tensors
    all_images = np.array(all_images)
    all_targets = np.array(all_targets)

    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Shuffle indices
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create train and test datasets
    train_images = tf.gather(all_images, train_indices)
    train_targets = tf.gather(all_targets, train_indices)
    test_images = tf.gather(all_images, test_indices)
    test_targets = tf.gather(all_targets, test_indices)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_targets))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_targets))
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset

def visualize_dataset_examples(dataset, title="Dataset Examples"):
    """
    Wrapper function to visualize examples from a dataset
    
    Parameters:
    dataset: A dataset with visualize_samples method
    title: Title for the visualization
    """
    dataset.visualize_samples()

def plot_class_distribution(dataset, title="Class Distribution"):
    """
    Wrapper function to plot class distribution for a dataset
    
    Parameters:
    dataset: A dataset with plot_class_distribution method
    title: Title for the plot
    """
    dataset.plot_class_distribution()