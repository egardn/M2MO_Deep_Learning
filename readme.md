# Deep Learning: Comparative Study of CNNs and Vision Transformers for Image Classification

## Project Overview
This project aims to analyze and compare two deep learning methods—Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs)—on image classification tasks. The focus is on understanding their strengths and weaknesses by designing datasets that highlight their differences, following the guidelines of the evaluation assignment.

## Objectives
- Compare CNN and ViT performance on image classification.
- Design or modify datasets to reveal method-specific advantages.
- Evaluate models using quantitative and qualitative metrics.
- Provide visualizations and analysis to interpret results.

## Project Structure
```
├── data/
│   ├── mnist_dataset.py        # MNIST and custom dataset loader/transformer
│   ├── relational_dataset.py   # Relational dataset for long-range dependencies
│   └── dataset_utils.py        # Utilities for dataset creation and visualization
├── models/
│   ├── cnn_model.py            # CNN model and hyperparameter tuning
│   ├── vit_model.py            # Vision Transformer model
│   └── model_trainer.py        # Training pipeline for models
├── utils/
│   ├── model_evaluator.py      # Evaluation, metrics, and visualization
│   └── visualization.py        # Plotting functions for samples and predictions
├── Model_comparison_v3.ipynb   # Main notebook for experiments and analysis
├── factory.py                  # Model/dataset factory utilities
├── kt_dir/                     # KerasTuner results for CNN tuning
└── readme.md                   # Project documentation
```

## Dataset Design
- **MNIST (Transformed):** Includes rotations, translations, noise, and random inversions to test robustness and texture-based classification (favoring CNNs).
- **Relational Dataset:** Designed to require long-range dependencies, highlighting the strengths of ViTs.
- Datasets are small and simple to clearly expose model differences.

## Model Architectures
- **CNN:** Customizable architecture with hyperparameter tuning (see `cnn_model.py`).
- **Vision Transformer (ViT):** Patch-based transformer model (see `vit_model.py`).

## Training & Evaluation Pipeline
- Training and evaluation are managed via `model_trainer.py` and `model_evaluator.py`.
- Metrics: Accuracy, loss, confusion matrix, and classification report.
- Visualizations: Training curves, sample predictions, and (for ViT) attention maps.

## How to Run
1. **Install Requirements:**
   - Python 3.10+
   - TensorFlow, NumPy, Matplotlib, scikit-learn, OpenCV, KerasTuner
   - Install with: `pip install -r requirements.txt` (create this file if missing)
2. **Prepare Datasets:**
   - Datasets are generated/augmented automatically by scripts in `data/`.
3. **Train Models:**
   - Use `Model_comparison_v3.ipynb` to run experiments, or call training scripts directly.
4. **Evaluate & Visualize:**
   - Use provided utilities to generate metrics and plots.

## Results & Analysis
- Results include quantitative metrics and qualitative visualizations.
- Analysis focuses on explaining observed differences, e.g., CNNs excelling on texture, ViTs on relational/occluded data.

## Report & Presentation
- **Report:** Submit an 8–15 page LaTeX report (see assignment for structure).
- **Oral Presentation:** 7 minutes + 3 minutes Q&A.
- **Deadline:** May 5, 2025.

## References
- Krizhevsky et al., 2012 (CNNs)
- Lu et al., 2022 (ViTs)
- Assignment instructions (see `Consignes_evaluation.pdf`)

## Credits
Work by Emmanuel Gardin and Joseph Ngueponwouo.

---
For more details, see code comments and the main notebook.
