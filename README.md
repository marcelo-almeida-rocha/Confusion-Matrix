# MNIST Classification with TensorFlow and Keras

This project demonstrates the use of TensorFlow and Keras for building a Convolutional Neural Network (CNN) model to classify handwritten digits from the MNIST dataset. The model is evaluated using various classification metrics including **Accuracy**, **Precision**, **Recall (Sensitivity)**, **Specificity**, and **F-score**.
Code based on Diego Renan's code for confusion matrix generation

## Requirements

To run this code, you need the following Python libraries:

- `tensorflow==2.12`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using `pip`:


## Dataset

The model is trained on the MNIST dataset, which consists of 60,000 28x28 grayscale images of handwritten digits for training, and 10,000 images for testing.

## Model Architecture

The model is built using the following layers:
1. **Convolutional Layer**: 32 filters, 3x3 kernel, ReLU activation
2. **MaxPooling Layer**: 2x2 pool size
3. **Convolutional Layer**: 64 filters, 3x3 kernel, ReLU activation
4. **MaxPooling Layer**: 2x2 pool size
5. **Convolutional Layer**: 64 filters, 3x3 kernel, ReLU activation
6. **Flatten Layer**
7. **Dense Layer**: 64 units, ReLU activation
8. **Output Layer**: 10 units (one for each digit), Softmax activation

## Metrics

After training, the following classification metrics are calculated:

- **Accuracy**: The proportion of correctly classified samples.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall (Sensitivity)**: The proportion of true positive predictions among all actual positives.
- **Specificity**: The proportion of true negative predictions among all actual negatives.
- **F-score**: The harmonic mean of precision and recall.

### Formulas Used:

- **Sensitivity (Recall)** = \( \frac{VP}{VP + FN} \)
- **Specificity** = \( \frac{VN}{FP + VN} \)
- **Accuracy** = \( \frac{VP + VN}{N} \)
- **Precision** = \( \frac{VP}{VP + FP} \)
- **F-score** = \( \frac{2 \times (\text{Precision} \times \text{Recall})}{\text{Precision} + \text{Recall}} \)

Where:
- **VP** = True Positive
- **FP** = False Positive
- **FN** = False Negative
- **VN** = True Negative
- **N** = Total number of samples
