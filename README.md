

# Custom Image Classification Model

This repository contains the implementation of an image classification model using PyTorch. The model is designed to train on a custom dataset, and includes functions for calculating accuracy and loss per batch and per epoch. It also provides a training pipeline with integrated evaluation metrics.

## Table of Contents

- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Accuracy and Loss Calculation](#accuracy-and-loss-calculation)
- [Saving the Model](#saving-the-model)

## Data Preparation

The dataset used was structured as follows:

```
dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── ...
```

### 1. Image Transformations

Transformations are applied to the images to resize them and normalize the pixel values:

```python
transformations_to_apply = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### 2. Load Dataset

The dataset is loaded using the `ImageFolder` class from `torchvision.datasets`:

```python
# Loading dataset
train_data = datasets.ImageFolder(
    root='/kaggle/working/dataset/train', 
    transform=transform
)

val_data = datasets.ImageFolder(
    root='/kaggle/working/dataset/val', 
    transform=transform
)
```


## Model Architecture

The model is defined using PyTorch's `nn.Module`. It consists of several convolutional layers followed by fully connected layers. It draws inspiration from the AlexNet Model. 

```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Sequential: 1-1                        --
|    ├─Conv2d: 2-1                       23,296
|    ├─ReLU: 2-2                         --
|    ├─BatchNorm2d: 2-3                  128
|    ├─MaxPool2d: 2-4                    --
|    ├─Conv2d: 2-5                       307,392
|    ├─ReLU: 2-6                         --
|    ├─BatchNorm2d: 2-7                  384
|    ├─MaxPool2d: 2-8                    --
|    ├─Conv2d: 2-9                       663,936
|    ├─ReLU: 2-10                        --
|    ├─BatchNorm2d: 2-11                 768
|    ├─Conv2d: 2-12                      884,992
|    ├─ReLU: 2-13                        --
|    ├─BatchNorm2d: 2-14                 512
|    ├─Conv2d: 2-15                      590,080
|    ├─ReLU: 2-16                        --
|    ├─BatchNorm2d: 2-17                 512
|    ├─MaxPool2d: 2-18                   --
|    └─AdaptiveAvgPool2d: 2-19           --
├─Sequential: 1-2                        --
|    ├─Dropout: 2-20                     --
|    ├─Linear: 2-21                      37,752,832
|    ├─ReLU: 2-22                        --
|    ├─BatchNorm1d: 2-23                 8,192
|    ├─Dropout: 2-24                     --
|    ├─Linear: 2-25                      16,781,312
|    ├─ReLU: 2-26                        --
|    ├─BatchNorm1d: 2-27                 8,192
|    ├─Dropout: 2-28                     --
|    ├─Linear: 2-29                      2,097,664
|    ├─ReLU: 2-30                        --
|    ├─BatchNorm1d: 2-31                 1,024
|    └─Linear: 2-32                      12,825
=================================================================
Total params: 59,134,041
Trainable params: 59,134,041
Non-trainable params: 0
=================================================================
```
The `MyModel` class defines a Convolutional Neural Network (CNN). It is composed of two main sections: feature extraction and classification. Here’s a breakdown of its architecture and its components:

#### 1. **Feature Extraction (`self.feature_extraction`)**

This section is responsible for extracting hierarchical features from the input image data.

- **`nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)`**
  - **Type**: Convolutional Layer
  - **Relevance**: Applies 64 convolutional filters of size 11x11 with a stride of 4 and padding of 2. Converts the 3-channel input image (RGB) into 64 feature maps. The large kernel size and stride help capture large-scale features and reduce spatial dimensions early on.

- **`nn.ReLU(inplace=True)`**
  - **Type**: Activation Function
  - **Relevance**: Introduces non-linearity into the model. The `inplace=True` argument modifies the input directly, which can be memory efficient.

- **`nn.BatchNorm2d(64)`**
  - **Type**: Batch Normalization
  - **Relevance**: Normalizes the output of the convolutional layer to improve training stability and speed up convergence by reducing internal covariate shift.

- **`nn.MaxPool2d(kernel_size=3, stride=2)`**
  - **Type**: Max Pooling Layer
  - **Relevance**: Reduces the spatial dimensions of the feature maps while retaining important features. The kernel size of 3x3 and stride of 2 help in reducing the dimensions effectively.

- **`nn.Conv2d(64, 192, kernel_size=5, padding=2)`**
  - **Type**: Convolutional Layer
  - **Relevance**: Applies 192 filters of size 5x5, with padding to maintain spatial dimensions. Extracts more complex features from the input.

- **`nn.BatchNorm2d(192)`**
  - **Type**: Batch Normalization
  - **Relevance**: Normalizes the output of the previous convolutional layer to stabilize and speed up training.

- **`nn.MaxPool2d(kernel_size=3, stride=2)`**
  - **Type**: Max Pooling Layer
  - **Relevance**: Further reduces the spatial dimensions of the feature maps.

- **`nn.Conv2d(192, 384, kernel_size=3, padding=1)`**
  - **Type**: Convolutional Layer
  - **Relevance**: Applies 384 filters of size 3x3 to capture finer features with increased depth.

- **`nn.Conv2d(384, 256, kernel_size=3, padding=1)`**
  - **Type**: Convolutional Layer
  - **Relevance**: Applies 256 filters of size 3x3, reducing the number of feature maps but retaining high-level features.

- **`nn.Conv2d(256, 256, kernel_size=3, padding=1)`**
  - **Type**: Convolutional Layer
  - **Relevance**: Continues to refine features with 256 filters of size 3x3.

- **`nn.MaxPool2d(kernel_size=3, stride=2)`**
  - **Type**: Max Pooling Layer
  - **Relevance**: Reduces spatial dimensions further before feeding into the classifier.

- **`nn.AdaptiveAvgPool2d((6, 6))`**
  - **Type**: Adaptive Average Pooling
  - **Relevance**: Pools the feature maps to a fixed size of 6x6 regardless of the input size. This step standardizes the output size before the fully connected layers.


#### 2. **Classifier (`self.classifier`)**

This section is responsible for classifying the extracted features into one of the predefined classes.

- **`nn.Dropout(0.1)`**
  - **Type**: Dropout
  - **Relevance**: Regularization technique that randomly sets a fraction (10%) of the input units to zero during training, helping to prevent overfitting.

- **`nn.Linear(256 * 6 * 6, 4096)`**
  - **Type**: Fully Connected (Dense) Layer
  - **Relevance**: Connects the flattened feature vector from the previous layer to 4096 neurons. This large number of neurons helps in learning complex patterns.

- **`nn.BatchNorm1d(4096)`**
  - **Type**: Batch Normalization
  - **Relevance**: Normalizes the activations in the fully connected layer to improve training stability.

- **`nn.Linear(4096, 4096)`**
  - **Type**: Fully Connected (Dense) Layer
  - **Relevance**: Further refines the learned features by connecting to another set of 4096 neurons.

- **`nn.BatchNorm1d(4096)`**
  - **Type**: Batch Normalization
  - **Relevance**: Normalizes the output of the previous fully connected layer.

- **`nn.Linear(4096, 512)`**
  - **Type**: Fully Connected (Dense) Layer
  - **Relevance**: Reduces the feature dimension to 512 neurons, further refining the learned features.

- **`nn.Linear(512, num_classes)`**
  - **Type**: Fully Connected (Dense) Layer
  - **Relevance**: Outputs the final class scores, with `num_classes` representing the number of classes for classification.

## Training Pipeline

The training loop iterates over a specified number of epochs to train and validate the model. Here is a step-by-step description:

1. **Epoch Iteration**:
    - For each epoch, the model is set to training mode with `model.train()`.
    - Lists to store predictions, probabilities, and labels for training are initialized.
    - A variable to keep track of the running loss is also initialized.

2. **Training Phase**:
    - **Progress Bar**: A progress bar is displayed using `tqdm` for the training process.
    - **Batch Processing**:
        - Inputs and labels are moved to the device (GPU/CPU).
        - Gradients are zeroed with `optimizer.zero_grad()`.
        - Forward pass: The model outputs predictions for the inputs.
        - Loss calculation: The loss between predictions and actual labels is computed.
        - Backward pass: Gradients are calculated and optimizer steps are taken.
        - The running loss is updated.
        - Predictions and probabilities are appended to their respective lists.
    - **Metrics Calculation**:
        - **Accuracy**: Computed using `accuracy_score`.
        - **Precision**: Computed using `precision_score` with macro average.
        - **Recall**: Computed using `recall_score` with macro average.
        - **F1 Score**: Computed using `f1_score` with macro average.
        - **ROC-AUC**: Computed using `roc_auc_score` with one-vs-rest strategy.
        - **Log Loss**: Computed using `log_loss`.
    - **Metrics are appended to** `train_metrics`.

3. **Validation Phase**:
    - **Model Evaluation**: The model is set to evaluation mode with `model.eval()`.
    - **Progress Bar**: A progress bar is displayed for the validation process.
    - **Batch Processing**:
        - Inputs and labels are moved to the device.
        - No gradients are computed during validation (`torch.no_grad()`).
        - Forward pass: The model outputs predictions for the inputs.
        - Loss calculation: The loss between predictions and actual labels is computed.
        - Predictions and probabilities are appended to their respective lists.
    - **Metrics Calculation**:
        - **Accuracy**: Computed using `accuracy_score`.
        - **Precision**: Computed using `precision_score` with macro average.
        - **Recall**: Computed using `recall_score` with macro average.
        - **F1 Score**: Computed using `f1_score` with macro average.
        - **ROC-AUC**: Computed using `roc_auc_score` with one-vs-rest strategy.
        - **Log Loss**: Computed using `log_loss`.
    - **Metrics are appended to** `val_metrics`.
    
4. **Print Metrics**:
    - The following metrics are printed for the validation phase:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC
        - Log Loss

5. **Completion**:
    - A message is printed to indicate that training is complete.

## Accuracy and Loss Calculation

Accuracy and loss are calculated using the scikit-learn library imported as,
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
```

The metrics mentioned above are plotted in a graph as follows after training the model to 10 epochs:
![Performance Metrics](https://github.com/chiragooner/bird_image_classification/blob/master/result.png?raw=true)


