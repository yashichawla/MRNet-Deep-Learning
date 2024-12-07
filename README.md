# SML-Project

# Knee MRI Deep Learning Analysis

Our study aims to implement deep-learning techniques that automate the interpretation of knee MRIs. The study aims to suggest methods to assist physicians in obtaining a precise diagnosis by differentiating between different kinds of abnormalities and to be able to prioritize high-risk patients. The project involves data pre-processing, model selection, hyperparameter tuning, and thorough evaluation to achieve optimal accuracy, sensitivity, and specificity results.

## Project Structure

To navigate through the project effectively, please follow the structure outlined below:

### 1. Data Exploration

**Folder:** `Data_Exploration`

- **Notebook:** 
  
  This notebook provides comprehensive insights into the dataset, including statistical analysis, visualization, and initial preprocessing steps.

- **Animations Folder:** `Animations`
  
  Contains `.mp4` videos illustrating the MRI scans from different axes. You can download and watch these animations to better understand the data distribution and anatomical structures.

### 2. 3D CNN Model

**Folder:** `3D_CNN`

- **Notebook:** 
  
  This notebook includes the implementation of the 3D Convolutional Neural Network (CNN) model. It covers the architecture design, training process, and evaluation metrics specific to the 3D CNN approach.

### 3. EfficientNet Model

**Folder:** `Efficientnet`

- **Notebook:** 
  
  This notebook details the EfficientNet model implementation. It includes model configuration, training routines, and performance evaluation tailored to the EfficientNet architecture.

### 4. ResNet Models

**Folder:** `ResNet`

This folder contains multiple notebooks implementing different variants of the ResNet architecture:

- **ResNet-200d:** 
  
  Implementation and training of the ResNet-200d model, including performance analysis and evaluation metrics.

- **3D ResNet:** 
  
  This notebook covers the 3D ResNet model, adapting the standard ResNet architecture for volumetric MRI data.

- **ResNet-18 & ResNet-50:** 
  
  Comparative analysis of ResNet-18 and ResNet-50 models, including their training processes, performance metrics, and insights into their generalization capabilities.

