# SML-Project

# Knee MRI Deep Learning Analysis

Our study aims to implement deep-learning techniques that automate the interpretation of knee MRIs. The study aims to suggest methods to assist physicians in obtaining a precise diagnosis by differentiating between different kinds of abnormalities and to be able to prioritize high-risk patients. The project involves data pre-processing, model selection, hyperparameter tuning, and thorough evaluation to achieve optimal accuracy, sensitivity, and specificity results.

## Report

The project report can be found here:
[MRNet___SML___Final_Report.pdf](MRNet___SML___Final_Report.pdf)

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

- **Notebook:** 

## How to Run the Notebooks

### Notebook Execution Instructions

#### 1. **Data Exploration**
   - **Folder:** `Data_Exploration`
   - **Steps to Run:**
     1. Navigate to the `Data_Exploration` folder:
        ```bash
        cd Data_Exploration
        ```
     2. Launch the notebook:
        ```bash
        jupyter notebook Data_Exploration.ipynb
        ```
     3. Follow the notebook to explore the dataset. Visualize statistical summaries, inspect MRI animations (available in the `Animations` folder), and perform initial preprocessing.

---

#### 2. **3D CNN Model**
   - **Folder:** `3D_CNN`
   - **Steps to Run:**
     1. Navigate to the `3D_CNN` folder:
        ```bash
        cd ../3D_CNN
        ```
     2. Launch the notebook:
        ```bash
        jupyter notebook SML_3D_CNN.ipynb
        ```
     3. The notebook includes the design, training, and evaluation of the 3D CNN architecture. Ensure the dataset is correctly preprocessed before running.

---

#### 3. **EfficientNet Model**
   - **Folder:** `Efficientnet`
   - **Steps to Run:**
     1. Navigate to the `Efficientnet` folder:
        ```bash
        cd ../Efficientnet
        ```
     2. Launch the notebook:
        ```bash
        jupyter notebook SMLEfficientNetB0.ipynb
        ```
     3. This notebook demonstrates the implementation and training of the EfficientNet model. Modify hyperparameters if needed and evaluate its performance.

---

#### 4. **ResNet Models**
   - **Folder:** `ResNet`
   - **Steps to Run:**
     1. Navigate to the `ResNet` folder:
        ```bash
        cd ../ResNet
        ```
     2. Launch any of the ResNet model notebooks:
        ```bash
        jupyter notebook ResNet_200d.ipynb
        jupyter notebook SML_3D_ResNet.ipynb
        jupyter notebook Resnet18&Resnet50.ipynb
        ```
     3. Each notebook corresponds to a specific ResNet variant:
        - `ResNet_200d.ipynb`: Implements and evaluates ResNet-200d.
        - `SML_3D_ResNet.ipynb`: Adapts ResNet for 3D volumetric MRI data.
        - `Resnet18&Resnet50.ipynb`: Provides a comparative analysis of ResNet-18 and ResNet-50.

---

## Summary of Model Performance

| **Model**             | **Validation Accuracy** | **Validation AUC** |
|------------------------|--------------------------|---------------------|
| Optimized 3D CNN       | 82.50%                  | 0.87                |
| ResNet-18 (Axial View) | 65.56%                  | 0.6202              |
| ResNet-50 (Axial View) | 64.78%                  | 0.6377              |
| ResNet-200d            | 32.50%                  | 0.6604              |
| 3D ResNet              | 29.17%                  | 0.6604              |
| EfficientNetB0         | 71.9%                   | -                   |

*Table: Summary of Model Performance*

---

## Future Work

Future work will focus on implementing and evaluating a multi-axes 3D ResNet architecture to capture features across different orientations, contingent on improved GPU resources. Efforts will also aim to further optimize the 3D CNN for enhanced class-specific performance through advanced regularization techniques and hyperparameter tuning. Additionally, refining ResNet and EfficientNet models to address overfitting and exploring larger variants of EfficientNet can improve their robustness. Expanding the dataset for better generalization, integrating transfer learning, and developing real-time deployment solutions for clinical use are also priorities. Finally, incorporating explainability techniques, such as Grad-CAM, and benchmarking against state-of-the-art methods will enhance the model's applicability and reliability in clinical settings.

---

## References

- **Stanford MRNet Dataset:** Stanford Machine Learning Group. *MRNet Dataset: Knee MRIs for Abnormality Detection and Anterior Cruciate Ligament (ACL) Tear Diagnosis.*  
  Available at: [https://stanfordmlgroup.github.io/competitions/mrnet/](https://stanfordmlgroup.github.io/competitions/mrnet/)

- Tran, D., Bourdev, L., Fergus, R., Torresani, L., and Paluri, M. (2015). *Learning Spatiotemporal Features with 3D Convolutional Networks.*  
  Proceedings of the IEEE International Conference on Computer Vision (ICCV), 4489-4497.  
  Available at: [https://arxiv.org/abs/1412.0767](https://arxiv.org/abs/1412.0767)

- He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition.*  
  Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.

- **EfficientNet:** *Rethinking Model Scaling for Convolutional Neural Networks.*  
  Available at: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)

- **Brain MRI Classification using PyTorch EfficientNetB0:**  
  Available at: [https://debuggercafe.com/brain-mri-classification-using-pytorch-efficientnetb0/](https://debuggercafe.com/brain-mri-classification-using-pytorch-efficientnetb0/)

---
