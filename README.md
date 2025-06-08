# Image Dataset Balancing with Data Augmentation

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Automated image dataset balancing using advanced data augmentation techniques for medical image classification**

## 📋 Overview

This project provides a comprehensive solution for balancing imbalanced image datasets using sophisticated data augmentation techniques. Originally designed for medical image classification with three classes (benign, malignant, normal), the tool can be adapted for any multi-class image classification task.

### 🎯 Problem Statement

Medical image datasets often suffer from class imbalance, which can significantly impact machine learning model performance. This tool addresses the imbalance by generating synthetic images through various augmentation techniques while preserving the essential characteristics of the original medical images.

### ✨ Key Features

- **🔄 Automated Dataset Balancing**: Automatically balances datasets to specified target counts
- **🎨 9 Augmentation Techniques**: Comprehensive set of image transformations
- **📊 Visual Analytics**: Generates comparison charts and detailed statistics
- **🚀 Google Colab Ready**: Optimized for cloud-based execution
- **📦 Export Functionality**: Creates downloadable zip files of balanced datasets
- **💾 Preservation of Originals**: Maintains all original images while adding augmented ones

## 🛠️ Augmentation Techniques

The tool implements nine different augmentation techniques to ensure diversity:

| Technique | Description | Parameters |
|-----------|-------------|------------|
| **Rotation** | Random rotation within safe angles | ±25 degrees |
| **Horizontal Flip** | Mirror image horizontally | Binary |
| **Vertical Flip** | Mirror image vertically | Binary |
| **Brightness** | Adjust image brightness | 0.8-1.2× factor |
| **Contrast** | Modify image contrast | 0.8-1.2× factor |
| **Sharpness** | Enhance or reduce sharpness | 0.7-1.3× factor |
| **Gaussian Blur** | Apply slight blur effect | 0.5-1.5 radius |
| **Zoom/Crop** | Zoom in by cropping and resizing | 5-20% crop |
| **Noise Injection** | Add Gaussian noise | σ=5, μ=0 |

## 🚀 Quick Start

### Prerequisites

```bash
pip install pillow matplotlib seaborn opencv-python numpy
```

### Google Colab Setup

1. **Upload your dataset to Google Drive** with the following structure:
```
your_dataset/
├── benign/     # Class 1 images
├── malignant/  # Class 2 images
└── normal/     # Class 3 images
```

2. **Open the notebook** in Google Colab

3. **Update the dataset paths**:

4. **Run all cells** and uncomment the final execution line

### Local Usage

```python
from image_dataset_balancer import ImageDatasetBalancer

# Initialize balancer
balancer = ImageDatasetBalancer(
    input_folder="path/to/your/dataset",
    output_folder="path/to/balanced/output",
    target_count=1050
)

# Run complete process
balancer.run_complete_process()
```

## 📊 Example Results

### Dataset Statistics

| Class | Before | After | Generated |
|-------|--------|-------|-----------|
| Benign | 467 | 1,050 | 583 |
| Malignant | 1,050 | 1,050 | 0 |
| Normal | 416 | 1,050 | 634 |
| **Total** | **1,933** | **3,150** | **1,217** |

### Performance Impact

- **Dataset Size Increase**: 63%
- **Class Balance Achievement**: 100%
- **Processing Time**: ~3-5 minutes (CPU)
- **Memory Usage**: <2GB RAM

## 🎨 Visualization

The tool automatically generates:

- 📊 **Before/After Comparison Charts**: Visual representation of class distribution
- 📈 **Statistical Summary**: Detailed breakdown of augmentation results
- 🎯 **Progress Tracking**: Real-time progress indicators during processing

![Sample Comparison Chart](https://via.placeholder.com/800x400/4ECDC4/FFFFFF?text=Before+vs+After+Comparison)

## 🔬 Methodology

### Why Image Augmentation over SMOTE?

Traditional SMOTE (Synthetic Minority Oversampling Technique) works well for tabular data but has limitations with image data:

- **SMOTE**: Interpolates between feature vectors, often creating unrealistic images
- **Image Augmentation**: Applies realistic transformations that preserve medical image integrity
- **Result**: Better model performance and clinically relevant synthetic data

### Scientific Approach

Our methodology follows current best practices in medical image preprocessing:

1. **Data Preservation**: All original images are maintained
2. **Realistic Augmentation**: Transformations mimic natural variations
3. **Balanced Distribution**: Ensures equal representation across classes
4. **Quality Control**: Maintains image quality and diagnostic features

## ⚙️ Configuration Options

### Basic Configuration

```python
balancer = ImageDatasetBalancer(
    input_folder="path/to/input",
    output_folder="path/to/output",
    target_count=1050  # Target images per class
)
```

### Advanced Configuration

```python
# Custom augmentation techniques
custom_techniques = ['rotation', 'brightness', 'contrast']

# Custom parameters
balancer.augmentation_techniques = custom_techniques
```

## 🐛 Troubleshooting

### Common Issues

1. **Google Drive Mount Issues**
   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```

2. **Path Not Found Errors**
   - Verify Google Drive paths are correct
   - Check folder permissions
   - Ensure dataset structure matches requirements

3. **Memory Issues**
   - Process smaller batches
   - Use CPU instead of GPU for this task
   - Clear unused variables

### Performance Optimization

- **CPU vs GPU**: CPU is sufficient and often faster for image augmentation
- **Batch Processing**: Tool processes images individually to manage memory
- **Progress Tracking**: Real-time progress indicators help monitor long-running tasks

## 📚 Use Cases

### Medical Imaging
- **Radiology**: X-rays, CT scans, MRI images
- **Pathology**: Histopathology slides, cell images
- **Dermatology**: Skin lesion classification

### General Computer Vision
- **Object Detection**: Balancing object classes
- **Image Classification**: Any multi-class scenarios
- **Quality Control**: Industrial defect detection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PIL/Pillow**: Image processing capabilities
- **Matplotlib**: Visualization tools
- **Google Colab**: Cloud computing platform
- **Medical Imaging Community**: Inspiration and use case validation

## 📈 Roadmap

- [ ] **Advanced Augmentation**: GAN-based synthetic image generation
- [ ] **Multi-format Support**: DICOM, NIFTI medical formats
- [ ] **Automated Parameter Tuning**: Optimal augmentation parameter selection
- [ ] **Performance Metrics**: Built-in evaluation tools
- [ ] **GUI Interface**: User-friendly graphical interface
- [ ] **Docker Support**: Containerized deployment

## 📊 Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/image-dataset-balancer?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/image-dataset-balancer?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/image-dataset-balancer)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/image-dataset-balancer)

---

**⭐ If this project helped you, please give it a star!**
