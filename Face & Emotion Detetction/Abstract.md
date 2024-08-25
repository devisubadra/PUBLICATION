
# Face Recognition and Emotion Detection Using Restricted Boltzmann Machines (RBM)

## Overview

This project implements a face recognition and emotion detection system using Restricted Boltzmann Machines (RBM). The system is designed to accurately recognize faces and detect basic human emotions from facial expressions. It utilizes the FERET database for face recognition and the MMI dataset for emotion detection.

## Features

- **Face Recognition**: Utilizes FERET database images for identifying and matching faces.
- **Emotion Detection**: Detects basic emotions using the MMI dataset.
- **RBM Model**: Employs a Restricted Boltzmann Machine to improve face recognition and emotion detection accuracy.

## Prerequisites

Ensure you have the following software installed:

- **Python 3.x**: The code is written in Python 3.7 or later.
- **NumPy**: For numerical computations.
- **SciPy**: For scientific computations and optimization.
- **OpenCV**: For image processing and computer vision.
- **TensorFlow or PyTorch**: For implementing and running the RBM model.
- **Matplotlib**: For plotting and visualizing results.

You can install the required Python packages using pip:

```bash
pip install numpy scipy opencv-python tensorflow matplotlib
```

## File Structure

- `face_recognition_emotion_detection.py`: Main script for face recognition and emotion detection.
- `rbm.py`: Contains the implementation of the Restricted Boltzmann Machine.
- `data_preprocessing.py`: Script for preprocessing the input data.
- `README.md`: This README file.
- `requirements.txt`: Lists the required Python packages.

## Usage

### Preparing the Data

1. **Download the FERET and MMI datasets** and place them in the `datasets` folder.
2. **Ensure the datasets are properly formatted** as described in the project.

### Running the Code

To run the face recognition and emotion detection system, execute the main script:

```bash
python face_recognition_emotion_detection.py
```

### Script Arguments

- `--feret_path`: Path to the FERET dataset.
- `--mmi_path`: Path to the MMI dataset.
- `--output_dir`: Directory where output results will be saved.

### Example

```bash
python face_recognition_emotion_detection.py --feret_path datasets/feret --mmi_path datasets/mmi --output_dir results
```

## Implementation Details

### Restricted Boltzmann Machine (RBM)

The RBM is implemented in `rbm.py`. It consists of:
- **Visible Layer**: Input layer that takes the pixel values of images.
- **Hidden Layer**: Layer of units that captures features from the input data.

### Data Preprocessing

Data preprocessing is handled by `data_preprocessing.py`, including:
- **Image Cropping**: Removing background and isolating facial features.
- **Normalization**: Scaling pixel values to a standard range.

## Results

The system will output:
- **Face Recognition Results**: Identified faces and accuracy metrics.
- **Emotion Detection Results**: Detected emotions and their accuracy.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **FERET Database**: For providing facial recognition data.
- **MMI Dataset**: For emotion detection data.
- **Restricted Boltzmann Machines**: For innovative machine learning models.

For any questions or support, please contact [devisv25@gmail.com].
