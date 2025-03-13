# Principles of Artificial Intelligence Coursework

This repository implements and analyzes two fundamental machine learning approaches on real-world datasets as part of the Principles of AI coursework. The implementation demonstrates the application of linear regression and support vector machines to practical prediction problems. It consists of two main components:

1. **Linear Regression**: Predicting California housing prices using Batch Gradient Descent and Stochastic Gradient Descent.

2. **Support Vector Machines (SVM)**: Human Activity Recognition using smartphone sensor data, converting a 6-class problem into binary classification.

## Potential Issues

- If you encounter errors about missing output directories, ensure the `linear_regression` and `svm` directories exist before running the scripts.
- Some visualizations may not display properly when running scripts from command line. Use an IDE for the best experience.

## Directory Structure
- `ai_env/`: Python virtual environment
- `linear_regression/`: Code and documentation for the California Housing dataset analysis
        - `california_housing.py`: Implementation of linear regression models
        - `income_vs_house_value.png`: Visualization of data distribution
        - `regression_results.png`: Visualization of model results
        - `README.md`: Documentation for the linear regression component
- `svm/`: Code and documentation for the Human Activity Recognition task
        - `human_activity_recognition.py`: Implementation of SVM models
        - `svm_results_subject_wise.png`: Visualization of classification results
        - `README.md`: Documentation for the SVM component
- `UCI HAR Dataset/`: Dataset for the SVM part of the assignment
        - `test/`: Test data files
        - `train/`: Training data files
        - `activity_labels.txt`: Mapping of activity IDs to names
        - `features.txt`: Feature names for the dataset
  - Other supporting files
        - `test_setup.py`: Script to verify environment setup
        - `requirements.txt`: List of Python dependencies

## Setup
See each subdirectory's README.md for specific implementation details.

## Installation

1. Clone this repository
2. Install the required dependencies:
pip install -r requirements.txt

3. Ensure you have Python 3.8+ installed

## Usage

It's recommended to run these scripts using an IDE such as VS Code or PyCharm for the best experience, including syntax highlighting and interactive plots.