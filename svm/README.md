# Support Vector Machines: Human Activity Recognition

This directory contains the implementation of Support Vector Machine (SVM) models for binary classification of human activities using smartphone sensor data.

## Implementation Details

- Conversion of 6-class activity labels into binary classification (active vs. inactive)
- Implementation of SVM with different kernels (linear, polynomial, RBF)
- Hyperparameter tuning using GridSearchCV
- Performance evaluation using confusion matrices and classification metrics
- Feature reduction techniques to manage the high-dimensional dataset

## Key Files
- `human_activity.py`: Main implementation file

# To run the SVM implementation
- python svm/human_activity_recognition.py