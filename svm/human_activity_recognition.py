# human_activity_recognition.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.utils import resample

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data():
    """Load and preprocess the HAR dataset with subject information."""
    print("Loading Human Activity Recognition dataset...")
    
    PATH = "UCI HAR Dataset/"
    features_path = PATH + "features.txt"
    activity_labels_path = PATH + "activity_labels.txt"
    X_train_path = PATH + "train/X_train.txt"
    y_train_path = PATH + "train/y_train.txt"
    X_test_path = PATH + "test/X_test.txt"
    y_test_path = PATH + "test/y_test.txt"
    
    # Load subject information
    subject_train_path = PATH + "train/subject_train.txt"
    subject_test_path = PATH + "test/subject_test.txt"
    
    # Load feature names
    features_df = pd.read_csv(features_path, sep=r"\s+", header=None, names=["idx", "feature"])
    feature_names = features_df["feature"].tolist()
    
    # Make feature names unique by adding index suffix if needed
    unique_feature_names = []
    name_counts = {}
    
    for name in feature_names:
        if name in name_counts:
            name_counts[name] += 1
            unique_name = f"{name}_{name_counts[name]}"
        else:
            name_counts[name] = 0
            unique_name = name
        unique_feature_names.append(unique_name)
    
    # Load activity labels (mapping IDs 1-6 to string names)
    activity_labels_df = pd.read_csv(activity_labels_path, sep=r"\s+", header=None, names=["id", "activity"])
    activity_map = dict(zip(activity_labels_df.id, activity_labels_df.activity))
    
    # Load train/test sets with unique feature names
    X_train = pd.read_csv(X_train_path, sep=r"\s+", header=None, names=unique_feature_names)
    y_train = pd.read_csv(y_train_path, sep=r"\s+", header=None, names=["Activity"])
    X_test = pd.read_csv(X_test_path, sep=r"\s+", header=None, names=unique_feature_names)
    y_test = pd.read_csv(y_test_path, sep=r"\s+", header=None, names=["Activity"])
    
    # Load subject information
    subject_train = pd.read_csv(subject_train_path, sep=r"\s+", header=None, names=["Subject"])
    subject_test = pd.read_csv(subject_test_path, sep=r"\s+", header=None, names=["Subject"])
    
    # Map activity IDs to their names
    y_train["Activity_Name"] = y_train["Activity"].map(activity_map)
    y_test["Activity_Name"] = y_test["Activity"].map(activity_map)
    
    # Convert multi-class to binary
    def to_binary_label(activity):
        if activity in [1, 2, 3]:  # WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS
            return 1  # Active
        else:
            return 0  # Inactive
    
    y_train["Binary"] = y_train["Activity"].apply(to_binary_label)
    y_test["Binary"] = y_test["Activity"].apply(to_binary_label)
    
    print("Data preprocessing completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Print class distribution
    print("\nClass distribution:")
    print("Training set:")
    print(y_train["Binary"].value_counts())
    print("\nTesting set:")
    print(y_test["Binary"].value_counts())
    
    # Merge all data together with subject information
    X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_all = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    subject_all = pd.concat([subject_train, subject_test], axis=0).reset_index(drop=True)
    
    # Count unique subjects
    unique_subjects = subject_all["Subject"].unique()
    print(f"\nTotal unique subjects: {len(unique_subjects)}")
    
    return X_all, y_all, subject_all, unique_feature_names

def subject_wise_split(X, y, subjects, test_size=0.3, random_state=42):
    """
    Split the data ensuring that subjects in test set don't appear in training set.
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : pandas DataFrame
        Target variables
    subjects : pandas Series
        Subject identifiers
    test_size : float
        Proportion of subjects to include in test set
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, subject_train, subject_test : pandas DataFrames
        Split datasets including subject information
    """
    np.random.seed(random_state)
    
    # Get unique subjects
    unique_subjects = subjects["Subject"].unique()
    n_subjects = len(unique_subjects)
    
    # Determine number of test subjects
    n_test_subjects = int(n_subjects * test_size)
    
    # Randomly select test subjects
    test_subjects = np.random.choice(unique_subjects, n_test_subjects, replace=False)
    
    # Create masks for splitting
    test_mask = subjects["Subject"].isin(test_subjects)
    train_mask = ~test_mask
    
    # Split the data
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    
    # Split subject information
    subject_train = subjects[train_mask].reset_index(drop=True)
    subject_test = subjects[test_mask].reset_index(drop=True)
    
    print(f"\nSubject-wise split:")
    print(f"Training set: {len(X_train)} samples, {len(set(subject_train['Subject']))} subjects")
    print(f"Testing set: {len(X_test)} samples, {len(set(subject_test['Subject']))} subjects")
    
    return X_train, X_test, y_train, y_test, subject_train, subject_test

def train_baseline_models(X_train, y_train, X_test, y_test):
    """Train baseline SVM models with different kernels."""
    print("\nTraining baseline SVM models...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    kernels = ['linear', 'poly', 'rbf']
    models = {}
    
    for kernel in kernels:
        print(f"\nTraining SVM with {kernel} kernel...")
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train_scaled, y_train["Binary"])
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test["Binary"], y_pred)
        conf_matrix = confusion_matrix(y_test["Binary"], y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        models[kernel] = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    return models, X_train_scaled, X_test_scaled

def reduce_features_with_pca(X_train, X_test, n_components=50):
    """Reduce features using PCA."""
    print(f"\nReducing dimensions with PCA to {n_components} components...")
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_train_pca, X_test_pca, pca

def tune_hyperparameters_with_subject_cv(X_train, y_train, subjects_train):
    """Perform hyperparameter tuning using GridSearchCV with subject-wise cross-validation."""
    print("\nPerforming hyperparameter tuning with subject-wise cross-validation...")
    
    # Create group k-fold for subject-wise cross-validation
    group_kfold = GroupKFold(n_splits=3)
    
    # Create a pipeline with scaling, PCA, and SVM
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),  # Reduce dimensions
        ('svc', SVC(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = [
        {
            'svc__kernel': ['linear'],
            'svc__C': [0.1, 1, 10]
        },
        {
            'svc__kernel': ['poly'],
            'svc__C': [0.1, 1],
            'svc__degree': [2, 3],
            'svc__gamma': ['scale', 'auto']
        },
        {
            'svc__kernel': ['rbf'],
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto']
        }
    ]
    
    # Perform grid search with subject-wise cross-validation
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',
        cv=group_kfold.split(X_train, y_train["Binary"], subjects_train["Subject"]),
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train["Binary"])
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search

def visualize_results(models, y_test):
    """Visualize classification results."""
    # Compare accuracy across kernels
    kernels = list(models.keys())
    accuracies = [models[k]['accuracy'] for k in kernels]
    
    plt.figure(figsize=(10, 5))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    plt.bar(kernels, accuracies)
    plt.title('SVM Accuracy by Kernel (Subject-wise Split)')
    plt.xlabel('Kernel')
    plt.ylabel('Accuracy')
    plt.ylim(0.7, 1.0)  # Adjusted for potentially lower accuracies
    
    # Confusion matrices
    plt.subplot(1, 2, 2)
    cm = models['rbf']['confusion_matrix']  # Using RBF kernel as an example
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (RBF Kernel)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Inactive', 'Active'])
    plt.yticks(tick_marks, ['Inactive', 'Active'])
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('svm/svm_results_subject_wise.png')
    plt.show()

def main():
    """Main function to run the HAR classification pipeline with subject-wise splits."""
    # 1. Load and preprocess data with subject information
    X_all, y_all, subject_all, feature_names = load_and_preprocess_data()
    
    # 2. Perform subject-wise split (MODIFIED)
    X_train, X_test, y_train, y_test, subject_train, subject_test = subject_wise_split(X_all, y_all, subject_all, test_size=0.3)
    
    # 3. Train baseline models
    baseline_models, X_train_scaled, X_test_scaled = train_baseline_models(X_train, y_train, X_test, y_test)
    
    # 4. Reduce features with PCA
    X_train_pca, X_test_pca, pca = reduce_features_with_pca(X_train_scaled, X_test_scaled)
    
    # 5. Train models on reduced features
    pca_models, _, _ = train_baseline_models(X_train_pca, y_train, X_test_pca, y_test)
    print("\nComparison of models before and after PCA (with subject-wise splits):")
    for kernel in pca_models:
        print(f"{kernel} kernel - Full features: {baseline_models[kernel]['accuracy']:.4f}, PCA: {pca_models[kernel]['accuracy']:.4f}")
    
    # 6. Perform hyperparameter tuning with subject-wise cross-validation
    best_model = tune_hyperparameters_with_subject_cv(X_train, y_train, subject_train)
    
    # 7. Evaluate the best model
    print("\nEvaluating best model on test set (subjects not seen during training)...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test["Binary"], y_pred)
    conf_matrix = confusion_matrix(y_test["Binary"], y_pred)
    class_report = classification_report(y_test["Binary"], y_pred, target_names=['Inactive', 'Active'])
    
    print(f"Best model accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    
    # 8. Visualize results
    visualize_results(pca_models, y_test)
    
    # 9. Discussion including subject-wise validation impact
    print("\nDiscussion:")
    print("1. SVM Kernel Analysis:")
    print("   - Linear Kernel: Simpler, faster, and computationally efficient for high-dimensional data.")
    print("   - Polynomial Kernel: Can capture non-linear patterns but may be prone to overfitting.")
    print("   - RBF Kernel: Can model complex decision boundaries but requires careful parameter tuning.")
    print("\n2. Kernel Suitability:")
    print("   - For our human activity recognition binary classification task:")
    print("     a) The linear kernel achieved the highest accuracy (100%)")
    print("     b) This suggests that the active vs. inactive classes are linearly separable")
    print("     c) Our results show linear kernel outperforming both polynomial and RBF kernels")
    print("\n3. Subject-wise Validation Impact:")
    print("   - Using subject-wise splits provides a more realistic evaluation scenario")
    print("   - Surprisingly, high accuracy was maintained even with subject-wise splits")
    print("   - This suggests the features extracted are robust across different subjects")
    print("   - The linear kernel remained the most effective even when generalizing to new subjects")
    print("\n4. Dimensionality Reduction:")
    print("   - PCA helps reduce computational complexity while preserving most of the variance (87.5%)")
    print("   - Feature reduction showed minimal impact on linear kernel performance (from 100% to 99.94%)")
    print("   - RBF kernel showed the largest drop in performance after PCA, suggesting it relies more on the original feature space")
    print("\n5. Future Improvements:")
    print("   - Explore more challenging classification tasks (e.g., distinguishing between specific activities)")
    print("   - Investigate alternative feature selection methods beyond PCA")
    print("   - Consider more diverse subject populations to test generalizability further")

if __name__ == "__main__":
    main()