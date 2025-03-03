# test_setup.py
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, linear_model, svm, metrics

# Print versions for confirmation
print("Python version:", sys.version)
print("\nLibrary versions:")
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)

# Test housing dataset access
print("\nTesting California Housing dataset access:")
try:
    housing = datasets.fetch_california_housing()
    print("✓ Successfully loaded California Housing dataset")
    print(f"  Dataset shape: {housing.data.shape}")
    print(f"  Feature names: {housing.feature_names[:3]}...")
except Exception as e:
    print("✗ Failed to load California Housing dataset:", e)

# Test UCI HAR Dataset access
print("\nTesting UCI HAR Dataset access:")
try:
    features_path = "UCI HAR Dataset/features.txt"
    features = pd.read_csv(features_path, sep=r"\s+", header=None, names=["idx", "feature"])
    print("✓ Successfully accessed UCI HAR Dataset")
    print(f"  Number of features: {len(features)}")
except Exception as e:
    print("✗ Failed to access UCI HAR Dataset:", e)

print("\nEnvironment setup test completed successfully!")