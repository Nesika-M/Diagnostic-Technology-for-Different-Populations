import pandas as pd
from sklearn.metrics import confusion_matrix

def calculate_diagnostic_accuracy(tp, tn, fp, fn):
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return sensitivity, specificity, ppv, npv, accuracy

def analyze_diagnostic_accuracy(dataset):
    # Calculate confusion matrix
    y_true = dataset['actual']
    y_pred = dataset['predicted']
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    
    # Calculate diagnostic accuracy metrics
    sensitivity, specificity, ppv, npv, accuracy = calculate_diagnostic_accuracy(tp, tn, fp, fn)
    
    return sensitivity, specificity, ppv, npv, accuracy

# Example dataset
dataset1 = pd.DataFrame({
    'actual': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'predicted': [1, 1, 1, 0, 1, 0, 0, 0, 1, 0]
})

dataset2 = pd.DataFrame({
    'actual': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'predicted': [1, 1, 0, 1, 1, 0, 0, 1, 0, 0]
})

# Analyze diagnostic accuracy for each dataset
sensitivity1, specificity1, ppv1, npv1, accuracy1 = analyze_diagnostic_accuracy(dataset1)
sensitivity2, specificity2, ppv2, npv2, accuracy2 = analyze_diagnostic_accuracy(dataset2)

# Print results
print("Dataset 1:")
print(f"Sensitivity: {sensitivity1:.3f}")
print(f"Specificity: {specificity1:.3f}")
print(f"PPV: {ppv1:.3f}")
print(f"NPV: {npv1:.3f}")
print(f"Accuracy: {accuracy1:.3f}")

print("\nDataset 2:")
print(f"Sensitivity: {sensitivity2:.3f}")
print(f"Specificity: {specificity2:.3f}")
print(f"PPV: {ppv2:.3f}")
print(f"NPV: {npv2:.3f}")
print(f"Accuracy: {accuracy2:.3f}")
