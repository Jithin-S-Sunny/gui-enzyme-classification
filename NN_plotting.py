import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

def plot_learning_curve(history, fold_idx):
    """
    Plot and save the learning curve for a specific fold.
    """
    plt.figure()
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold_idx}: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"fold_{fold_idx}_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold_idx}: Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"fold_{fold_idx}_accuracy_curve.png")
    plt.close()

def plot_pca_variance(pca):
    """
    Plot and save the cumulative explained variance by PCA.
    """
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title('Cumulative Explained Variance by PCA')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.savefig("pca_variance.png")
    plt.close()

def plot_roc_curve(y_true, y_scores, fold_idx=None):
    """
    Plot and save the ROC curve for the provided labels and confidence scores.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('ROC Curve' + (f' - Fold {fold_idx}' if fold_idx else ''))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    filename = f"roc_curve_{fold_idx}.png" if fold_idx else "roc_curve_independent.png"
    plt.savefig(filename)
    plt.close()

def save_independent_predictions(predictions, labels, output_file):
    """
    Save the independent dataset predictions with confidence scores to a CSV file.
    """
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    results_df = pd.DataFrame({
        "True Label": labels,
        "Predicted Label": predicted_classes,
        "Confidence Score": confidence_scores
    })
    results_df.to_csv(output_file, index=False)
    print(f"Independent predictions saved to {output_file}")

print("Loading data from Pipeline 1...")
with open("pipeline_data.pkl", "rb") as f:
    data = pickle.load(f)  # Corrected

cv_results = data['cv_results']
pca = data['pca']
independent_labels = data.get('independent_labels', None)
independent_predictions = data.get('independent_predictions', None)

print("Generating plots...")

for fold_idx, result in enumerate(cv_results, 1):
    plot_learning_curve(result['history'], fold_idx)
    if 'val_accuracy' in result['history']:
        y_val_true = result.get('y_val_true')
        y_val_scores = result.get('y_val_scores')
        if y_val_true is not None and y_val_scores is not None:
            plot_roc_curve(y_val_true, y_val_scores, fold_idx)

plot_pca_variance(pca)
if independent_labels is not None and independent_predictions is not None:
    y_true = independent_labels
    y_scores = independent_predictions[:, 1]  # Assuming binary classification and class 1 scores
    plot_roc_curve(y_true, y_scores)
    save_independent_predictions(independent_predictions, independent_labels, "independent_predictions.csv")

print("All plots and files saved.")

