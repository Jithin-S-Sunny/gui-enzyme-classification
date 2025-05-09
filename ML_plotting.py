import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
features_reduced = pd.read_csv("features_reduced.csv").to_numpy()
labels = pd.read_csv("features_labels.csv")['Label'].to_numpy()
plt.figure(figsize=(8, 6))
plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=labels, cmap='viridis', s=10)
plt.title('PCA-transformed Features (First Two Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class Label')
plt.savefig("pca_features_plot.png")
plt.show()

explained_variance = PCA(n_components=features_reduced.shape[1]).fit(features_reduced).explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.savefig("cumulative_variance_plot.png")
plt.show()
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(features_reduced)
plt.figure(figsize=(8, 6))
plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=clusters, cmap='tab10', s=10)
plt.title('K-Means Clustering on PCA-Reduced Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.savefig("kmeans_clustering_plot.png")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_roc_auc_from_csv(csv_path, save_path="roc_auc_plot.png"):
    roc_df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 8))
    for _, row in roc_df.iterrows():
        fpr = eval(row["FPR"])  # Convert string back to list
        tpr = eval(row["TPR"])
        auc_value = row["AUC"]
        classifier_name = row["Classifier"]

        plt.plot(fpr, tpr, label=f'{classifier_name} (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.show()
    print(f"ROC AUC plot saved to {save_path}")

if __name__ == "__main__":
    plot_roc_auc_from_csv("roc_auc_data.csv")

