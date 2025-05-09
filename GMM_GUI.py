import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

def load_matrix():
    file_path = filedialog.askopenfilename(title="Select TM-align Matrix", filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            global tm_score_matrix, distance_matrix, symmetrized_distance_matrix
            tm_score_matrix = pd.read_csv(file_path, header=None)
            tm_score_matrix = tm_score_matrix.drop(0, axis=1).drop(0, axis=0).astype(float).reset_index(drop=True)
            tm_score_matrix.columns = range(tm_score_matrix.shape[1])
            distance_matrix = 1 - tm_score_matrix
            symmetrized_distance_matrix = (distance_matrix + distance_matrix.T) / 2
            messagebox.showinfo("Success", "TM-align matrix loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load matrix: {e}")
def reduce_dimensions():
    method = dim_method.get()
    try:
        global features
        if method == "MDS":
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
            features = mds.fit_transform(symmetrized_distance_matrix)
        elif method == "PCA":
            pca = PCA(n_components=2)
            features = pca.fit_transform(tm_score_matrix)
        elif method == "t-SNE":
            tsne = TSNE(n_components=2, metric='euclidean', random_state=0)
            features = tsne.fit_transform(tm_score_matrix)
        else:
            raise ValueError("Invalid dimensionality reduction method.")
        messagebox.showinfo("Success", f"Dimensionality reduction with {method} completed!")
    except Exception as e:
        messagebox.showerror("Error", f"Dimensionality reduction failed: {e}")
def cluster_data():
    try:
        global spectral_labels, kmeans_labels, gmm_labels
        spectral = SpectralClustering(n_clusters=5, affinity='precomputed', random_state=0)
        spectral_labels = spectral.fit_predict(tm_score_matrix)

        kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
        kmeans_labels = kmeans.fit_predict(features)

        gmm = GaussianMixture(n_components=5, random_state=0, n_init=10)
        gmm_labels = gmm.fit_predict(features)

        messagebox.showinfo("Success", "Clustering completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Clustering failed: {e}")

def plot_clusters(labels, title):
    try:
        special_indices_red = list(map(int, red_indices.get().split(','))) if red_indices.get() else []
        special_indices_green = list(map(int, green_indices.get().split(','))) if green_indices.get() else []

        plt.figure(figsize=(8, 8))
        unique_labels = np.unique(labels)
        grey_shades = [str(0.2 * (i+1)) for i in range(len(unique_labels))] 
        for index in range(features.shape[0]):
            if index in special_indices_red:
                color = 'red'
                plt.scatter(features[index, 0], features[index, 1], color=color)
            elif index in special_indices_green:
                color = 'green'
                plt.scatter(features[index, 0], features[index, 1], color=color)
                plt.text(features[index, 0], features[index, 1], f'Node {index}', fontsize=9, color='green')
            else:
                label = labels[index]
                color = grey_shades[unique_labels.tolist().index(label)]        
                plt.scatter(features[index, 0], features[index, 1], color=color)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Plotting failed: {e}")
root = tk.Tk()
root.title("Clustering Pipeline GUI")
load_button = tk.Button(root, text="Load TM-align Matrix", command=load_matrix)
load_button.pack(pady=10)
dim_method_label = tk.Label(root, text="Select Dimensionality Reduction Method:")
dim_method_label.pack()
dim_method = tk.StringVar(value="MDS")
dim_mds = tk.Radiobutton(root, text="MDS", variable=dim_method, value="MDS")
dim_mds.pack()
dim_pca = tk.Radiobutton(root, text="PCA", variable=dim_method, value="PCA")
dim_pca.pack()
dim_tsne = tk.Radiobutton(root, text="t-SNE", variable=dim_method, value="t-SNE")
dim_tsne.pack()

reduce_button = tk.Button(root, text="Perform Dimensionality Reduction", command=reduce_dimensions)
reduce_button.pack(pady=10)
cluster_button = tk.Button(root, text="Perform Clustering", command=cluster_data)
cluster_button.pack(pady=10)
red_label = tk.Label(root, text="Enter Node Indices to Highlight in Red (comma-separated):")
red_label.pack()
red_indices = tk.Entry(root)
red_indices.pack()

green_label = tk.Label(root, text="Enter Node Indices to Highlight in Green (comma-separated):")
green_label.pack()
green_indices = tk.Entry(root)
green_indices.pack()
plot_spectral_button = tk.Button(root, text="Plot Spectral Clustering", command=lambda: plot_clusters(spectral_labels, "Spectral Clustering"))
plot_spectral_button.pack(pady=5)

plot_kmeans_button = tk.Button(root, text="Plot K-Means Clustering", command=lambda: plot_clusters(kmeans_labels, "K-Means Clustering"))
plot_kmeans_button.pack(pady=5)

plot_gmm_button = tk.Button(root, text="Plot GMM Clustering", command=lambda: plot_clusters(gmm_labels, "GMM Clustering"))
plot_gmm_button.pack(pady=5)

root.mainloop()

