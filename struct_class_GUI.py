import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)

root = tk.Tk()
root.title("Structural Classification Tool")
root.geometry("800x500")
df_train, df_ind, trained_models, scaler = None, None, {}, None
def load_training_data():
    global df_train
    file_path = filedialog.askopenfilename(title="Select Training Dataset", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    df_train = pd.read_csv(file_path)
    log_message(f"Loaded Training Data: {file_path}")
def load_independent_data():
    global df_ind
    file_path = filedialog.askopenfilename(title="Select Independent Dataset", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    df_ind = pd.read_csv(file_path)
    log_message(f"Loaded Independent Data: {file_path}")
def preprocess_data(df):
    df.columns = df.columns.str.strip().str.replace("`", "")
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x.mean()), axis=0)
    return df
def train_models():
    global df_train, trained_models, scaler
    if df_train is None:
        messagebox.showerror("Error", "Please load the training dataset first!")
        return

 
    df_train = preprocess_data(df_train)
    df_train['label'] = df_train['enzymes'].apply(lambda x: 1 if x.startswith('pr') else 0)

    X_train = df_train.iloc[:, 1:-1].values  
    y_train = df_train['label'].values


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    log_message("Training models...")


    models = {
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}
    y_scores = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        y_pred_train = model.predict(X_train_scaled)
        accuracy = accuracy_score(y_train, y_pred_train)
        f1 = f1_score(y_train, y_pred_train)
        roc_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1]) if name != "KNN" else None
        y_scores[name] = model.predict_proba(X_train_scaled)[:, 1] if name != "KNN" else None

        auc_display = f"{roc_auc:.3f}" if roc_auc is not None else "N/A"
        log_message(f"{name} - Accuracy: {accuracy:.3f}, F1-score: {f1:.3f}, AUC: {auc_display}")

    log_message("Model training complete!")


    plt.figure(figsize=(12, 4))
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred_train = model.predict(X_train_scaled)
        cm = confusion_matrix(y_train, y_pred_train)

        plt.subplot(1, 3, i+1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-pr", "pr"], yticklabels=["Non-pr", "pr"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{name} - Confusion Matrix")

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8, 6))
    for name, y_score in y_scores.items():
        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_train, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for SVM & RF")
    plt.legend()
    plt.show()


def predict_independent():
    global df_ind, trained_models, scaler
    if df_ind is None:
        messagebox.showerror("Error", "Please load the independent dataset first!")
        return
    if not trained_models:
        messagebox.showerror("Error", "Train the models before running predictions!")
        return

    df_ind = preprocess_data(df_ind)
    X_ind = df_ind.iloc[:, 1:].values
    X_ind_scaled = scaler.transform(X_ind)

    log_message("Running predictions on independent dataset...")

    predictions = {}
    for name, model in trained_models.items():
        preds = model.predict(X_ind_scaled)
        predictions[name] = preds
        df_ind[f'Pred_{name}'] = preds

    output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if output_file:
        df_ind.to_csv(output_file, index=False)
        log_message(f"Predictions saved to {output_file}")

def log_message(msg):
    log_text.insert(tk.END, msg + "\n")
    log_text.see(tk.END)
frame_top = tk.Frame(root)
frame_top.pack(pady=10)

btn_load_train = tk.Button(frame_top, text="Load Training Data", command=load_training_data)
btn_load_train.grid(row=0, column=0, padx=10)

btn_train_models = tk.Button(frame_top, text="Train Models", command=train_models)
btn_train_models.grid(row=0, column=1, padx=10)

btn_load_indep = tk.Button(frame_top, text="Load Independent Data", command=load_independent_data)
btn_load_indep.grid(row=1, column=0, padx=10, pady=5)

btn_predict = tk.Button(frame_top, text="Run Predictions", command=predict_independent)
btn_predict.grid(row=1, column=1, padx=10, pady=5)

frame_log = tk.Frame(root)
frame_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

log_text = tk.Text(frame_log, height=10, width=80)
log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
log_scroll = tk.Scrollbar(frame_log, command=log_text.yview)
log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
log_text.config(yscrollcommand=log_scroll.set)

root.mainloop()
