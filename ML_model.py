import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from Bio import SeqIO
from itertools import product
import pandas as pd
from sklearn.cluster import KMeans


def calculate_aac(sequence): 
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq_len = len(sequence)
    composition = {aa: sequence.count(aa) / seq_len for aa in amino_acids}
    return composition
    

def calculate_cksapp(sequence, k=1): 
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    kspaced_pairs = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    cksaap_counts = {pair: 0 for pair in kspaced_pairs}
    for i in range(len(sequence) - k - 1):
        pair = sequence[i] + sequence[i + k + 1]
        if pair in cksaap_counts:
            cksaap_counts[pair] += 1
    total_pairs = sum(cksaap_counts.values())
    cksaap_composition = {pair: count / total_pairs for pair, count in cksaap_counts.items()}
    return cksaap_composition

def calculate_ctd(sequence): 
    hydrophobic = "AVLIMFWY"
    hydrophilic = "RNDQEKSTCHGP"
    hydrophobic_count = sum(sequence.count(aa) for aa in hydrophobic)
    hydrophilic_count = sum(sequence.count(aa) for aa in hydrophilic)
    total_count = len(sequence)

    composition = [hydrophobic_count / total_count, hydrophilic_count / total_count]
    transitions = sum(
        1 for i in range(len(sequence) - 1) if
        (sequence[i] in hydrophobic and sequence[i + 1] in hydrophilic) or
        (sequence[i] in hydrophilic and sequence[i + 1] in hydrophobic)
    )
    transition = [transitions / (len(sequence) - 1)]

    def calculate_distribution(positions, total_length):
        if not positions:
            return [0] * 5
        return [
            min(positions) / total_length,
            np.percentile(positions, 25) / total_length,
            np.percentile(positions, 50) / total_length,
            np.percentile(positions, 75) / total_length,
            max(positions) / total_length,
        ]

    hydrophobic_positions = [sequence.find(aa) + 1 for aa in hydrophobic if sequence.find(aa) != -1]
    hydrophilic_positions = [sequence.find(aa) + 1 for aa in hydrophilic if sequence.find(aa) != -1]

    hydrophobic_distribution = calculate_distribution(hydrophobic_positions, len(sequence))
    hydrophilic_distribution = calculate_distribution(hydrophilic_positions, len(sequence))

    distribution = hydrophobic_distribution + hydrophilic_distribution
    return composition + transition + distribution

def calculate_g_gap_dipeptide(sequence, g=1): 
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    gap_dipeptide_counts = {dipeptide: 0 for dipeptide in dipeptides}

    for i in range(len(sequence) - g - 1):
        dipeptide = sequence[i] + sequence[i + g + 1]
        if dipeptide in gap_dipeptide_counts:
            gap_dipeptide_counts[dipeptide] += 1

    total_dipeptides = sum(gap_dipeptide_counts.values())
    gap_dipeptide_composition = {dipeptide: count / total_dipeptides for dipeptide, count in gap_dipeptide_counts.items()}
    return gap_dipeptide_composition

def extract_features_and_labels(fasta_file, k=1, g=1): 
    features = []
    labels = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        header = record.description
        if header.startswith('lac|'):
            labels.append(0)
        elif header.startswith('pr|'):
            labels.append(1)
        else:
            continue

        aac = calculate_aac(sequence)
        cksaap = calculate_cksapp(sequence, k=k)
        ctd = calculate_ctd(sequence)
        g_gap_dipeptide = calculate_g_gap_dipeptide(sequence, g=g)

        feature_vector = list(aac.values()) + list(cksaap.values()) + ctd + list(g_gap_dipeptide.values())
        features.append(feature_vector)

    return np.array(features), np.array(labels)

def scale_and_reduce(features, n_components=60):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_scaled)

    return features_reduced, scaler, pca

def cross_validate_models(features, labels, classifiers, cv=5):
    results = {}
    for name, clf in classifiers.items():
        print(f"\nPerforming cross-validation for {name}...")
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_results = cross_validate(clf, features, labels, cv=cv, scoring=scoring)
        
        results[name] = {
            "Accuracy": np.mean(cv_results['test_accuracy']),
            "Precision": np.mean(cv_results['test_precision_weighted']),
            "Recall": np.mean(cv_results['test_recall_weighted']),
            "F1 Score": np.mean(cv_results['test_f1_weighted']),
        }
    return results

def evaluate_on_independent_data(independent_fasta, classifiers, scaler, pca, k=1, g=1):
    independent_features, independent_labels = extract_features_and_labels(independent_fasta, k=k, g=g)
    independent_features_scaled = scaler.transform(independent_features)
    independent_features_reduced = pca.transform(independent_features_scaled)

    results = {}
    for name, clf in classifiers.items():
        if hasattr(clf, "predict_proba"):
            confidences = clf.predict_proba(independent_features_reduced).max(axis=1)
        else:
            confidences = clf.decision_function(independent_features_reduced)
            confidences = np.abs(confidences)  

        predictions = clf.predict(independent_features_reduced)
        acc = accuracy_score(independent_labels, predictions)
        precision = precision_score(independent_labels, predictions, average='weighted')
        recall = recall_score(independent_labels, predictions, average='weighted')
        f1 = f1_score(independent_labels, predictions, average='weighted')
        cm = confusion_matrix(independent_labels, predictions)

        results[name] = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": cm,
        }
        output_file = f"{name}_independent_predictions.csv"
        sequence_ids = [record.id for record in SeqIO.parse(independent_fasta, "fasta")]
        df = pd.DataFrame({
            "Sequence ID": sequence_ids,
            "Predicted Label": predictions,
            "Confidence Score": confidences
        })
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    return results

if __name__ == "__main__":

    fasta_file = "lac_perox.fasta"
    independent_fasta = "independant_lacperox.fasta"
    k = 1
    g = 1

    print("Extracting features and labels...")
    features, labels = extract_features_and_labels(fasta_file, k=k, g=g)
    features_df = pd.DataFrame(features)
    features_df['Label'] = labels
    features_df.to_csv("features_labels.csv", index=False)
    print("Features and labels saved to features_labels.csv")

    print("Scaling and reducing features...")
    features_reduced, scaler, pca = scale_and_reduce(features, n_components=60)
    pd.DataFrame(features_reduced).to_csv("features_reduced.csv", index=False)

    print("Defining classifiers...")
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='linear', probability=True, random_state=42)
    }

    print("\nPerforming cross-validation...")
    cv_results = cross_validate_models(features_reduced, labels, classifiers, cv=5)
    pd.DataFrame(cv_results).to_csv("cross_validation_results.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(features_reduced, labels, test_size=0.2, random_state=42)
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pd.DataFrame(clf.predict(X_test), columns=["Predicted"]).to_csv(f"{name}_predictions.csv", index=False)

    print("\nEvaluating on independent dataset...")
    independent_results = evaluate_on_independent_data(independent_fasta, classifiers, scaler, pca, k=k, g=g)
    pd.DataFrame(independent_results).to_csv("independent_results.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(features_reduced, labels, test_size=0.2, random_state=42)
training_results = []

for name, clf in classifiers.items():
    print(f"Fitting {name}...")
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    training_results.append({
        "Model": name,
        "Training Accuracy": train_acc
    })


training_results_df = pd.DataFrame(training_results)
training_results_df.to_csv("training_accuracy_results.csv", index=False)
print("Training accuracy results saved to training_accuracy_results.csv")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def calculate_roc_auc(classifiers, X_test, y_test):
    roc_data = []
    for name, clf in classifiers.items():
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        
        roc_data.append({
            "Classifier": name,
            "FPR": fpr.tolist(),
            "TPR": tpr.tolist(),
            "AUC": roc_auc
        })


    roc_df = pd.DataFrame(roc_data)
    roc_df.to_csv("roc_auc_data.csv", index=False)
    print("ROC AUC data saved to roc_auc_data.csv")
    return roc_df

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = train_test_split(features_reduced, labels, test_size=0.2, random_state=42)

    print("\nFitting classifiers...")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)

    print("\nCalculating ROC and AUC...")
    calculate_roc_auc(classifiers, X_test, y_test)


