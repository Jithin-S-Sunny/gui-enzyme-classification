import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from esm import pretrained
from Bio import SeqIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import torch
import os
import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def clean_sequence(sequence):
    """
    Removes unsupported characters from the sequence.
    """
    valid_chars = "ACDEFGHIKLMNPQRSTVWY"  # Standard amino acids
    return ''.join([char for char in sequence if char in valid_chars])


def generate_esm2_embeddings(sequences):
    """
    Generate ESM2 embeddings for a list of sequences.
    """
    print("Loading ESM2 model...")
    model, alphabet = pretrained.esm2_t33_650M_UR50D()  
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    embeddings = []
    for seq in sequences:
        data = [("seq", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33]) 
        cls_embedding = results["representations"][33][0, 0, :].numpy()  
        embeddings.append(cls_embedding)

    print(f"Generated embeddings for {len(sequences)} sequences.")
    return np.array(embeddings)


def build_nn_model(input_dim):
    """
    Build a feedforward neural network for classification with regularization.
    """
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
        Dropout(0.4),  # Increased dropout for better generalization
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


fasta_file = "lac_perox.fasta"
independent_fasta = "independant_lacperox.fasta"
print("Extracting sequences and labels...")
sequences = []
labels = []
for record in SeqIO.parse(fasta_file, "fasta"):
    sequence = clean_sequence(str(record.seq))
    if record.description.startswith('lac|'):
        labels.append(0)
    elif record.description.startswith('pr|'):
        labels.append(1)
    else:
        continue
    sequences.append(sequence)
print("Generating ESM2 embeddings...")
features = generate_esm2_embeddings(sequences)
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=50) 
reduced_features = pca.fit_transform(features)
labels = np.array(labels)
print("Performing cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for train_idx, val_idx in kf.split(reduced_features):
    X_train, X_val = reduced_features[train_idx], reduced_features[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    model = build_nn_model(input_dim=reduced_features.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=32,
                        class_weight=class_weights_dict,
                        callbacks=[early_stopping, lr_scheduler],
                        verbose=0)

    val_predictions = np.argmax(model.predict(X_val), axis=1)
    acc = accuracy_score(y_val, val_predictions)
    f1 = f1_score(y_val, val_predictions, average='weighted')
    precision = precision_score(y_val, val_predictions, average='weighted')
    recall = recall_score(y_val, val_predictions, average='weighted')

    cv_results.append({
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'history': history.history  
    })
print("\nProcessing the independent dataset...")
independent_sequences = []
independent_labels = []
for record in SeqIO.parse(independent_fasta, "fasta"):
    sequence = clean_sequence(str(record.seq))
    if record.description.startswith('lac|'):
        independent_labels.append(0)
    elif record.description.startswith('pr|'):
        independent_labels.append(1)
    else:
        continue
    independent_sequences.append(sequence)

if independent_sequences:
    print("Generating ESM2 embeddings for independent dataset...")
    independent_features = generate_esm2_embeddings(independent_sequences)

    print("Reducing dimensions for independent dataset using PCA...")
    independent_features_reduced = pca.transform(independent_features)

    print("Making predictions on independent dataset...")
    independent_predictions = model.predict(independent_features_reduced)
else:
    print("No valid independent sequences found.")
    independent_predictions = None
    independent_labels = None
print("Saving data for plotting...")
with open("pipeline_data.pkl", "wb") as f:
    pickle.dump({
        'cv_results': cv_results,
        'reduced_features': reduced_features,
        'labels': labels,
        'pca': pca,
        'independent_predictions': independent_predictions,
        'independent_labels': independent_labels
    }, f)

