import json
import numpy as np
import torch
from dataloader_GMM import DARA_dataset
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from loguru import logger

def dataloader_to_numpy(data, labels):
    """Converts tensors in data loaders to numpy arrays for processing with GMM."""
    data_np = [d.cpu().numpy() for d in data]  # Assuming data is already a tensor
    labels_np = [l.cpu().numpy() for l in labels]  # Assuming labels is already a tensor
    return np.vstack(data_np), np.concatenate(labels_np)


def get_num_unique_labels(labels):
    """Return the number of unique labels."""
    return np.unique(labels).size


def train_and_evaluate(full_dataset, covariance_type='full', random_state=42):
    # Split the dataset
    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset))
    
    # Extract data and labels from loaders
    train_data, train_labels = dataloader_to_numpy(*zip(*train_loader))
    eval_data, eval_labels = dataloader_to_numpy(*zip(*eval_loader))

    # Standardize the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    eval_data = scaler.transform(eval_data)

    # Train GMM
    n_components = np.unique(train_labels).size
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    gmm.fit(train_data)

    # Evaluate GMM
    predicted_labels = gmm.predict(eval_data)
    accuracy = accuracy_score(eval_labels, predicted_labels)
    precision = precision_score(eval_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(eval_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(eval_labels, predicted_labels, average='weighted')

    return accuracy, precision, recall, f1

def cross_validate_model(full_dataset):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    import numpy as np

    skf = StratifiedKFold(n_splits=5)
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    # Convert list to NumPy array for indexing
    data = np.array(full_dataset.data, dtype=object)
    labels = np.array(full_dataset.labels)

    for train_index, test_index in skf.split(data, labels):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        scaler = StandardScaler()
        train_data = scaler.fit_transform(np.vstack(train_data))  # Ensure data is appropriately reshaped
        test_data = scaler.transform(np.vstack(test_data))

        n_components = get_num_unique_labels(train_labels)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(train_data)

        predicted_labels = gmm.predict(test_data)
        logger.debug(f"test_labels: {test_labels}")
        logger.debug(f"predicted_labels: {predicted_labels}")
        
        accuracies.append(accuracy_score(test_labels, predicted_labels))
        precisions.append(precision_score(test_labels, predicted_labels, average='weighted', zero_division=0))
        recalls.append(recall_score(test_labels, predicted_labels, average='weighted', zero_division=0))
        f1_scores.append(f1_score(test_labels, predicted_labels, average='weighted'))

    # Log average scores
    print(f"Average Accuracy: {np.mean(accuracies):.2f}")
    print(f"Average Precision: {np.mean(precisions):.2f}")
    print(f"Average Recall: {np.mean(recalls):.2f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.2f}")


def run():
    full_dataset = DARA_dataset('./data_cleaned.json', 'arch')  # Adjust path and label_type as needed
    cross_validate_model(full_dataset)

if __name__ == "__main__":
    run()
