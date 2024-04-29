import json
import random
import torch

class DARA_dataset:
    def __init__(self, dict_path: str, label_type: str) -> None:
        self.label_type = label_type
        self.model_dict = self.load_dict(dict_path)  # Load model vectors
        self.model_names = list(self.model_dict.keys())  # List of model names
        self.num_classes = len(set(self.model_dict[model_name][self.label_type] for model_name in self.model_names))  # Number of unique classes
        self.data, self.labels = self.processing()  # Process and split data and labels
        self.label_to_index = self.create_label_mapping()  # Create label mapping
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}  # Create index to label mapping

        self.convert_labels_to_indices()  # Convert labels to indices
        self.shuffle_data()  # Shuffle data for training
        self.train_data, self.test_data = self.split_data()  # Split data into training and testing sets

    def load_dict(self, dict_path: str) -> dict:
        with open(dict_path, 'r') as f:
            return json.load(f)

    def processing(self):
        data = []
        labels = []
        for model_name in self.model_names:
            l_tensor = torch.tensor(self.model_dict[model_name]['l'], dtype=torch.float).unsqueeze(0)
            p_tensor = torch.tensor(self.model_dict[model_name]['p'], dtype=torch.float).unsqueeze(0)
            vec = torch.cat((l_tensor, p_tensor), dim=1)
            label = self.model_dict[model_name][self.label_type]
            data.append(vec)
            labels.append(label)
        return data, labels

    def create_label_mapping(self):
        unique_labels = sorted(set(self.labels))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def convert_labels_to_indices(self):
        label_indices = [self.label_to_index[label] for label in self.labels]
        self.labels = label_indices

    def shuffle_data(self):
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)

    def split_data(self, train_ratio=0.8):
        split_index = int(len(self.data) * train_ratio)
        self.train_data = self.data[:split_index]
        self.test_data = self.data[split_index:]
        return self.train_data, self.test_data

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

    def get_num_classes(self):
        return len(self.label_to_index)

    def get_label_mapping(self):
        return self.index_to_label

    def get_data_shape(self):
        return self.data[0].size()
