# ECE695-GenModel

This is the code for my ECE 695 final project. The project is about

> Hybrid DNN+GMM Approach for Detecting Anomalies in Pre-trained Model Architectures. 

# Environment

The environment can be installed using `requirements.txt` file. 
```bash
pip install -r requirements.txt
```

# Motivation
As innovation in deep learning continues, many engineers seek to adopt Pre-Trained Models (PTMs) as components in computer systems.
Researchers publish PTMs, which engineers adapt for quality or performance prior to deployment.
PTM authors should choose appropriate metadata for their PTMs, which would facilitate model discovery and reuse.
HoIver, prior research has reported that model architectures are not always Ill chosen --- and are sometimes erroneous.


# Problem Statement 
This project present a solution to detect naming anomalies and introduce a novel automated DNN Architecture Assessment technique, capable of detecting PTM naming anomalies. 
This work envisions future works on leveraging meta-features of PTMs to improve model reuse and trustworthiness.


# Implementation

The process begins with open-smyce pre-trained Iights. 
First, I load each Iight and feed dummy inputs to outline the graph architecture of each Pre-trained Model (PTM). Next, I transform the computational graph into an abstract architecture and implement n-gram feature extraction. Utilizing these features, I train a CNN classifier to identify naming anomalies.

## Feature Extraction

It is insufficient to identify naming anomalies solely based on the identifiers. 
In this section, I present my feature extraction techniques, including graph conversion, delineation of abstract architecture, and n-gram feature selection.

### Graph Conversion
I developed an automated pipeline to transform pre-trained Iights into computational graphs.
Existing work shows that Pytorch is the dominant framework in Hugging Face, and using Hugging Face API I will load the PyTorch model by default.
Due to PyTorch's dynamic structure, I introduce *dummy inputs* to the models to construct and trace the computational graphs effectively. 
I refined my automated pipeline by iteratively adjusting *dummy inputs*: if a conversion failed due to the input, I incorporated a new input type into my input list. Consequently, my finalized pipeline employs fixed language inputs while utilizing random tensors for models in the vision and audio domains.


I conceptualized all neural network graphs as directed acyclic graphs (DAGs), enhancing the comprehensive and efficient application of algorithms. To handle cycles in models like LSTM, I remove upstream or backward edges, turning every network into a DAG with ``complete reachability''. 
In this configuration, each node is connected through a path from an input to an output, guaranteeing that my algorithm covers all functional layers.

### Abstract Architecture
The abstract architecture not only includes information about layers, parameters, and input/output dimensions but also incorporates detailed information about the connections betIen nodes, offering a fuller depiction of the model’s architecture. 
This enhancement is pivotal for my n-gram feature selection method.

### N-gram Feature Selection
I propose a method for vectorizing PTM architectural information using two fundamental components that define their architecture: layer connections,
and layer parameters. 
These aspects include the structural configuration of each PTM, such as pairs of connected layers (e.g., `(Conv2D, BatchNormalization)`), and specific layer attributes (`kernel_size: [1, 1]` in a Conv2D layer).
I apply an N-gram feature extraction method to convert the architectural characteristics of each DNN into vector form.
Specifically, I employ 2-grams for analyzing layer connections and 1-grams for detailing layer parameter information. 
My goal is to find a good unsupervised feature for PTM architectures accurately and efficiently.
I also experimented with 3-gram methods for feature extraction; hoIver, I found that the process was notably slow, and the resulting features are overly sparse and high-dimensional, suggesting this method is unsuitable for extracting features from PTM architectures.
Subsequently, I apply padding to standardize the length of features across all PTMs.
This innovative approach provides a structured and detailed basis for analyzing and comparing PTM architectures.


## Dataset Preparation

My evaluation focuses on two key naming labels: `model_type` and `architecture`. 
I evaluate the pipeline on a sampled data from the PeaTMOSS dataset by calculating the accuracy of the prediction of the trained classifier.
There are totally 132 architectures which has over 50 PTM instances. I randomly selected 50 architectures from the dataset (50/132, 38%).
During feature extraction, we were unable to load the PTMs from four architectures, \eg `Bloom`, because their model sizes are too large to be loaded on a single GPU.
This resulted in 46 unique architectures. 
I was also unable to load another 600 PTMs across these 46 categories because either the dummy inputs could not be used to those models, missing configuration files, or their architecture included customized code which we cannot fully trust.
Overall, we collected 1700 PTM packages from 46 architectures and 30 model types.

I also changed the `model_type` and `architectures` of the categories which had less than 20 data as `Others`. This reduced the total number of `architectures` to 40, and `model_types` to 27.

## Training and Evaluation

I designed a Deep Neural Network (DNN) classifier for anomaly detection in the PTM dataset because this approach has been shown effective on similar tasks.
I also implemented a DNN+GMM model to improve the performance of the classifier. The GMM model is used to filter out the noise in the data and improve the performance of the DNN model.

For the training, I use an 80-20 split for training and evaluation. The classifier consists of four fully connected layers. 
I systematically implemented a grid search to optimize the hyperparameters for DARA.
The final training parameters include 40 epochs, a learning rate of 1e-3, Cross Entropy loss, and a step-wise LR scheduler. Batch sizes are 256 for training and 32 for evaluation. 
I improved model validation using 5-fold cross-validation on the shuffled dataset.




# Code Structure

The code and scripts is organized as follows:

<!-- Create a table -->
| File/Folder | Description | Script  |
| --- | --- | --- |
| `data_analysis.py` | Get the statistics of the data | `python data_analysis.py` |
| `plot_peatmoss.py` | Plot the distribution of model architectures in the PeaTMOSS dataset | `python plot_peatmoss.py` |
| `data_pre.py` | Prepare the data from PeaTMOSS dataset. Convert extracted features into vector. | `python data_pre.py` |
| `data_cleaned_filtered.json` | The cleaned and filtered data from PeaTMOSS dataset which is used as the training and eval data. |  |
| `dataloader.py`, `dataloader_GMM.py` | Load the data |  |
| `modeling.py`, `modeling_GMM.py` | Define the model architecture |  |
| `run.py` | Train and evaluate the vanilla DNN model | ``` python run.py``` |
| `run_DNNwithGMM.py` | Train and evaluate the DNN+GMM model | `python run_DNNwithGMM.py` |
| `pca_plot.py` | Plot the PCA of the data | `python pca_plot.py` |
|`logs/` | The logs of the training and evaluation |  |

