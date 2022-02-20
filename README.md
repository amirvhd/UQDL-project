# UQDL-project

This repository contains two codes. The first part is for section 5.1 form report (Linear case analysis of BNN with MFVI layers), and the second part is for section 5.5 form report (comparison between MFVI and full covariance) 

## section 5.1 
The relevant files for this part are Main.ipynb and Model.py.
Model.py contains the model architecture for the BNN. 

It gets two inputs:
- Number of layers (layer_size)
- Whether to use activations function or not (activation)


In Main.ipynb:
After loading the data and splitting the data, three models are defined. First, the results after training of each model are saved (e.g., Models/model_1_net.pth).
For each model, the 10000 different product matrices were produced. and the covariance matrix for each model is calculated and saved (e.g., Covariance_matrices/cov_1.npy)

## section 5.5

The relevant files for this part are Iris.ipynb and Model_iris.py.
Model_iris has the same structure as Model.py

In Iris.ipynb:
After loading and splitting the data, 8 different models are defined. The best cross-entropy loss for each case is saved. The training process for each model is repeated for 10 different random seeds. The results are saved in "results.npy". So this file contains 80 different values for 10 seeds and 8 models. The mean values over seeds for all models (8 different values) are illustrated.  
