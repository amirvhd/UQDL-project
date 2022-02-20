# UQDL-project

This repository contains two codes. First part is for section 5.1 form report (Linear case analysis of BNN with MFVI layers) and second part is for section 5.5 form report (comparison between MFVI and full covariance) 

## section 5.1 
The relevant files for this part are Main.ipynb and Model.py.
Model.py contains the model architecture for the BNN. 

Ite gets two inputs:
- Number of layers (layer_size)
- Whether to use activations function or not (activation)


In Main.ipynb:
After loading the data and spliting the data, three models are defined. The results after training of each models are saved (e.g. Models/model_1_net.pth).
For each model, the 10000 different product matrices produced. and the covariance matrix for each model is calculted and saved (e.g. Covariance_matrices/cov_1.npy)

## section 5.5

The relevant files for this part are Iris.ipynb and Model_iris.py.
Model_iris has the same structure as Model.py

In Iris.ipynb:
After loading and spliting the data, 8 different models are defined. The best cross-entropy loss for each case is saved. The process of training for each model is repeated for 10 different random seeds. the results is saved in results.npy. So this files contains 80 different values for 10 seeds and 8 models. The mean values over seeds for all models (8 different values) are illustrated.  
