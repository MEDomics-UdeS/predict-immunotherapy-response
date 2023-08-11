# Predict immunotherapy response project
This project contains the code which predicts response to immune checkpoint inhibition treatment from biomarkers of patients suffering from cancers. The goal is to highlight the predictive property of mutational signatures in immune checkpoint inhibition treatment.

This project is in collaboration with the *MEDomics UdeS* laboratory from the Sherbrooke University, and a biology laboratory of Sherbrooke University.

The internship report is available on : https://bit.ly/3KBhwb2 

## Installation
Before running codes, please follow these steps to make sure you have all dependencies :
1. Clone the repository
2. Move to the root of this repository
3. Create a Python virtual environment. The creation and activation commands depending on your OS are available on : https://docs.python.org/3/library/venv.html 
4. Install the requirements : ```pip install -r requirements.txt```

You are ready to begin !

## Executions

The pipelines needs some arguments, described below :
- --sigmut : to include mutational signatures or not, and how.
    - `no-sigmut` : no mutational signatures. Only the biomarkers from the reference article are included.
    - `only-sigmut-sbs` : include only SBS mutational signatures.
    - `only-sigmut-indel` : include only INDEL mutational signatures. 
    - `only-sigmut-comb` : include only SBS and INDEL mutational signatures.
    - `comb` : include biomarkers from the reference article and the mutational signatures.
- --architecture : the model architecture :
    - `logistic-regression` : the logistic regression, for binary classification problem.
    - `gcn` : the Graph Convolutional Network, for binary classification problem.
    - `gat` : the Graph Attention Network, for binary classification problem.
    - `cox` : the Cox Model, for survival analysis problem.
    - `gcn-cox` : the Graph Convolutional Network Cox Model, for survival analysis problem.
    - `gat-cox` : the Graph Attention Network Cox Model, for survival analysis problem.
- --n_features : the n_features most important features will be considered for each patient
- --n_epochs : the number of epochs to train the models. The Cox Model is trained on 1 epoch.
- --lr : the learning rate used during the Stochastic Gradient Descent. Not used in Cox Model.
- --reg : the regularization factor used during the loss computation. Not used in Cox Model.
- --max_neighbors (only for Graph Neural Networks) : the maximum of neighbors for each node of the graph.

### Classification problem

#### Python scripts

The Python file containing the whole pipeline is the `main-classification.py` file.

For the **Logistic Regression**, an example of running is the following :

```
python3 main-classification.py --sigmut comb --architecture logistic-regression --n_features 5 --n_epochs 150 --lr 0.005 --reg 0.005
```

For the **Graph Neural Network**, an example of running is the following :

```
python3 main-classification.py --sigmut no-sigmut --architecture gcn --n_features 6 --n_epochs 100 --lr 0.005 --reg 0.005 --max_neighbors 2
```

#### Jupyter notebooks examples

To see examples of executions and results, you can read the following notebooks :
- `A.1-clf-logreg-sigmut.ipynb` : comparison between without mutational signatures, with only mutational signatures, and with combination of reference article biomarkers and mutational signatures, with the logistic regression. 
- `A.2-clf-logreg-sbs-indel.ipynb` : comparison between with only SBS signatures, with only INDEL signatures, and with combination of both, with the logistic regression. 
- `A.3-clf-gcn-sigmut.ipynb` : comparison between without mutational signatures, with only mutational signatures, and with combination of reference article biomarkers and mutational signatures, with the Graph Convolutional Network.
- `A.4-clf-gcn-sbs-indel.ipynb` : comparison between with only SBS signatures, with only INDEL signatures, and with combination of both, with the Graph Neural Network. 

### Survival analysis problem

#### Python script

The Python file containing the whole pipeline is the `main-survival-analysis.py` file.

For the **Cox Model**, an example of running is the following :

```
python3 main-survival-analysis.py --sigmut comb --architecture cox --n_features 5 --n_epochs 150 --lr 0.005 --reg 0.005
```

For the **Graph Neural Network Cox Model**, an example of running is the following :

```
python3 main-survival-analysis.py --sigmut no-sigmut --architecture gcn --n_features 6 --n_epochs 100 --lr 0.005 --reg 0.005 --max_neighbors 2
```

#### Jupyter notebooks examples

To see examples of executions and results, you can read the following notebooks :
- `B.1-survival-analysis-cox-sigmut.ipynb` : comparison between without mutational signatures, with only mutational signatures, and with combination of reference article biomarkers and mutational signatures, with the Cox Model. 
- `B.2-survival-analysis-cox-sbs-indel.ipynb` : comparison between with only SBS signatures, with only INDEL signatures, and with combination of both, with the Cox Model. 
- `B.3-survival-analysis-gcn-cox-sigmut.ipynb` : comparison between without mutational signatures, with only mutational signatures, and with combination of reference article biomarkers and mutational signatures, with the Graph Convolutional Network Cox Model.
- `B.4-survival-analysis-gcn-cox-sbs-indel.ipynb` : comparison between with only SBS signatures, with only INDEL signatures, and with combination of both, with the Graph Neural Network Cox Model. 

## Project Tree
```
├── data
│   ├── cohort-dataset.xlsx                         <- dataset from the reference article with mutational signatures
│   ├── cohort-dataset-with-logreg-output.xlsx      <- dataset from the reference article with mutational signatures and the output of the logistic regression model (class 1 probability)
|
├── evaluation                           
│   ├── ClassificationMetrics.py                    <- class containing evaluation metrics for binary classification
│   ├── SurvivalMetrics.py                          <- class containing evaluation metrics for survival analysis
|
├── manage                           
│   ├── CoxTrainTestManager.py                      <- class handling train-test process of the Cox Model
│   ├── GNNClassifierTrainTestManager.py            <- class handling train-test process of the GNN classifier
│   ├── GNNCoxTrainTestManager.py                   <- class handling train-test process of the GNN Cox Model
│   ├── LogisticRegressionTrainTestManager.py       <- class handling train-test process of the logistic regression model
|
├── models                           
│   ├── BuildGraph.py                               <- class handling the building of the graph
│   ├── CoxModel.py                                 <- class handling the Cox Model architecture and its forward pass
│   ├── GATClassifier.py                            <- class handling the GAT architecture and its forward pass
│   ├── GCNClassifier.py                            <- class handling the GCN architecture and its forward pass
│   ├── LogisticRegression.py                       <- class handling the logistic regression architecture and its forward pass
|
├── results                                         <- .png files containing the results of Python scripts                           
|
├── utils                           
│   ├── featureSelection.py                         <- class handling the selection of most important features
│   ├── preProcessing.py                            <- class handling the data preprocessing
|
├── .gitignore
├── A.1-clf-logreg-sigmut.ipynb                     <- notebook example for classification with logistic regression for studying mutational signatures impact    
├── A.2-clf-logreg-sbs-indel.ipynb                  <- notebook example for classification with logistic regression for studying SBS / INDEL signatures impact  
├── A.3-clf-gcn-sigmut.ipynb                        <- notebook example for classification with GCN for studying mutational signatures impact     
├── A.4-clf-gcn-sbs-indel.ipynb                     <- notebook example for classification with GCN for studying SBS / INDEL signatures impact   
├── B.1-survival-analysis-cox-sigmut.ipynb          <- notebook example for survival analysis with Cox Model for studying mutational signatures impact    
├── B.2-survival-analysis-cox-sbs-indel.ipynb       <- notebook example for survival analysis with Cox Model for studying SBS / INDEL mutational signatures impact    
├── B.3-survival-analysis-gcn-cox-sigmut.ipynb      <- notebook example for survival analysis with GCN Cox Model for studying mutational signatures impact    
├── B.4-survival-analysis-gcn-cox-sbs-indel.ipynb   <- notebook example for survival analysis with GCN Cox Model for studying SBS / INDEL mutational signatures impact  
├── LICENSE 
├── main-classification.py                          <- main script for classification problem    
├── main-survivalAnalysis.py                        <- main script for survival analysis problem  
├── README.md
├── requirements.txt
```