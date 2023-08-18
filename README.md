# Predict immunotherapy response project
This project contains the code which predicts response to immune checkpoint inhibition treatment from biomarkers of patients suffering from cancers. The goal is to highlight the predictive property of mutational signatures in immune checkpoint inhibition treatment.

This project is in collaboration with the *MEDomics UdeS* laboratory from the Université de Sherbrooke, and a biology laboratory of Université de Sherbrooke.

The internship report is available on : https://bit.ly/3KBhwb2 

## Project summary
The goal of this project is to create a machine learning model which **predicts immune checkpoint inhibitor treatment response for patients suffering from cancers**. For that, we collected data from the following article https://doi.org/10.1158/1078-0432.CCR-20-1163, which contains 82 patients with 6 features called biomarkers (age, exome TMB, genome TMB, CD8+ T cell score, M1M2 expression, CD274 expression), and we added new features / biomarkers called mutational signatures (SBS and INDEL). We want to know if the addition of mutational signatures gives better performances. Data are available in the `data/` folder of this repository.

First, we built a **classification model** which predicts if the disease will progress by a time t. 3 models have been implemented : 
- Logistic Regression : notebooks A.1 and A.2;
- Graph Convolutional Network (GCN) : notebooks A.3 and A.4 ;
- Graph Attention Network (GAT) : notebooks A.3 and A.4 with putting 'gat' in 'architecture' variable.

Next, we built a **survival analysis model** which predicts the risk level for a patient to die and for a disease to progress, and which estimates the survival probability and the no-progression of the disease through days. 3 models have been implemented : 
- Cox Model (which is equivalent to logistic regression but in a survival analysis context) : notebooks B.1 and B.2 ;
- GCN Cox Model : notebooks B.3 and B.4 ;
- GAT Cox Model : notebooks B.3 and B.4 with putting 'gat' in 'architecture' variable.

To know if mutational signatures are predictive features, we tested the models with and without mutational signatures. 

To have more details, you can read the internship report (link above).

## Installation
First, follow these steps to make sure you have all dependencies :
1. Open a terminal
2. Clone the repository : 
    1. With HTTPS : `git clone https://github.com/MEDomics-UdeS/predict-immunotherapy-response.git`
    2. With SSH : `git clone git@github.com:MEDomics-UdeS/predict-immunotherapy-response.git`
3. Move to the root of this repository : `cd predict-immunotherapy-response`
4. Create a Python virtual environment, called venv : `python3 -m venv venv`
5. Activate the virtual environment :
    1. Linux (bash) : `source venv/bin/activate`
    2. Windows (powershell) : `venv\Scripts\activate.ps1` 
6. Install the requirements : ```pip install -r requirements.txt```

## Executions

### Classification problem

#### Python scripts

The Python file containing the whole pipeline is the `main-classification.py` file. To execute it with default arguments, you can enter this command :

```
python3 main-classification.py
```

You can also add arguments manually :
- --sigmut (default : `comb`) : to include mutational signatures or not, and how.
    - `no-sigmut` : no mutational signatures. Only the biomarkers from the reference article are included.
    - `only-sigmut-sbs` : include only SBS mutational signatures.
    - `only-sigmut-indel` : include only INDEL mutational signatures. 
    - `only-sigmut-comb` : include only SBS and INDEL mutational signatures.
    - `comb` : include biomarkers from the reference article and the mutational signatures.
- --architecture (default : `logistic-regression`) : the model architecture :
    - `logistic-regression` : the logistic regression, for binary classification problem.
    - `gcn` : the Graph Convolutional Network, for binary classification problem, or survival analysis problem.
    - `gat` : the Graph Attention Network, for binary classification problem, or survival analysis problem.
    - `cox` : the Cox Model, for survival analysis problem.
- --n_features (default : `5`) : the n_features most important features will be considered for each patient
- --n_epochs (default : `150`) : the number of epochs to train the models. The Cox Model is trained on 1 epoch.
- --lr (default : `0.005`) : the learning rate used during the Stochastic Gradient Descent. Not used in Cox Model.
- --reg (default : `0.005`) : the regularization factor used during the loss computation. Not used in Cox Model.
- --max_neighbors (only for Graph Neural Networks) (default : `2`) : the maximum of neighbors for each node of the graph.
- --threshold (default : `0.5`) : the probability used as threshold between 0 and 1 classes. Must be between 0 and 1. Greater the threshold is, less 
patients are associated to class 1.

For the **Logistic Regression**, an example of running is the following :

```
python3 main-classification.py --sigmut comb --architecture logistic-regression --n_features 5 --n_epochs 150 --lr 0.005 --reg 0.005 --threshold 0.6
```

For the **Graph Neural Network**, an example of running is the following :

```
python3 main-classification.py --sigmut no-sigmut --architecture gcn --n_features 6 --n_epochs 100 --lr 0.005 --reg 0.005 --max_neighbors 2 --threshold 0.6
```

#### Jupyter notebooks examples

To see examples of executions and results, you can read the following notebooks :
- `A.1-clf-logreg-sigmut.ipynb` : comparison between without mutational signatures, with only mutational signatures, and with combination of reference article biomarkers and mutational signatures, with the logistic regression. 
- `A.2-clf-logreg-sbs-indel.ipynb` : comparison between with only SBS signatures, with only INDEL signatures, and with combination of both, with the logistic regression. 
- `A.3-clf-gcn-sigmut.ipynb` : comparison between without mutational signatures, with only mutational signatures, and with combination of reference article biomarkers and mutational signatures, with the Graph Convolutional Network.
- `A.4-clf-gcn-sbs-indel.ipynb` : comparison between with only SBS signatures, with only INDEL signatures, and with combination of both, with the Graph Convolutional Network. 

### Survival analysis problem

#### Python scripts

The Python file containing the whole pipeline is the `main-survival-analysis.py` file. To execute it with default arguments, you can enter this command :

```
python3 main-survival-analysis.py
```

You can also add arguments manually, which are the same as classification problem. But there are 2 differences :
- you can choose `architecture` arguments between `cox` (default), `gcn` and `gat`.
- the `threshold` arguments represents the **quantile** used as threshold between low risk and high risk patients.
you can add a quantile used as threshold between low risk and high risk patients. Must be between 0 and 1 (default : 0.5). Greater the quantile is, less patients are associated to high risk class.

For the **Cox Model**, an example of running is the following :

```
python3 main-survival-analysis.py --sigmut comb --architecture cox --n_features 5 --n_epochs 150 --lr 0.005 --reg 0.005 --threshold 0.8
```

For the **Graph Neural Network Cox Model**, an example of running is the following :

```
python3 main-survival-analysis.py --sigmut no-sigmut --architecture gcn --n_features 6 --n_epochs 100 --lr 0.005 --reg 0.005 --max_neighbors 2 --threshold 0.8
```

#### Jupyter notebooks examples

To see examples of executions and results, you can read the following notebooks :
- `B.1-survival-analysis-cox-sigmut.ipynb` : comparison between without mutational signatures, with only mutational signatures, and with combination of reference article biomarkers and mutational signatures, with the Cox Model. 
- `B.2-survival-analysis-cox-sbs-indel.ipynb` : comparison between with only SBS signatures, with only INDEL signatures, and with combination of both, with the Cox Model. 
- `B.3-survival-analysis-gcn-cox-sigmut.ipynb` : comparison between without mutational signatures, with only mutational signatures, and with combination of reference article biomarkers and mutational signatures, with the Graph Convolutional Network Cox Model.
- `B.4-survival-analysis-gcn-cox-sbs-indel.ipynb` : comparison between with only SBS signatures, with only INDEL signatures, and with combination of both, with the Graph Convolutional Network Cox Model. 

### Hyper parameters optimization
If you want to optimize hyper parameters for the classification problem, you can read the `optimize-hyper-parameters` notebook. 

## Project Tree
```
├── data                                                <- datasets used by the models
│   ├── cohort-dataset.xlsx                             <- dataset from the reference article with mutational signatures
│   ├── cohort-dataset-with-logreg-output.xlsx          <- dataset from the reference article with mutational signatures and the output of the logistic regression model (class 1 probability)
|
├── notebooks                                           <- examples of model execution
│   ├── A.1-clf-logreg-sigmut.ipynb                     <- notebook example for classification with logistic regression for studying mutational signatures impact    
│   ├── A.2-clf-logreg-sbs-indel.ipynb                  <- notebook example for classification with logistic regression for studying SBS / INDEL signatures impact  
│   ├── A.3-clf-gcn-sigmut.ipynb                        <- notebook example for classification with GCN for studying mutational signatures impact     
│   ├── A.4-clf-gcn-sbs-indel.ipynb                     <- notebook example for classification with GCN for studying SBS / INDEL signatures impact   
│   ├── B.1-survival-analysis-cox-sigmut.ipynb          <- notebook example for survival analysis with Cox Model for studying mutational signatures impact    
│   ├── B.2-survival-analysis-cox-sbs-indel.ipynb       <- notebook example for survival analysis with Cox Model for studying SBS / INDEL mutational signatures impact    
│   ├── B.3-survival-analysis-gcn-cox-sigmut.ipynb      <- notebook example for survival analysis with GCN Cox Model for studying mutational signatures impact    
│   ├── B.4-survival-analysis-gcn-cox-sbs-indel.ipynb   <- notebook example for survival analysis with GCN Cox Model for studying SBS / INDEL mutational signatures impact  
|
├── results                                             <- .png files containing the results of Python scripts                           
|
├── src                                                 <- core of the models
│   ├── evaluation                                      <- evaluation methods for models
│       ├── ClassificationMetrics.py                    <- class containing evaluation metrics for binary classification
│       ├── SurvivalMetrics.py                          <- class containing evaluation metrics for survival analysis
│   ├── manage                                          <- train-test managers of models
│       ├── CoxTrainTestManager.py                      <- class handling train-test process of the Cox Model
│       ├── GNNClassifierTrainTestManager.py            <- class handling train-test process of the GNN classifier
│       ├── GNNCoxTrainTestManager.py                   <- class handling train-test process of the GNN Cox Model
│       ├── LogisticRegressionTrainTestManager.py       <- class handling train-test process of the logistic regression model
|   ├── models                                          <- prediction models
│       ├── GraphBuilder.py                             <- class handling the building of the graph
│       ├── CoxModel.py                                 <- class handling the Cox Model architecture and its forward pass
│       ├── GATClassifier.py                            <- class handling the GAT architecture and its forward pass
│       ├── GCNClassifier.py                            <- class handling the GCN architecture and its forward pass
│       ├── LogisticRegression.py                       <- class handling the logistic regression architecture and its forward pass
|   ├── utils                                           <- some functions for feature selection and preprocessing
│       ├── FeatureSelector.py                          <- class handling the selection of most important features
│       ├── PreProcessor.py                             <- class handling the data preprocessing
│       ├── FeaturesNames.py                            <- file storing names of the features used in the project
├── .gitignore
├── LICENSE 
├── main-classification.py                          <- main script for classification problem    
├── main-survivalAnalysis.py                        <- main script for survival analysis problem  
├── README.md
├── requirements.txt
```