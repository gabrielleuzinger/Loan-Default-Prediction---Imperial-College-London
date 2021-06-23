This repository contains a **Model to predict if a loan default will occur** based on the dataset 'Loan Default Prediction - Imperial College London' available at [Kaggle](https://www.kaggle.com/c/loan-default-prediction/overview). The original kaggle competition, participants must predict loan default and the respective loss. In this notebook, we are only interested in predicting if a loan default will occur. This is why we do not use the test set from kaggle and work exclusively with the train set. Thus, the test set is created from the train set. This project was set up following the premises of [Reproducible Research](https://pt.coursera.org/learn/reproducible-research), so that anyone can achieve the same results as me using the steps I followed in Jupyter Notebook.

# Project libraries


All dependencies can be found in the file  `requirements.txt`, but are also listed below:
* Numpy
* Scikit-Learn
* Imbalanced-Learn
* Pandas
* Jupyter Notebook
* Matplotlib

To install the dependencies run in the project's main folder : `pip install -r requirements.txt`. 

To access the Jupyter Notebook that I created, run in the root folder of the `jupyter notebook` project. Soon after, your browser will open and just select the file `Algerian Forest Fires DataSet.ipynb`.  

# Project structure

```{sh}
  .
  |-report
  |  |- markdown
  |  |  |- Loan Default Prediction - Imperial College London.md
  |-data
  |- Loan Default Prediction - Imperial College London.ipynb
  |- requirements.txt
```

The folder `report` contains an md file with a version of the report generated from the study done on that project. This file contains **all the insights and studies that were done, as well as a detailed description of how the project was developed**.

**All references used to create this project are described in the report**.
