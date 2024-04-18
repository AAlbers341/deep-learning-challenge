# Alphabet Soup Predictor Model

## Overview

For this model, a nonprofit foundation called Alphabet Soup needs a tool to help select applicants for funding with the best chance of success in their ventures. This model uses machine learning and neural network techniques to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Dependencies

1. `sklearn.model_selection`
   - `train_test_split`
2. `sklearn.preprocessing`
   - `StandardScaler`
3. `Pandas`
4. `tensorflow`
5. `matplotlib`

## Data Preprocessing

In the first preprocessing step, we binned the 'APPLICATION_TYPE' and 'CLASSIFICATION' features. Binning these features was necessary to address the issue of high cardinality, where there were numerous unique values for each feature. By binning these features, we reduced the number of unique values, which helped to simplify the model complexity and prevent overfitting.

Another preprocessing step we performed involved converting categorical data to numeric format using the pd.get_dummies function. This transformation was necessary because most machine learning algorithms require numeric input data. By converting categorical variables into a set of binary variables (dummy variables), we preserved the categorical information while making it compatible with the numerical requirements of the model.


### Targets

Target: `IS_SUCCESSFUL`

### Features

Features: `['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE','ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS','ASK_AMT']`

### Dropped Features

Dropped Features: `['EIN', 'NAME']`

## Compiling, Training, and Evaluating the Model

### Model 1
![Model 1 Accuracy Plot](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Plots/model1_plot.png)
![Model 1 Evaluation](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Evaluations/model1_evaluation.PNG)

[Description or analysis of Model 1]

### Model 2
![Model 2 Accuracy Plot](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Plots/model2_plot.png)
![Model 2 Evaluation](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Evaluations/model2_evaluation.PNG)

[Description or analysis of Model 2]

### Model 3
![Model 3 Accuracy Plot](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Plots/model3_plot.png)
![Model 3 Evaluation](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Evaluations/model3_evaluation.PNG)

[Description or analysis of Model 3]

### Model 4
![Model 4 Accuracy Plot](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Plots/model4_plot.png)
![Model 4 Evaluation](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Evaluations/model4_evaluation.PNG)

[Description or analysis of Model 4]

### Model 5
![Model 5 Accuracy Plot](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Plots/model5_plot.png)
![Model 5 Evaluation](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Evaluations/model5_evaluation.PNG)

[Description or analysis of Model 5]

## Conclusion

[Summary of findings and conclusion]
