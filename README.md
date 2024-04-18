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

Number of neurons: 43

Layer 1:
- Number of nodes: 10
- Activation Function: relu

Layer 2:
- Number of nodes: 5
- Activation Function: relu

Outer Layer:
- Number of nodes: 1
- Activation Function: Sigmoid


The model achieved a loss of approximately **0.554** and an accuracy of approximately **72.92%**. Although the model's performance did not achieve the 75%, the model performed reasonably well for the first iteration keeping in mind to keep things simple and build upon future iterations.

### Model 2
![Model 2 Accuracy Plot](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Plots/model2_plot.png)
![Model 2 Evaluation](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Evaluations/model2_evaluation.PNG)

Number of neurons: 43

Layer 1:
- Number of nodes: 60
- Activation Function: tanh

Outer Layer:
- Number of nodes: 1
- Activation Function: Sigmoid

The model achieved a loss of approximately **0.554** and an accuracy of approximately **72.91%**. The 2nd model did not achieve the 75%, and the change of performance from Model 1 is negligible. Alterations to Model 2, was removing the 2nd hidden layer and changing the activation function on the 1st hidden layer to 'tanh'. This was more or less to see what happens with our accruacy score as an experiment. 

### Model 3
![Model 3 Accuracy Plot](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Plots/model3_plot.png)
![Model 3 Evaluation](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Evaluations/model3_evaluation.PNG)

Number of neurons: 43

Layer 1:
- Number of nodes: 20
- Activation Function: tanh

Layer 2:
- Number of nodes: 10
- Activation Function: relu

Layer 3:
- Number of nodes: 8
- Activation Function: relu

Outer Layer:
- Number of nodes: 1
- Activation Function: Sigmoid

The model achieved a loss of approximately **0.553** and an accuracy of approximately **72.75%**. The 3rd model did not achieve the 75%, and the change from Model 1 and Model 2 was small, but increased. Alterations to Model 3, was adding a 3rd layer to increase the model's capacity to learn more complex patterns and relations in the data. 

### Model 4
![Model 4 Accuracy Plot](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Plots/model4_plot.png)
![Model 4 Evaluation](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Evaluations/model4_evaluation.PNG)

Number of neurons: 43

Layer 1:
- Number of nodes: 20
- Activation Function: relu

Layer 2:
- Number of nodes: 10
- Activation Function: relu
- Dropout: 0.25

Outer Layer:
- Number of nodes: 1
- Activation Function: Sigmoid

The model achieved a loss of approximately **0.552** and an accuracy of approximately **73.06%**. The 4th model did not achieve the 75%, and the change from Model 1, Model 2 and Model 3 increased by 1%. Alterations to Model 4, was adding a 'Dropout' for the 2nd hidden layer. Dopouts are used to prevent overfitting and improve generalization performance. When the data is being trained, dropout will randomly drop a fraction of the neurons in the networks. This seemed to have improved the model marginally. 

### Model 5
![Model 5 Accuracy Plot](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Plots/model5_plot.png)
![Model 5 Evaluation](https://github.com/AAlbers341/deep-learning-challenge/blob/main/Visuals/Evaluations/model5_evaluation.PNG)

Number of neurons: 43

Layer 1:
- Number of nodes: 50
- Activation Function: relu

Outer Layer:
- Number of nodes: 1
- Activation Function: Sigmoid

The model achieved a loss of approximately **0.553** and an accuracy of approximately **73.08%**. The 5th and final model did not achieve the 75%, and the change from Model's 1, 2, 3, and 4 was meant with a slight increase in accuracy. Alterations to Model 5, was removing the 2nd hidden layer and adding the 'Dropout' for the 1st hidden layer as it was meant with success in Model 4. The improvement was marginal, but measureable. 

## Conclusion

Overall, the series of deep learning models have shown a consistent performance in terms of loss and accuracy, with incremental improvements observed in later iterations. However, none of the models were able to meet the target accuracy of 75%. Given the observations from the series of models, a different approach using a more complex architecture or a different type of neural network model could potentially solve this classification problem more effectively such as Convolutional Neural Network (CNN).
