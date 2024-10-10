# Boston Housing Regression

## Objective
The objective of this project is to develop a neural network model for predicting housing prices based on various features. The model will be implemented using PyTorch and will undergo training and validation processes to evaluate its performance.

## Dataset Description
The dataset used in this project is a CSV file containing various features related to housing properties. Key features include:

- **Area**: The total area of the house.
- **Furnishing Status**: The status of the furnishing (e.g., furnished, semi-furnished).
- **Main Road**: Indicates whether the property is located on a main road.
- **Price**: The target variable representing the price of the house.



## Steps to Run the Code in Colab
1. **Upload the Dataset**:
   - Upload the dataset CSV file to the Colab environment.

2. **Clone or Create a New Notebook**:
   - Create a new notebook in Google Colab or clone this repository.

3. **Install Dependencies**:
   - Ensure that the necessary libraries are installed.
     

4. **Import the Libraries**:
   - Import the required libraries at the beginning of your notebook:
     ```python
      import pandas as pd
      import torch
      import torch.nn as nn
      from sklearn.preprocessing import MinMaxScaler
      from torchsummary import summary
      from sklearn.model_selection import train_test_split
      import torch.nn.functional as F
      import torch.optim as optim
      from torch.utils.data import TensorDataset, DataLoader
      import numpy as np
      from tqdm import tqdm
      from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
      import matplotlib.pyplot as plt
      import seaborn as sns
     ```

5. **Load the Dataset**:
   - Load the dataset using pandas:
     ```python
     df = pd.read_csv('path/to/your/dataset.csv')
     ```

6. **Preprocess the Data**:
   - Encode categorical features. ex: Main Road

7. **Define the Model**:
   - Create a neural network model using PyTorch.

8. **Set Up Training**:
   - Define the loss function and optimizer, and set the number of epochs.

9. **Train the Model**:
   - Implement the training loop with training and validation phases.

10. **Evaluate the Model**:
   - Assess the model's performance using metrics like MSE and MAE.

11. **Visualize Predictions**:
   - Plot the predicted vs. actual prices to visualize the model performance.

## Plot after 200 epoch:
![Boston Housing Plot](https://github.com/user-attachments/assets/28dc2c76-c5bf-4cf1-9769-80a842cd6d5e)


