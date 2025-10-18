# CO Concentration Prediction using a Neural Network in MatLab

This project is a practical implementation of a shallow neural network from scratch in MatLab to solve a regression problem. The goal is to predict Carbon Monoxide (CO) concentrations based on other atmospheric sensor data.

This was developed as a university project for the "Optimizations" course at Politehnica University Bucharest.

## Project Objective

The core objective was to implement and compare two fundamental gradient-based optimization algorithms:
1.  **Batch Gradient Descent**
2.  **Stochastic Gradient Descent (SGD)**

The comparison focuses on both model accuracy (using $R^2$ and MSE metrics) and training performance (e.g., time to convergence).

## Dataset

* **Source:** "Air Quality" Dataset from UC Irvine Machine Learning Repository.
* **Description:** The dataset contains ~9,000 hourly sensor readings from a field in an Italian city.
* **Target Variable:** `CO(GT)` - Carbon Monoxide concentration.
* **Features:** Other sensor readings for benzene ($C_6H_6$), $NO_2$, ozone ($O_3$), etc..

## Methodology

### 1. Data Preprocessing

Before training, the data underwent several key preprocessing steps:
* **Data Cleaning:** Samples with missing target variable (`CO(GT) == -200`) were removed.
* **Train/Test Split:** The data was split into an 80% training set and a 20% test set.
* **Missing Value Imputation:** Missing feature values (marked as -200) were replaced with the mean of that feature *from the training set only*.
* **Standardization:** All features were standardized (scaled to mean 0 and standard deviation 1) based on the training set's statistics. The same transformation was applied to the test set.
* **Bias Term:** A bias column (vector of ones) was added to the data.

### 2. Model Architecture

A shallow neural network with a single hidden layer was implemented.
* **Hidden Layer:** 20 neurons.
* **Activation Function:** A custom cubic activation function, $g(z) = z^3$ (with $a=1$ fixed).
* **Loss Function:** The model was trained to minimize the Mean Squared Error (MSE), $L(e,y)=\frac{1}{2N}||e-y||^{2}$.

### 3. Optimization Algorithms

Both algorithms were implemented from scratch to find the optimal network weights (X and x).

* **Batch Gradient Descent:** The gradient was calculated using the *entire* training dataset in each iteration. A constant learning rate of $\alpha = 10^{-3}$ was used.
* **Stochastic Gradient Descent (SGD):** The gradient was calculated using a small, randomly selected mini-batch of 10 samples in each iteration. The same learning rate of $\alpha = 10^{-3}$ was used.

## Results & Analysis

The model's performance was evaluated on the unseen test set, yielding the following results:

| Optimization Method | $R^2$ (Test Set) | MSE (Test Set) | Training Time (100k Iterations) |
| :--- | :--- | :--- | :--- |
| **Batch Gradient Descent** | 0.9616 | 0.051 | ~15 seconds |
| **Stochastic Gradient Descent** | 0.9638 | 0.0481 | ~3 seconds |

### Key Takeaways

1.  **Accuracy:** Both methods achieved high and very similar predictive accuracy. SGD performed slightly better ($R^2 = 0.9638$ vs 0.9616).
2.  **Performance:** The most significant difference was in training speed. **SGD was approximately 5 times faster** than the batch method, achieving superior results in a fraction of the time.
3.  **Convergence:** As expected, the convergence plots for Batch Gradient Descent show a smooth decrease in the loss function. In contrast, the SGD plots are more chaotic due to the mini-batch sampling, yet still trend towards a strong minimum.

## How to Run

The MatLab script is divided into runnable sections.

1.  **Load Data:** Run the first section (`CITIREA DATELOR SI INITIALIZARI`) to load, clean, and preprocess the dataset.
2.  **Train Model:** You can then run *either*:
    * The `METODA GRADIENT` section to train using Batch Gradient Descent.
    * The `METODA GRADIENT STOCHASTIC` section to train using Stochastic Gradient Descent.

Both training sections will automatically run predictions on the test set and output the final $R^2$ and MSE metrics to the console.
