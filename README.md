# A basic example of a Physic-Informed Neural Network (PINN)

The system is governed by the following ODE:

$`\frac{d\alpha}{dt} = k_1 (1-\alpha)`$.

The script collects data and trains a NN by minimizing:

$`\mathcal{L}(\theta) = \mathcal{L}_{data}(\theta) + \lambda \mathcal{L}_{ODE}(\theta)`$.

By modifying the hyperparameter $`\lambda`$, one can choose the trade of between 
fitting the data and satisfying the ODE. 