import torch
import matplotlib.pyplot as plt


def grad(outputs, inputs):
    """Computes the partial derivative of
    an output with respect to an input."""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )

def physics_loss(model: torch.nn.Module):
    """The physics loss of the model"""
    r = 0.005
    temp_env = .25
    # create points in time (to evaluate the model ODE)
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True)
    # get points in temperature (to evaluate the model ODE)
    temps = model(ts)
    # get the gradient
    dT = grad(temps, ts)[0]
    # compute the ODE
    ode = dT - r * (temp_env - temps)
    # MSE of ODE
    return torch.mean(ode**2)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer_1 = torch.nn.Linear(1,5)
        self.hidden_layer_2 = torch.nn.Linear(5, 20)
        self.hidden_layer_3 = torch.nn.Linear(20, 5)
        self.output_layer = torch.nn.Linear(5,1)

    def forward(self, t):
        layer_hidden_1 = self.hidden_layer_1(t)
        sigmoid_layer_hidden_1 = torch.relu(layer_hidden_1)
        layer_hidden_2 = self.hidden_layer_2(sigmoid_layer_hidden_1)
        sigmoid_layer_hidden_2 = torch.relu(layer_hidden_2)
        layer_hidden_3 = self.hidden_layer_3(sigmoid_layer_hidden_2)
        sigmoid_layer_hidden_3 = torch.relu(layer_hidden_3)
        output = self.output_layer(sigmoid_layer_hidden_3)
        return output


class SystemTemperature(torch.nn.Module):
    def __init__(self, temp_init):
        super().__init__()
        self.r = 0.005
        self.temp_env = .25
        self.temp_init = temp_init

    def forward(self, ts):
        temp_current = (self.temp_init - self.temp_env) * torch.exp(-self.r * ts) + self.temp_env
        return temp_current


# Hyperparameters:
alpha = 1000

# Generate the dataset:
true_model = SystemTemperature(temp_init=1)
t_data = torch.linspace(0, 300, 10).unsqueeze(1)
temp_data = true_model(t_data)

# Model to train:
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_MSE = torch.nn.MSELoss()


for epoch in range(int(1e5)):
    optimizer.zero_grad()
    temp = model(t_data)
    loss_1 = loss_MSE(temp, temp_data)
    loss_2 = physics_loss(model)
    loss = loss_1 + alpha * loss_2
    if epoch % 1000 == 0:
        print("Epoch: %i ---||--- Loss MSE: %.4f --- Loss Physics: %.4f" % (epoch, 1e6*loss_1, 1e6*loss_2))
    loss.backward()
    optimizer.step()

# Plot after training:
t_data_ext = torch.linspace(0, 1000, 50).unsqueeze(1)
temp_data_ext = true_model(t_data_ext)
temp_ext = model(t_data_ext)

plt.scatter(t_data, temp_data, label="Data")
plt.plot(t_data_ext, temp_data_ext, label="True model")
plt.plot(t_data_ext, temp_ext.detach(), label="Trained model")
plt.legend()
plt.show()
