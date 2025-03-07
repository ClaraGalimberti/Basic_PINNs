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
    # make collocation points
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True)
    # run the collocation points through the network
    temps = model(ts)
    # get the gradient
    dT = grad(temps, ts)[0]
    # compute the ODE
    ode = dT - model.r * (model.Tenv - temps)
    # MSE of ODE
    return torch.mean(ode**2)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = torch.nn.Linear(2,5)
        self.output_layer = torch.nn.Linear(5,1)

    def forward(self, t):
        layer_out = torch.sigmoid(self.hidden_layer(t))
        output = self.output_layer(layer_out)
        return output


class SystemTemperature(torch.nn.Module):
    def __init__(self, temp_init):
        super().__init__()
        self.r = 0.005
        self.temp_env = 25
        self.temp_init = temp_init

    def forward(self, ts):
        temp_current = (self.temp_init - self.temp_env) * torch.exp(-self.r * ts) + self.temp_env
        return temp_current


# Generate the dataset:
true_model = SystemTemperature(temp_init=100)
t_data = torch.linspace(0, 300, 10)
temp_data = true_model(t_data)
plt.scatter(t_data, temp_data)
plt.show()

#