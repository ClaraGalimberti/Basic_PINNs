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
    # create points in time (to evaluate the model ODE)
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True)
    # get points in x (to evaluate the model ODE)
    xs = model(ts)
    # get the gradient
    dT = grad(xs, ts)[0]
    # compute the ODE
    ode = dT - model.r * (model.x_bar - xs)
    # MSE of ODE
    return torch.mean(ode**2)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer_1 = torch.nn.Linear(1,5)
        self.hidden_layer_2 = torch.nn.Linear(5, 5)
        self.hidden_layer_3 = torch.nn.Linear(5, 5)
        self.output_layer = torch.nn.Linear(5,1)

        self.r = torch.nn.Parameter(0.1 * torch.ones(1))
        self.x_bar = torch.nn.Parameter(0.5 * torch.ones(1))

    def forward(self, t):
        layer_hidden_1 = self.hidden_layer_1(t)
        sigmoid_layer_hidden_1 = torch.tanh(layer_hidden_1)
        layer_hidden_2 = self.hidden_layer_2(sigmoid_layer_hidden_1)
        sigmoid_layer_hidden_2 = torch.tanh(layer_hidden_2)
        layer_hidden_3 = self.hidden_layer_3(sigmoid_layer_hidden_2)
        sigmoid_layer_hidden_3 = torch.tanh(layer_hidden_3)
        output = self.output_layer(sigmoid_layer_hidden_3)
        return output


class System(torch.nn.Module):
    def __init__(self, x_init):
        super().__init__()
        self.r = 0.005
        self.x_bar = 1.
        self.x_init = x_init

    def forward(self, ts):
        x_current = (self.x_init - self.x_bar) * torch.exp(-self.r * ts) + self.x_bar
        return x_current


# Hyperparameters:
lambda_ = 100

# Generate the dataset:
true_model = System(x_init=0)
t_data = torch.linspace(0, 300, 10).unsqueeze(1)
x_data = true_model(t_data)

# Model to train:
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_MSE = torch.nn.MSELoss()


for epoch in range(int(2e4)):
    optimizer.zero_grad()
    x = model(t_data)
    loss_1 = loss_MSE(x, x_data)
    loss_2 = physics_loss(model)
    loss = loss_1 + lambda_ * loss_2
    if epoch % 1000 == 0:
        print("Epoch: %i ---||--- Loss MSE: %.4f --- Loss Physics: %.4f ---||--- r = %.4f --- xbar = %.2f"
              % (epoch, 1e6*loss_1, 1e6*loss_2, model.r, model.x_bar))
    loss.backward()
    optimizer.step()

# Plot after training:
t_data_ext = torch.linspace(0, 1000, 50).unsqueeze(1)
x_data_ext = true_model(t_data_ext)
x_ext = model(t_data_ext)
# Loss after training:
loss_1 = loss_MSE(x, x_data)
loss_2 = physics_loss(model)
loss = loss_1 + lambda_ * loss_2
print("Epoch: %i ---||--- Loss MSE: %.4f --- Loss Physics: %.4f ---||--- r = %.4f --- xbar = %.2f"
      % (epoch, 1e6*loss_1, 1e6*loss_2, model.r, model.x_bar))

plt.scatter(t_data, x_data, label="Data")
plt.plot(t_data_ext, x_data_ext, label="True model")
plt.plot(t_data_ext, x_ext.detach(), label="Trained model")
ax = plt.gca()
ax.text(0.95, 0.02, r'$k_1 = %.5f, \,\, \bar{\alpha} = %.4f$' % (model.r, model.x_bar),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes)
plt.legend()
plt.savefig("plot.pdf", format="pdf")
plt.show()
