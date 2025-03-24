import torch
import matplotlib.pyplot as plt


# Physically consistent NN
# Di Natale, L., Svetozarevic, B., Heer, P., & Jones, C. N. (2023).
# Towards scalable physically consistent neural networks:
# An application to data-driven multi-zone thermal building models.
# Applied Energy, 340, 121071.

class Net(torch.nn.Module):
    def __init__(self, x_bar_trainable=False):
        super(Net, self).__init__()
        self.hidden_layer_1 = torch.nn.Linear(1,5)
        self.hidden_layer_2 = torch.nn.Linear(5, 5)
        self.hidden_layer_3 = torch.nn.Linear(5, 5)
        self.output_layer = torch.nn.Linear(5,1)

        self.r = torch.nn.Parameter(0.1 * torch.ones(1))
        if x_bar_trainable:
            self.x_bar = torch.nn.Parameter(0.5 * torch.ones(1))
        else:
            self.x_bar = torch.ones(1)
        self.x_ini = torch.zeros(1)

    def forward(self, t):
        layer_hidden_1 = self.hidden_layer_1(t)
        sigmoid_layer_hidden_1 = torch.tanh(layer_hidden_1)
        layer_hidden_2 = self.hidden_layer_2(sigmoid_layer_hidden_1)
        sigmoid_layer_hidden_2 = torch.tanh(layer_hidden_2)
        layer_hidden_3 = self.hidden_layer_3(sigmoid_layer_hidden_2)
        sigmoid_layer_hidden_3 = torch.tanh(layer_hidden_3)
        output_1 = self.output_layer(sigmoid_layer_hidden_3)

        output_2 = self.x_bar + (self.x_ini - self.x_bar) * torch.exp(-self.r*t)

        return output_1 + output_2


class System(torch.nn.Module):
    def __init__(self, x_init, r=None, x_bar=None):
        super().__init__()
        self.x_init = x_init
        if r is None:
            self.r = 0.005
        else:
            self.r = r
        if x_bar is None:
            self.x_bar = 1.
        else:
            self.x_bar = x_bar

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_MSE = torch.nn.MSELoss()


for epoch in range(int(2e4)):
    optimizer.zero_grad()
    x = model(t_data)
    loss_1 = loss_MSE(x, x_data)
    loss = loss_1
    if epoch % 1000 == 0:
        print("Epoch: %i ---||--- Loss MSE: %.4f ---||--- r = %.4f --- xbar = %.2f"
              % (epoch, 1e6*loss_1, model.r, model.x_bar))
    loss.backward()
    optimizer.step()

# Plot after training:
t_data_ext = torch.linspace(0, 1000, 50).unsqueeze(1)
x_data_ext = true_model(t_data_ext)
if model.x_bar.requires_grad:
    learnt_model = System(x_init=0, r=model.r, x_bar=model.x_bar)
else:
    learnt_model = System(x_init=0, r=model.r)
x_ode = learnt_model(t_data_ext)
x_ext = model(t_data_ext)
# Loss after training:
loss_1 = loss_MSE(x, x_data)
loss = loss_1
print("Epoch: %i ---||--- Loss MSE: %.4f ---||--- r = %.4f --- xbar = %.2f"
      % (epoch, 1e6*loss_1, model.r, model.x_bar))

plt.scatter(t_data, x_data, label="Data")
plt.plot(t_data_ext, x_data_ext, label="True model")
plt.plot(t_data_ext, x_ext.detach(), label=r"$\alpha_{NN}(t)$")
plt.plot(t_data_ext, x_ode.detach(), label=r"$\alpha_{ODE}(t, k_1)$")
ax = plt.gca()
if model.x_bar.requires_grad:
    ax.text(0.95, 0.02, r'$k_1 = %.5f, \,\, \bar{\alpha} = %.4f$' % (model.r, model.x_bar),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes)
else:
    ax.text(0.95, 0.02, r'$k_1 = %.5f$' % (model.r),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes)
plt.legend()
plt.savefig("plot.pdf", format="pdf")
plt.show()
