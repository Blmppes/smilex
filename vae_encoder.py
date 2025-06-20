import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE(nn.Module):
    def __init__(self, input_dim=243, hidden_dim=50, hidden2_dim=12, latent_dim=32):
        super(VAE, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.lin_bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden2_dim)
        self.lin_bn2 = nn.BatchNorm1d(hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, hidden2_dim)
        self.lin_bn3 = nn.BatchNorm1d(hidden2_dim)

        self.fc1 = nn.Linear(hidden2_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, hidden2_dim)
        self.fc_bn4 = nn.BatchNorm1d(hidden2_dim)

        self.linear4 = nn.Linear(hidden2_dim, hidden2_dim)
        self.lin_bn4 = nn.BatchNorm1d(hidden2_dim)
        self.linear5 = nn.Linear(hidden2_dim, hidden_dim)
        self.lin_bn5 = nn.BatchNorm1d(hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, input_dim)
        self.lin_bn6 = nn.BatchNorm1d(input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.lin_bn1(self.linear1(x)))
        h = self.relu(self.lin_bn2(self.linear2(h)))
        h = self.relu(self.lin_bn3(self.linear3(h)))
        h = self.relu(self.bn1(self.fc1(h)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        return mu

    def decode(self, z):
        h = self.relu(self.fc_bn3(self.fc3(z)))
        h = self.relu(self.fc_bn4(self.fc4(h)))
        h = self.relu(self.lin_bn4(self.linear4(h)))
        h = self.relu(self.lin_bn5(self.linear5(h)))
        return self.lin_bn6(self.linear6(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE_Loss(nn.Module):
    def __init__(self):
        super(VAE_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, recon_x, x, mu, logvar):
        mse = self.mse(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld


def weights_init(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def run_vae_encoding(input_csv, output_file, latent_dim=32, epochs=1000):
    print(f"Reading {input_csv}...")
    df = np.loadtxt(input_csv, delimiter=",", skiprows=1)  # assumes CSV input

    tensor_data = torch.tensor(df, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    input_dim = df.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.apply(weights_init)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = VAE_Loss()

    print("Training VAE...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch in loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = criterion(recon, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} / {epochs} - Loss: {train_loss / len(loader.dataset):.4f}")

    print("Extracting latent representation...")
    model.eval()
    latent_list = []
    with torch.no_grad():
        for batch in loader:
            data = batch[0].to(device)
            _, mu, _ = model(data)
            latent_list.append(mu.cpu())

    latent_vectors = torch.cat(latent_list, dim=0).numpy()
    np.save(output_file, latent_vectors)
    print(f"Saved encoded features to {output_file}")
