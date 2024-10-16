import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data_processing
import data_per
import runwalk


# LSTM-VAE模型参数
timesteps = 1  # 时间步
features = 8  # 特征数量
latent_dim = 2  # 潜在空间维度
hidden_dim_1 = 64  # LSTM第一层隐藏单元
hidden_dim_2 = 32  # LSTM第二层隐藏单元


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, timesteps, features, hidden_dim_1, hidden_dim_2, latent_dim):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(features, hidden_dim_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.z_mean = nn.Linear(hidden_dim_2, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim_2, latent_dim)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, (h_n, _) = self.lstm2(x)
        z_mean = self.z_mean(h_n.squeeze(0))
        z_log_var = self.z_log_var(h_n.squeeze(0))
        return z_mean, z_log_var


# 重参数化技巧
def reparameterize(z_mean, z_log_var):
    std = torch.exp(0.5 * z_log_var)
    epsilon = torch.randn_like(std)
    return z_mean + epsilon * std


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, timesteps, features, hidden_dim_1, hidden_dim_2, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim_2)
        self.lstm1 = nn.LSTM(hidden_dim_2, hidden_dim_2, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim_2, hidden_dim_1, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim_1, features)

    def forward(self, z):
        z = self.fc(z).unsqueeze(1).repeat(1, timesteps, 1)  # Repeat vector
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        out = self.fc_out(z)
        return out


# 定义完整的VAE模型
class LSTMVAE(nn.Module):
    def __init__(self, timesteps, features, hidden_dim_1, hidden_dim_2, latent_dim):
        super(LSTMVAE, self).__init__()
        self.encoder = Encoder(timesteps, features, hidden_dim_1, hidden_dim_2, latent_dim)
        self.decoder = Decoder(timesteps, features, hidden_dim_1, hidden_dim_2, latent_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = reparameterize(z_mean, z_log_var)
        recon_x = self.decoder(z)
        return recon_x, z_mean, z_log_var


# 定义损失函数
def vae_loss(recon_x, x, z_mean, z_log_var, beta=1):
    reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return reconstruction_loss + beta * kl_loss


# 训练步骤
def train_vae(model, data_loader, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        train_loss = 0
        for x_batch in data_loader:
            optimizer.zero_grad()
            recon_x, z_mean, z_log_var = model(x_batch)
            loss = vae_loss(recon_x, x_batch, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(data_loader.dataset)}')


# 假设xcode是处理后的数据，形状为 (num_samples, timesteps, features)
xcode_tensor = torch.tensor(runwalk.X_processed, dtype=torch.float32)

# 数据加载器
batch_size = 32
train_loader = torch.utils.data.DataLoader(xcode_tensor, batch_size=batch_size, shuffle=True)

# 初始化模型
vae_model = LSTMVAE(timesteps, features, hidden_dim_1, hidden_dim_2, latent_dim)

# 训练模型
train_vae(vae_model, train_loader)


# 生成新数据
def generate_data(model, num_samples):
    model.eval()
    with torch.no_grad():
        z_samples = torch.randn(num_samples, latent_dim)
        generated_data = model.decoder(z_samples)
    return generated_data


# 生成10个样本
new_health_data = generate_data(vae_model, 10)
print(new_health_data)
