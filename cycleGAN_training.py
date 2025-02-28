import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

# -------------------------
#  Network Components
# -------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial Convolution Block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        # Downsampling
        in_features = 64
        for _ in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        # Residual Blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        # Upsampling
        for _ in range(2):
            out_features = in_features // 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        # Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
#  Dataset
# -------------------------

class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.files_A = sorted(os.listdir(root_A))
        self.files_B = sorted(os.listdir(root_B))
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        img_A = Image.open(os.path.join(self.root_A, self.files_A[index % len(self.files_A)])).convert("RGB")
        img_B = Image.open(os.path.join(self.root_B, self.files_B[index % len(self.files_B)])).convert("RGB")
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return img_A, img_B

# -------------------------
#  Training Loop
# -------------------------

def train_cycle_gan(dataset_A, dataset_B, num_epochs=300, batch_size=1, lr=0.0002):
    # dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    # dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize networks
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # -----------------------------
    #  Define the LR schedulers
    # -----------------------------
    # Keep LR constant for first half (125 epochs), then linearly decay to 0 for second half.
    def lambda_rule(epoch):
        start_decay_epoch = 125
        total_decay_epochs = 125  # from epoch 125 to 249 (inclusive)
        if epoch < start_decay_epoch:
            return 1.0
        else:
            # e.g. epoch=125 => factor=1, epoch=249 => factor~0
            return 1.0 - float(epoch - start_decay_epoch) / float(total_decay_epochs)

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
    scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)

    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    
    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(dataloader):
        #for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Adversarial ground truths (size may vary based on image dimensions and architecture)
            # valid = torch.ones(real_A.size(0), 1, 30, 30).to(device)
            # fake = torch.zeros(real_A.size(0), 1, 30, 30).to(device)
            valid = torch.ones(real_A.size(0), 1, 15, 15).to(device)
            fake  = torch.zeros(real_A.size(0), 1, 15, 15).to(device)
            
            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()
            
            # Identity loss (encourages preservation of color/content)
            loss_id_A = criterion_cycle(G_BA(real_A), real_A)
            loss_id_B = criterion_cycle(G_AB(real_B), real_B)
            
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            
            # Cycle consistency loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            
            loss_G = loss_GAN_AB + loss_GAN_BA + 10*(loss_cycle_A + loss_cycle_B) + 5*(loss_id_A + loss_id_B)
            loss_G.backward()
            optimizer_G.step()
            
            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            loss_real_A = criterion_GAN(D_A(real_A), valid)
            loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = 0.5 * (loss_real_A + loss_fake_A)
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            loss_real_B = criterion_GAN(D_B(real_B), valid)
            loss_fake_B = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = 0.5 * (loss_real_B + loss_fake_B)
            loss_D_B.backward()
            optimizer_D_B.step()
            
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}] "
                  f"[D_A: {loss_D_A.item():.4f}] [D_B: {loss_D_B.item():.4f}] "
                  f"[G: {loss_G.item():.4f}]")
        
        # -----------------------
        #  Step the Schedulers
        # -----------------------
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
        
        # Optionally, save checkpoints every few epochs:
        if (epoch + 1) % 25 == 0:
            torch.save(G_AB.state_dict(), f'G_AB_epoch_{epoch+1}.pth')
            torch.save(G_BA.state_dict(), f'G_BA_epoch_{epoch+1}.pth')

# -------------------------
#  Example Usage
# -------------------------

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = ImageDataset(root_A="dataset/typed", root_B="dataset/handwritten", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


    
    train_cycle_gan(dataset, dataset, num_epochs=300, batch_size=1)