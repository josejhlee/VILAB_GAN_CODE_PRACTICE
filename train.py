import torch
import torch.nn as nn
import torch.optim as optim
# from args import *
from model import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import Discriminator, Generator, inialize_weights


transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize()
    ])

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(args.noise_dim, args.channel_n, FEATURES_GEN).to(device)
disc = Discriminator(args.channel_n, FEATURES_DISC).to(device)

dataset = datasets.MNIST(root="data_directory", train=True, transform=transforms,
                       download=True)

opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
loss = nn.BCELoss()

gen.train()
disc.train()

for epoch in range(args.epochs):
    for batch_idx, (real,_) in enumerate(dataloader):
        noise = torch.randn(args.batch_size, args.noise_dim,1,1).to
        fake = gen(noise)

        #Discriminator training
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()
        
        #Generator training
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()