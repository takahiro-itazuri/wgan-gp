import os
import sys

import torch
from torch import nn, optim
from torch.autograd import grad

def initialize_weights(model):
	for m in model.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.02)
			m.bias.data.zero_()


class LayerNorm(nn.Module):
	def __init__(self, num_features, eps=1e-5, affine=True):
		super(LayerNorm, self).__init__()
		self.eps = eps
		self.affine = affine

		if self.affine:
			self.gamma = nn.Parameter(torch.ones(num_features))
			self.beta = nn.Parameter(torch.zeros(num_features))
	
	def forward(self, x):
		shape = [-1] + [1] * (x.dim() - 1)
		mean = x.view(x.size(0), -1).mean(1).view(*shape)
		std = x.view(x.size(0), -1).std(1).view(*shape)
		x = (x - mean) / (std + self.eps)
		
		if self.affine:
			shape = [1, -1] + [1] * (x.dim() - 2)
			x = self.gamma.view(*shape) * x + self.beta.view(*shape)

		return x


class DownConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
		super(DownConv, self).__init__()
		self.conv = nn.Sequential(
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		)

	def forward(self, x):
		return self.conv(x)


class ConvDown(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
		super(ConvDown, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
			nn.AvgPool2d(kernel_size=2)
		)

	def forward(self, x):
		return self.conv(x)


class UpConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
		super(UpConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		)
	
	def forward(self, x):
		return self.conv(x)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, NormLayer, resample=None):
		super(ResidualBlock, self).__init__()
		self.need_shortcut_conv = (in_channels != out_channels) or (resample is not None)

		if resample == 'down':
			self.conv = nn.Sequential(
				NormLayer(in_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
				NormLayer(out_channels),
				nn.ReLU(inplace=True),
				ConvDown(out_channels, out_channels, kernel_size=3, padding=1)
			)
			self.shortcut_conv = ConvDown(in_channels, out_channels, kernel_size=3, padding=1)
		elif resample == 'up':
			self.conv = nn.Sequential(
				NormLayer(in_channels),
				nn.ReLU(inplace=True),
				UpConv(in_channels, out_channels, kernel_size=3, padding=1),
				NormLayer(out_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
			)
			self.shortcut_conv = UpConv(in_channels, out_channels, kernel_size=3, padding=1)
		elif resample is None:
			self.conv = nn.Sequential(
				NormLayer(out_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
				NormLayer(out_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			)
			if self.need_shortcut_conv:
				self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

	def forward(self, x):
		if self.need_shortcut_conv:
			shortcut = self.shortcut_conv(x)
		else:
			shortcut = x

		x = self.conv(x)
		x += shortcut
		return x


class OptimizedBlock(nn.Module):
	def __init__(self, in_channels, out_channels, NormLayer=LayerNorm):
		super(OptimizedBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			LayerNorm(out_channels),
			nn.ReLU(inplace=True),
			ConvDown(out_channels, out_channels)
		)
		self.shortcut = DownConv(in_channels, out_channels)
	
	def forward(self, x):
		shortcut = self.shortcut(x)
		x = self.conv(x)
		x += shortcut
		return x


class Generator(nn.Module):
	def __init__(self, nz=128, nc=3, ngf=128):
		super(Generator, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ngf = ngf

		self.fc1 = nn.Linear(nz, 4 * 4 * ngf)
		self.block2 = ResidualBlock(ngf, ngf, nn.BatchNorm2d, 'up')
		self.block3 = ResidualBlock(ngf, ngf, nn.BatchNorm2d, 'up')
		self.block4 = ResidualBlock(ngf, ngf, nn.BatchNorm2d, 'up')
		self.block5 = nn.Sequential(
			nn.BatchNorm2d(ngf),
			nn.ReLU(inplace=True),
			nn.Conv2d(ngf, nc, kernel_size=3, padding=1),
			nn.Tanh()
		)

	def forward(self, z):
		x = self.fc1(z)
		x = x.view(-1, self.ngf, 4, 4)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		return x


class Discriminator(nn.Module):
	def __init__(self, nz=128, nc=3, ndf=128):
		super(Discriminator, self).__init__()
		self.nz = nz
		self.nc = nc
		self.ndf = ndf
		
		self.block1 = OptimizedBlock(nc, ndf)
		self.block2 = ResidualBlock(ndf, ndf, LayerNorm, 'down')
		self.block3 = ResidualBlock(ndf, ndf, LayerNorm, None)
		self.block4 = ResidualBlock(ndf, ndf, LayerNorm, None)
		self.pool5 = nn.Sequential(
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=8)
		)
		self.fc6 = nn.Linear(ndf, 1)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.pool5(x).view(x.size(0), -1)
		x = self.fc6(x)
		return x


class WGAN_GP(object):
	def __init__(self, opt):
		# hyperparameters
		self.num_samples = 64
		self.n_critic = 5
		self.lambd = 10
		self.nz = 128
		self.nc = 3
		self.ngf = 128
		self.ndf = 128

		self.device = opt.device

		# model
		self.G = Generator(self.nz, self.nc, self.ngf).to(self.device)
		self.D = Discriminator(self.nz, self.nc, self.ndf).to(self.device)
		if opt.train:
			initialize_weights(self.G)
			initialize_weights(self.D)
			self.G_optimizer = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
			self.D_optimizer = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
		
		print(self.G)
		print(self.D)

		self.sample_z = torch.randn((self.num_samples, self.nz)).to(self.device)

	def train(self, loader, opt, writer=None):
		self.y_real = torch.ones(opt.batch_size, 1).to(self.device)
		self.y_fake = torch.zeros(opt.batch_size, 1).to(self.device)

		self.D.train()
		n_itrs = 0
		D_running_loss = 0
		G_running_loss = 0
		GP_running_loss = 0

		while True:
			self.G.train()

			for itr, (x_real, _) in enumerate(loader):
				if x_real.size(0) != opt.batch_size:
					break

				x_real = x_real.to(opt.device)
				z = torch.randn((opt.batch_size, self.nz)).to(self.device)
				
				# === update D network === #
				self.D_optimizer.zero_grad()

				D_real = self.D(x_real)
				D_real_loss = -torch.mean(D_real)

				x_fake = self.G(z)
				D_fake = self.D(x_fake)
				D_fake_loss = torch.mean(D_fake)

				# gradient penalty
				alpha = torch.rand((opt.batch_size, 1, 1, 1)).to(self.device)
				x_hat = alpha * x_real.data + (1 - alpha) * x_fake.data
				x_hat.requires_grad = True

				D_hat = self.D(x_hat)
				gradients = grad(outputs=D_hat, inputs=x_hat, grad_outputs=torch.ones(D_hat.size()).to(self.device), create_graph=True, retain_graph=True, only_inputs=True)[0]

				gradient_penalty = self.lambd * ((gradients.view(gradients.size(0), -1).norm(2, 1) - 1) ** 2).mean()

				D_loss = D_real_loss + D_fake_loss + gradient_penalty
				D_loss.backward()
				self.D_optimizer.step()
				D_running_loss += D_loss.item()
				GP_running_loss += gradient_penalty.item()

				if ((itr + 1) % self.n_critic) == 0:
					n_itrs += 1

					# === update G === #
					self.G_optimizer.zero_grad()

					x_fake = self.G(z)
					D_fake = self.D(x_fake)
					
					G_loss = -torch.mean(D_fake)
					G_loss.backward()
					self.G_optimizer.step()
					G_running_loss += G_loss.item()

					# log
					if writer is not None:
						writer.add_scalars('Loss', {'G_loss': G_loss.item(), 'D_loss': D_loss.item()}, global_step=n_itrs)
						samples = self.generate()
						writer.add_image('generated samples', samples, global_step=n_itrs)

					# standard output
					if n_itrs % 100 == 0:
						commandline_output = '\r\033[K[itr {:d}] G_loss: {:.4f}, D_loss: {:.4f}, GP_loss: {:.4f}\n'.format(n_itrs, G_running_loss / 100, D_running_loss / (self.n_critic * 100), GP_running_loss / (self.n_critic * 100))
						D_running_loss = 0.0
						G_running_loss = 0.0
						GP_running_loss = 0.0
					else:
						commandline_output = '\r\033[K[itr {:d}] G_loss: {:.4f}, D_loss: {:.4f}, GP_loss: {:.4f}'.format(n_itrs, G_loss.item(), D_loss.item(), gradient_penalty.item())
					sys.stdout.write(commandline_output)
					sys.stdout.flush()

				# save model
				if n_itrs != 0 and n_itrs % opt.checkpoint == 0:
					G_path = os.path.join(opt.log_dir, 'G_{:d}itr.pkl'.format(n_itrs))
					D_path = os.path.join(opt.log_dir, 'D_{:d}itr.pkl'.format(n_itrs))
					self.save(G_path, D_path)

				if n_itrs == opt.num_itrs:
					break
			
			if n_itrs == opt.num_itrs:
				break

	def generate(self, fix=True):
		self.G.eval()

		if fix:
			samples = self.G(self.sample_z)
		else:
			samples = self.G(torch.rand((self.num_samples, self.nz)).to(self.device))

		return (samples + 1.) / 2.

	def save(self, G_path, D_path):
		torch.save(self.G.state_dict(), G_path)
		torch.save(self.D.state_dict(), D_path)

	def load(self, G_path, D_path):
		self.G.load_state_dict(torch.load(G_path))
		self.D.load_state_dict(torch.load(D_path))
