import os
import argparse
import json
import torch

class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		# dataset
		# parser.add_argument('--dataset', type=str, required=True, help='cifar10 | cifar100 | tiny_imagenet')
		parser.add_argument('--num_workers', type=int, default=12, help='number of workers')
		# hyperparameter
		parser.add_argument('--batch_size', type=int, default=64, help='batch size')
		# log
		parser.add_argument('--log_dir', type=str, required=True, help='log directory')
		# GPU
		parser.add_argument('--use_gpu', action='store_true', default=False, help='GPU mode ON/OFF')
		parser.add_argument('--device_id', type=int, default=None, help='GPU device ID')

		self.initialized = True
		return parser

	def gather_options(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '---------------------------- Options --------------------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: {}]'.format(str(default))
			message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
		message += '---------------------------- End ------------------------------'
		print(message)

		os.makedirs(opt.log_dir, exist_ok=True)
		with open(os.path.join(opt.log_dir, 'options.txt'), 'wt') as f:
			command = ''
			for k, v in sorted(vars(opt).items()):
				command += '--{} {} '.format(k, str(v))
			command += '\n'
			f.write(command)
			f.write(message)
			f.write('\n')

	def parse(self):
		opt = self.gather_options()
		self.print_options(opt)

		if self.__class__.__name__ == 'TrainOptions':
			opt.train = True
		else:
			opt.train = False

		# GPU
		if opt.use_gpu and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			if opt.device_id:
				opt.device = torch.device('cuda:{}'.format(opt.device_id))
			else:
				opt.device = torch.device('cuda')
		else:
			opt.device = torch.device('cpu')

		self.opt = opt
		return self.opt

class TrainOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# hyperparameter
		parser.add_argument('--num_itrs', type=int, default=50000, help='number of iterations')
		parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for Adam')
		parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
		parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
		parser.add_argument('--num_critic', type=int, default=5, help='number of critic iteration for one generator iteration')
		parser.add_argument('--lambda_gp', type=float, default=10, help='coefficient of gradient penalty')
		# log
		parser.add_argument('--checkpoint', type=int, default=5000, help='checkpoint iteration')
		return parser

class TestOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# model
		parser.add_argument('--weight', type=str, required=True, help='path to model weight')
		return parser
