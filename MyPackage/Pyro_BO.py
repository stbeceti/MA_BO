import torch

import pyro
from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.models.gpr import GPRegression
from pyro.contrib.gp.util import train


class ConfidenceBound:


	def __init__(self, input_dim, kappa=2., maximize=True):

		self.input_dim = input_dim
		self.kappa = kappa
		self.maximize = maximize


	def get_acq_values(self, mu_vector, sigma_vector):

		if self.maximize:
			return mu_vector + self.kappa*sigma_vector
		else:
			return mu_vector - self.kappa*sigma_vector


	def get_next_sample(self, X_test, mu_vector, sigma_vector):

		if self.maximize:
			return X_test[int(self.get_acq_values(mu_vector, sigma_vector).argmax().item())].reshape(-1, X_test.shape[-1])
		else:
			return X_test[int(self.get_acq_values(mu_vector, sigma_vector).argmin().item())].reshape(-1, X_test.shape[-1])



class TransformationKernel(RBF):


	def __init__(self, input_dim, trans_mappings=None):
		
		super().__init__(input_dim)
		
		if trans_mappings is None:
			self.mappings = OrderedDict()

			for dim_idx in range(input_dim):
				self.mappings[len(self.mappings)] = lambda x: x

		else:
			self.mappings = trans_mappings
		
		for attr_idx, key in zip(range(len(self.mappings)), list(self.mappings.keys())):
	                setattr(self, 'module_{}'.format(attr_idx), self.mappings[key])

	
	def forward(self, X, Z=None, diag=False):
		
		keys = list(self.mappings.keys())
		X_cpy = torch.zeros(X.size(), dtype=torch.float64)
		
		if Z is not None:
			Z_cpy = torch.zeros(Z.size(), dtype=torch.float64)
		
		col_idx = 0
		for key in keys:
			
			X_cpy.T[[col_idx]] = self.mappings[key](X.T[[key]].T).T.to(torch.float64)

			if Z is not None:
				Z_cpy.T[[col_idx]] = self.mappings[key](Z.T[[key]].T).T.to(torch.float64)
			col_idx = col_idx+1

		if Z is not None:
			Z = Z_cpy

		return super().forward(X_cpy, Z, diag)



class PyroBO(GPRegression):

	def __init__(self, X, y, kernel, noise=None, jitter=1e-06, acq_fun=None):
		
		super().__init__(X, y, kernel, noise=noise, jitter=jitter)

		self.input_dim = kernel.input_dim

		if acq_fun is None:
			self.acq_fun = ConfidenceBound(input_dim=self.input_dim)
		else:
			self.acq_fun = acq_fun


	def next_sample(self, X_test):

		prediction = self(X_test)
		mu_vector = prediction[0].data
		var_vector = prediction[1].data

		return self.acq_fun.get_next_sample(X_test, mu_vector, var_vector.sqrt())

	def get_acq_values(self, X_test):
		
		prediction = self(X_test)
		mu_vector = prediction[0].data
		var_vector = prediction[1].data

		return self.acq_fun.get_acq_values(mu_vector, var_vector.sqrt())

	def add_Observation(self, xnew, ynew):

		new_X = torch.cat((self.X, xnew), dim=0)
		new_y = torch.cat((self.y, ynew), dim=0)
		
		self.set_data(new_X, new_y)

	def update(self):
		train(self)
