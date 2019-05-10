from abc import abstractmethod, ABC
from enum import Enum

import copy
#import intertools

import numpy as np
import torch

from helpers.configuration_container import ConfigurationContainer
#from helpers.db_logger import DbLogger
import logging
import os

from distribution.state_encoder import StateEncoder
from helpers.pytorch_helpers import to_pytorch_variable, is_cuda_enabled, size_splits, noise
from networks.smugan_loss import HeuristicLoss


class CompetetiveNet(ABC):

    _logger = logging.getLogger(__name__)


    def __init__(self, loss_function, net, data_size, optimize_bias=True, loss_function_name = ""):
        self.data_size = data_size
        self.loss_function = loss_function
        self.net = net.cuda() if is_cuda_enabled() else net
        self.optimize_bias = optimize_bias
    
        self.loss_function_name = loss_function_name
        self.n_weights = np.sum([l.weight.numel() for l in self.net if hasattr(l, 'weight')])
        # Calculate split positions; cumulative sum needed because split() expects positions, not chunk sizes
        self.split_positions_weights = [l.weight.numel() for l in self.net if hasattr(l, 'weight')]

        if optimize_bias:
            self.split_positions_biases = [l.bias.numel() for l in self.net if hasattr(l, 'bias')]
        
        self.cc = ConfigurationContainer.instance()
        #experiment_id = self.cc.settings['general']['logging'].get('experiment_id', None)
        #self.db_logger = DbLogger(current_experiment=experiment_id)

    @abstractmethod
    def compute_loss_against(self, opponent, input):
        """
        :return: (computed_loss, output_data (optional))
        """
        pass

    def clone(self):
        return eval(self.__class__.__name__)(self.loss_function, copy.deepcopy(self.net),
                                             self.data_size, self.optimize_bias)

    @property
    @abstractmethod
    def name(self):
        pass
 
    @property
    @abstractmethod
    def default_fitness(self):
        pass

    @property
    def encoded_parameters(self):
        """
        :return: base64 encoded representation of the networks state dictionary
        """
        return StateEncoder.encode(self.net.state_dict())

    @encoded_parameters.setter
    def encoded_parameters(self, value):
        """
        :param value: base64 encoded representation of the networks state dictionary
        """
        #self._logger.info("Encoded paramenter: {} {} ".format(value, StateEncoder.decode(value)))
        self.net.load_state_dict(StateEncoder.decode(value))

    @property
    def parameters(self):
        """
        :return: 1d-ndarray[nr_of_layers * (nr_of_weights_per_layer + nr_of_biases_per_layer)]
        """
        weights = torch.cat([l.weight.data.view(l.weight.numel()) for l in self.net if hasattr(l, 'weight')])
        if self.optimize_bias:
            biases = torch.cat([l.bias.data for l in self.net if hasattr(l, 'bias')])
            return torch.cat((weights, biases))
        else:
            return weights

    @parameters.setter
    def parameters(self, value):
        """
        :param value: 1d-ndarray[nr_of_layers * (nr_of_weights_per_layer + nr_of_biases_per_layer)]
        """
 
        if self.optimize_bias:
            (weights, biases) = value.split(self.n_weights)
        else:
            weights = value

        # Update weights
        layered_weights = size_splits(weights, self.split_positions_weights)
        for i, layer in enumerate([l for l in self.net if hasattr(l, 'weight')]):
            self._update_layer_field(layered_weights[i], layer.weight)

        # Update biases
        if self.optimize_bias:
            layered_biases = size_splits(biases, self.split_positions_biases)
            for i, layer in enumerate([l for l in self.net if hasattr(l, 'bias')]):
                self._update_layer_field(layered_biases[i], layer.bias)

    @staticmethod
    def _update_layer_field(source, target):
        # Required because it's recommended to only use in-place operations on PyTorch variables
        target.data.zero_()
        if len(target.data.shape) == 1:
            target.data.add_(source)
        else:
            target.data.add_(source.view(target.size()))


class GeneratorNet(CompetetiveNet):
    def __init__(self, loss_function, net, data_size, optimize_bias=True):
        #self.bceloss = torch.nn.BCELoss()
        #self.mseloss = torch.nn.MSELoss()
        #self.heuristicloss = HeuristicLoss()
        super().__init__(loss_function, net, data_size, optimize_bias, "Loss not conficured yet")

        if 'SMuGANLoss' in loss_function.__class__.__name__:
            prob = np.random.uniform()
            if prob < 0.33:
                self.loss_function_to_apply =  torch.nn.BCELoss()
                self.loss_function_name = "SMuGANLoss - BCE"
            elif prob < 0.66 :
                self.loss_function_to_apply =  torch.nn.MSELoss()
                self.loss_function_name = "SMuGANLoss - MSE"
            else:
                self.loss_function_to_apply = HeuristicLoss()
                self.loss_function_name = "SMuGANLoss - HEU"
        else:
            self.loss_function_to_apply = self.loss_function
            self.loss_function_name = self.loss_function.__class__.__name__
        
        self._logger.info("Generator - Selected loss function: {}".format(self.loss_function_name))


        #self.bceloss = torch.nn.BCELoss()
        #self.mseloss = torch.nn.MSELoss()
        #self.heuristicloss = HeuristicLoss()
  
    @property
    def name(self):
        return 'Generator'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input):
        batch_size = input.size(0)

        real_labels = to_pytorch_variable(torch.ones(batch_size))

        z = noise(batch_size, self.data_size)
        fake_images = self.net(z)
        outputs = opponent.net(fake_images).view(-1)

        # Compute BCELoss using D(G(z))
        if False: #self.loss_function.__class__.__name__ == 'SMuGANLoss':
            prob = np.random.uniform()
            if prob < 0.33:
                loss_selected = "BCE"
                loss = self.bceloss(outputs, real_labels)
            elif prob < 0.66 :
                loss = self.mseloss(outputs, real_labels)
                loss_selected = "MSE"
            else:
                loss = self.heuristicloss(outputs, real_labels)
                loss_selected = "HEU"
            return loss, fake_images
        else:
            #self._logger.info("Applying: {}".format(self.loss_function_name))
            return self.loss_function_to_apply(outputs, real_labels), fake_images
         


class DiscriminatorNet(CompetetiveNet):
    def __init__(self, loss_function, net, data_size, optimize_bias=True):
        super().__init__(loss_function, net, data_size, optimize_bias, "Loss not conficured yet")

        if 'MSELoss' in loss_function.__class__.__name__:
            self.loss_function_to_apply =  torch.nn.MSELoss()
            self.loss_function_name = "MSE"
        else:
            self.loss_function_to_apply = torch.nn.BCELoss() 
            self.loss_function_name = self.loss_function.__class__.__name__

        self._logger.info("Discriminator - Selected loss function: {}".format(self.loss_function_name))
    
    @property
    def name(self):
        return 'Discriminator'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input):
        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        batch_size = input.size(0)
        real_labels = to_pytorch_variable(torch.ones(batch_size))
        fake_labels = to_pytorch_variable(torch.zeros(batch_size))

        outputs = self.net(input).view(-1)
        #d_loss_real = self.loss_function(outputs, real_labels)
        d_loss_real = self.loss_function_to_apply(outputs, real_labels)

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = noise(batch_size, self.data_size)
        fake_images = opponent.net(z)
        outputs = self.net(fake_images).view(-1)
        d_loss_fake = self.loss_function_to_apply(outputs, fake_labels)

        return d_loss_real + d_loss_fake, None
