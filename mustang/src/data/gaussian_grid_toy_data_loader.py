import random

import torch
from torch.utils.data import Dataset
import matplotlib

from helpers.pytorch_helpers import to_pytorch_variable

matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np

from data.data_loader import DataLoader

N_RECORDS = 50000
N_VALUES_PER_RECORD = 2
N_MODES = 12

N_ROWS = 4
N_COLUMNS = 6
DISTANCE = 2
SIGMA = 0.01



class GaussianGridToyDataLoader(DataLoader):
    """
    A dataloader that returns samples from a simple toyproblem distribution multiple fixed points in a GaussianGrid
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(GaussianGridToyDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return N_VALUES_PER_RECORD

    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)


class GaussianGridToyDataSet(Dataset):

    def __init__(self, **kwargs):
        self.xs, self.ys = self.points()
        points_array = np.array((self.xs, self.ys), dtype=np.float).T
        self.data = torch.from_numpy(points_array).float()

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return N_RECORDS

    def points(self):
        points_x_coordinate = []
        points_y_coordinate = []
        centers = [(float(DISTANCE) * x, float(DISTANCE) * y) for x in range(N_COLUMNS) for y in range(N_ROWS)]
        for i in range(N_RECORDS):
            # For random samples from N(mu, sigma^2), use: sigma * np.random.randn() + mu
            point = np.random.randn(2) * SIGMA
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            points_x_coordinate.append(point[0])
            points_y_coordinate.append(point[1])

        return points_x_coordinate, points_y_coordinate 

    colors = None

    def save_images(self, tensor, filename, discriminator=None):
        plt.interactive(False)
        if not isinstance(tensor, list):
            plt.style.use('ggplot')
            plt.clf()
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            data = tensor.data.cpu().numpy() if hasattr(tensor, 'data') else tensor.cpu().numpy()
            x, y = np.split(data, 2, axis=1)
            x = x.flatten()
            y = y.flatten()
            for i in range(len(x)):
                rand_x = random.gauss(mu=x, sigma=0.1)
                rand_y = random.gauss(mu=y, sigma=0.1)
                ax1.scatter(rand_x, rand_y, c='red', marker='.', s=1)
            x_original, y_original = self.xs, self.ys #self.points()
            ax1.scatter(x_original, y_original, c='lime')
        
        else:
            if GaussianGridToyDataSet.colors is None:
                GaussianGridToyDataSet.colors = [np.random.rand(3, ) for _ in tensor]

            plt.style.use('ggplot')
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            # Plot generator
            x_original, y_original = self. xs, self.ys
            ax1.scatter(x_original, y_original, zorder=len(tensor) + 1, color='b')
            cm = plt.get_cmap('gist_rainbow')
            number_of_colors = 10
            ax1.set_prop_cycle('color', [cm(1. * i /  number_of_colors)  for i in range(number_of_colors)])
            for i, element in enumerate(tensor):
                data = element.data.cpu().numpy() if hasattr(element, 'data') else element.cpu().numpy()
                x, y = np.split(data, 2, axis=1)
                ax1.scatter(x.flatten(), y.flatten(), color=GaussianGridToyDataSet.colors[i],
                            zorder=len(tensor) - i, marker='x')

        plt.savefig(filename)

