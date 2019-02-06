# a predicted, b true
# data loader is in such a way that true y should be indexed from floatTensor
import torch


class HeuristicLoss(torch.nn.Module):

    def __init__(self):
        super(HeuristicLoss, self).__init__()

    def forward(self, input, target):  
        return -0.5 * torch.mean(torch.log(input),dim=0)


class SMuGANLoss(torch.nn.Module):

    def __init__(self):
        super(SMuGANLoss, self).__init__()

    def forward(self, input, target):
        pass  
        

