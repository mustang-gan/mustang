# Mustangs

MUtation SpaTial gANs training method (MUSTANGs)

## Summary

Mustangs is an generative adversarial networks (GANs) training framework that combines E-GANs [1], which apply the principles of evolutionary computing to train GANs by generating diversity in terms of (gradient-based) mutations applied to the generator, and Lipizzaner [2], which uses a spatially distributed coevolutioary algorithms to optimize two populations of networks (generators and discriminators). Mustan mitigates problems such as instability and mode collapse during the training process. 

This method has been presented in the paper **Spatial Evolutionary Generative Adversarial Networks**, which has been accepted/published in **GECCO'19**. The information about the paper can be seen below.

## How-To

As the method is principally developed over Lipizzaner, the installation instructions are the same than Lipizzaner and are included in the `./mustang/` folder. 

In order to configure our system to apply the probabilistic Mustangs loss functions, the user has to set `smuganloss` as the loss function in the configuration files (`.yml`). Therefore the `network` section of the configuration file should include the following information:

   ```
      network:
         name: convolutional
         loss: smuganloss 
   ```

## GECCO'19 Paper Information

#### Title: 
**Spatial Evolutionary Generative Adversarial Networks**

#### Abstract: 
Generative adversary networks (GANs) suffer from training pathologies such as instability and mode collapse. These pathologies mainly arise from a lack of diversity in their adversarial interactions. Evolutionary generative adversarial networks apply the principles of evolutionary computation to mitigate these problems. We hybridize two of these approaches that promote training diversity. One, E-GAN, at each batch, injects mutation diversity by training the (replicated) generator with three independent objective functions then selecting the resulting best performing generator for the next batch. The other, Lipizzaner, injects population diversity by training a two-dimensional grid of GANs with a distributed evolutionary algorithm that includes neighbor exchanges of additional training adversaries, performance based selection and population-based hyper-parameter tuning. We propose to combine mutation and population approaches to diversity improvement. We contribute a superior evolutionary GANs training method, Mustangs, that eliminates the single loss function used across Lipizzaner ’s grid. Instead, each training round, a loss function is selected with equal probability, from among the three E-GAN uses. Experimental analyses on standard benchmarks, MNIST and CelebA, demonstrate that Mustangs provides a statistically faster training method resulting in more accurate networks.

#### ACM Reference Format:

Jamal Toutouh, Erik Hemberg, and Una-May O’Reilly. 2019. Re-purposing Heterogeneous Generative Ensembles with Evolutionary Computation. In *Genetic and Evolutionary Computation Conference (GECCO ’20), 2020.* ACM, New York, NY, USA, 9 pages. [https://doi.org/10.1145/3377930.3390229](https://doi.org/10.1145/3377930.3390229)

#### Bibtex Reference Format:

```
@inproceedings{Toutouh_GECO2020,
author = {Toutouh, Jamal and Hemberg, Erik and O’Reilly, Una-May},
title = {Re-purposing Heterogeneous Generative Ensembles with Evolutionary Computation},
year = {2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3377930.3390229},
doi = {10.1145/3377930.3390229},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
numpages = {9},
series = {GECCO ’2020}
}
```


