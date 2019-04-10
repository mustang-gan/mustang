# Mustang

Mustang is an generative adversarial networks (GANs) training framework that combines E-GANs [1], which apply the principles of evolutionary computing to train GANs by generating diversity in terms of (gradient-based) mutationts applied to the generator, and Lipizzaner [2], which uses a spatially distributed coevolutioary algorithms to optimize two populations of networks (generators and discriminators). Mustan mitigates problems such as instability and mode collapse during the training process. 

[1] Wang, C., Xu, C., Yao, X., & Tao, D. (2019). Evolutionary generative adversarial networks. *IEEE Transactions on Evolutionary Computation*, 2019.

[2] Schmiedlechner, T., Ng Zhi Yong, I., Al-Dujaili, A., Hemberg, E., O'Reilly, U., “Lipizzaner: A System That Scales Robust Generative Adversarial Network Training,” NeurIPS 2018 Workshop on System for Machine Learning, 2018.
