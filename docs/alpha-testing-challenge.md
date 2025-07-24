1. clone the repository (https://github.com/Lindsay-Lab/DynVision/tree/alpha-testing) to your computer and read the documentation (https://github.com/Lindsay-Lab/DynVision/blob/alpha-testing/docs/index.md) to complete the following steps.
2. install toolbox with all its dependencies in a conda/mamba environment
3. setup paths and configs to run a first minimal example: training the `DyRCNNx2` model on the MNIST dataset
    - use config settings: `use_ffcv=False`, 
    - depending on the compute resources of your computer this might take a long time, the important thing is to get it to train without errors
4. edit the model by modifying the configuration, for example try different recurrence types and other biologically inspired features
5. scale up the training by using the larger model `DyRCNNx4` and the larger dataset `cifar10`
    - to speed up the dataloader switch to the ffcv dataloader with config setting `use_ffcv=True`
6. ... (running things on a compute cluster)