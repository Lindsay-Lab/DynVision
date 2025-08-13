# User Guide

Welcome to the DynVision User Guide! This section provides task-oriented guides to help you accomplish specific goals with DynVision.

## Contents

### Core Functionality

- [**Installation**](installation.md): Detailed instructions for installing DynVision on different platforms
- [**Custom Models**](custom-models.md): How to create your own neural network architectures
- [**Data Processing**](data-processing.md): Working with datasets and data loaders
- [**Workflow Management**](workflows.md): Using Snakemake for experiment orchestration
- [**Visualization**](visualization.md): Visualizing and analyzing results
- [**Cluster Integration**](cluster-integration.md): Running DynVision on high-performance computing clusters

<!-- ### Advanced Topics

- [**Model Evaluation**](evaluation.md): Evaluating model performance and biological plausibility
- [**Hyperparameter Optimization**](hyperparameter-optimization.md): Systematic parameter tuning
- [**Custom Experiments**](custom-experiments.md): Designing and implementing new experiments
- [**Performance Optimization**](performance-optimization.md): Making DynVision run faster
- [**Transfer Learning**](transfer-learning.md): Using pre-trained models -->

### Common Tasks

Below are step-by-step instructions for common tasks in DynVision:

#### Training a Model on a Custom Dataset

1. Place your dataset in the `data/raw/your_dataset/` directory.
2. Add dataset statistics in `config_data.yaml`.
3. Run the data preparation workflow:
   ```bash
   snakemake project_paths.data.raw/your_dataset/train
   ```
4. Convert the dataset to FFCV format:
   ```bash
   snakemake project_paths.data.processed/your_dataset/train_all/train.beton
   ```
5. Train a model on your dataset:
   ```bash
   snakemake train_model --config \
     model_name=DyRCNNx4 \
     data_name=your_dataset \
     model_args="{rctype:full}"
   ```

#### Comparing Different Recurrence Types

1. Set up your experiment in `config_experiments.yaml` (or use an existing one).
2. Run the experiment with multiple recurrence types:
   ```bash
   snakemake all_experiments --config \
     experiment=contrast \
     model_args="{rctype:[full,self,depthpointwise,pointdepthwise]}"
   ```
3. Generate comparative visualizations:
   ```bash
   snakemake plot_experiments_on_models
   ```
4. Analyze the results in the `reports/figures/contrast/` directory.

#### Extracting Neural Responses

1. Test a model and store responses:
   ```bash
   snakemake test_model --config \
     model_name=DyRCNNx4 \
     model_args="{rctype:full}" \
     data_name=cifar100 \
     data_group=invertebrates \
     store_responses=100 \
     data_loader=StimulusDuration \
     data_args="{tsteps:100,stim:15}"
   ```
2. Load and analyze the responses in Python:
   ```python
   import torch
   import matplotlib.pyplot as plt
   
   # Load responses
   responses = torch.load('models/DyRCNNx4/DyRCNNx4:rctype=full_0001_cifar100_trained_StimulusDuration:tsteps=100+stim=15_invertebrates_test_responses.pt')
   
   # Extract layer responses
   v1_response = responses['V1'].mean(dim=(0, 2, 3, 4))  # Average over all dimensions except time
   
   # Plot response time course
   plt.figure(figsize=(10, 6))
   plt.plot(v1_response.cpu())
   plt.title('V1 Response Time Course')
   plt.xlabel('Time (timesteps)')
   plt.ylabel('Average Activation')
   plt.savefig('v1_response.png')
   ```

#### Creating a Custom Recurrence Type

1. Create a new recurrence class in `model_components/recurrence.py`:
   ```python
   class CustomRecurrence(nn.Module):
       def __init__(
           self,
           in_channels,
           kernel_size,
           bias=False,
           max_weight_init=0.05,
           **kwargs
       ):
           super().__init__()
           self.max_weight_init = max_weight_init
           
           # Implement your custom recurrence
           self.conv = nn.Conv2d(
               in_channels=in_channels,
               out_channels=in_channels,
               kernel_size=kernel_size,
               padding=kernel_size//2,
               bias=bias
           )
       
       def forward(self, x):
           return self.conv(x)
       
       def _init_parameters(self):
           nn.init.uniform_(
               self.conv.weight, a=-self.max_weight_init, b=self.max_weight_init
           )
   ```

2. Add your recurrence type to `RecurrentConnectedConv2d` in `recurrence.py`:
   ```python
   # Inside _define_architecture method
   elif self.recurrence_type == "custom":
       self.recurrence = CustomRecurrence(
           in_channels=self.out_channels,
           **recurrence_params,
       )
   ```

3. Use your custom recurrence type in a model:
   ```bash
   snakemake experiment --config \
     model_name=DyRCNNx4 \
     model_args="{rctype:custom}" \
     experiment=contrast
   ```

## FAQ

**Q: How do I run DynVision without using Snakemake?**  
A: While Snakemake provides the most integrated experience, you can use DynVision components directly in Python scripts. See [Direct API Usage](direct-api-usage.md) for examples.

**Q: Can I use DynVision with my existing models?**  
A: Yes, you can wrap existing PyTorch models with DynVision's `LightningBase` class to leverage its training and evaluation infrastructure. See [Integrating Existing Models](integrating-existing-models.md).

**Q: How do I tune hyperparameters efficiently?**  
A: DynVision supports parameter sweeps through Snakemake's config system. See [Hyperparameter Optimization](hyperparameter-optimization.md) for more details.

**Q: Can I use DynVision without a GPU?**  
A: Yes, DynVision works on CPU, but training will be significantly slower. Use smaller models and datasets for experimentation on CPU.

**Q: How can I contribute to DynVision?**  
A: See the [Contributing Guide](../contributing.md) for information on how to contribute to the project.

## Troubleshooting

For common issues and their solutions, see the [Troubleshooting Guide](troubleshooting.md).

If you encounter problems not covered in the documentation, please [open an issue](https://github.com/yourusername/dynvision/issues) on GitHub.
