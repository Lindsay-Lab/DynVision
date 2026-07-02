# Basic Model Training

This tutorial will guide you through the process of training a recurrent neural network using DynVision. By the end, you'll have trained a DyRCNNx4 model on the CIFAR-10 dataset and analyzed its response dynamics.

## Prerequisites

Before starting this tutorial, make sure you have:

1. DynVision installed (see the [Installation Guide](../user-guide/installation.md))
2. A CUDA-compatible GPU (recommended, though CPU will work for small models)
3. Basic familiarity with PyTorch and neural networks

## Step 1: Set Up Your Environment

First, activate your DynVision environment:

```bash
conda activate dynvision
```

Then, navigate to your DynVision working directory:

```bash
cd /path/to/your/working/directory
```

## Step 2: Download and Prepare the Dataset

DynVision provides a Snakemake workflow for dataset preparation. For CIFAR-10:

```bash
# Download CIFAR-10
snakemake project_paths.data.raw/cifar10/train

# Create symbolic links for the full dataset
snakemake project_paths.data.interim/cifar10/train_all.ready
snakemake project_paths.data.interim/cifar10/test_all.ready

# Convert to FFCV format for faster loading
snakemake project_paths.data.processed/cifar10/train_all/train.beton
```

This will download CIFAR-10, organize it in the data directory structure, and convert it to the FFCV format for efficient loading.

## Step 3: Initialize a Model

Now, let's initialize a DyRCNNx4 model with full recurrence:

```bash
snakemake project_paths.models/DyRCNNx4/DyRCNNx4:rctype=full_0001_cifar10_init.pt --config \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}" \
  data_name=cifar10 \
  seed=0001
```

This creates an initialized model file with full recurrent connections.

## Step 4: Train the Model

Now, let's train the model:

```bash
snakemake project_paths.models/DyRCNNx4/DyRCNNx4:rctype=full_0001_cifar10_trained.pt --config \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}" \
  data_name=cifar10 \
  seed=0001 \
  epochs=50 \
  batch_size=128 \
  learning_rate=0.001
```

This will train the model for 50 epochs using the Adam optimizer with a learning rate of 0.001.

The training process will:

1. Load the initialized model
2. Prepare the data loaders using FFCV
3. Configure the optimizer and learning rate scheduler
4. Train the model with PyTorch Lightning
5. Save the trained model

You can monitor the training progress with the output logs, which show:

- Training loss and accuracy
- Validation loss and accuracy
- Learning rate changes
- Time per epoch

## Step 5: Test the Model

After training, let's test the model on the CIFAR-10 test set:

```bash
snakemake project_paths.reports/DyRCNNx4/DyRCNNx4:rctype=full_0001_cifar10_trained_StandardDataLoader_all_test_outputs.csv --config \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}" \
  data_name=cifar10 \
  data_group=all \
  status=trained \
  data_loader=StandardDataLoader
```

This will run the model on the test set and save the results as a CSV file.

## Step 6: Run Dynamics Experiments

Now, let's run some experiments to analyze the model's temporal dynamics:

```bash
# Response experiment
snakemake experiment --config \
  experiment=response \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}" \
  data_name=cifar10 \
  seed=0001
```

This will run the basic response experiment, which measures how the model responds to a static input over time.

Let's also run the contrast experiment to see how the model responds to different contrast levels:

```bash
# Contrast experiment
snakemake experiment --config \
  experiment=contrast \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}" \
  data_name=cifar10 \
  seed=0001
```

## Step 7: Visualize the Results

Generate visualizations for the experiment results:

```bash
# Generate visualizations
snakemake plot_adaption --config \
  experiment=contrast \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}" \
  data_name=cifar10 \
  seed=0001
```

This will create visualizations showing how the model's responses change with different contrast levels.

## Step 8: Compare Different Recurrence Types

For comparison, let's train and evaluate a model with self recurrence:

```bash
# Initialize and train a model with self recurrence
snakemake project_paths.models/DyRCNNx4/DyRCNNx4:rctype=self_0001_cifar10_trained.pt --config \
  model_name=DyRCNNx4 \
  model_args="{rctype:self}" \
  data_name=cifar10 \
  seed=0001 \
  epochs=50 \
  batch_size=128 \
  learning_rate=0.001

# Run contrast experiment
snakemake experiment --config \
  experiment=contrast \
  model_name=DyRCNNx4 \
  model_args="{rctype:self}" \
  data_name=cifar10 \
  seed=0001
```

Now, let's generate comparative visualizations:

```bash
# Generate comparative visualizations
snakemake plot_experiments_on_models --config \
  experiment=contrast \
  model_args="{rctype:[full,self]}" \
  data_name=cifar10 \
  seed=0001
```

This will create visualizations comparing the response properties of models with different recurrence types.

## Step 9: Analyze the Results

The visualizations will be saved in the `reports/figures/` directory. Let's examine them to understand the model's behavior:

1. **Basic Response Properties**:
      - Look at the response time courses in each layer (V1, V2, V4, IT)
      - Note the response latencies (when activity starts to increase)
      - Observe the peak times and response durations

2. **Contrast Response Properties**:
      - Examine how response magnitude changes with contrast
      - Observe whether higher contrast leads to faster responses
      - Compare the contrast sensitivity across layers

3. **Recurrence Type Comparison**:
      - Compare full recurrence vs. self recurrence
      - Note differences in response magnitude, timing, and shape
      - Consider which better captures biological properties

## Step 10: Customizing the Training

You can customize the training process by modifying the configuration parameters:

```bash
# Train with custom parameters
snakemake train_model --config \
  model_name=DyRCNNx4 \
  model_args="{rctype:full,dt:1,tau:10,tff:12,trc:5}" \
  data_name=cifar10 \
  seed=0001 \
  epochs=100 \
  batch_size=256 \
  learning_rate=0.0005 \
  loss="[CrossEntropyLoss,ActivityLoss]"
```

This example:

- Uses a smaller time step (dt=1ms)
- Changes the time constant (tau=10ms)
- Adjusts the feedforward (tff=12ms) and recurrent (trc=5ms) delays
- Trains for more epochs (100)
- Uses a larger batch size (256)
- Uses a lower learning rate (0.0005)
- Adds an activity loss term to promote stable activity

You can also control temporal data presentation patterns:

```bash
# Train with stimulus/null presentation pattern and reaction time masking
snakemake train_model --config \
  model_name=DyRCNNx4 \
  model_args="{rctype:full,pattern:1011,shufflepattern:true,lossrt:4}" \
  data_name=cifar10 \
  seed=0001 \
  epochs=50
```

This example adds:

- `pattern:1011`: Alternating stimulus (1) and null (0) presentation
- `shufflepattern:true`: Randomly shuffle the pattern per batch
- `lossrt:4`: Mask labels for 4ms after stimulus onset

For details on temporal presentation options, see the [Temporal Data Presentation Guide](../user-guide/temporal-data-presentation.md).

## Conclusion

Congratulations! You've successfully:

1. Prepared a dataset for DynVision
2. Trained a recurrent neural network model
3. Evaluated its performance
4. Analyzed its temporal dynamics
5. Compared different recurrence types

Next Steps:

- Try other datasets like CIFAR-100 or ImageNet
- Experiment with different model architectures
- Explore other recurrence types
- Create your own custom model

For more advanced usage, check out the [Custom Models Guide](../user-guide/custom-models.md).
