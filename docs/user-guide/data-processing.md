# Data Processing and Management

This guide explains how to use DynVision's data processing pipeline, including dataset preparation, management, and loading for experiments. 

Most operations described in this documentation are automatically handled when executing the full workflow through Snakemake and usually don't need to run manually. For custom modifications and extensions to the pipeline, refer to the [Customization of Data Processing](#customization-of-data-processing) section.

## Overview

DynVision implements a comprehensive data management system that:

1. Handles standard image datasets (CIFAR10, CIFAR100, MNIST, ImageNet)
2. Organizes data into groups for specialized experiments
3. Uses symbolic links for efficient storage and organization
4. Provides performance-optimized data loading with FFCV
5. Implements specialized data loaders for neuroscience experiments

### Data Directory Structure

DynVision organizes data into a structured hierarchy:

```
data/
├── raw/                 # Original datasets in original format
│   ├── cifar10/
│   ├── cifar100/
│   └── mnist/
├── external/            # External data not part of standard datasets
├── interim/             # Processed data, organized by experiments
│   ├── cifar100/
│   │   ├── train_all/
│   │   └── test_invertebrates/
│   └── mnist/
└── processed/           # Final data ready for model consumption
    ├── cifar100/
    │   ├── train_all/
    │   │   ├── train.beton     # FFCV-formatted data
    │   │   └── val.beton
    └── mnist/
```

### Data Processing Flow

The following flow chart illustrates the complete data processing pipeline in DynVision:

```
data/raw/                      data/interim/                   data/processed/              Usage
(Original Datasets)            (Organized Data)                (Optimized Data)
      |                             |                               |
      |                             |                               |
[Data Acquisition]------------>[Data Organization]---------->[FFCV Processing]----------->[Training]
      |                             |                               |                         |
Download/Mount                 Create Symlinks              Convert to .beton                 |

  - CIFAR10                    - Group by Classes           Configure Loader                  |
  - CIFAR100                   - Train/Test Split           - Extended Timesteps              |
  - MNIST                      - Create PyTorch             - GPU Optimization                |
  - ImageNet                     Dataset Objects            - Memory Efficiency               |
  - Custom Datasets                 |                                                         |
                                    |                                                         |
                                    |                                                         |
                                    +--------------->[PyTorch Processing]---------------->[Testing]
                                                    - Specialized DataLoader
                                                    - Temporal Presentation
                                                    - Experimental Conditions
```

Key Processing Steps:

1. Raw Data Storage: Original datasets stored in their native format
2. Interim Organization: Data split into train/test and organized in testing groups
3. FFCV Optimization: Conversion to .beton format for efficient loading
4. Usage Paths: 
   - Training: uses FFCV-optimized data and dataloader
   - Testing: uses pytorch dataloader on symlinked groups


## Downloading and Preparing Datasets

DynVision uses Snakemake rules to download and prepare datasets:

```bash
# Download and prepare CIFAR100 dataset
snakemake <project_paths.data.raw>/cifar100/train/
```

This command triggers the `get_data` rule, which downloads CIFAR100 and organizes it in the raw data directory.

For custom datasets, place them in the `data/raw` directory following the same structure.

## Configuring the Dataset

DynVision supports organizing datasets into groups of classes, which is useful for experiments with a scaled-down dataset that can vary in difficulty depending on the grouping:

### Configuring Data Groups

Data groups are defined in `config_data.yaml`:

```yaml
data_groups:
    cifar100:
        invertebrates:
            - 6  # bee
            - 7  # beetle
            - 14 # butterfly
            - 18 # caterpillar
            - 24 # cockroach
            - 26 # crab
            - 45 # lobster
            - 77 # snail
            - 79 # spider
            - 99 # worm
    mnist:
        '01':
            - 0
            - 1
        '89':
            - 8
            - 9
```

### Class Index Files

For datasets whose on-disk class folders do not match the numeric labels used in `data_groups` (e.g., ImageNet or TinyImageNet), map class indices to folder names via `class_index_files`:

```yaml
class_index_files:
    imagenet: imagenet_class_index.json
    imagenette: imagenet_class_index.json
    tinyimagenet: tinyimagenet_class_index.json
```

The paths are resolved relative to `project_paths.references`. Each JSON entry should map a numeric index to the corresponding class identifier (e.g., WordNet ID). If a dataset is omitted from this section, DynVision assumes the folder names already match the class IDs you specify in `data_groups`.

### Creating Data Group Symlinks

To create symlinks for a data group:

```bash
# Create symlinks for CIFAR100 invertebrates group
snakemake <project_paths.data.interim>/cifar100/test_invertebrates.ready
```

The default group containing all classes is called `all`:

```bash
snakemake <project_paths.data.interim>/cifar100/train_all.ready
```

Each target builds the `<data_subset>_<data_group>` directory of per-class symlinks and then writes a `.ready` flag file. Downstream rules depend on the flag while consuming the actual directory (e.g., `.../train_all/`) for data loading.


### Converting to FFCV Format

DynVision uses FFCV for optimized data loading during training. To convert a dataset to FFCV format:

```bash
# Convert CIFAR100 to FFCV format
snakemake <project_paths.data.processed>/cifar100/train_all/train.beton
```

This creates `.beton` files that can be loaded more efficiently during training.


## Configuring Transforms

### Data Transforms

DynVision provides configurable data transformations:

```python
from dynvision.data.transforms import get_data_transform

# Get a predefined transform
transform = get_data_transform(transform='train')  # Standard training augmentations

# Get dataset-specific transform
transform = get_data_transform(transform='train', data_name='cifar100')

# Create custom transform combination
transform = get_data_transform(transform=['train', 'test'])
```

The `transform` input can single str, list, or dict of keys from the transform presets defined in `dynvision.data.transforms`, which you can edit and extend.

Available transform presets include:

- `train`: Standard training augmentations (random flips, etc.)
- `test`: Standard test transformations
- `ffcv_train`: Optimized transforms for FFCV training
- `ffcv_test`: Optimized transforms for FFCV testing
- Dataset-specific transforms (e.g., `mnist`, `imagenet`)

### Target Transforms

For data grouping, target transforms map original labels to new indices:

```python
from dynvision.data.transforms import get_target_transform

# Transform for CIFAR100 invertebrates group
target_transform = get_target_transform('cifar100_invertebrates')
```

## Data Loaders

With the dataset files and symlinks organized, the data can be compiled into pytorch dataset.

```python
from dynvision.data.dataloader import get_data_loader
from dynvision.data.datasets import get_dataset

# get dataset
dataset = get_dataset(
    data_path="<project_paths.data.interim>/{data_name}/test_{data_group}",
    data_name="{data_name}",
    data_transform="test",
    target_transform="{data_group}",
)
```

The selected data can be presented to the model in various ways by varying how an image appears over time, by changing the image's contrast, or other manipulations.
These can be realized by the dataloader at runtime.
DynVision provides several specialized data loaders for neuroscience experiments:

**Standard Loader**

Optionally repeating the image presentation over multiple timesteps.

```python
loader = get_data_loader(
    dataset=dataset,
    n_timesteps=1,  # Repeat data over time dimension
    **kwargs        # other optional data loader args
)
```

**Stimulus Duration Loader**

This loader presents a stimulus for a specified duration, with intro and outro periods of void values.

```python
loader = get_data_loader(
    dataloader='StimulusDuration',
    dataset=dataset,
    n_timesteps=100,        # Total sequence length
    stimulus_duration=20,   # Duration of the stimulus
    intro_duration=2,       # Duration before stimulus
    void_value=0            # Pixel value shown in absence of image
    non_label_index=-1,     # Label index for void input
    **kwargs                # other optional data loader args
)
```


**Stimulus Interval Loader**

This loader presents a stimulus twice with a specified interval between presentations.

```python
loader = get_data_loader(
    dataloader='StimulusInterval',
    n_timesteps=100,        # Total sequence length
    stimulus_duration=5,    # Duration of each stimulus
    intro_duration=1,       # Duration before the first stimulus
    interval_duration=10,   # Interval between the two stimuli
    void_value=0,           # Pixel value shown in absence of image
    non_label_index=-1,     # Label index for void input
    **kwargs                # other optional data loader args
)
```

**Stimulus Contrast Loader**

This loader presents a stimulus with adjustable contrast. The contrast level can be set to simulate different experimental conditions.

```python
loader = get_data_loader(
    dataloader='StimulusContrast',
    dataset=dataset,
    n_timesteps=100,        # Total sequence length
    stimulus_duration=15,   # Duration of the stimulus
    intro_duration=2,       # Duration before the stimulus
    stimulus_contrast=0.5,  # Contrast of the stimulus (0-1)
    void_value=0,           # Pixel value shown in absence of image
    non_label_index=-1,     # Label index for void input
    **kwargs                # Other optional data loader arguments
)
```

### Using Data Loaders in Experiments

DynVision's workflow system sets up appropriate data loaders based on experiment configurations:

```yaml
# In config_experiments.yaml
experiment_config:
  contrast:
    status: trained
    parameter: contrast
    data_loader: StimulusContrast
    data_args:
      tsteps: 100
      stim: 15
      contrast: [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]  
```

To run this experiment:

```bash
snakemake --config experiment=contrast
```

The system automatically sets up the `StimulusContrastDataLoader` with the specified parameters.

### FFCV Data Loader

For optimal performance during training use FFCV:

```python
from dynvision.data.ffcv_dataloader import get_ffcv_dataloader

# Create FFCV data loader
loader = get_ffcv_dataloader(
    path='data/processed/cifar100/train_all/train.beton',
    n_timesteps=20,
    resolution=32,
    normalize=([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    batch_size=128,
)
```

You can toggle the use of ffcv for training with the config `use_ffcv=False`.

## Handling Large Datasets

DynVision implements several optimizations for large datasets:

### Memory-Efficient Processing

For large datasets like ImageNet:

```python
# In config_data.yaml
mounted_datasets:
    - imagenet
```

This configuration tells DynVision to use the dataset from its mounted location (`/imagenet/) rather than copying it.

### Automatic Optimization

When working with large datasets, DynVision automatically adjusts parameters:

```python
# For large datasets (high resolution or many timesteps)
optimized_params = DatasetParams(
    batch_size=min(batch_size, MAX_BATCH_SIZE),    # Reduce batch size
    batches_ahead=min(batches_ahead, 2),           # Reduce prefetching
    order=OrderOption.QUASI_RANDOM,                # Better memory efficiency
    os_cache=False,                                # Disable OS cache
    dtype=torch.float16,                           # Use mixed precision
    num_workers=min(num_workers, MAX_WORKERS_LARGE_DATASET)  # Limit workers
)
```

A large dataset is detected by multiple criteria, that are set in the script `data/ffcv_dataloader.py`:
```python
MAX_RESOLUTION = 112
MAX_TIMESTEPS = 20
```

1. Image Resolution: Datasets with resolution > 112x112 pixels
2. Temporal Dimension: Sequences with > 20 timesteps

These criteria trigger automatic parameter adjustments to ensure efficient processing.

## Customization of Data Processing

While most data processing is automated through the workflow system, you may need to customize certain aspects for your specific research needs. This section details the key points of customization:

### Adding New Datasets

1. Prepare your dataset in the standard format:
   ```
   data/raw/your_dataset/
   ├── train/
   │   ├── class_1/
   │   │   ├── image1.ext
   │   │   └── image2.ext
   │   └── class_2/
   │       ├── image1.ext
   │       └── image2.ext
   └── test/
       └── [same structure as train]
   ```

2. Configure dataset parameters in `config_data.yaml`:
   ```yaml
   # Set resolution for efficient preprocessing with ffcv
   data_resolution:
       your_dataset: 64  # desired image size
   
   # Define dataset statistics
   data_statistics:
       your_dataset:
           mean: [0.5, 0.5, 0.5]  # channel-wise mean
           std: [0.25, 0.25, 0.25]  # channel-wise std
   ```

### Adding New Data Groups

1. Define class groups in `config_data.yaml`:
   ```yaml
   data_groups:
       your_dataset:
           group_name:
               - class_id_1  # class index or name
               - class_id_2
               # Add more classes as needed
   ```

2. The workflow will automatically:
   - Create appropriate symlinks
   - Generate group-specific target transforms
   - Set up data loaders

### Adding New Data Transforms

1. Add your transform to `dynvision/data/transforms.py`:
   ```python
   # In transform_presets dictionary
   'your_transform': [
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       # Add your custom transforms
   ]
   ```

### Adding New Experiments

1. (Optional) Create a custom data loader in `dynvision/data/dataloader.py`
   if needed for specialized data presentation.

2. Define the experiment in `config_experiments.yaml`:
   ```yaml
   experiment_config:
       your_experiment:
           status: trained
           data_loader: YourDataLoader
           parameter: param2
           data_args:
               param1: value1
               param2: [0.1, 0.2, 0.3]  # parameter range
   ```

The workflow system will automatically integrate your customizations into the pipeline, maintaining consistency with existing functionality.


## Best Practices

### Memory Management

1. **Store large datasets externally**: Use `mounted_datasets` for large datasets
2. **Use FFCV**: Enable `use_ffcv: True` for efficient data loading
3. **Adjust batch size**: Use smaller batches for large models/datasets
4. **Use mixed precision**: Set `dtype=torch.float16` for data loading

### Performance Optimization

1. **Optimize worker count**: Set `num_workers` based on CPU cores (typically CPU cores / 2)
2. **Enable pin memory**: Use `pin_memory=True` for faster GPU transfer
3. **Adjust prefetch factor**: Use `prefetch_factor=2` for balanced loading
4. **Use channels-last memory format**: For GPU optimization

### Data Organization

1. **Use symbolic links**: Organize data with symlinks instead of duplicating
2. **Keep raw data intact**: Don't modify original datasets
3. **Create appropriate data groups**: Organize categories that make sense together
4. **Use consistent naming**: Follow the established naming pattern

## Troubleshooting

### Data Loading Errors

If you encounter errors loading data:

1. Check if the dataset exists in the expected location
2. Verify that symlinks are correctly created
3. Ensure transforms are appropriate for the dataset
4. Check for sufficient disk space and memory

### FFCV Issues

If FFCV data loading fails:

1. Verify FFCV is correctly installed with required dependencies
2. Check that `.beton` files exist and are not corrupted
3. Try reducing the number of workers
4. Set `use_ffcv: False` temporarily to isolate the issue

## Next Steps

- Learn about temporal data presentation in [Temporal Data Presentation Guide](temporal-data-presentation.md)
- Explore workflow management in [Workflow Guide](workflows.md)
- See configuration options in [Configuration Reference](../reference/configuration.md)
