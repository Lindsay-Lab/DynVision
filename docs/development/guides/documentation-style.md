# DynVision Documentation Style Guidelines

These guidelines establish the documentation standards for the DynVision project. All documentation should follow these conventions to ensure consistency, clarity, and usefulness for the project's users.

## Documentation Framework

DynVision documentation follows the **Diátaxis framework** (https://diataxis.fr/), which organizes documentation into four distinct types:

1. **Tutorials**: Learning-oriented guides that help newcomers get started
2. **How-to Guides**: Task-oriented instructions for solving specific problems
3. **Reference**: Information-oriented technical descriptions
4. **Explanation**: Understanding-oriented conceptual discussions

Each type serves a different purpose and requires a different writing approach.

## General Writing Guidelines

### Voice and Tone

- Use **active voice** whenever possible
- Maintain a **professional but approachable** tone
- Address the reader directly using "you" rather than "the user"
- For tutorials, use first-person plural ("we") to guide the reader through examples
- For reference documentation, use a neutral, descriptive tone

### Structure and Organization

- Begin each document with a clear **introduction** stating its purpose
- Use **hierarchical headings** (H1, H2, H3, etc.) to organize content
- Keep paragraphs **short and focused** (3-5 sentences)
- Use **bullet points** and **numbered lists** for clarity when appropriate
- Include a **conclusion or summary** for longer documents
- Add **cross-references** to related documentation where helpful

### Language Conventions

- Use **American English** spelling and grammar
- Write in **present tense** whenever possible
- Avoid jargon, or define it when first used
- Use **consistent terminology** throughout all documentation
- Prefer **simple language** over complex alternatives
- Keep sentences **concise** (aim for 15-25 words)

## Document-Specific Guidelines

### 1. Tutorials

Tutorials should:

- Begin with clear learning objectives
- Present a complete, working example
- Progress step-by-step in a logical sequence
- Include complete code examples that work when copied exactly
- Explain what the reader is doing and why
- End with a summary of what was learned and suggested next steps

Structure:
```
# Title (What the reader will learn)
## Introduction
  - Prerequisites
  - What will be accomplished
## Step 1: [Clear action statement]
  - Explanation
  - Code example
  - Expected outcome
## Step 2: [Clear action statement]
  ...
## Summary
  - What was learned
  - Next steps
```

### 2. How-to Guides

How-to guides should:

- Focus on solving a specific problem
- Provide a clear sequence of steps
- Address practical real-world use cases
- Be concise and direct
- Assume basic knowledge of DynVision
- Include troubleshooting tips where appropriate

Structure:
```
# How to [Accomplish Specific Task]
## Overview
  - Problem definition
  - Expected outcome
## Prerequisites
  - Required setup/knowledge
## Steps
  1. [Clear instruction]
     - Code example
     - Explanation

  2. [Clear instruction]
     ...
## Common Issues and Solutions
## Related Resources
```

### 3. Reference Documentation

Reference documentation should:

- Describe components accurately and completely
- Use consistent formatting for parameters, return values, and examples
- Include type information for all parameters and return values
- Provide brief examples showing common usage
- Document exceptions and edge cases
- Maintain alphabetical ordering where applicable

Structure for module/class references:
```
# [Module/Class Name]

## Description
  - Purpose and scope
  - Key features

## Class/Function Signatures
  - Parameters with types and descriptions
  - Return values with types and descriptions
  - Exceptions

## Examples
  - Minimal examples showing common usage

## Notes
  - Implementation details, constraints, or caveats
```

### 4. Explanation

Explanation documents should:

- Provide context and background information
- Explain concepts, design decisions, and trade-offs
- Connect ideas to broader principles or research
- Use diagrams and visualizations where helpful
- Avoid tutorial-like instructions
- Include references to research papers or external resources

Structure:
```
# [Concept Name]
## Introduction
  - High-level overview
  - Importance and context
## Background
  - Theoretical foundations
  - Related concepts
## Implementation in DynVision
  - Design decisions
  - Advantages and limitations
## Advanced Topics
  - Deeper considerations
  - Research connections
## Conclusion
## References
```

## Code Example Guidelines

All code examples should:

- Be **complete** and **runnable** whenever possible
- Include **comments** explaining key concepts or non-obvious operations
- Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python
- Use **descriptive variable names** that reflect their purpose
- Include **error handling** when appropriate
- Show **expected output** where helpful
- Avoid **unnecessary complexity**

Code block format:
```python
import torch
from dynvision.models import DyRCNNx4

# Create a 4-layer RCNN with recurrent connections
model = DyRCNNx4(
    n_classes=10,              # Number of output classes
    input_dims=(20, 3, 64, 64),  # (timesteps, channels, height, width)
    recurrence_type="full",    # Full recurrent connectivity
    dt=2,                      # Integration time step (ms)
    tau=8                      # Neural time constant (ms)
)

# Forward pass with a batch of inputs
batch = torch.randn(1, 20, 3, 64, 64)  # (batch, timesteps, channels, height, width)
outputs = model(batch)

print(f"Output shape: {outputs.shape}")  # Expected output shape: [1, 20, 10]
```

## File Organization

- Each documentation file should be a Markdown (`.md`) file
- Name files using **lowercase kebab-case** (e.g., `getting-started.md`)
- Group related files in appropriate directories:
  - `/docs/tutorials/`
  - `/docs/user-guide/`
  - `/docs/reference/`
  - `/docs/explanation/`
- Use `index.md` for directory overview pages
- Place shared assets in `/docs/assets/`

## Visual Elements

### Diagrams and Images

- Use **SVG format** for diagrams when possible
- Maintain consistent **style and colors** across diagrams
- Include **alt text** for all images
- Keep file sizes **reasonable** (optimize when necessary)
- Use the following syntax for images:
  ```markdown
  ![Alt text description](../assets/image-name.png "Optional title")
  ```

### Tables

- Use tables for **structured data** and comparisons
- Provide **headers** for all columns
- Align columns appropriately (left for text, right for numbers)
- Example:
  ```markdown
  | Parameter | Type | Default | Description |
  |-----------|------|---------|-------------|
  | n_classes | int  | 10      | Number of output classes |
  | dt        | float| 2.0     | Integration time step (ms) |
  ```

## Mathematical Notation

- Use **LaTeX syntax** for mathematical formulas
- Use inline math for simple expressions: `$\tau \frac{dx}{dt}$`
- Use display math for complex formulas:
  ```markdown
  $$\tau \frac{dx}{dt} = -x + \Phi[f(t, r_n, r_{n-1})]$$
  ```
- Define variables on first use
- Maintain consistent notation throughout all documents

## Version-Specific Documentation

- Clearly mark version-specific features with a note:
  ```markdown
  > **Note:** This feature is available in DynVision 1.2.0 and later.
  ```
- Use admonitions for deprecated features:
  ```markdown
  !!! warning "Deprecated"
      This feature is deprecated in version 1.3.0 and will be removed in 2.0.0.
  ```
- Maintain documentation for current and previous major versions

## API Documentation

- Generate API documentation from docstrings
- Follow NumPy docstring format:
  ```python
  def function_name(param1, param2):
      """Short description of function purpose.
      
      Extended description with more details if needed.
      
      Parameters
      ----------
      param1 : type
          Description of param1
      param2 : type
          Description of param2
          
      Returns
      -------
      type
          Description of return value
          
      Examples
      --------
      >>> function_name(1, 2)
      3
      """
  ```

## Cross-Referencing

- Use relative links for cross-references within the documentation
- Link to API reference when mentioning classes, methods, or functions
- Use descriptive link text rather than "click here" or "this link"
- Example:
  ```markdown
  For more details, see the [Recurrence Types](../reference/recurrence-types.md) reference.
  ```

## TODOs and Placeholders

- When documentation is incomplete:
  - Mark clearly with `TODO: description of what needs to be added`
  - Provide at least basic information even for incomplete sections
  - For missing images or diagrams, add a placeholder note:
    ```markdown
    [Placeholder for diagram showing the data processing pipeline]
    ```
- When I realize missing features or inconsistencies in the current codebase:
  - I don't hallucinate features or relationships, but instead add a corresponding comment `TODO: suggestion of what could be changed`
  - When reasonable, I ask for additional information or access to additional files
  - I reflect on the correct usage of the involved packages and request clarification where necessary

## Review Process

Before submitting documentation for review, ensure:

- All code examples have been tested and work
- All links are valid and point to correct destinations
- Spelling and grammar have been checked
- Formatting is consistent and renders correctly
- Information is accurate and up-to-date

## Example Documentation File

```markdown
# Creating Custom Models

This guide explains how to create custom neural network architectures with DynVision's components and base classes.

## Overview

Creating a custom model in DynVision typically involves these steps:

1. Inheriting from the appropriate base class
2. Defining the model architecture
3. Implementing required methods
4. Customizing parameters and behaviors

## Base Classes

DynVision provides several base classes for model creation:

1. **UtilityBase**: Core functionality for neural network models
2. **LightningBase**: Integration with PyTorch Lightning for training and evaluation
3. **DyRCNN**: Base class specifically for recurrent convolutional networks

For most custom models, you'll want to inherit from `LightningBase` or `DyRCNN`.

## Simple Custom Model Example

Let's create a simple 2-layer recurrent model:

```python
import torch
import torch.nn as nn
from dynvision.model_components import RecurrentConnectedConv2d, EulerStep
from dynvision.model_components import LightningBase

class SimpleRCNN(LightningBase):
    def __init__(
        self, 
        n_classes=10, 
        input_dims=(20, 3, 32, 32),
        recurrence_type="self",
        **kwargs
    ):
        super().__init__(
            n_classes=n_classes, 
            input_dims=input_dims,
            recurrence_type=recurrence_type,
            **kwargs
        )
        self._define_architecture()
    
    def _define_architecture(self):
        """Define the model architecture."""
        # Implementation details...
```

## Using the Custom Model

You can now use your custom model just like built-in models:

```python
from my_models import SimpleRCNN

# Create model
model = SimpleRCNN(
    n_classes=10,
    input_dims=(20, 3, 32, 32),
    recurrence_type="full",
    dt=2,
    tau=8
)
```

## Best Practices

When creating custom models, follow these best practices:

1. **Proper Initialization**: Always implement proper parameter initialization
2. **Reset Method**: Implement a comprehensive `reset()` method
3. **Setup Method**: Implement a `setup()` method for initialization before training

## Related Resources

- [Base Classes Reference](../../reference/model-components.md#base-classes)
- [Recurrence Types](../../reference/recurrence-types.md)
- [Example Models](../../tutorial/custom-model.md)
```