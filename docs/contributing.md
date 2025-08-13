# Contributing to DynVision

Thank you for your interest in contributing to DynVision! This document provides guidelines and workflows for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Tracking](#issue-tracking)

## Code of Conduct

DynVision adopts the Contributor Covenant Code of Conduct. By participating in this project, you agree to abide by its terms. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A GitHub account

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/dynvision.git
   cd dynvision
   ```
3. Add the original repository as a remote:
   ```bash
   git remote add upstream https://github.com/original-owner/dynvision.git
   ```
4. Create a virtual environment:
   ```bash
   conda create -n dynvision-dev python=3.12
   conda activate dynvision-dev
   ```
5. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

This will install all development dependencies along with DynVision itself.

## Development Workflow

1. Ensure your master branch is up-to-date:
   ```bash
   git checkout master
   git pull upstream master
   ```
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes, adhering to the [Coding Standards](#coding-standards)
4. Write or update tests as needed
5. Run tests locally:
   ```bash
   pytest
   ```
6. Update documentation as needed
7. Commit your changes with a clear commit message:
   ```bash
   git commit -m "Add feature: brief description of changes"
   ```
8. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
9. Open a pull request from your fork to the main repository

## Pull Request Process

1. Ensure your PR addresses a specific issue. If no issue exists, create one first.
2. Include a clear description of the changes and their purpose
3. Update relevant documentation
4. Ensure all tests pass
5. Request a review from at least one maintainer
6. Address any feedback or requested changes
7. Once approved, your PR will be merged by a maintainer

## Coding Standards

DynVision follows these coding standards:

- **PEP 8** for Python code style
- **Type Hints** for function signatures
- **Docstrings** following the NumPy docstring format
- **Imports** organized in the following order: standard library, third-party packages, local modules
- **Line Length** limited to 88 characters (using Black formatter)

We use several tools to enforce these standards:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for style guide enforcement
- **mypy** for type checking

You can run these tools locally:

```bash
# Format code
black dynvision tests

# Sort imports
isort dynvision tests

# Check code style
flake8 dynvision tests

# Type checking
mypy dynvision
```

<!-- ## Testing

DynVision uses pytest for testing. Tests are located in the `tests/` directory, mirroring the structure of the `dynvision/` package.

When adding new features, please include appropriate tests:
- **Unit tests** for individual functions and classes
- **Integration tests** for interactions between components
- **End-to-end tests** for full workflows when appropriate

Run tests locally with:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/path/to/test_file.py

# Run with coverage report
pytest --cov=dynvision tests/
``` -->

## Documentation

Good documentation is crucial for the usability of DynVision. When contributing, please:

1. Update or add docstrings to all public functions, classes, and methods
2. Update relevant user guides, tutorials, and reference documentation
3. Add examples for new features
4. Ensure documentation builds correctly

The documentation follows the "Diátaxis" system with four types of documentation:
- **Tutorials**: Learning-oriented guides for beginners
- **How-to Guides**: Task-oriented guides for specific problems
- **Reference**: Information-oriented technical descriptions
- **Explanation**: Understanding-oriented conceptual discussions

## Issue Tracking

We use GitHub Issues to track bugs, feature requests, and other project tasks.

### Reporting Bugs

When reporting bugs, please include:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- System information (OS, Python version, PyTorch version, etc.)
- Any relevant logs or screenshots

### Feature Requests

When suggesting features, please include:
- A clear, descriptive title
- Detailed description of the proposed feature
- Rationale: why this feature would be useful
- Example use cases
- Any references to similar implementations, if applicable

## Adding New Models or Components

DynVision is designed to be modular and extensible. If you'd like to add a new model or component:

1. Follow the existing patterns in similar modules
2. Ensure proper integration with the rest of the codebase
3. Add comprehensive documentation
4. Include tests that verify functionality
5. Update relevant configuration files if needed

### Model Components Checklist

When adding new model components:
- [ ] Implement the component following the existing architecture
- [ ] Add type hints and docstrings
- [ ] Write unit tests
- [ ] Document the component in the API reference
- [ ] Add examples in tutorials or how-to guides
- [ ] Update the model zoo if applicable

Thank you for contributing to DynVision!
