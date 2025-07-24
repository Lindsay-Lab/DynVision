# DynVision Toolbox: Responsibility Distribution Analysis

## Current State Overview

### Parameter Handling Flow
```
Config Files (YAML)
├── config_defaults.yaml (lowest priority)
├── config_data.yaml
├── config_experiments.yaml  
└── config_workflow.yaml (highest priority)
    ↓
Snakemake Rules
├── CLI argument overrides
├── Parameter sweep expansion
└── Rule-specific resource allocation (in executor config)
    ↓
CLI Scripts (init_model.py, train_model.py, test_model.py)
├── argparse CLI parsing (via pydantic)
└── Direct parameter passing
    ↓
Pydantic Parameter Classes
├── BaseParams, ModelParams, TrainerParams, DataParams
├── Validation and type conversion
├── Derived parameter computation
└── Component separation
    ↓
Target Systems
├── PyTorch Lightning (Trainer, LightningModule)
├── FFCV Dataloaders
├── Models (DynVision architectures)
└── Logging/Monitoring (WandB, etc.)
```

### Current Responsibility Distribution

#### **dynvision/project_paths.py**
- **Responsibilities:**
  - Environment detection (cluster vs local)
  - Path configuration for all project directories
  - Environment variable setting (WANDB_DIR)
  - Hardcoded project/user configuration
- **Dependencies:** 
  - OS environment, pathlib
- **Issues:** 
  - Hardcoded values, mixed concerns (paths + env detection + env vars)

#### **dynvision/configs/ (YAML files)**
- **Responsibilities:**
  - Hierarchical parameter defaults
  - Debug mode parameter alternatives
  - All parameter types mixed together
- **Dependencies:** 
  - YAML, manual hierarchy management
- **Issues:** 
  - Flat structure mixes different parameter types, manual precedence

#### **dynvision/workflow/ (Snakemake)**
- **Responsibilities:**
  - Parameter expansion for sweeps
  - CLI argument override handling  
  - Resource allocation (GPU/CPU/distributed)
  - Executor script coordination
  - Environment variable setup for distributed training
- **Dependencies:** 
  - Snakemake, shell commands, cluster systems
- **Issues:** 
  - Parameter logic mixed with workflow logic

#### **dynvision/runtime/ (CLI Scripts)**
- **Responsibilities:**
  - Basic CLI argument parsing
  - Parameter passing to components
  - Script-specific logic
- **Dependencies:** 
  - argparse, pydantic classes
- **Issues:** 
  - Minimal parameter handling, relies heavily on downstream validation

#### **Pydantic Parameter Classes**
- **Responsibilities:**
  - Parameter validation and type conversion
  - Computed/derived parameter calculation
  - Component-specific parameter separation
  - CLI argument processing (advanced)
- **Dependencies:** 
  - Pydantic, typing system
- **Issues:** 
  - Doing too much - validation + computation + parsing

#### **dynvision/model_components/lightning_base.py**
- **Responsibilities:**
  - Dtype consistency checking and casting
  - Parameter initialization from config
  - Device coordination with PyTorch Lightning
  - Custom logging setup
- **Dependencies:** 
  - PyTorch, PyTorch Lightning
- **Issues:** 
  - Redundant dtype handling, logging scattered

#### **Individual Modules (data/, models/, etc.)**
- **Responsibilities:**
  - Module-specific logging creation (`logging.getLogger(__name__)`)
  - Component-specific parameter handling
  - Local configuration needs
- **Dependencies:** 
  - Standard logging, component-specific dependencies
- **Issues:** 
  - No centralized logging control, scattered configuration

#### **External Dependencies**
- **PyTorch Lightning:** Device management, training coordination, some dtype handling
- **FFCV:** Data loading optimization, simple distributed boolean flag
- **Snakemake:** Workflow orchestration, parameter expansion, resource management
- **WandB:** Logging coordination via environment variables

---

## Ideal Responsibility Distribution

### Core Principles
1. **Single Responsibility Principle:** Each module should have one clear purpose
2. **Centralized Configuration:** Common concerns handled in one place
3. **Clear Dependency Flow:** Explicit parameter flow without circular dependencies
4. **Environment Abstraction:** Environment-specific logic isolated and configurable

### Proposed Module Structure

#### **1. dynvision/config/ (New - Configuration Management)**
```
dynvision/config/
├── __init__.py
├── environment.py      # Environment detection and adaptation
├── paths.py           # Path configuration and management  
├── logging_config.py  # Centralized logging setup
├── base_params.py     # Base parameter handling (current BaseParams)
└── config_loader.py   # Hierarchical config loading and merging
```

**Responsibilities:**
- **environment.py:** Environment detection, cluster vs local logic, environment variable management
- **paths.py:** Path configuration based on environment, no hardcoded values
- **logging_config.py:** Global logging configuration, level management, format standardization
- **base_params.py:** Core parameter validation, type conversion, CLI parsing
- **config_loader.py:** YAML hierarchy loading, precedence rules, debug mode switching

**Dependencies:** `pathlib`, `logging`, `pydantic`, `yaml`, `os`

#### **2. dynvision/params/ (New - Parameter Classes)**
```
dynvision/params/
├── __init__.py
├── base.py          # BaseParams class
├── model.py         # ModelParams  
├── trainer.py       # TrainerParams
├── data.py          # DataParams
├── workflow.py      # WorkflowParams (Snakemake-specific)
└── composite.py     # TrainConfig, TestConfig, InitConfig
```

**Responsibilities:**
- Pure parameter validation and type safety
- Computed property calculation
- Component-specific parameter grouping
- No CLI parsing or file I/O

**Dependencies:** `pydantic`, `typing`

#### **3. dynvision/runtime/ (Enhanced - Script Orchestration)**
```
dynvision/runtime/
├── __init__.py
├── cli_parser.py    # Advanced CLI argument parsing
├── orchestrator.py  # Script coordination and setup
├── init_model.py    # Model initialization script
├── train_model.py   # Training script  
└── test_model.py    # Testing script
```

**Responsibilities:**
- **cli_parser.py:** Advanced CLI parsing, alias resolution, unknown argument handling
- **orchestrator.py:** Script setup, configuration assembly, component coordination
- Scripts focused on their core logic, minimal parameter handling

**Dependencies:** `argparse`, `dynvision.config`, `dynvision.params`

#### **4. dynvision/model_components/lightning_base.py (Simplified)**
**Responsibilities:**
- Model architecture definition
- PyTorch Lightning integration
- Training/validation/test step logic
- Model-specific utilities

**Dependencies:** `pytorch_lightning`, `torch`, `dynvision.params`

**Removed Responsibilities:** Dtype checking (→ config), logging setup (→ config), parameter validation (→ params)

#### **5. dynvision/workflow/ (Simplified - Pure Workflow)**
**Responsibilities:**
- Snakemake rule definitions  
- Workflow DAG coordination
- Resource allocation coordination
- Parameter sweep expansion (using config module)

**Dependencies:** `snakemake`, `dynvision.config`

**Removed Responsibilities:** Parameter precedence logic (→ config), environment detection (→ config)

#### **6. dynvision/data/ (Enhanced)**
**Responsibilities:**
- Data loading and preprocessing
- FFCV integration
- Dataset management
- Data-specific parameter validation

**Dependencies:** `ffcv`, `pytorch`, `dynvision.params.data`

### Centralized Cross-Cutting Concerns

#### **Logging Management**
```python
# dynvision/config/logging_config.py
class LoggingManager:
    @classmethod
    def setup_global_logging(cls, level: str, format: str = None):
        """Set up logging for entire DynVision toolbox."""
        
    @classmethod  
    def get_script_logger(cls, name: str):
        """Get configured logger for specific module."""
```

**Usage:** All modules use `LoggingManager.get_script_logger(__name__)`

#### **Environment Management**
```python
# dynvision/config/environment.py  
class EnvironmentManager:
    @classmethod
    def detect_environment(cls) -> EnvironmentType:
        """Detect current environment (local/cluster/docker)."""
        
    @classmethod
    def get_env_config(cls) -> EnvironmentConfig:
        """Get environment-specific configuration."""
        
    @classmethod
    def setup_env_variables(cls, config: EnvironmentConfig):
        """Set up environment variables for current context."""
```

#### **Path Management**
```python
# dynvision/config/paths.py
class PathManager:
    def __init__(self, env_config: EnvironmentConfig, project_config: ProjectConfig):
        """Initialize paths based on environment and project configuration."""
        
    def get_paths(self) -> ProjectPaths:
        """Get all project paths for current environment."""
```

#### **Device and Dtype Coordination**
```python
# dynvision/config/device_config.py
class DeviceManager:
    @classmethod
    def setup_device_config(cls, trainer_params: TrainerParams) -> DeviceConfig:
        """Coordinate device settings across all components."""
        
    @classmethod
    def ensure_dtype_consistency(cls, model, dataloader, trainer):
        """Ensure consistent dtypes across all components."""
```

### Benefits of Ideal Distribution

1. **Clear Separation of Concerns:** Each module has a single, well-defined responsibility
2. **Centralized Configuration:** Common concerns (logging, paths, environment) handled centrally
3. **Reduced Redundancy:** No duplicate dtype checking, parameter validation, or logging setup
4. **Better Testability:** Each component can be tested independently
5. **Easier Maintenance:** Changes to parameter handling, logging, or environment detection happen in one place
6. **Consistent Behavior:** Same parameter parsing, logging format, and environment detection across all scripts
7. **Flexible Extension:** Easy to add new parameter types, environments, or configuration sources

### Migration Strategy

1. **Phase 1:** Create new config module with current functionality
2. **Phase 2:** Migrate parameter classes to new structure  
3. **Phase 3:** Update scripts to use new orchestration pattern
4. **Phase 4:** Simplify existing modules by removing duplicated responsibilities
5. **Phase 5:** Add new centralized features (global logging, device coordination)

This structure maintains all current functionality while providing a much cleaner, more maintainable architecture that follows software engineering best practices.