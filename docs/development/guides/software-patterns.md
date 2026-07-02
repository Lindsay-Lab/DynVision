# Software Design Patterns for Scientific Computing and Neural Network Frameworks

## Introduction

This reference guide presents software design patterns particularly relevant to scientific computing and neural network frameworks like DynVision. Each pattern includes a brief description, when to use it, and a minimal Python example demonstrating its implementation. Patterns are organized into architectural, creational, structural, and behavioral categories.

## Architectural Patterns

Architectural patterns define the high-level organization of software systems, addressing concerns related to overall structure, component interactions, and quality attributes.

### 1. Layered Architecture

**Description**: Organizes components into horizontal layers where each layer provides services to the layer above it and uses services from the layer below.

**When to use**:

- For complex systems that benefit from separation of concerns
- When different aspects of functionality need to evolve independently
- For systems requiring clear boundaries between components (e.g., presentation, business logic, data)

**Example**:
```python
# DynVision uses layered architecture with clear separation

class DataLayer:
    def load_data(self, dataset_path):
        # Handle data loading, preprocessing
        pass

class ModelLayer:
    def __init__(self, data_layer):
        self.data_layer = data_layer
        
    def create_model(self, architecture_params):
        # Create neural network using data from data layer
        pass

class TrainingLayer:
    def __init__(self, model_layer):
        self.model_layer = model_layer
        
    def train(self, training_params):
        # Train the model with specific parameters
        pass
        
class VisualizationLayer:
    def __init__(self, model_layer):
        self.model_layer = model_layer
        
    def visualize_activations(self, input_data):
        # Generate visualizations of model activations
        pass
```

### 2. Pipeline Architecture

**Description**: Organizes the system as a series of processing stages where the output of one stage is the input to the next.

**When to use**:

- For data processing workflows with well-defined stages
- When operations need to be chained sequentially
- For parallel processing of multiple data streams

**Example**:
```python
class Pipeline:
    def __init__(self, stages=None):
        self.stages = stages or []
        
    def add_stage(self, stage):
        self.stages.append(stage)
        
    def process(self, data):
        result = data
        for stage in self.stages:
            result = stage.process(result)
        return result
        
# Usage example for neural data processing
pipeline = Pipeline([
    DataLoadingStage(),
    PreprocessingStage(),
    RecurrentNetworkStage(),
    AnalysisStage(),
    VisualizationStage()
])
result = pipeline.process(input_data)
```

### 3. Domain-Driven Design

**Description**: Focuses on modeling the domain and defining bounded contexts that encapsulate domain logic.

**When to use**:

- For complex domains with rich business rules and constraints
- When collaborating with domain experts
- When building systems that need to align closely with real-world concepts

**Example**:
```python
# Domain model for neural modeling
class NeuronModel:
    def __init__(self, time_constant, resting_potential):
        self.time_constant = time_constant
        self.resting_potential = resting_potential
        self.membrane_potential = resting_potential
        self.inputs = []
        
    def add_input(self, input_connection):
        self.inputs.append(input_connection)
        
    def update(self, dt):
        # Update membrane potential based on inputs and time constant
        input_current = sum(inp.get_current() for inp in self.inputs)
        d_v = (-self.membrane_potential + self.resting_potential + input_current) / self.time_constant
        self.membrane_potential += d_v * dt
        return self.membrane_potential

# Service layer that uses the domain model
class NeuralSimulationService:
    def __init__(self, neuron_repository):
        self.neuron_repository = neuron_repository
        
    def run_simulation(self, simulation_params):
        neurons = self.neuron_repository.get_neurons()
        # Run simulation with domain objects
        pass
```

### 4. Event-Driven Architecture

**Description**: Components communicate through events, allowing for loose coupling and flexibility.

**When to use**:

- For systems with asynchronous behavior
- When components need to react to changes in state
- For building responsive, real-time systems

**Example**:
```python
class EventBus:
    def __init__(self):
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event_type, data):
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(data)
                
# Usage in neural network training
event_bus = EventBus()

# Log loss values
event_bus.subscribe('epoch_completed', lambda data: print(f"Epoch {data['epoch']}: Loss = {data['loss']}"))

# Save checkpoints
event_bus.subscribe('epoch_completed', 
                   lambda data: save_checkpoint(data) if data['epoch'] % 10 == 0 else None)

# Early stopping
event_bus.subscribe('epoch_completed',
                   lambda data: stop_training() if data['no_improvement_count'] > 5 else None)

# During training
event_bus.publish('epoch_completed', {'epoch': 23, 'loss': 0.342, 'no_improvement_count': 2})
```

## Creational Patterns

Creational patterns deal with object creation mechanisms, encapsulating knowledge about which concrete classes the system uses.

### 1. Factory Method

**Description**: Defines an interface for creating an object, but lets subclasses decide which class to instantiate.

**When to use**:

- When a class can't anticipate the type of objects it must create
- When you want to delegate responsibility to subclasses
- For dynamic selection of implementation classes

**Example**:
```python
from abc import ABC, abstractmethod

class RecurrenceFactory(ABC):
    @abstractmethod
    def create_recurrence(self, input_shape):
        pass
        
class FullRecurrenceFactory(RecurrenceFactory):
    def create_recurrence(self, input_shape):
        return FullRecurrence(input_shape)
        
class SelfRecurrenceFactory(RecurrenceFactory):
    def create_recurrence(self, input_shape):
        return SelfRecurrence(input_shape)
        
class DepthwiseRecurrenceFactory(RecurrenceFactory):
    def create_recurrence(self, input_shape):
        return DepthwiseRecurrence(input_shape)
        
# Usage
factory_map = {
    'full': FullRecurrenceFactory(),
    'self': SelfRecurrenceFactory(),
    'depthwise': DepthwiseRecurrenceFactory()
}

def build_model(recurrence_type, input_shape):
    factory = factory_map.get(recurrence_type)
    if not factory:
        raise ValueError(f"Unknown recurrence type: {recurrence_type}")
    recurrence = factory.create_recurrence(input_shape)
    return Model(recurrence)
```

### 2. Abstract Factory

**Description**: Provides an interface for creating families of related or dependent objects without specifying their concrete classes.

**When to use**:

- When the system needs to be independent of how its products are created
- When families of related products are designed to be used together
- When you want to provide a library of products and reveal only their interfaces

**Example**:
```python
from abc import ABC, abstractmethod

# Abstract factory interface
class NeuralComponentFactory(ABC):
    @abstractmethod
    def create_activation(self):
        pass
        
    @abstractmethod
    def create_recurrence(self):
        pass
        
    @abstractmethod
    def create_pooling(self):
        pass

# Concrete factory for biologically plausible components
class BiologicalComponentFactory(NeuralComponentFactory):
    def create_activation(self):
        return SupralinearActivation(alpha=2.0)
        
    def create_recurrence(self):
        return LateralRecurrence(kernel_size=3)
        
    def create_pooling(self):
        return AdaptivePooling(time_constant=20)
        
# Concrete factory for standard ML components
class StandardComponentFactory(NeuralComponentFactory):
    def create_activation(self):
        return ReLUActivation()
        
    def create_recurrence(self):
        return ConvLSTMRecurrence()
        
    def create_pooling(self):
        return MaxPooling()
        
# Client code
class NeuralNetworkBuilder:
    def __init__(self, factory: NeuralComponentFactory):
        self.factory = factory
        
    def build_network(self):
        activation = self.factory.create_activation()
        recurrence = self.factory.create_recurrence()
        pooling = self.factory.create_pooling()
        
        return NeuralNetwork(activation, recurrence, pooling)
        
# Usage
biological_builder = NeuralNetworkBuilder(BiologicalComponentFactory())
bio_network = biological_builder.build_network()

standard_builder = NeuralNetworkBuilder(StandardComponentFactory())
standard_network = standard_builder.build_network()
```

### 3. Builder

**Description**: Separates the construction of complex objects from their representation, allowing the same construction process to create different representations.

**When to use**:

- When the construction process is complex with many optional parameters
- When different representations of an object need to be created
- To encapsulate code for construction and representation

**Example**:
```python
class RCNNModelBuilder:
    def __init__(self):
        self.reset()
        
    def reset(self, input_shape: Optional[Tuple[int, ...]] = None) :
        self.model = RCNNModel()
        
    def set_recurrence_type(self, recurrence_type):
        self.model.recurrence_type = recurrence_type
        return self
        
    def set_time_constants(self, time_constants):
        self.model.time_constants = time_constants
        return self
        
    def set_layer_sizes(self, layer_sizes):
        self.model.layer_sizes = layer_sizes
        return self
        
    def set_activation_function(self, activation):
        self.model.activation = activation
        return self
        
    def set_learning_rate(self, learning_rate):
        self.model.learning_rate = learning_rate
        return self
        
    def build(self):
        return self.model
        
# Usage
builder = RCNNModelBuilder()
model = builder.set_recurrence_type('full') \
               .set_time_constants([10, 20, 30, 40]) \
               .set_layer_sizes([64, 128, 256, 512]) \
               .set_activation_function('supralinear') \
               .set_learning_rate(0.001) \
               .build()
```

### 4. Singleton

**Description**: Ensures a class has only one instance and provides a global point of access to it.

**When to use**:

- When exactly one instance of a class is needed
- When you need centralized access to a resource
- For managing shared state or configuration

**Example**:
```python
class ConfigurationManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance.config = {}
        return cls._instance
        
    def load_config(self, config_path):
        # Load configuration from file
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get(self, key, default=None):
        return self.config.get(key, default)
        
# Usage
config = ConfigurationManager()
config.load_config('path/to/config.yaml')
learning_rate = config.get('learning_rate', 0.01)

# Later in another module
config = ConfigurationManager()  # Same instance
batch_size = config.get('batch_size', 32)
```

### 5. Prototype

**Description**: Creates new objects by copying an existing object, known as the prototype.

**When to use**:

- When creating a new object is more expensive than copying an existing one
- When objects have many possible configurations
- When the system needs to be independent of how objects are created

**Example**:
```python
import copy

class NeuralLayer:
    def __init__(self, size, activation='relu', use_bias=True):
        self.size = size
        self.activation = activation
        self.use_bias = use_bias
        self.weights = None
        self.bias = None
        
    def initialize(self, input_size):
        import numpy as np
        self.weights = np.random.randn(input_size, self.size) * 0.01
        if self.use_bias:
            self.bias = np.zeros(self.size)
            
    def clone(self):
        return copy.deepcopy(self)
        
# Usage
prototype_layer = NeuralLayer(128, activation='tanh', use_bias=True)
prototype_layer.initialize(256)

# Create copies with modifications
layer1 = prototype_layer.clone()
layer2 = prototype_layer.clone()
layer2.activation = 'sigmoid'
layer3 = prototype_layer.clone()
layer3.size = 64
layer3.initialize(256)  # Reinitialize with new size
```

## Structural Patterns

Structural patterns deal with how classes and objects are composed to form larger structures.

### 1. Adapter

**Description**: Converts the interface of a class into another interface clients expect.

**When to use**:

- When you need to use an existing class with an incompatible interface
- When you want to reuse existing functionality without modifying the source code
- When integrating with external libraries or systems

**Example**:
```python
# External library class with incompatible interface
class ExternalTensorLibrary:
    def create_tensor(self, data_array):
        # Creates tensor in specific format
        pass
        
    def tensor_operation(self, tensor1, tensor2, operation_type):
        # Performs operations in specific way
        pass
        
# Our system's expected interface
class TensorOperations:
    def create(self, data):
        pass
        
    def add(self, t1, t2):
        pass
        
    def multiply(self, t1, t2):
        pass

# Adapter to make ExternalTensorLibrary compatible with our system
class TensorLibraryAdapter(TensorOperations):
    def __init__(self, external_library):
        self.lib = external_library
        
    def create(self, data):
        # Convert our data format to external library format
        return self.lib.create_tensor(data)
        
    def add(self, t1, t2):
        return self.lib.tensor_operation(t1, t2, 'add')
        
    def multiply(self, t1, t2):
        return self.lib.tensor_operation(t1, t2, 'multiply')
        
# Usage
external_lib = ExternalTensorLibrary()
tensor_ops = TensorLibraryAdapter(external_lib)

# Now use through our expected interface
t1 = tensor_ops.create([1, 2, 3])
t2 = tensor_ops.create([4, 5, 6])
result = tensor_ops.add(t1, t2)
```

### 2. Facade

**Description**: Provides a unified interface to a set of interfaces in a subsystem.

**When to use**:

- When you need a simple interface to a complex subsystem
- When there are many dependencies between clients and implementation classes
- When you want to layer your subsystems

**Example**:
```python
# Complex subsystem classes
class DataLoader:
    def load_data(self, path):
        pass
        
class DataPreprocessor:
    def normalize(self, data):
        pass
        
    def augment(self, data):
        pass
        
class ModelTrainer:
    def train(self, model, data, epochs):
        pass
        
class ModelEvaluator:
    def evaluate(self, model, test_data):
        pass
        
class ModelSerializer:
    def save(self, model, path):
        pass
        
    def load(self, path):
        pass

# Facade providing a simplified interface
class MachineLearningFacade:
    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.serializer = ModelSerializer()
        
    def train_and_evaluate(self, data_path, model_type, epochs=10):
        # Handle the entire workflow with a simple interface
        data = self.loader.load_data(data_path)
        processed_data = self.preprocessor.normalize(data)
        augmented_data = self.preprocessor.augment(processed_data)
        
        model = self._create_model(model_type)
        self.trainer.train(model, augmented_data, epochs)
        metrics = self.evaluator.evaluate(model, processed_data['test'])
        
        self.serializer.save(model, f"models/{model_type}_model.pkl")
        return metrics
        
    def _create_model(self, model_type):
        # Factory method to create appropriate model
        pass
        
# Usage
ml_facade = MachineLearningFacade()
results = ml_facade.train_and_evaluate("data/experiment_1", "rcnn", epochs=50)
print(f"Accuracy: {results['accuracy']}")
```

### 3. Composite

**Description**: Composes objects into tree structures to represent part-whole hierarchies.

**When to use**:

- When you want to represent part-whole hierarchies of objects
- When clients should be able to treat individual objects and compositions uniformly
- For tree-like structures where components can contain other components

**Example**:
```python
from abc import ABC, abstractmethod

# Component interface
class NeuralComponent(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass
        
    @abstractmethod
    def parameters(self):
        pass

# Leaf nodes
class Convolution(NeuralComponent):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weights = None
        self.bias = None
        
    def forward(self, inputs):
        # Implement convolution operation
        pass
        
    def parameters(self):
        return {'weights': self.weights, 'bias': self.bias}
        
class Activation(NeuralComponent):
    def __init__(self, function_type):
        self.function_type = function_type
        
    def forward(self, inputs):
        # Apply activation function
        pass
        
    def parameters(self):
        return {}  # No trainable parameters

# Composite component
class Sequential(NeuralComponent):
    def __init__(self, components=None):
        self.components = components or []
        
    def add(self, component):
        self.components.append(component)
        
    def forward(self, inputs):
        result = inputs
        for component in self.components:
            result = component.forward(result)
        return result
        
    def parameters(self):
        params = {}
        for i, component in enumerate(self.components):
            component_params = component.parameters()
            for key, value in component_params.items():
                params[f"component_{i}_{key}"] = value
        return params

# Usage
model = Sequential([
    Convolution(3, 32, 3),
    Activation('relu'),
    Convolution(32, 64, 3),
    Activation('relu')
])

# Can add more components
model.add(Convolution(64, 128, 3))
model.add(Activation('relu'))

# Use uniformly
output = model.forward(input_data)
params = model.parameters()
```

### 4. Decorator

**Description**: Attaches additional responsibilities to an object dynamically.

**When to use**:

- When you need to add responsibilities to objects dynamically and transparently
- When extending functionality by subclassing is impractical
- When you want to keep new functionality separate

**Example**:
```python
from abc import ABC, abstractmethod

# Component interface
class DataLoader(ABC):
    @abstractmethod
    def load_batch(self, batch_idx):
        pass

# Concrete component
class BasicDataLoader(DataLoader):
    def __init__(self, dataset_path, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        
    def load_batch(self, batch_idx):
        # Load data from disk
        start = batch_idx * self.batch_size
        end = start + self.batch_size
        # Simplified implementation
        return {'data': f"Data from {start} to {end}"}

# Base decorator
class DataLoaderDecorator(DataLoader):
    def __init__(self, wrapped_loader):
        self.wrapped_loader = wrapped_loader
        
    def load_batch(self, batch_idx):
        return self.wrapped_loader.load_batch(batch_idx)

# Concrete decorators
class CachingDataLoader(DataLoaderDecorator):
    def __init__(self, wrapped_loader, cache_size=10):
        super().__init__(wrapped_loader)
        self.cache = {}
        self.cache_size = cache_size
        
    def load_batch(self, batch_idx):
        if batch_idx in self.cache:
            print(f"Cache hit for batch {batch_idx}")
            return self.cache[batch_idx]
            
        data = self.wrapped_loader.load_batch(batch_idx)
        
        # Manage cache size
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[batch_idx] = data
        return data

class AugmentingDataLoader(DataLoaderDecorator):
    def __init__(self, wrapped_loader, augmentation_params=None):
        super().__init__(wrapped_loader)
        self.augmentation_params = augmentation_params or {}
        
    def load_batch(self, batch_idx):
        data = self.wrapped_loader.load_batch(batch_idx)
        # Apply augmentation
        augmented_data = self._augment(data)
        return augmented_data
        
    def _augment(self, data):
        # Apply various augmentations based on parameters
        # This is a simplified implementation
        data['augmented'] = True
        return data

# Usage
loader = BasicDataLoader('data/training', batch_size=32)

# Wrap with decorators
cached_loader = CachingDataLoader(loader, cache_size=10)
augmented_cached_loader = AugmentingDataLoader(cached_loader, 
                                              {'flip': True, 'rotate': 15})

# Use the decorated object
batch = augmented_cached_loader.load_batch(5)
```

### 5. Bridge

**Description**: Separates an abstraction from its implementation so that both can vary independently.

**When to use**:

- When you want to avoid a permanent binding between an abstraction and its implementation
- When both the abstraction and implementation should be extensible by subclassing
- When changes in the implementation should not impact the client code

**Example**:
```python
from abc import ABC, abstractmethod

# Implementation interface
class RecurrenceImplementation(ABC):
    @abstractmethod
    def apply_recurrence(self, current_input, previous_state):
        pass

# Concrete implementations
class FullRecurrenceImpl(RecurrenceImplementation):
    def apply_recurrence(self, current_input, previous_state):
        # Implement full recurrence
        print("Applying full recurrence")
        return current_input + previous_state * 0.5
        
class SelfRecurrenceImpl(RecurrenceImplementation):
    def apply_recurrence(self, current_input, previous_state):
        # Implement self recurrence
        print("Applying self recurrence")
        return current_input + previous_state * 0.3

# Abstraction
class RecurrentLayer(ABC):
    def __init__(self, implementation):
        self.implementation = implementation
        
    @abstractmethod
    def process(self, inputs, previous_state):
        pass

# Refined abstractions
class BasicRecurrentLayer(RecurrentLayer):
    def process(self, inputs, previous_state):
        # Basic processing with the implementation
        return self.implementation.apply_recurrence(inputs, previous_state)
        
class GatedRecurrentLayer(RecurrentLayer):
    def process(self, inputs, previous_state):
        # More complex processing with gates
        gate = self._compute_gate(inputs, previous_state)
        recurrent_output = self.implementation.apply_recurrence(inputs, previous_state)
        return gate * recurrent_output
        
    def _compute_gate(self, inputs, previous_state):
        # Simple gate computation
        return 0.8  # Simplified for demonstration

# Usage
full_recurrence = FullRecurrenceImpl()
self_recurrence = SelfRecurrenceImpl()

basic_full_layer = BasicRecurrentLayer(full_recurrence)
gated_self_layer = GatedRecurrentLayer(self_recurrence)

# Use either combination
result1 = basic_full_layer.process(inputs=1.0, previous_state=2.0)
result2 = gated_self_layer.process(inputs=1.0, previous_state=2.0)
```

## Behavioral Patterns

Behavioral patterns are concerned with algorithms and the assignment of responsibilities between objects.

### 1. Strategy

**Description**: Defines a family of algorithms, encapsulates each one, and makes them interchangeable.

**When to use**:

- When you need different variants of an algorithm
- When you want to isolate the algorithm from the code that uses it
- When you have multiple conditional statements in your code

**Example**:
```python
from abc import ABC, abstractmethod

# Strategy interface
class RecurrenceStrategy(ABC):
    @abstractmethod
    def compute_recurrence(self, inputs, hidden_state):
        pass

# Concrete strategies
class FullRecurrenceStrategy(RecurrenceStrategy):
    def compute_recurrence(self, inputs, hidden_state):
        # Implementation for full recurrence
        return f"Full recurrence: {inputs} + {hidden_state}"
        
class SelfRecurrenceStrategy(RecurrenceStrategy):
    def compute_recurrence(self, inputs, hidden_state):
        # Implementation for self recurrence
        return f"Self recurrence: {inputs} + {hidden_state}"
        
class DepthwiseRecurrenceStrategy(RecurrenceStrategy):
    def compute_recurrence(self, inputs, hidden_state):
        # Implementation for depthwise recurrence
        return f"Depthwise recurrence: {inputs} + {hidden_state}"

# Context using the strategy
class RecurrentLayer:
    def __init__(self, recurrence_strategy: RecurrenceStrategy):
        self.strategy = recurrence_strategy
        self.hidden_state = None
        
    def set_strategy(self, recurrence_strategy: RecurrenceStrategy):
        self.strategy = recurrence_strategy
        
    def forward(self, inputs):
        if self.hidden_state is None:
            # Initialize hidden state
            self.hidden_state = 0
            
        self.hidden_state = self.strategy.compute_recurrence(inputs, self.hidden_state)
        return self.hidden_state

# Usage
layer = RecurrentLayer(FullRecurrenceStrategy())
output1 = layer.forward("Input 1")

# Change strategy at runtime
layer.set_strategy(DepthwiseRecurrenceStrategy())
output2 = layer.forward("Input 2")
```

### 2. Observer

**Description**: Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

**When to use**:

- When a change to one object requires changing others, and you don't know how many objects need to change
- When an object should be able to notify other objects without making assumptions about them
- For event handling systems

**Example**:
```python
from abc import ABC, abstractmethod

# Observer interface
class TrainingObserver(ABC):
    @abstractmethod
    def update(self, metrics):
        pass

# Concrete observers
class LossPlotter(TrainingObserver):
    def update(self, metrics):
        # Plot the loss
        print(f"Plotting loss: {metrics['loss']}")
        
class CheckpointSaver(TrainingObserver):
    def __init__(self, save_path, save_frequency=10):
        self.save_path = save_path
        self.save_frequency = save_frequency
        
    def update(self, metrics):
        epoch = metrics['epoch']
        if epoch % self.save_frequency == 0:
            print(f"Saving checkpoint at epoch {epoch} to {self.save_path}")
            
class EarlyStoppingObserver(TrainingObserver):
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        
    def update(self, metrics):
        current_loss = metrics['val_loss']
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            print(f"Early stopping triggered! No improvement for {self.patience} epochs")

# Subject (Observable)
class ModelTrainer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.observers = []
        self.training = False
        
    def register_observer(self, observer):
        self.observers.append(observer)
        
    def remove_observer(self, observer):
        self.observers.remove(observer)
        
    def notify_observers(self, metrics):
        for observer in self.observers:
            observer.update(metrics)
            
    def train(self, epochs):
        self.training = True
        
        for epoch in range(epochs):
            if not self.training:
                print("Training stopped early")
                break
                
            # Simulated training loop
            train_loss = 1.0 / (epoch + 1)  # Dummy loss that decreases
            val_loss = 1.2 / (epoch + 1)    # Dummy validation loss
            
            metrics = {
                'epoch': epoch,
                'loss': train_loss,
                'val_loss': val_loss
            }
            
            # Notify all observers
            self.notify_observers(metrics)
            
            # Check if early stopping observer signaled to stop
            for observer in self.observers:
                if isinstance(observer, EarlyStoppingObserver) and observer.should_stop:
                    self.training = False
                    break

# Usage
model = "DummyModel"
data_loader = "DummyDataLoader"
trainer = ModelTrainer(model, data_loader)

# Register observers
trainer.register_observer(LossPlotter())
trainer.register_observer(CheckpointSaver(save_path="models/checkpoints", save_frequency=5))
trainer.register_observer(EarlyStoppingObserver(patience=3))

# Start training
trainer.train(epochs=20)
```

### 3. Command

**Description**: Encapsulates a request as an object, allowing you to parameterize clients with different requests, queue or log requests, and support undoable operations.

**When to use**:

- When you want to parameterize objects with operations
- When you want to queue operations, schedule their execution, or execute them remotely
- When you need to support undoable operations

**Example**:
```python
from abc import ABC, abstractmethod

# Command interface
class ModelCommand(ABC):
    @abstractmethod
    def execute(self):
        pass
        
    @abstractmethod
    def undo(self):
        pass

# Concrete commands
class TrainModelCommand(ModelCommand):
    def __init__(self, model, data_loader, epochs):
        self.model = model
        self.data_loader = data_loader
        self.epochs = epochs
        self.previous_weights = None
        
    def execute(self):
        print(f"Training model for {self.epochs} epochs")
        self.previous_weights = self.model.get_weights()  # Save current weights
        self.model.train(self.data_loader, self.epochs)
        return f"Training completed with loss: {self.model.loss}"
        
    def undo(self):
        print("Reverting to previous weights")
        self.model.set_weights(self.previous_weights)
        
class EvaluateModelCommand(ModelCommand):
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        self.results = None
        
    def execute(self):
        print("Evaluating model")
        self.results = self.model.evaluate(self.test_data)
        return f"Evaluation complete. Accuracy: {self.results['accuracy']}"
        
    def undo(self):
        # Evaluation doesn't change state, so no undo needed
        print("Nothing to undo for evaluation")
        
class SaveModelCommand(ModelCommand):
    def __init__(self, model, file_path):
        self.model = model
        self.file_path = file_path
        
    def execute(self):
        print(f"Saving model to {self.file_path}")
        self.model.save(self.file_path)
        return f"Model saved to {self.file_path}"
        
    def undo(self):
        import os
        print(f"Deleting saved model at {self.file_path}")
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

# Invoker
class ModelManager:
    def __init__(self):
        self.history = []
        
    def execute_command(self, command):
        result = command.execute()
        self.history.append(command)
        return result
        
    def undo_last_command(self):
        if self.history:
            command = self.history.pop()
            command.undo()
            return f"Undid {command.__class__.__name__}"
        return "No commands to undo"

# Usage
class DummyModel:
    def __init__(self):
        self.weights = [0, 0, 0]
        self.loss = None
        
    def get_weights(self):
        return self.weights.copy()
        
    def set_weights(self, weights):
        self.weights = weights.copy()
        
    def train(self, data_loader, epochs):
        # Simulate training
        self.weights = [w + 0.1 * epochs for w in self.weights]
        self.loss = 1.0 / (epochs + 1)
        
    def evaluate(self, test_data):
        # Simulate evaluation
        return {'accuracy': sum(self.weights) / len(self.weights)}
        
    def save(self, file_path):
        # Simulate saving
        print(f"Model would save weights {self.weights} to {file_path}")

# Client code
model = DummyModel()
manager = ModelManager()

# Execute commands
manager.execute_command(TrainModelCommand(model, "data_loader", 10))
manager.execute_command(EvaluateModelCommand(model, "test_data"))
manager.execute_command(SaveModelCommand(model, "model.h5"))

# Undo last command
manager.undo_last_command()
```

### 4. Template Method

**Description**: Defines the skeleton of an algorithm in a method, deferring some steps to subclasses.

**When to use**:

- When you want to let clients extend only particular steps of an algorithm
- When you have several classes that contain almost identical algorithms with minor variations
- To implement the invariant parts of an algorithm once and leave the variable parts to subclasses

**Example**:
```python
from abc import ABC, abstractmethod

# Abstract class with template method
class ModelTrainingPipeline(ABC):
    def train_model(self, data_path):
        """Template method defining the algorithm skeleton"""
        data = self.load_data(data_path)
        preprocessed_data = self.preprocess_data(data)
        model = self.create_model()
        trained_model = self.train(model, preprocessed_data)
        metrics = self.evaluate(trained_model, preprocessed_data['test'])
        self.save_results(trained_model, metrics)
        return trained_model, metrics
        
    @abstractmethod
    def load_data(self, data_path):
        pass
        
    @abstractmethod
    def preprocess_data(self, data):
        pass
        
    @abstractmethod
    def create_model(self):
        pass
        
    def train(self, model, data):
        """Default implementation for training"""
        print("Training model with default parameters")
        # Basic training logic
        return model
        
    def evaluate(self, model, test_data):
        """Default implementation for evaluation"""
        print("Evaluating model with default metrics")
        # Basic evaluation logic
        return {'accuracy': 0.85}
        
    def save_results(self, model, metrics):
        """Hook method with default implementation"""
        print(f"Saving model and metrics: {metrics}")
        # Default saving logic

# Concrete implementation
class CNNImageClassificationPipeline(ModelTrainingPipeline):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def load_data(self, data_path):
        print(f"Loading image data from {data_path}")
        # Specific implementation for loading image data
        return {'images': [1, 2, 3]}
        
    def preprocess_data(self, data):
        print("Preprocessing image data with normalization and augmentation")
        # Specific preprocessing for images
        return {
            'train': {'x': [1, 2], 'y': [0, 1]},
            'test': {'x': [3], 'y': [1]}
        }
        
    def create_model(self):
        print(f"Creating CNN model for {self.num_classes} classes with input shape {self.input_shape}")
        # Create CNN model
        return "CNN Model"
        
    def train(self, model, data):
        # Override default training with CNN-specific training
        print("Training CNN with data augmentation and early stopping")
        # CNN-specific training logic
        return model

# Another concrete implementation
class RNNTextClassificationPipeline(ModelTrainingPipeline):
    def __init__(self, vocab_size, num_classes):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
    def load_data(self, data_path):
        print(f"Loading text data from {data_path}")
        # Specific implementation for loading text data
        return {'texts': ["text1", "text2"]}
        
    def preprocess_data(self, data):
        print("Preprocessing text data with tokenization and padding")
        # Specific preprocessing for text
        return {
            'train': {'x': [[1, 2], [3, 4]], 'y': [0, 1]},
            'test': {'x': [[5, 6]], 'y': [1]}
        }
        
    def create_model(self):
        print(f"Creating RNN model with vocabulary size {self.vocab_size}")
        # Create RNN model
        return "RNN Model"
        
    # Use default train and evaluate methods

# Usage
cnn_pipeline = CNNImageClassificationPipeline(input_shape=(224, 224, 3), num_classes=10)
cnn_model, cnn_metrics = cnn_pipeline.train_model("/data/images")

rnn_pipeline = RNNTextClassificationPipeline(vocab_size=10000, num_classes=5)
rnn_model, rnn_metrics = rnn_pipeline.train_model("/data/texts")
```

### 5. State

**Description**: Allows an object to alter its behavior when its internal state changes.

**When to use**:

- When an object's behavior depends on its state, and it must change behavior at runtime
- When operations have large, multipart conditional statements that depend on the object's state
- To avoid duplication of state-specific code across multiple methods

**Example**:
```python
from abc import ABC, abstractmethod

# State interface
class NeuronState(ABC):
    @abstractmethod
    def update(self, neuron, input_current):
        pass
        
    @abstractmethod
    def get_description(self):
        pass

# Concrete states
class RestingState(NeuronState):
    def update(self, neuron, input_current):
        if input_current > neuron.threshold:
            neuron.membrane_potential += input_current
            return FiringState()
        else:
            # Stay in resting state
            return self
            
    def get_description(self):
        return "Neuron is at rest"
        
class FiringState(NeuronState):
    def __init__(self):
        self.duration = 0
        
    def update(self, neuron, input_current):
        self.duration += 1
        neuron.membrane_potential = neuron.spike_value
        
        if self.duration >= neuron.refractory_period:
            return RefractoryState()
        else:
            return self
            
    def get_description(self):
        return f"Neuron is firing (duration: {self.duration})"
        
class RefractoryState(NeuronState):
    def __init__(self):
        self.cool_down = 5  # How long the neuron remains in refractory state
        
    def update(self, neuron, input_current):
        self.cool_down -= 1
        neuron.membrane_potential = neuron.resting_potential
        
        if self.cool_down <= 0:
            return RestingState()
        else:
            return self
            
    def get_description(self):
        return f"Neuron is in refractory period (cool down: {self.cool_down})"

# Context
class Neuron:
    def __init__(self):
        self.state = RestingState()
        self.membrane_potential = -70.0  # mV
        self.resting_potential = -70.0   # mV
        self.threshold = -55.0           # mV
        self.spike_value = 30.0          # mV
        self.refractory_period = 3       # time steps
        
    def receive_input(self, input_current):
        self.state = self.state.update(self, input_current)
        return self.membrane_potential
        
    def get_status(self):
        return self.state.get_description()

# Usage
neuron = Neuron()
print(f"Initial state: {neuron.get_status()}")

# Simulate neuron over time
inputs = [0, 20, 0, 0, 0, 0, 15, 0, 0, 0]
for t, input_current in enumerate(inputs):
    potential = neuron.receive_input(input_current)
    print(f"Time {t}, Input: {input_current}, Potential: {potential}, State: {neuron.get_status()}")
```

### 6. Iterator

**Description**: Provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.

**When to use**:

- When you want to access an aggregate object's contents without exposing its internal structure
- When you want to support multiple traversal methods for an aggregate object
- When you want to provide a uniform interface for traversing different structures

**Example**:
```python
from abc import ABC, abstractmethod

# Iterator interface
class DataIterator(ABC):
    @abstractmethod
    def has_next(self):
        pass
        
    @abstractmethod
    def next(self):
        pass

# Concrete iterator for batch data
class BatchIterator(DataIterator):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_idx = 0
        
    def has_next(self):
        return self.current_idx < len(self.dataset)
        
    def next(self):
        if not self.has_next():
            raise StopIteration("No more data")
            
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        batch = self.dataset[start_idx:end_idx]
        
        self.current_idx = end_idx
        return batch

# Concrete iterator for time series data
class TimeWindowIterator(DataIterator):
    def __init__(self, time_series, window_size, stride=1):
        self.time_series = time_series
        self.window_size = window_size
        self.stride = stride
        self.current_idx = 0
        
    def has_next(self):
        return self.current_idx + self.window_size <= len(self.time_series)
        
    def next(self):
        if not self.has_next():
            raise StopIteration("No more time windows")
            
        window = self.time_series[self.current_idx:self.current_idx + self.window_size]
        self.current_idx += self.stride
        return window

# Aggregate interface
class Dataset(ABC):
    @abstractmethod
    def create_iterator(self):
        pass

# Concrete aggregate
class TabularDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def create_iterator(self, batch_size=1):
        return BatchIterator(self.data, batch_size)
        
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def create_iterator(self, window_size=10, stride=1):
        return TimeWindowIterator(self.data, window_size, stride)

# Usage
tabular_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dataset = TabularDataset(tabular_data)
iterator = dataset.create_iterator(batch_size=3)

print("Iterating through batches:")
while iterator.has_next():
    batch = iterator.next()
    print(f"Batch: {batch}")
    
time_series = [t for t in range(100)]
ts_dataset = TimeSeriesDataset(time_series)
ts_iterator = ts_dataset.create_iterator(window_size=5, stride=2)

print("\nIterating through time windows:")
window_count = 0
while ts_iterator.has_next() and window_count < 5:  # Limit to 5 windows for brevity
    window = ts_iterator.next()
    print(f"Window: {window}")
    window_count += 1
```

### 7. Visitor

**Description**: Represents an operation to be performed on the elements of an object structure.

**When to use**:

- When you need to perform operations on all elements of a complex object structure
- When the classes defining the object structure rarely change, but operations performed on them change frequently
- When you want to keep related operations together instead of spreading them across classes

**Example**:
```python
from abc import ABC, abstractmethod

# Element interface
class NeuralComponent(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass

# Concrete elements
class Layer(NeuralComponent):
    def __init__(self, name, units):
        self.name = name
        self.units = units
        self.activation = None
        
    def accept(self, visitor):
        return visitor.visit_layer(self)
        
class RecurrentConnection(NeuralComponent):
    def __init__(self, source, target, weight=1.0):
        self.source = source
        self.target = target
        self.weight = weight
        
    def accept(self, visitor):
        return visitor.visit_recurrent_connection(self)
        
class Activation(NeuralComponent):
    def __init__(self, function_type):
        self.function_type = function_type
        
    def accept(self, visitor):
        return visitor.visit_activation(self)

# Visitor interface
class Visitor(ABC):
    @abstractmethod
    def visit_layer(self, layer):
        pass
        
    @abstractmethod
    def visit_recurrent_connection(self, connection):
        pass
        
    @abstractmethod
    def visit_activation(self, activation):
        pass

# Concrete visitors
class ModelAnalysisVisitor(Visitor):
    def __init__(self):
        self.layer_count = 0
        self.connection_count = 0
        self.activation_functions = set()
        self.total_units = 0
        
    def visit_layer(self, layer):
        self.layer_count += 1
        self.total_units += layer.units
        
    def visit_recurrent_connection(self, connection):
        self.connection_count += 1
        
    def visit_activation(self, activation):
        self.activation_functions.add(activation.function_type)
        
    def get_summary(self):
        return {
            'layer_count': self.layer_count,
            'connection_count': self.connection_count,
            'activation_functions': list(self.activation_functions),
            'total_units': self.total_units
        }

class DiagramGenerationVisitor(Visitor):
    def __init__(self):
        self.diagram = []
        
    def visit_layer(self, layer):
        self.diagram.append(f"[{layer.name} ({layer.units} units)]")
        
    def visit_recurrent_connection(self, connection):
        self.diagram.append(
            f"{connection.source} --> {connection.target} (weight: {connection.weight})"
        )
        
    def visit_activation(self, activation):
        self.diagram.append(f"Activation: {activation.function_type}")
        
    def get_diagram(self):
        return "\n".join(self.diagram)

# Neural network structure
class NeuralNetwork:
    def __init__(self):
        self.components = []
        
    def add_component(self, component):
        self.components.append(component)
        
    def accept(self, visitor):
        results = []
        for component in self.components:
            results.append(component.accept(visitor))
        return results

# Usage
network = NeuralNetwork()
network.add_component(Layer("Input", 32))
network.add_component(Layer("Hidden", 64))
network.add_component(Activation("relu"))
network.add_component(RecurrentConnection("Hidden", "Hidden", 0.5))
network.add_component(Layer("Output", 10))
network.add_component(Activation("softmax"))

# Use analysis visitor
analysis_visitor = ModelAnalysisVisitor()
network.accept(analysis_visitor)
print("Network Analysis:")
print(analysis_visitor.get_summary())

# Use diagram visitor
diagram_visitor = DiagramGenerationVisitor()
network.accept(diagram_visitor)
print("\nNetwork Diagram:")
print(diagram_visitor.get_diagram())
```

## Scientific Computing-Specific Patterns

These patterns are particularly relevant to scientific computing and neural network frameworks.

### 1. Computation Graph

**Description**: Represents computational operations as a directed graph where nodes are operations and edges represent data flow.

**When to use**:

- For building neural networks with automatic differentiation
- For creating complex computation pipelines
- When operations can be optimized through graph transformations

**Example**:
```python
class ComputationNode:
    def __init__(self, operation=None, name=None):
        self.operation = operation
        self.name = name or str(id(self))
        self.inputs = []
        self.outputs = []
        self.value = None
        self.gradient = None
        
    def connect_to(self, node):
        self.outputs.append(node)
        node.inputs.append(self)
        
    def forward(self):
        if self.operation is None or not self.inputs:
            return self.value
            
        # Get input values
        input_values = [node.forward() for node in self.inputs]
        # Compute and store result
        self.value = self.operation(*input_values)
        return self.value
        
    def backward(self, gradient=1.0):
        self.gradient = gradient
        
        if not self.inputs or self.operation is None:
            return
            
        # Compute gradients for inputs (simplified)
        input_gradients = [1.0] * len(self.inputs)  # Placeholder for real gradients
        
        # Propagate gradients to inputs
        for i, input_node in enumerate(self.inputs):
            input_node.backward(gradient * input_gradients[i])

class ComputationGraph:
    def __init__(self):
        self.nodes = []
        self.input_nodes = []
        self.output_nodes = []
        
    def add_node(self, node, is_input=False, is_output=False):
        self.nodes.append(node)
        
        if is_input:
            self.input_nodes.append(node)
            
        if is_output:
            self.output_nodes.append(node)
            
    def forward(self, input_values):
        # Set input values
        for node, value in zip(self.input_nodes, input_values):
            node.value = value
            
        # Compute forward pass for output nodes
        results = [node.forward() for node in self.output_nodes]
        
        return results if len(results) > 1 else results[0]
        
    def backward(self, output_gradients=None):
        if output_gradients is None:
            output_gradients = [1.0] * len(self.output_nodes)
            
        # Initialize backward pass from output nodes
        for node, gradient in zip(self.output_nodes, output_gradients):
            node.backward(gradient)

# Usage example: simple neural network computation
def add(a, b): return a + b
def multiply(a, b): return a * b
def relu(x): return max(0, x)
def sigmoid(x): return 1.0 / (1.0 + (0.0 - x))

# Build a simple computation graph: f(x, y) = sigmoid(relu(x * w1 + y * w2))
graph = ComputationGraph()

# Input nodes
x = ComputationNode(name="x")
y = ComputationNode(name="y")
w1 = ComputationNode(name="w1")
w2 = ComputationNode(name="w2")
graph.add_node(x, is_input=True)
graph.add_node(y, is_input=True)
graph.add_node(w1, is_input=True)
graph.add_node(w2, is_input=True)

# Computation nodes
mul1 = ComputationNode(multiply, name="x*w1")
mul2 = ComputationNode(multiply, name="y*w2")
add_node = ComputationNode(add, name="add")
relu_node = ComputationNode(relu, name="relu")
sigmoid_node = ComputationNode(sigmoid, name="sigmoid")
graph.add_node(mul1)
graph.add_node(mul2)
graph.add_node(add_node)
graph.add_node(relu_node)
graph.add_node(sigmoid_node, is_output=True)

# Connect nodes
x.connect_to(mul1)
w1.connect_to(mul1)
y.connect_to(mul2)
w2.connect_to(mul2)
mul1.connect_to(add_node)
mul2.connect_to(add_node)
add_node.connect_to(relu_node)
relu_node.connect_to(sigmoid_node)

# Use the graph
result = graph.forward([2.0, 3.0, 0.5, -0.5])
print(f"Forward pass result: {result}")

# Compute gradients
graph.backward()
print(f"Gradient of x: {x.gradient}")
print(f"Gradient of y: {y.gradient}")
print(f"Gradient of w1: {w1.gradient}")
print(f"Gradient of w2: {w2.gradient}")
```

### 2. Lazy Evaluation

**Description**: Delays the evaluation of expressions until their values are needed, allowing for optimization opportunities.

**When to use**:

- When computations are expensive and might not be needed
- For handling large datasets that don't fit in memory
- To optimize computational graphs before execution

**Example**:
```python
class LazyTensor:
    def __init__(self, operation=None, operands=None, value=None):
        self.operation = operation
        self.operands = operands or []
        self._value = value
        self._evaluated = value is not None
        
    @property
    def value(self):
        if not self._evaluated:
            self._value = self._evaluate()
            self._evaluated = True
        return self._value
        
    def _evaluate(self):
        if self.operation is None:
            return self._value
            
        # Evaluate operands if needed
        operand_values = [operand.value for operand in self.operands]
        return self.operation(*operand_values)
        
    def __add__(self, other):
        if not isinstance(other, LazyTensor):
            other = LazyTensor(value=other)
        return LazyTensor(operation=lambda a, b: a + b, operands=[self, other])
        
    def __mul__(self, other):
        if not isinstance(other, LazyTensor):
            other = LazyTensor(value=other)
        return LazyTensor(operation=lambda a, b: a * b, operands=[self, other])
        
    def __neg__(self):
        return LazyTensor(operation=lambda a: -a, operands=[self])
        
    def __sub__(self, other):
        if not isinstance(other, LazyTensor):
            other = LazyTensor(value=other)
        return LazyTensor(operation=lambda a, b: a - b, operands=[self, other])
        
    def relu(self):
        return LazyTensor(operation=lambda a: max(0, a), operands=[self])
        
    def sigmoid(self):
        return LazyTensor(operation=lambda a: 1.0 / (1.0 + (0.0 - a)), operands=[self])

# Lazy-loading dataset
class LazyDataset:
    def __init__(self, data_loader_fn, transform_fn=None):
        self.data_loader_fn = data_loader_fn
        self.transform_fn = transform_fn
        self._data = None
        
    @property
    def data(self):
        if self._data is None:
            self._data = self.data_loader_fn()
            if self.transform_fn:
                self._data = self.transform_fn(self._data)
        return self._data
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def __len__(self):
        return len(self.data)

# Usage example
def load_large_dataset():
    print("Loading large dataset (expensive operation)...")
    return list(range(1000))

def normalize_data(data):
    print("Normalizing data...")
    max_val = max(data)
    return [x / max_val for x in data]

# Create lazy dataset
dataset = LazyDataset(load_large_dataset, normalize_data)
print("Dataset created but not loaded yet")

# Define lazy computation
x = LazyTensor(value=2.0)
y = LazyTensor(value=3.0)
w1 = LazyTensor(value=0.5)
w2 = LazyTensor(value=-0.3)

# Build computation graph
z = (x * w1 + y * w2).relu().sigmoid()
print("Computation defined but not executed yet")

# Force evaluation
result = z.value
print(f"Computation result: {result}")

# Accessing dataset forces loading
first_ten = dataset[:10]
print(f"First ten elements: {first_ten}")
```

### 3. Parameter Management

**Description**: Centralizes the management of model parameters for easier optimization, serialization, and tracking.

**When to use**:

- For complex models with many parameters
- When parameters need to be optimized jointly
- For tracking parameter changes during training

**Example**:
```python
import numpy as np

class Parameter:
    def __init__(self, value, requires_grad=True, name=None):
        self.value = np.array(value)
        self.grad = np.zeros_like(self.value)
        self.requires_grad = requires_grad
        self.name = name
        
    def zero_grad(self):
        self.grad = np.zeros_like(self.value)
        
    def __str__(self):
        return f"Parameter(name={self.name}, shape={self.value.shape})"

class ParameterManager:
    def __init__(self):
        self.parameters = {}
        
    def add(self, param, name=None):
        name = name or f"param_{len(self.parameters)}"
        param.name = name
        self.parameters[name] = param
        return param
        
    def get_all(self, requires_grad=None):
        if requires_grad is None:
            return list(self.parameters.values())
        return [p for p in self.parameters.values() if p.requires_grad == requires_grad]
        
    def zero_grad(self):
        for param in self.parameters.values():
            param.zero_grad()
            
    def get_grads_dict(self):
        return {name: param.grad for name, param in self.parameters.items() 
                if param.requires_grad}
                
    def set_values_dict(self, values_dict):
        for name, value in values_dict.items():
            if name in self.parameters:
                self.parameters[name].value = np.array(value)
                
    def get_values_dict(self):
        return {name: param.value for name, param in self.parameters.items()}
        
    def save(self, path):
        values_dict = self.get_values_dict()
        np.savez(path, **values_dict)
        
    def load(self, path):
        data = np.load(path)
        for name in data.files:
            if name in self.parameters:
                self.parameters[name].value = data[name]

# Simple optimizer example
class SGDOptimizer:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate
        
    def step(self):
        for param in self.parameters:
            if param.requires_grad:
                param.value -= self.learning_rate * param.grad

# Neural network layer using parameter management
class LinearLayer:
    def __init__(self, input_size, output_size, param_manager=None):
        self.input_size = input_size
        self.output_size = output_size
        
        # Create or use parameter manager
        self.param_manager = param_manager or ParameterManager()
        
        # Initialize parameters
        self.weights = self.param_manager.add(
            Parameter(np.random.randn(input_size, output_size) * 0.01),
            name=f"linear_{input_size}x{output_size}_W"
        )
        
        self.bias = self.param_manager.add(
            Parameter(np.zeros(output_size)),
            name=f"linear_{input_size}x{output_size}_b"
        )
        
    def forward(self, x):
        return np.dot(x, self.weights.value) + self.bias.value

# Usage example
param_manager = ParameterManager()

# Create layers with shared parameter manager
layer1 = LinearLayer(10, 20, param_manager)
layer2 = LinearLayer(20, 5, param_manager)

# Print all parameters
print("All model parameters:")
for param in param_manager.get_all():
    print(param)

# Simulate a forward pass
x = np.random.randn(1, 10)
hidden = layer1.forward(x)
output = layer2.forward(hidden)
print(f"Output shape: {output.shape}")

# Simulate backward pass (manually set gradients)
param_manager.zero_grad()
for param in param_manager.get_all():
    param.grad = np.ones_like(param.value) * 0.1

# Create optimizer and update parameters
optimizer = SGDOptimizer(param_manager.get_all(requires_grad=True), learning_rate=0.01)
optimizer.step()

# Save and load parameters
param_manager.save("/tmp/model_params.npz")
param_manager.load("/tmp/model_params.npz")
```

### 4. Data Pipeline

**Description**: Separates data loading, preprocessing, and augmentation into a modular and efficient pipeline.

**When to use**:

- When dealing with complex data processing workflows
- For handling large datasets efficiently
- To ensure reproducible data processing

**Example**:
```python
from abc import ABC, abstractmethod
import numpy as np

# Base class for all pipeline stages
class PipelineStage(ABC):
    @abstractmethod
    def process(self, data):
        pass
        
    def __call__(self, data):
        return self.process(data)

# Data loading stage
class DataLoader(PipelineStage):
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def process(self, dataset):
        indices = np.arange(len(dataset))
        
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield [dataset[idx] for idx in batch_indices]

# Data preprocessing stages
class Normalize(PipelineStage):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
        
    def process(self, data):
        return [(x - self.mean) / self.std for x in data]
        
class Resize(PipelineStage):
    def __init__(self, size):
        self.size = size
        
    def process(self, data):
        # In a real implementation, this would resize images
        print(f"Resizing data to {self.size}")
        return data

# Data augmentation stages
class RandomFlip(PipelineStage):
    def __init__(self, probability=0.5):
        self.probability = probability
        
    def process(self, data):
        # In a real implementation, this would flip images with probability p
        print(f"Random flip with p={self.probability}")
        return data
        
class RandomRotate(PipelineStage):
    def __init__(self, max_angle=30):
        self.max_angle = max_angle
        
    def process(self, data):
        # In a real implementation, this would rotate images
        print(f"Random rotation with max angle {self.max_angle}")
        return data

# Batch processing stage
class BatchProcessor(PipelineStage):
    def process(self, batch):
        # In a real implementation, this would convert a batch to needed format
        # like separating features and labels
        features = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return features, labels

# Complete pipeline
class DataPipeline:
    def __init__(self, stages=None):
        self.stages = stages or []
        
    def add_stage(self, stage):
        self.stages.append(stage)
        return self
        
    def process(self, data):
        result = data
        for stage in self.stages:
            result = stage(result)
            if hasattr(result, '__iter__') and not isinstance(result, (list, tuple)):
                # Handle generator stages (like DataLoader)
                for batch in result:
                    # Process remaining pipeline on each batch
                    remaining_pipeline = DataPipeline(self.stages[self.stages.index(stage)+1:])
                    yield remaining_pipeline.process(batch)
                return
        return result

# Usage example
class DummyDataset:
    def __init__(self, size=100):
        self.data = [(np.random.randn(28, 28), np.random.randint(0, 10)) 
                     for _ in range(size)]
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def __len__(self):
        return len(self.data)

# Create dataset
dataset = DummyDataset(size=100)

# Define training pipeline
train_pipeline = DataPipeline([
    Resize((32, 32)),
    RandomFlip(0.5),
    RandomRotate(30),
    DataLoader(batch_size=16, shuffle=True),
    Normalize(mean=0.5, std=0.5),
    BatchProcessor()
])

# Process data
print("Processing training data...")
for i, (features, labels) in enumerate(train_pipeline.process(dataset)):
    if i < 3:  # Show only first 3 batches
        print(f"Batch {i}: Features shape: {len(features)}, Labels shape: {len(labels)}")
    else:
        break

# Define evaluation pipeline (without augmentation)
eval_pipeline = DataPipeline([
    Resize((32, 32)),
    DataLoader(batch_size=32, shuffle=False),
    Normalize(mean=0.5, std=0.5),
    BatchProcessor()
])

print("\nProcessing evaluation data...")
for i, (features, labels) in enumerate(eval_pipeline.process(dataset)):
    if i < 2:  # Show only first 2 batches
        print(f"Batch {i}: Features shape: {len(features)}, Labels shape: {len(labels)}")
    else:
        break
```

### 5. Experiment Tracking

**Description**: Manages experiment configuration, logging, and results tracking for reproducible research.

**When to use**:

- For tracking multiple experiment runs
- To ensure reproducibility of results
- For comparing different model configurations

**Example**:
```python
import os
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self, name, description=None, base_dir="experiments"):
        self.name = name
        self.description = description
        self.base_dir = base_dir
        self.start_time = datetime.now()
        self.end_time = None
        
        # Generate unique experiment ID
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.id = f"{name}_{timestamp}"
        
        # Create experiment directory
        self.exp_dir = os.path.join(base_dir, self.id)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Initialize config and metrics
        self.config = {}
        self.metrics = {}
        self.artifacts = {}
        
    def set_config(self, config):
        """Set experiment configuration parameters"""
        self.config.update(config)
        self._save_config()
        return self
        
    def log_metric(self, name, value, step=None):
        """Log a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
            
        entry = {"value": value}
        if step is not None:
            entry["step"] = step
            
        self.metrics[name].append(entry)
        self._save_metrics()
        return self
        
    def log_artifact(self, name, artifact, artifact_type=None):
        """Log an artifact (model, figure, etc.)"""
        artifact_dir = os.path.join(self.exp_dir, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        
        artifact_path = os.path.join(artifact_dir, name)
        
        if artifact_type == "figure":
            plt.figure(artifact)
            plt.savefig(artifact_path)
            plt.close()
        elif artifact_type == "numpy":
            np.save(artifact_path, artifact)
        elif artifact_type == "json":
            with open(f"{artifact_path}.json", "w") as f:
                json.dump(artifact, f, indent=2)
        else:
            # Default: try to save as pickle
            import pickle
            with open(f"{artifact_path}.pkl", "wb") as f:
                pickle.dump(artifact, f)
                
        self.artifacts[name] = {
            "path": artifact_path,
            "type": artifact_type
        }
        
        return self
        
    def finish(self):
        """Mark experiment as complete"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        summary = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": duration
        }
        
        summary_path = os.path.join(self.exp_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary
        
    def _save_config(self):
        """Save configuration to file"""
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
            
    def _save_metrics(self):
        """Save metrics to file"""
        metrics_path = os.path.join(self.exp_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

class ExperimentManager:
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def create_experiment(self, name, description=None):
        """Create and return a new experiment"""
        return Experiment(name, description, self.base_dir)
        
    def list_experiments(self):
        """List all experiments"""
        experiments = []
        for exp_dir in os.listdir(self.base_dir):
            summary_path = os.path.join(self.base_dir, exp_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                    experiments.append(summary)
        return experiments
        
    def load_experiment(self, experiment_id):
        """Load experiment by ID"""
        exp_dir = os.path.join(self.base_dir, experiment_id)
        
        if not os.path.exists(exp_dir):
            raise ValueError(f"Experiment {experiment_id} not found")
            
        # Load summary
        summary_path = os.path.join(exp_dir, "summary.json")
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        # Load config
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Load metrics
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        return {"summary": summary, "config": config, "metrics": metrics}
        
    def compare_experiments(self, experiment_ids, metric_name):
        """Compare metric across experiments"""
        results = {}
        
        for exp_id in experiment_ids:
            exp_data = self.load_experiment(exp_id)
            if metric_name in exp_data["metrics"]:
                values = [entry["value"] for entry in exp_data["metrics"][metric_name]]
                results[exp_id] = values
                
        return results

# Usage example
manager = ExperimentManager()

# Create and configure experiment
experiment = manager.create_experiment(
    name="rcnn_model",
    description="Testing recurrence types"
)

# Set configuration
experiment.set_config({
    "model_type": "RCNN",
    "recurrence_type": "full",
    "layers": [64, 128, 256],
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
})

# Simulate training and log metrics
for epoch in range(10):
    # Simulate training
    train_loss = 1.0 / (epoch + 1)
    val_loss = 1.2 / (epoch + 1)
    accuracy = 0.5 + (epoch / 20.0)
    
    # Log metrics
    experiment.log_metric("train_loss", train_loss, step=epoch)
    experiment.log_metric("val_loss", val_loss, step=epoch)
    experiment.log_metric("accuracy", accuracy, step=epoch)
    
    # Simulate artifact saving
    if epoch % 5 == 0:
        # Create a figure
        plt.figure(figsize=(10, 5))
        plt.plot(range(epoch+1), [1.0 / (e + 1) for e in range(epoch+1)], label="Train Loss")
        plt.plot(range(epoch+1), [1.2 / (e + 1) for e in range(epoch+1)], label="Validation Loss")
        plt.legend()
        plt.title(f"Training Progress - Epoch {epoch}")
        
        # Log the figure
        experiment.log_artifact(f"loss_plot_epoch_{epoch}", plt.gcf(), "figure")
        
        # Simulate model checkpoint
        model_state = {"weights": np.random.randn(10, 10), "epoch": epoch}
        experiment.log_artifact(f"model_checkpoint_epoch_{epoch}", model_state, "json")

# Finish experiment
summary = experiment.finish()
print(f"Experiment completed: {summary['id']}")

# List all experiments
experiments = manager.list_experiments()
print(f"Total experiments: {len(experiments)}")

# Load experiment data
exp_data = manager.load_experiment(summary['id'])
print(f"Loaded experiment config: {exp_data['config']}")

# Print final metrics
final_accuracy = exp_data['metrics']['accuracy'][-1]['value']
print(f"Final accuracy: {final_accuracy}")
```

## Conclusion

These design patterns provide a strong foundation for building complex scientific computing applications like DynVision. They promote code reusability, maintainability, and scalability through proven architectural solutions. When applying these patterns:

1. **Consider the context**: Choose patterns that match your project's specific requirements
2. **Combine patterns**: Most real-world applications use multiple complementary patterns
3. **Start simple**: Introduce patterns as complexity demands them, not preemptively
4. **Document usage**: Make pattern implementations clear to other developers

Through thoughtful application of these patterns, scientific software projects can achieve a balance of flexibility, performance, and code clarity.

## References

1. Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). Design Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley.
2. Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design. Prentice Hall.
3. Martelli, A. (2000). Python in a Nutshell. O'Reilly Media.
4. Abadi, M., et al. (2016). TensorFlow: A System for Large-Scale Machine Learning. OSDI.
5. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
