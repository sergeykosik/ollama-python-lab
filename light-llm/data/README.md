# Training Data for Light LLM

This directory contains sample training data and utilities for training the Light LLM model on code generation tasks.

## Directory Structure

```
data/
├── __init__.py           # Package initialization
├── dataset.py            # Dataset classes and data loading utilities
├── samples/              # Sample code files for training
│   ├── python/          # Python code examples
│   │   ├── basics.py
│   │   ├── classes.py
│   │   ├── data_structures.py
│   │   └── web_api.py
│   ├── javascript/      # JavaScript code examples
│   │   ├── basics.js
│   │   ├── classes.js
│   │   └── async.js
│   └── java/            # Java code examples
│       ├── Basics.java
│       └── Classes.java
└── README.md            # This file
```

## Dataset Classes

### CodeDataset

Loads code files from a directory and prepares them for training.

```python
from data.dataset import CodeDataset
from light_llm.tokenizer import CodeTokenizer

tokenizer = CodeTokenizer.load("./tokenizer")
dataset = CodeDataset(
    data_dir="./data/samples",
    tokenizer=tokenizer,
    max_length=512,
    file_extensions=[".py", ".js", ".java"],
)
```

### SimpleTextDataset

Creates a dataset from a list of text examples.

```python
from data.dataset import SimpleTextDataset

texts = ["def hello(): print('hello')", "class Foo: pass"]
dataset = SimpleTextDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_length=512,
)
```

## Data Loading Utilities

### create_dataloaders

Creates train and validation dataloaders from a directory of code files.

```python
from data.dataset import create_dataloaders

train_loader, val_loader = create_dataloaders(
    data_dir="./data/samples",
    tokenizer=tokenizer,
    batch_size=4,
    max_length=512,
    num_workers=4,
    train_ratio=0.9,
    file_extensions=[".py", ".js", ".java"],
)
```

### collect_code_files

Collects code files into a single text file (useful for training tokenizers).

```python
from data.dataset import collect_code_files

num_files = collect_code_files(
    data_dir="./data/samples",
    output_file="./corpus.txt",
    file_extensions=[".py", ".js", ".java"],
)
```

## Sample Data

The `samples/` directory contains example code files in multiple programming languages:

### Python Examples (`samples/python/`)
- **basics.py**: Basic Python functions (factorial, fibonacci, prime checking, string operations)
- **classes.py**: OOP examples (Rectangle, Circle, BankAccount, Stack, Queue, LinkedList)
- **data_structures.py**: Algorithms and data structures (binary search, sorting, linked list)
- **web_api.py**: Flask web API example with authentication and CRUD operations

### JavaScript Examples (`samples/javascript/`)
- **basics.js**: Basic JavaScript functions and array operations
- **classes.js**: ES6 classes and data structures
- **async.js**: Asynchronous programming patterns (Promises, async/await, error handling)

### Java Examples (`samples/java/`)
- **Basics.java**: Basic Java methods and algorithms
- **Classes.java**: Java OOP examples with generics

## Using Your Own Data

To train on your own code:

1. **Prepare your data directory**:
   ```bash
   mkdir -p my_code_data
   # Add your code files to this directory
   ```

2. **Train a tokenizer** (if needed):
   ```bash
   python scripts/train.py --data_dir ./my_code_data --tokenizer_path ./my_tokenizer
   ```

3. **Train the model**:
   ```bash
   python scripts/train.py \
       --data_dir ./my_code_data \
       --tokenizer_path ./my_tokenizer \
       --model_size small \
       --batch_size 4 \
       --num_epochs 10
   ```

## Data Format

The dataset classes prepare data in the following format for training:

- **input_ids**: Token IDs (shape: `[batch_size, max_length]`)
- **attention_mask**: Attention mask where 1 = real token, 0 = padding (shape: `[batch_size, max_length]`)
- **labels**: Target labels for language modeling, same as input_ids but with padding masked as -100 (shape: `[batch_size, max_length]`)

## Supported File Extensions

By default, the following file extensions are supported:
- Python: `.py`
- JavaScript: `.js`
- Java: `.java`
- C++: `.cpp`

You can customize this by passing a different list to the `file_extensions` parameter.

## Best Practices

1. **Data Quality**: Use clean, well-formatted code with good practices
2. **Diversity**: Include various programming patterns and structures
3. **Size**: Start with 10-100K examples for initial training
4. **Preprocessing**: Remove sensitive information (API keys, passwords)
5. **Validation Split**: Use 10-20% of data for validation
6. **Max Length**: Choose max_length based on your typical code snippet size (512-1024 tokens)

## Example Training Pipeline

```python
from light_llm import LightLLM, get_config
from light_llm.tokenizer import CodeTokenizer
from light_llm.trainer import Trainer
from data.dataset import create_dataloaders

# Load tokenizer
tokenizer = CodeTokenizer.load("./tokenizer")

# Create model
config = get_config("small")
config.vocab_size = len(tokenizer)
model = LightLLM(config)

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    data_dir="./data/samples",
    tokenizer=tokenizer,
    batch_size=4,
    max_length=512,
)

# Train model
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    learning_rate=3e-4,
    num_epochs=10,
)
trainer.train()
```

## Notes

- The sample data provided is for demonstration purposes
- For production use, you should train on a much larger dataset
- Consider using multiple programming languages for better generalization
- Monitor validation loss to prevent overfitting
- Use gradient accumulation for larger effective batch sizes on limited hardware
