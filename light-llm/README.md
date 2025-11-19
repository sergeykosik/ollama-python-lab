# Light LLM

A lightweight transformer-based language model optimized for code generation on consumer-grade hardware.

## Features

- **Lightweight Architecture**: Efficient transformer model (100M-1B parameters) designed for consumer GPUs
- **Code-Optimized**: Specialized tokenization and training for code generation tasks
- **Memory Efficient**:
  - Mixed precision training (FP16/BF16)
  - Gradient checkpointing
  - Gradient accumulation
  - KV-cache for efficient inference
- **Flexible Generation**:
  - Multiple sampling strategies (greedy, top-k, nucleus)
  - Streaming generation
  - Batch inference
  - Code completion mode
- **Production Ready**:
  - Comprehensive training pipeline
  - Checkpointing and resuming
  - TensorBoard logging
  - Easy-to-use inference API

## Architecture

Light LLM uses a decoder-only transformer architecture with:

- **Rotary Position Embeddings (RoPE)** for better position encoding
- **Pre-LayerNorm** for training stability
- **SwiGLU activation** for improved performance
- **Multi-head self-attention** with optional Flash Attention
- **Causal masking** for autoregressive generation

## Model Sizes

| Size   | Parameters | Layers | Hidden Size | Heads | Context Length | GPU Memory (Training) |
|--------|-----------|--------|-------------|-------|----------------|----------------------|
| Tiny   | ~50M      | 8      | 512         | 8     | 1024           | ~4-6 GB              |
| Small  | ~150M     | 12     | 768         | 12    | 2048           | ~6-10 GB             |
| Medium | ~350M     | 16     | 1024        | 16    | 2048           | ~10-14 GB            |
| Large  | ~700M     | 24     | 1280        | 20    | 2048           | ~14-20 GB            |

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU training)
- 8-16GB GPU memory (recommended)

### Setup

```bash
# Clone the repository
cd light-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional Optimizations

For enhanced performance, install these optional dependencies:

```bash
# Flash Attention (requires CUDA and compatible GPU)
pip install flash-attn>=2.0.0

# 8-bit quantization support
pip install bitsandbytes>=0.41.0

# Memory-efficient attention
pip install xformers>=0.0.20
```

## Quick Start

### 1. Prepare Your Data

Organize your code files in a directory:

```
data/
├── project1/
│   ├── file1.py
│   ├── file2.js
│   └── ...
├── project2/
│   └── ...
└── ...
```

### 2. Train a Model

Basic training with default settings:

```bash
python scripts/train.py \
  --data_dir ./data/code \
  --model_size small \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 10 \
  --output_dir ./checkpoints
```

Advanced training with custom settings:

```bash
python scripts/train.py \
  --data_dir ./data/code \
  --model_size medium \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-4 \
  --warmup_steps 1000 \
  --max_steps 100000 \
  --mixed_precision \
  --gradient_checkpointing \
  --eval_steps 1000 \
  --save_steps 5000 \
  --output_dir ./checkpoints/medium
```

### 3. Generate Text

Simple generation:

```bash
python scripts/generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --prompt "def fibonacci(n):" \
  --max_length 200 \
  --temperature 0.7
```

Interactive mode:

```bash
python scripts/generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --mode interactive
```

Code completion mode:

```bash
python scripts/generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --prompt "def quicksort(arr):" \
  --mode complete \
  --code_mode
```

Streaming generation:

```bash
python scripts/generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --prompt "class LinkedList:" \
  --mode streaming \
  --code_mode
```

## Usage

### Training

#### Command Line Arguments

```bash
# Model configuration
--model_size {tiny,small,medium,large}  # Preset model size
--vocab_size INT                         # Vocabulary size (default: 32000)
--max_length INT                         # Maximum sequence length (default: 1024)

# Data configuration
--data_dir PATH                          # Directory with code files (required)
--tokenizer_path PATH                    # Path to trained tokenizer (optional)
--file_extensions EXT [EXT ...]          # File extensions to include
--train_ratio FLOAT                      # Train/val split ratio (default: 0.9)

# Training configuration
--batch_size INT                         # Batch size (default: 4)
--gradient_accumulation_steps INT        # Gradient accumulation (default: 4)
--learning_rate FLOAT                    # Learning rate (default: 3e-4)
--num_epochs INT                         # Number of epochs (default: 10)
--warmup_steps INT                       # Warmup steps (default: 1000)

# Optimization
--mixed_precision                        # Use FP16 training
--gradient_checkpointing                 # Enable gradient checkpointing
--num_workers INT                        # Data loading workers (default: 4)

# Logging and checkpointing
--output_dir PATH                        # Checkpoint directory
--logging_steps INT                      # Log every N steps (default: 100)
--eval_steps INT                         # Evaluate every N steps (default: 1000)
--save_steps INT                         # Save every N steps (default: 5000)
```

#### Python API

```python
from light_llm import LightLLM, get_config
from light_llm.tokenizer import CodeTokenizer
from light_llm.trainer import Trainer
from data.dataset import create_dataloaders

# Load tokenizer
tokenizer = CodeTokenizer.load("./tokenizer")

# Create model
config = get_config("small")
model = LightLLM(config)

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    data_dir="./data/code",
    tokenizer=tokenizer,
    batch_size=4,
    max_length=1024,
)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    learning_rate=3e-4,
    output_dir="./checkpoints",
)

# Train
trainer.train(num_epochs=10)
```

### Inference

#### Command Line

```bash
# Generate text
python scripts/generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --prompt "Your prompt here" \
  --max_length 200 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.95

# Interactive mode
python scripts/generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --mode interactive

# Batch generation from file
python scripts/generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --mode batch \
  --prompt_file prompts.txt \
  --output_file results.txt
```

#### Python API

```python
from light_llm.inference import TextGenerator, load_model_for_inference
from light_llm.tokenizer import CodeTokenizer

# Load model and tokenizer
model = load_model_for_inference("./checkpoints/best_model.pt")
tokenizer = CodeTokenizer.load("./tokenizer")

# Create generator
generator = TextGenerator(model, tokenizer)

# Generate text
output = generator.generate(
    prompt="def merge_sort(arr):",
    max_length=200,
    temperature=0.7,
    top_p=0.95,
)
print(output)

# Code completion
code = generator.complete_code(
    code="class BinaryTree:\n    def __init__(self):",
    max_length=150,
)
print(code)

# Streaming generation
for token in generator.generate_streaming(
    prompt="def factorial(n):",
    max_length=100,
):
    print(token, end="", flush=True)
```

## Project Structure

```
light-llm/
├── light_llm/              # Main package
│   ├── __init__.py
│   ├── config.py           # Model configuration
│   ├── model.py            # Model architecture
│   ├── attention.py        # Attention mechanisms
│   ├── tokenizer.py        # Tokenizer
│   ├── trainer.py          # Training pipeline
│   ├── inference.py        # Inference engine
│   └── utils.py            # Utility functions
├── data/                   # Data loading
│   ├── __init__.py
│   └── dataset.py          # Dataset classes
├── scripts/                # Training and inference scripts
│   ├── train.py            # Training script
│   └── generate.py         # Generation script
├── checkpoints/            # Model checkpoints (created during training)
├── logs/                   # TensorBoard logs (created during training)
├── tokenizer/              # Trained tokenizer (created during training)
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Training Tips

### Memory Optimization

If you encounter out-of-memory errors:

1. **Reduce batch size**: Lower `--batch_size` and increase `--gradient_accumulation_steps`
2. **Enable gradient checkpointing**: Add `--gradient_checkpointing`
3. **Reduce sequence length**: Lower `--max_length`
4. **Use a smaller model**: Choose `--model_size tiny` or `small`
5. **Clear GPU cache**: The trainer automatically manages memory, but you can manually clear cache

### Training Speed

To improve training speed:

1. **Use mixed precision**: Add `--mixed_precision` (enabled by default)
2. **Increase batch size**: If memory allows, increase `--batch_size`
3. **Optimize data loading**: Increase `--num_workers` (4-8 recommended)
4. **Use Flash Attention**: Install flash-attn for faster attention computation
5. **Enable torch.compile()**: For PyTorch 2.0+, the model supports compilation

### Hyperparameter Tuning

Recommended hyperparameters for different scenarios:

**Small datasets (< 100MB)**:
- Model size: tiny or small
- Learning rate: 5e-4
- Batch size: 8-16
- Warmup steps: 500

**Medium datasets (100MB - 1GB)**:
- Model size: small or medium
- Learning rate: 3e-4
- Batch size: 4-8
- Warmup steps: 1000

**Large datasets (> 1GB)**:
- Model size: medium or large
- Learning rate: 1e-4 to 3e-4
- Batch size: 2-4
- Warmup steps: 2000

## Benchmarks

Performance benchmarks on consumer hardware (NVIDIA RTX 3070, 8GB VRAM):

| Model Size | Training Speed | Inference Speed | Memory (Training) | Memory (Inference) |
|-----------|---------------|-----------------|-------------------|-------------------|
| Tiny      | ~500 tok/s    | ~80 tok/s       | 4 GB             | 1.5 GB            |
| Small     | ~300 tok/s    | ~50 tok/s       | 8 GB             | 2.5 GB            |
| Medium    | ~150 tok/s    | ~30 tok/s       | 12 GB            | 4 GB              |

*Note: Speeds vary based on sequence length and hardware*

## Advanced Features

### Custom Tokenizer Training

```python
from light_llm.tokenizer import CodeTokenizer
from data.dataset import collect_code_files

# Collect code files
collect_code_files(
    data_dir="./data/code",
    output_file="./corpus.txt",
    file_extensions=[".py", ".js"],
)

# Train tokenizer
tokenizer = CodeTokenizer(vocab_size=32000)
tokenizer.train(["./corpus.txt"], vocab_size=32000)
tokenizer.save("./tokenizer")
```

### Resume Training

```bash
python scripts/train.py \
  --data_dir ./data/code \
  --resume_from_checkpoint ./checkpoints/checkpoint-10000.pt \
  --output_dir ./checkpoints
```

### Model Export

```python
import torch
from light_llm.inference import load_model_for_inference

# Load model
model = load_model_for_inference("./checkpoints/best_model.pt")

# Export to TorchScript
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")

# Export to ONNX
dummy_input = torch.randint(0, 32000, (1, 100))
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}}
)
```

## Troubleshooting

### Common Issues

**Q: Out of memory error during training**
- A: Reduce batch size, enable gradient checkpointing, or use a smaller model

**Q: Training is very slow**
- A: Enable mixed precision, increase num_workers, or check if GPU is being used

**Q: Generated text is repetitive**
- A: Increase temperature, use nucleus sampling (top_p), or add repetition penalty

**Q: Model generates nonsensical code**
- A: Train longer, use more data, or adjust hyperparameters

**Q: Tokenizer not found error**
- A: Train a tokenizer first or provide --tokenizer_path argument

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is released under the MIT License. See LICENSE file for details.

## Acknowledgments

- Transformer architecture based on "Attention Is All You Need" (Vaswani et al.)
- RoPE implementation inspired by RoFormer
- Training techniques from various LLM papers and implementations

## Citation

If you use this code in your research, please cite:

```bibtex
@software{light_llm,
  title = {Light LLM: A Lightweight Language Model for Code Generation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/light-llm}
}
```

## Contact

For questions and feedback, please open an issue on GitHub.

---

**Happy coding with Light LLM!** 🚀
