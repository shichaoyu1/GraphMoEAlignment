# GraphMoEAlignment

Research code for multimodal glioma MRI semantic alignment, topology-guided anchor retrieval, and graph-fusion experiments.

## Setup

Clone the repository into a directory named `glioma`. Run the Python commands below from its parent directory.

```bash
python -m pip install -r glioma/requirements.txt
python -m pytest glioma/tests -q
```

Python 3.11 or 3.12 is recommended for the full suite. The general authority projection dependencies install on Python 3.11 and newer. On GPU servers, install a matching PyTorch and torchvision pair for your CUDA environment before installing the requirements. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Documentation

- [Project guide](docs/project.md) for data layout, CLI usage, and output formats.
- [Documentation index](docs/README.md) for Paper 2, Paper 4, Idea4.5, and archived material pending deletion review.

Data, checkpoints, experiment outputs, and logs live in local or server storage.
