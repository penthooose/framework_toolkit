# HAI-DEV-TOOLS

A comprehensive toolkit for preparing data, extracting information, and fine-tuning subsymbolic AI models for integration into hybrid AI applications.

> Note: The tools of this framework are still under development. Many of these tools are currently customized for the development of my own application and may not all be easily generalizable.

## Overview

This framework provides a suite of specialized tools focused on the data preparation and model fine-tuning pipeline for subsymbolic AI components. The toolkit is designed to support the creation of custom language models that can later be integrated into hybrid AI systems combining both symbolic and subsymbolic approaches.

## Components

### PII Removal

Tools for identifying and removing personally identifiable information (PII) from datasets:

- Automated detection of sensitive information
- Configurable redaction strategies
- Support for multiple languages
- Compliance with privacy regulations

### Model Tools

Utilities for working with and manipulating AI models:

- Model conversion between different formats
- Quantization tools for model compression
- Performance benchmarking
- Integration helpers for various deployment environments

### Data Preparation

> Note: These tools are currently tightly coupled to specific application projects and may not be suitable for direct external use without adaptation.

Tools for preprocessing and transforming raw data:

- Text normalization and cleaning
- Data validation and error detection
- Format conversion utilities
- Contextual data enrichment

### Information Extraction

Tools designed for creating specialized fine-tuning datasets for different tasks:

- Named entity recognition data preparation
- Text classification dataset creation
- Question-answering pair generation
- Semantic relationship extraction

### Dataset Building

Utilities for constructing instruction-following datasets:

- JSONL file generation for fine-tuning tasks
- Instruction template management
- Data augmentation capabilities
- Quality assurance tools for dataset validation

### Fine-Tuning

An Erlang-to-PyTorch bridge for model fine-tuning:

- ErlPort integration for seamless Elixir/Python interoperability
- JSON-based configuration for fine-tuning parameters
- Support for various optimization strategies
- Checkpoint management and evaluation metrics

## Getting Started

Some component include their own README with specific installation and usage instructions. Most components are implemented as standalone Elixir applications with appropriate dependencies.

> **Important:** Many tools in this framework rely on ErlPort for Elixir-Python integration. Proper setup of Python paths and environment variables is required for these components to function correctly. Make sure your Python environment contains all the necessary dependencies and is correctly configured in your system PATH.

```bash
# Example: To use the fine-tuning module
cd fine_tuning
mix deps.get
mix compile
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You are free to:

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:

- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial — You may not use the material for commercial purposes.

For the full license text, see: [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)

## Acknowledgements

This framework is developed as part of a Master's Thesis at Otto-Friedrich-Universität Bamberg.
