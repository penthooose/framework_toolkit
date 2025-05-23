# Information Extraction Module

## Overview

This module provides tools for extracting structured information from unstructured text sources using both subsymbolic AI approaches (transformer-based models) and symbolic AI methods (regular expressions and rule-based systems). The extracted information can be used for subsequent fine-tuning processes.

## Capabilities

- **Subsymbolic Information Extraction**: Leverage deep learning models to extract complex patterns and semantic information from text.
- **Symbolic Information Extraction**: Use regular expressions and rule-based approaches for extracting well-defined patterns.
- **Hybrid Approaches**: Combine both methods for optimal information extraction results.
- **Processing Pipeline**: Process prepared files efficiently in a standardized workflow.

## Requirements

### Ollama Setup

To use the subsymbolic information extraction functionality, you must have Ollama properly set up on your system:

1. Install Ollama from [https://ollama.com/](https://ollama.com/)
2. Ensure Ollama is running as a service
3. Pull the required models for your specific extraction tasks

```bash
# Example commands to pull models
ollama pull mistral
ollama pull llama2
```

## Usage

Basic extraction workflow:

1. Prepare your text data in the required format
2. Apply extraction methods (symbolic, subsymbolic, or hybrid)
3. Export structured results for further processing or fine-tuning
