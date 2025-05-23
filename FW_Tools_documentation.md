# Data Preparation Tools

The Data Preparation module provides tools for extracting, converting, filtering, and organizing document data for training AI models. These tools form a processing pipeline to transform raw documents into structured training data.

### DataPrepareController (`data_prepare_controller.ex`)

**Purpose:** Central API that orchestrates all data preparation tools through a unified interface.

**Functionality:**

- Provides two access patterns:
  - Direct function calls: `DataPrepareController.function_name()`
  - Module import: `use DataPrepareController` to import all functions
- Controls workflow between different data preparation steps
- Enables both interactive and programmatic usage of tools

### Extract Files (`dp_extract_files.ex`)

**Purpose:** Locates and extracts document files from a structured database directory for further processing.

**Functionality:**

- Scans directories recursively for files matching specific patterns (e.g., `*GA*01.docm`)
- Copies matched files to a dedicated processing directory
- Returns detailed statistics about the extraction process
- Handles errors and provides logging for troubleshooting

### Convert Files (`dp_convert_files.ex`)

**Purpose:** Transforms proprietary document formats into standardized markdown format for easier text processing.

**Functionality:**

- Uses Pandoc for document format conversion (DOCM to Markdown)
- Implements multiple execution strategies for cross-platform compatibility
- Handles Windows path conversions and escaping
- Provides detailed logging of the conversion process
- Includes fallback mechanisms for different execution environments

### Extract MDB Data (`dp_extract_mdb_data.ex`)

**Purpose:** Extracts structured data from Microsoft Access databases for integration with document content.

**Functionality:**

- Interfaces with MDB Tools through Windows Subsystem for Linux (WSL)
- Handles multiple character encodings (cp1252, iso8859-1, etc.)
- Detects and processes binary data appropriately
- Sanitizes data for JSON compatibility
- Exports structured data in both JSON and JSONL formats
- Performs intelligent date and currency format conversions

### Filter Files (`dp_filter_files.ex`)

**Purpose:** Identifies relevant document files based on content analysis and keyword matching.

**Functionality:**

- Analyzes document content to identify files containing specific keywords
- Implements pattern matching with tolerance for OCR errors and spacing issues
- Supports regular expressions for flexible text matching
- Organizes files into relevant and non-relevant categories
- Preserves original files while creating a filtered subset

### Partition Data (`dp_partition_data.ex`)

**Purpose:** Breaks down documents into logical sections for granular processing and training data creation.

**Functionality:**

- Identifies document structure through intelligent heading detection
- Extracts metadata and references sections
- Splits documents into chapters and subchapters
- Maintains hierarchical document structure
- Creates individual files for each section
- Handles complex formatting issues and inconsistencies
- Performs content cleanup and normalization
- Preserves both individual sections and the complete document

### Statistics (`statistics.ex`)

**Purpose:** Provides quantitative analysis of datasets to inform training strategies and identify quality issues.

**Functionality:**

- Calculates token lengths for efficient model training
- Analyzes chapter distributions and document structures
- Identifies outliers and problematic content patterns
- Reports on data quality metrics (completeness, consistency)
- Extracts patterns across different document types
- Creates detailed reports for dataset optimization
- Identifies documents requiring revision
- Generates structured JSON analytics for downstream processing
- Supports both supervised and unsupervised data analysis

### Revising Data (`dp_revising_data.ex`)

**Purpose:** Corrects and updates processed content to address quality issues and improve training data.

**Functionality:**

- Integrates revised summaries into processed documents
- Updates documents with corrected content
- Merges multiple data revision sources
- Maintains document structure during revision
- Tracks changes between original and revised content
- Re-triggers downstream processing after revisions

### Remove Overlength Files (`dp_remove_overlength_files.ex`)

**Purpose:** Manages document length constraints to optimize model training and prevent token limit issues.

**Functionality:**

- Identifies files exceeding specified token thresholds
- Filters content based on maximum token limits
- Splits or removes overlength documents
- Optimizes content for specific model context windows
- Transfers properly sized files to training directories
- Processes both individual chapters and complete documents

### Training Data Preparation (`dp_prepare_training_data.ex`)

**Purpose:** Formats and structures content for optimal model training with consistent patterns.

**Functionality:**

- Standardizes formatting across diverse document sources
- Normalizes text representations (headings, quotes, emphasis)
- Processes paragraph structures for context preservation
- Prepares both markdown and JSON formats for training
- Handles special characters and escape sequences
- Creates consistent training examples from varied inputs
- Preserves semantic structure while standardizing format

### QA Data Preparation (`dp_prepare_qa_data.ex`)

**Purpose:** Creates structured question-answer pairs from document content for supervised learning.

**Functionality:**

- Extracts well-formed question-answer pairs from documents
- Validates and normalizes question formats
- Manages letter prefixes and question identifiers
- Estimates token lengths for efficient training
- Detects and merges related question subitems
- Creates fine-tuning layouts with controlled token counts
- Avoids duplicate question patterns
- Filters invalid or malformed QA content

### Following Chapters (`dp_prepare_following_chapters.ex`)

**Purpose:** Creates contextual relationships between document sections to enhance model understanding of content flow.

**Functionality:**

- Identifies logical chapter sequences in documents
- Combines related chapters within token constraints
- Creates document structures for context-aware training
- Determines optimal chapter groupings based on content
- Supports both supervised and unsupervised formats
- Defines chapter hierarchies and dependencies
- Maintains chapter relationships while respecting token limits
- Ensures proper context window utilization

---

## Information Extraction Tools

The Information Extraction module provides tools for systematically extracting structured information from unstructured or semi-structured document content. It implements a hybrid approach combining rule-based (symbolic) and AI-powered (sub-symbolic) techniques.

### InformationExtractor (`information_extractor.ex`)

**Purpose:** Central controller that orchestrates both symbolic and sub-symbolic extraction processes through a unified interface.

**Functionality:**

- Manages hybrid information extraction pipeline
- Coordinates workflow between symbolic and sub-symbolic extraction stages
- Provides configurable file selection through inclusion/exclusion patterns
- Handles content preprocessing for optimal extraction
- Implements cross-file information merging and consolidation
- Produces both JSON and human-readable markdown outputs
- Supports both batch processing and targeted extraction
- Maintains detailed logging with uncaptured content tracking
- Measures performance metrics for process optimization
- Ensures consistent transformation from raw content to structured data

### SymbolicExtractor (`symbolic_extractor.ex`)

**Purpose:** Rule-based system for extracting precise metadata using predefined patterns from technical documents.

**Functionality:**

- Implements 90+ specialized extraction rules for technical and document metadata
- Uses sophisticated regex patterns with context awareness
- Applies automatic data transformations and normalization to captured values
- Handles multiple data formats including text, dates, and monetary values
- Implements exception handling for edge cases and missing data
- Detects uncaptured important content for continuous improvement
- Supports hierarchical data extraction across nested document structures
- Preserves extraction provenance for traceability
- Offers configurable extraction tolerance with customizable transformation rules
- Efficiently processes German-language technical documentation

### MetaDataProcessor (`meta_data_processor.ex`)

**Purpose:** Systematically detects presence of extracted metadata across document subchapters and enriches document summaries.

**Functionality:**

- Analyzes document structure to locate metadata instances across chapters
- Maps extracted metadata to relevant document sections
- Implements fuzzy matching for content with varying formats
- Uses intelligent word-level detection with partial matching
- Calculates confidence scores for metadata matches
- Updates document summaries with metadata distribution analytics
- Preserves original metadata values while tracking occurrences
- Supports cross-chapter metadata coherence validation
- Processes document trees hierarchically from root to leaf nodes
- Provides statistical measurement of metadata distribution patterns

### SubSymbolicExtractor (`subsymbolic_extractor.ex`)

**Purpose:** AI-powered information extraction using large language models for comprehensive content analysis.

**Functionality:**

- Processes document content using contextual language understanding
- Extracts structured information through prompt-based question answering
- Analyzes both full reports and individual document chapters
- Categorizes and processes content based on document structure
- Preserves semantic relationships between extracted information
- Supports both summary-only and detailed information extraction modes
- Intelligently limits token usage through content chunking
- Implements robust error handling and retry mechanisms for LLM processing
- Generates both machine-readable (JSON) and human-friendly outputs
- Adapts extraction strategies based on content categories
- Preserves document classification metadata for downstream processing

### PromptCreation (`prompt_creation.ex`)

**Purpose:** Creates specialized, context-aware prompts to optimize information extraction from domain-specific content.

**Functionality:**

- Generates tailored prompts based on document classification and content type
- Implements few-shot learning approaches with category-specific examples
- Creates comparison prompts for technical content validation
- Manages prompt complexity through intelligent content chunking
- Estimates token usage for optimal prompt sizing
- Supports different prompt variants for different extraction scenarios
- Maintains a repository of extraction templates for consistent processing
- Performs smart question grouping for related information extraction
- Formats extraction results for both machine and human consumption
- Handles special technical content comparison (text vs. structured data)

---

# Dataset Building Tools

The Dataset Building module provides specialized tools for creating, formatting, and organizing training datasets for different model training scenarios. It transforms processed documents into optimized instruction-following datasets with proper structuring and distribution.

## Coherence Datasets (Submodule)

### MixedCoherenceDS (`ds_mixed_coherence.ex`)

**Purpose:** Creates balanced, multi-source datasets that combine supervised and unsupervised learning examples for coherence training of merged models. Specifically designed to support models that have been merged using mergekit SLERP (Spherical Linear Interpolation) to ensure proper alignment and coherence between identical architectures combined into a single model.

**Functionality:**

- Integrates multiple dataset sources with controlled proportions (60% supervised, 40% unsupervised)
- Samples instructions from diverse sources to ensure representative training data
- Filters instructions based on maximum token limits for model compatibility
- Automatically detects and handles different instruction formats (supervised vs. unsupervised)
- Creates statistically balanced training, validation, and test splits (70/18/12)
- Ensures equal representation across source datasets through proportional sampling
- Handles both input/output pairs and raw text formats in a unified framework
- Produces JSONL-formatted datasets ready for model training

---

## Idea & Testing (Submodule)

### InstructionCreation (`instruction_creation.ex`)

**Purpose:** Generates specialized training instructions from document content using dynamic templates and contextual awareness.

**Functionality:**

- Creates instruction-response pairs from document chapters using customizable prompt templates
- Dynamically selects appropriate templates based on content categories and chapter types
- Implements category-specific metadata filtering using predefined mapping rules
- Processes question-answer pairs into standardized training formats
- Specializes instruction creation for technical data chapters vs. narrative content
- Formats content with consistent bracket notation (`[INST]`/`[/INST]`)
- Intelligently includes relevant context from previous chapter content when appropriate
- Implements task rotation to create diverse instruction phrasings for similar content
- Manages template-based prompt engineering with variable substitution
- Ensures proper JSON formatting and encoding for training compatibility

### DatasetBuilder (`dataset_builder.ex`)

**Purpose:** Orchestrates the end-to-end process of transforming document content into structured fine-tuning datasets.

**Functionality:**

- Processes document folders to extract chapter content, summaries, and metadata
- Maps document chapters to appropriate instruction categories
- Generates diverse instruction variations using rotated task descriptions
- Creates comprehensive chapter maps with metadata associations
- Implements category-aware dataset organization for specialized training
- Ensures proper distribution of examples across training/validation/test splits (75/20/5)
- Maintains category balance using round-robin distribution with adjustable priorities
- Produces both category-specific and combined dataset outputs
- Creates JSONL-formatted files ready for fine-tuning pipelines
- Implements intelligent sampling to handle varying dataset sizes

### MetaDataCategorization (`meta_data_for_categories.json`)

**Purpose:** Defines the mapping between document categories and relevant metadata fields for context-aware instruction creation.

**Functionality:**

- Maps 12+ document categories to relevant metadata fields
- Enables intelligent selection of contextually appropriate metadata
- Supports specialized data extraction for technical chapters
- Ensures consistent metadata inclusion across similar document types
- Provides category-specific context enrichment for training examples
- Optimizes instruction relevance through targeted metadata selection
- Maintains structured relationships between document categories and metadata fields

## Instruction Utilities (Submodule)

### MergeInstructions (`merge_instructions.ex`)

**Purpose:** Combines and redistributes instruction datasets from multiple sources into unified dataset splits with balanced representation.

**Functionality:**

- Merges instruction data from multiple source directories into a single combined dataset
- Creates statistically balanced training, validation, and test splits (80/15/5)
- Shuffles instructions to ensure random distribution across all splits
- Preserves original JSONL format while combining datasets
- Verifies data integrity by ensuring all instructions are accounted for after splitting
- Produces detailed distribution reports of instruction counts across splits
- Handles large-scale dataset merging with memory-efficient file operations
- Supports the creation of mixed datasets from diverse instruction sources
- Maintains consistent file naming conventions across dataset versions

### InstructionPreparation (`instruction_preparation.ex`)

**Purpose:** Analyzes and optimizes instruction datasets through token length estimation and filtering to ensure compatibility with model context windows.

**Functionality:**

- Filters instructions based on configurable maximum token length thresholds
- Intelligently detects and processes both supervised (input/output) and unsupervised (text-only) dataset formats
- Estimates token lengths using character-to-token ratio approximations with configurable ratios
- Generates comprehensive token length statistics by dataset split and instruction entry
- Creates detailed reports of token length distribution in structured JSON format
- Identifies and filters instructions exceeding token limits to prevent training issues
- Sorts instructions by token count for optimal batching and attention to outliers
- Preserves original instruction structure while applying token-based filtering
- Implements multi-format parsing with robust error handling for malformed entries
- Provides statistical analysis tools for dataset optimization before training

---

## Supervised Datasets (Submodule)

### QADataset (`ds_qa_dataset.ex`)

**Purpose:** Creates structured question-answer datasets from source documents with appropriate context for supervised fine-tuning.

**Functionality:**

- Extracts question-answer pairs from document structures following predefined layouts
- Groups related questions with their relevant context passages for coherent training examples
- Implements intelligent handling of unavailable answers (N.A.) with diverse replacement phrases
- Creates balanced dataset splits for training (80%), validation (15%), and test (5%)
- Provides consistent question formatting with standardized labeling (A, B, C)
- Organizes datasets by category for domain-specific training approaches
- Implements comprehensive content sanitization and text normalization
- Processes structured finetuning layouts to control context and question grouping
- Ensures optimal token distribution across question-answer examples
- Manages training data in both category-specific and combined formats

### MixedChapters (`ds_mixed_chapters.ex`)

**Purpose:** Builds comprehensive chapter-based training datasets that preserve document hierarchies and relationships between content sections.

**Functionality:**

- Processes hierarchical chapter structures with sophisticated parent-child relationship detection
- Handles complex chapter numbering systems including multi-level hierarchies (e.g., 5.2.1)
- Implements intelligent chapter type classification (main chapters, sub-chapters, technical sections)
- Creates training examples that preserve document structure context across chapter boundaries
- Integrates metadata with chapter content for enhanced contextual understanding
- Ensures balanced dataset creation across diverse chapter types using proportional sampling
- Manages technical vs. narrative content with specialized processing strategies
- Implements advanced chapter relationship analysis for context preservation
- Supports both content-only and summary-based dataset creation approaches
- Creates category-specific datasets with configurable inclusion/exclusion patterns

### QAInstructionCreation (`sv_qa_instruction_creation.ex`)

**Purpose:** Transforms question-answer content into standardized instruction-following format optimized for fine-tuning language models on Q&A tasks.

**Functionality:**

- Implements template-based instruction formatting with clearly defined sections (KONTEXT, FRAGEN, ANTWORTEN)
- Rotates through multiple task instruction variants to create diverse training examples
- Enforces structured prompt templates with consistent bracketed sections
- Ensures proper context presentation with enumerated content passages
- Processes both questions and answers through comprehensive sanitization pipeline
- Handles complex formatting requirements including nested JSON escape sequences
- Creates instruction pairs in standardized input/output format for training
- Implements robust error handling for malformed content processing
- Preserves proper formatting while removing potentially problematic elements (footnotes, images)
- Generates training examples that follow consistent patterns while maintaining content diversity

### MixedInstructionCreation (`sv_mixed_instruction_creation.ex`)

**Purpose:** Generates diverse instruction-following examples from chapter-based content with sophisticated context handling and format specialization.

**Functionality:**

- Supports multiple instruction types through template rotation for training diversity
- Implements specialized handling for technical chapters vs narrative content
- Creates context-aware instructions incorporating previous chapter content
- Builds hierarchical chapter relationships with proper formatting and tagging
- Processes metadata to provide relevant context for instruction generation
- Manages complex chapter relationships through parent-child hierarchy detection
- Implements comprehensive content sanitization and normalization pipeline
- Preserves document structure through intelligent chapter type identification
- Handles special formatting requirements for input/output instruction pairs
- Creates instructions that maintain consistent formatting while preserving document semantics
- Supports both summary-based and full-content instruction creation approaches
- Implements chapter-specific formatting based on position in document hierarchy

---

## Unsupervised Datasets (Submodule)

### SingleChaptersDS (`ds_single_chapters.ex`)

**Purpose:** Creates unsupervised training datasets from individual document chapters, preserving the document's categorical structure while ensuring proper distribution across training splits.

**Functionality:**

- Processes individual chapter files from document repositories into plain text format
- Implements intelligent content validation with minimum word count thresholds (10+ words)
- Performs content normalization through whitespace standardization and newline handling
- Creates category-specific datasets while maintaining original document classifications
- Implements proportional data splitting with 75/20/5 distribution for training/validation/test
- Uses round-robin sampling to ensure balanced category representation
- Handles special case distribution for small category samples (1-3 items)
- Adaptively adjusts distribution ratios to ensure minimum representation in each split
- Produces structured JSONL output in both category-specific and combined formats

### MultipleChaptersDS (`ds_multiple_chapters.ex`)

**Purpose:** Builds training datasets from consolidated multi-chapter document content, optimizing for context-aware language modeling with larger content windows.

**Functionality:**

- Processes documents where multiple chapters have been combined into single files
- Employs uniform 80/15/5 split distribution across training/validation/test sets
- Implements global shuffling strategy for better cross-category representation
- Applies consistent content validation with word count thresholds
- Performs text normalization through whitespace and newline standardization
- Creates comprehensive datasets that preserve broader document context
- Supports cross-referential learning through multi-chapter context windows
- Implements memory-efficient file processing for handling large document collections
- Produces standardized JSONL output compatible with PyTorch and HuggingFace training

### MDBDatasetDS (`ds_mdb_dataset.ex`)

**Purpose:** Transforms structured database records into natural language text for unsupervised pretraining on domain-specific structured data patterns.

**Functionality:**

- Converts structured MDB database records into natural language text format
- Transforms rows from multiple database tables into consistent textual representations
- Implements sophisticated JSON sanitization for handling complex character escaping
- Applies intelligent filtering to remove empty or zero-value fields based on context
- Formats database records with table names and field labels for semantic preservation
- Creates category-specific datasets based on database table classifications
- Implements balanced 75/20/5 distribution for training/validation/test splits
- Preserves domain-specific technical terminology through minimal text transformation
- Produces JSONL-formatted datasets with consistently structured text representations
- Supports category-aware round-robin sampling for balanced representation
- Implements robust error handling for malformed database records and JSON parsing issues

---

# Fine-Tuning Tools

The Fine-Tuning module provides tools for customizing foundation models through parameter-efficient training, model merging, and inference optimization. This module enables adaptation of large language models to domain-specific knowledge while maintaining general capabilities.

## FineTuningController (`fine_tuning_controller.ex`)

**Purpose:** Orchestrates the fine-tuning process through a unified Elixir interface that connects to Python-based machine learning tools.

**Functionality:**

- Coordinates unsupervised and supervised fine-tuning workflows
- Manages model merging operations for adapter integration
- Provides configuration management through JSON parameter files
- Implements real-time progress reporting during training
- Supports interactive and programmatic fine-tuning control
- Handles error conditions with graceful recovery mechanisms
- Controls both training and inference with consistent interfaces
- Optimizes training hyperparameters for different scenarios
- Maintains separate parameter sets for different training phases

## GatewayAPI (`gateway_API.ex`)

**Purpose:** Creates a reliable communication bridge between Elixir and Python environments for machine learning operations.

**Functionality:**

- Establishes lazy-loaded connections to Python processes
- Implements robust error handling with formatted error messages
- Provides real-time progress callbacks for long-running operations
- Supports hot-reloading of Python modules during development
- Manages Python process lifecycle with automated restart capabilities
- Implements module-specific verification for connection integrity
- Handles serialization/deserialization between Elixir and Python
- Processes sequential or parallel execution of Python functions
- Creates isolated Python environments for concurrent operations
- Provides debugging and logging capabilities for cross-language calls

## PyTorch Fine-Tuning (`pytorch_finetuning.py`)

**Purpose:** Implements parameter-efficient fine-tuning of large language models using advanced PyTorch techniques.

**Functionality:**

- Supports multiple training modes (unsupervised, supervised, mixed)
- Implements LoRA (Low-Rank Adaptation) for parameter-efficient training
- Manages dataset preparation with specialized tokenization strategies
- Optimizes memory usage through gradient accumulation and offloading
- Implements layer freezing patterns for targeted model adaptation
- Provides comprehensive progress reporting and metrics tracking
- Handles checkpoint management for training resumption
- Supports FlashAttention 2 for accelerated training
- Implements quantization-aware training (4-bit, 8-bit precision)
- Provides custom text generation capabilities for model evaluation
- Creates robust serialization for model checkpoint compatibility

## Model Merging (`merge_models.py`)

**Purpose:** Combines multiple fine-tuned models or integrates adapters with base models to create specialized capabilities.

**Functionality:**

- Merges LoRA adapters into base models for deployment
- Implements SLERP (Spherical Linear Interpolation) for model merging
- Supports layer-specific merging weights for heterogeneous adaptation
- Manages GPU memory efficiently during large model operations
- Handles precise data typing during model merging operations
- Creates consistent model serialization with proper metadata
- Supports both adapter integration and model-to-model merging
- Implements advanced checkpoint loading with security considerations
- Provides detailed progress reporting during memory-intensive operations
- Supports different merge strategies (SLERP, TIES-Merging, TaskArithmetic)

## Configuration Management

**Purpose:** Manages the complex configuration requirements of fine-tuning through structured parameter files.

**Functionality:**

### Fine-Tuning Parameters (`ft_params.json`)

- Defines training hyperparameters for different fine-tuning phases
- Configures dataset paths and preprocessing options
- Sets PEFT (Parameter-Efficient Fine-Tuning) configuration
- Controls optimization strategies (learning rates, weight decay)
- Defines model architecture modifications (layer freezing)
- Manages checkpoint and evaluation schedules
- Specifies training modes and specialized training techniques
- Controls hardware acceleration options
- Manages token length constraints for efficient processing

### Merge Parameters (`merge_params.json` & `merge_models_params.yaml`)

- Defines base models and adapter paths for integration
- Configures layer-specific interpolation weights for SLERP
- Controls precision during merging operations (FP16/32)
- Specifies output paths for merged models
- Manages model slicing for complex merging patterns
- Configures memory optimization during merging
- Defines element-wise operations for model tensor combination
- Provides advanced configuration for heterogeneous model merging
- Supports both direct adapter merging and inter-model merging
- Defines filter patterns for selective parameter merging

---

# Model Tools

The Model Tools module provides utilities for working with large language models in different formats, optimizing them for deployment, and managing their integration with inference systems. These tools leverage the llama.cpp library to enable efficient model conversion, quantization, and deployment in Ollama.

## ModelToolsController (`model_tools_controller.ex`)

**Purpose:** Central interface that coordinates model conversion and quantization operations through a unified API.

**Functionality:**

- Provides simplified access to model conversion and quantization capabilities
- Coordinates the workflow between different model processing steps
- Enables both programmatic and interactive usage of model tools
- Manages integration of processed models with Ollama inference system
- Supports parameter-based configuration through JSON files

## ModelConverter (`model_converter.ex`)

**Purpose:** Transforms models between different formats to optimize for deployment and inference.

**Functionality:**

- Converts models from HuggingFace safetensors format to GGUF (GPT-Generated Unified Format)
- Integrates converted models with Ollama inference system using customizable parameters
- Handles LoRA adapter conversion and merging with base models
- Supports various quantization options during conversion (F16, F32, etc.)
- Implements configurable model parameters through Modelfile creation
- Processes model configurations from structured JSON parameter files
- Manages Python interoperability through Erlport for conversion operations
- Handles Windows and cross-platform path conversions
- Sanitizes model names for proper deployment integration
- Creates temporary working directories for model integration operations

## ModelQuantizer (`model_quantizer.ex`)

**Purpose:** Reduces model size and optimizes inference speed through various quantization techniques.

**Functionality:**

- Implements multiple quantization types with different precision-speed tradeoffs
- Supports extensive quantization options (Q4_0 through Q8_0, IQ2/3/4, and F16/BF16/F32)
- Normalizes quantization type specifications for consistent processing
- Processes models from both GGUF and safetensors formats
- Generates properly named output files based on quantization type
- Integrates quantized models with Ollama for immediate deployment
- Manages Python interoperability for quantization operations
- Provides detailed logging of the quantization process
- Handles complex error conditions with appropriate reporting

## Python Wrapper Modules

### ConvertWrapper (`convert_wrapper.py`)

**Purpose:** Provides Python-level interface to llama.cpp conversion utilities with enhanced error handling.

**Functionality:**

- Executes llama.cpp's conversion scripts with proper parameter handling
- Supports conversion between model formats (safetensors to GGUF)
- Handles LoRA adapter merging with base models
- Implements robust error handling with detailed reporting
- Manages path conversions between Elixir and Python environments
- Automatically detects conversion script locations across different environments
- Captures and formats conversion output for structured reporting
- Provides detailed debug information about the execution environment
- Supports character encoding management for cross-language compatibility

### QuantizeWrapper (`quantize_wrapper.py`)

**Purpose:** Provides Python-level interface to llama.cpp quantization utilities.

**Functionality:**

- Executes llama.cpp's quantization tools with structured parameter handling
- Supports multiple quantization algorithms for optimizing models
- Implements comprehensive error reporting and output capturing
- Handles path management between different execution environments
- Provides detailed information about the quantization process
- Supports both GGUF and safetensors format models for quantization

## Configuration Files

### ConvParams (`conv_params.json`)

**Purpose:** Defines configuration parameters for model conversion operations.

**Functionality:**

- Specifies source and destination paths for model conversion
- Configures model integration parameters for Ollama deployment
- Defines quantization settings for conversion operations
- Supports multiple parameter sets for different conversion scenarios
- Provides Modelfile configuration parameters for inference optimization
- Enables flexible parameter management through a structured JSON format

### QuantParams (`quant_params.json`)

**Purpose:** Defines configuration parameters for model quantization operations.

**Functionality:**

- Specifies source and destination paths for model quantization
- Configures quantization algorithm and precision settings
- Defines integration settings for deployment after quantization
- Manages format-specific parameters (GGUF vs. safetensors)
- Supports multiple parameter sets for different quantization scenarios
- Enables consistent parameter management across quantization operations

---

# Scripts

The Scripts module provides standalone utilities for common tasks related to model management, data acquisition, and environment setup. These scripts can be run independently as needed during various stages of the AI framework's lifecycle.

## Model Download (`download_model.ipynb`)

**Purpose:** Simplifies the acquisition of pre-trained models or datasets from popular AI model repositories for further fine-tuning or inference.

**Functionality:**

- Downloads models from multiple sources including Kaggle datasets and Hugging Face repositories
- Provides automated authentication handling with repository-specific tokens
- Saves models to standardized locations for consistent framework access
- Supports both interactive notebook execution and headless operation
- Downloads model files to appropriate cache directories based on source platform
- Handles large file downloads with appropriate progress tracking
- Supports specific model version targeting when available
- Can be extended to support additional model repositories through similar interfaces
- Verifies downloaded model integrity when supported by the source platform
