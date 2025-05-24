# PII Removal Tool

## Overview

The PII Removal Tool is a web application designed to identify and anonymize Personally Identifiable Information (PII) from text content. This tool is part of the Framework for Developing (Hybrid) AI Applications.

## Features

- **Text Input Processing**: Easily paste text directly into the web interface for PII removal
- **File Processing**: Upload files containing text for automatic PII detection and anonymization
- **Flexible Anonymization Options**: Configure how different types of PII should be handled
- **Phoenix-based Web Interface**: User-friendly UI built with Phoenix Framework
- **Microsoft Presidio Integration**: Leverages the powerful PII detection and anonymization capabilities of Microsoft Presidio

## PII Types Detected

- Names (first, last, full)
- Email addresses
- Phone numbers
- Addresses
- Social security numbers
- Credit card numbers
- Dates of birth
- IP addresses
- And more...

## Setup Instructions

### Prerequisites

- Elixir 1.16 or later
- Erlang/OTP 27 or later
- Phoenix Framework 1.7 or later
- Python 3.10 or later (for Presidio integration)

### Installation

1. **Clone the repository**

2. **Navigate to the Phoenix UI directory**

   ```bash
   cd phoenix_UI
   ```

3. **Install Elixir dependencies**

   ```bash
   mix deps.get
   mix deps.compile
   ```

4. **Set up the Python environment** (if needed for Presidio)

   ```bash
   cd ../Presidio
   # Create a virtual environment
   python -m venv venv
   # Activate the environment
   .\venv\Scripts\Activate.ps1

   ```

5. **Return to the Phoenix UI directory**

   ```bash
   cd ../phoenix_UI
   ```

6. **Start the Phoenix server**

   ```bash
   iex -S mix phx.server
   ```

7. **Access the web application**
   Open your browser and navigate to: [http://localhost:4000](http://localhost:4000)

## Usage

### Text Input Processing

1. Navigate to the "Text Input" tab
2. Paste your text containing PII into the provided text area
3. Configure anonymization options if needed
4. Click "Process Text"
5. View and copy the anonymized text from the results area

### File Processing

1. Navigate to the "File Upload" tab
2. Click "Choose File" and select a text file to process
3. Configure anonymization options if needed
4. Click "Upload and Process"
5. Download the processed file with PII removed

## Configuration

The tool allows customization of detection and anonymization settings through the web interface. Advanced configurations can be modified in the application settings.

## Troubleshooting

### Common Issues

- **Server won't start**: Ensure all dependencies are installed with `mix deps.get` and compiled with `mix deps.compile`
- **Presidio integration issues**: Check that Python and required packages are correctly installed
- **Performance issues with large files**: Large files may take longer to process; consider splitting them into smaller chunks

### Resolving Python Path Issues

Python path configuration is crucial for the ErlPort integration to work correctly. Environment variables can differ between VS Code, terminal sessions, and system settings.

#### Windows

Reset environment variables in PowerShell to ensure consistency:

```powershell
# Reset PATH to system + user values
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")

# Reset Python environment variables
$env:PYTHONHOME = [System.Environment]::GetEnvironmentVariable("PYTHONHOME", "Machine")
$env:PYTHONPATH = [System.Environment]::GetEnvironmentVariable("PYTHONPATH", "Machine")

# Add your Python virtual environment to PATH (for ErlPort to find the correct Python)
$pythonVenvPath = "C:\Users\YourUsername\path\to\FW_TOOLS\PII_removal\Presidio\venv\Scripts"
$env:PATH = "$pythonVenvPath;$env:PATH"
```

For persistent settings in PowerShell, add to your profile:

```powershell
# Open your PowerShell profile
if (!(Test-Path -Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force
}
notepad $PROFILE

# Add the environment settings to your profile so they're set every time PowerShell starts
```

#### Linux (Ubuntu)

Reset and configure Python paths in Bash:

```bash
# Reset PATH to system default
export PATH=$(getconf PATH)

# Reset Python environment variables
unset PYTHONHOME
unset PYTHONPATH

# Add the virtual environment's Python to your PATH
export PATH="/path/to/FW_TOOLS/PII_removal/Presidio/venv/bin:$PATH"
```

For persistent settings, add to your `~/.bashrc` or `~/.bash_profile`.

#### VS Code-Specific Configuration

VS Code may use different environment variables than your terminal:

1. Create a `.env` file in your project root:

   ```
   PYTHON_EXECUTABLE=C:/Users/YourUsername/path/to/FW_TOOLS/PII_removal/Presidio/venv/Scripts/python.exe
   ```

2. Configure VS Code settings in `.vscode/settings.json`:
   ```json
   {
     "terminal.integrated.env.windows": {
       "PATH": "${workspaceFolder}/FW_TOOLS/PII_removal/Presidio/venv/Scripts;${env:PATH}"
     },
     "terminal.integrated.env.linux": {
       "PATH": "${workspaceFolder}/FW_TOOLS/PII_removal/Presidio/venv/bin:${env:PATH}"
     }
   }
   ```

#### Troubleshooting Python Path

If you're still experiencing path issues:

1. **Verify Python executable path**:

   ```powershell
   # In PowerShell
   (Get-Command python).Path
   ```

2. **Print Python search paths in your Python scripts**:

   ```python
   import sys
   print("Python search paths:", sys.path)
   ```

3. **Check Python environment details**:

   ```powershell
   python -m site
   ```

4. **Use absolute paths** for maximum reliability when configuring Python paths.

### Log Location

Logs can be found in the standard Elixir/Phoenix location. Check the console output when running with `iex -S mix phx.server`.
