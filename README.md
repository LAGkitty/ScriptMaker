# ScriptMaker

A versatile GUI application for automatically converting natural language commands into functional code in multiple programming languages.

## Overview

ScriptMaker is a desktop application built with Python and Tkinter that allows users to write simple commands in natural language and convert them into functioning code in various programming languages. It supports both rule-based translation and AI-powered code generation.

Created by Claude (Anthropic)

## Features

- **Multi-language Support**: Generate code in Lua, Python, JavaScript, C#, and Java
- **Multiple Translation Modes**:
  - Rule-based translation (works offline, no dependencies)
  - AI-powered translation with OpenAI's API (requires API key)
  - Local AI-powered translation using transformer models (runs offline)
- **User-friendly Interface**: Clean and intuitive GUI built with Tkinter
- **Code Management**: Copy to clipboard or save to file with proper extension
- **Examples**: Pre-defined examples to help you get started

## Installation

### Prerequisites

- Python 3.6 or higher

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ScriptMaker.git
cd ScriptMaker

# Install the required dependencies
pip install tkinter
```

### Enhanced Installation (with AI capabilities)

For online AI features:
```bash
pip install openai
```

For offline AI features:
```bash
pip install transformers torch
```

For GPU acceleration (if you have a compatible NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

1. Run the application:
   ```bash
   python ScriptMaker.py
   ```

2. Select your desired translation mode:
   - **Rule-Based**: Fast and works offline
   - **AI-Powered (Online)**: Better quality but requires OpenAI API key
   - **AI-Powered (Offline)**: Good quality without external API but requires local models

3. Choose your target programming language

4. Enter your natural language commands (one per line)
   - Example: "create a button", "make button blue", "add click event"

5. Click "Convert Script" to generate the code

6. Save or copy the generated code

## Example Commands

```
create button
make button size 200x50
set button color blue
add text 'Click Me'
when clicked show message
```

## Advanced Usage

### Using Local AI Models

1. Select "AI-Powered (Offline)" mode
2. Choose a model from the dropdown or specify a custom model path
3. Select your device (CPU or GPU)
4. Click "Download/Setup Model" to initialize the model
5. Convert your script using the local AI model

### Using OpenAI API

1. Select "AI-Powered (Online)" mode
2. Enter your OpenAI API key
3. Select the model (e.g., gpt-3.5-turbo, gpt-4)
4. Test the connection
5. Convert your script using the OpenAI API

## Requirements

- Python 3.6+
- tkinter (included in standard Python installation)
- openai (optional, for online AI)
- transformers (optional, for offline AI)
- torch (optional, for offline AI)

## License

This project is open source and available under the MIT License.

## Acknowledgements

- This tool was created with Python and Tkinter
- AI capabilities are powered by OpenAI API and Hugging Face Transformers
- Created by Claude (Anthropic)
