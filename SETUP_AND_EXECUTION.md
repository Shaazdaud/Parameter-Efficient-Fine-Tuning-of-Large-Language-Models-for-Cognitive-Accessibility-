# Setup and Execution Instructions

This document outlines the software requirements and the step-by-step instructions to execute the AccessibleText Simplification application.

## 1. Software & Tools Used

### Core Frameworks & Languages
- **Python (3.9+)**: The primary programming language used for the backend logic and data orchestration.
- **Flask**: The web framework used to serve the dashboard and handle API requests (PDF uploads, text simplification).
- **PyTorch**: Used for deep learning tensor operations and loading neural networks on the GPU (CUDA).

### Natural Language Processing (NLP) & Deep Learning Libraries
- **Transformers (Hugging Face)**: Used to load and inference Seq2Seq models (`AutoModelForSeq2SeqLM`, `AutoTokenizer`), specifically the Flan-T5 models.
- **PEFT (Parameter-Efficient Fine-Tuning)**: Used to merge custom LoRA adapters back into the base T5 model on the fly.
- **Unsloth**: A specialized library used for fast, memory-efficient inference of quantized large language models (Llama family).
- **Ollama (Python Client)**: Used to communicate with the local instance of Ollama to inference the raw/baseline `llama3.2:3b` model.

### Utility & Evaluation Libraries
- **PyMuPDF (`fitz`)**: Used to extract raw text context from user-uploaded PDF files.
- **Textstat**: Used to calculate the Flesch-Kincaid grade level for measuring text readability.
- **Evaluate (Hugging Face)**: Used alongside the `bertscore` metric to evaluate semantic similarity between the original complex text and the simplified text.

### External Software
- **Ollama Desktop/CLI**: Required to run the local baseline Llama model natively in the background.

---

## 2. Instructions to Execute the Code

### Prerequisites
1. **Python Installation:** Ensure you have Python installed (preferably version 3.9, 3.10, or 3.11).
2. **CUDA / GPU Drivers:** For optimal performance, an NVIDIA GPU with appropriate CUDA drivers installed is highly recommended. The code will default to `cpu` if CUDA is unavailable, but this will significantly slow down model inference.

### Step 1: Install Python Dependencies
Open your terminal (or Command Prompt / PowerShell) in the project directory and run the following command to install all required libraries:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version as needed
pip install flask textstat ollama evaluate transformers accelerate peft unsloth PyMuPDF bert_score
```

*Note: The `unsloth` package requires specific Linux-based compilers, but the code implements a Windows disabling patch for `.dynamo`. Please consult the Unsloth GitHub repository for exact Windows/CUDA installation steps if native pip fails.*

### Step 2: Set Up Ollama
1. Download and install **Ollama** from `https://ollama.com/`.
2. Open a terminal and pull the required baseline model by running:
   ```bash
   ollama run llama3.2:3b
   ```
   *Leave this running in the background, or ensure the Ollama service is active. The application will query it via the API.*

### Step 3: Verify Local Model Checkpoints
Ensure that the following fine-tuned model directories exist within your root project folder (these were created during training):
- `./merged_accessible_text_model` (V1 Seq2Seq)
- `./final_edge_model_v2` (V2 PEFT Adapters)
- `./unsloth_llama_accessible` (Custom Llama Model)

### Step 4: Run the Application
In your terminal, navigate to the project directory containing `app.py` and start the Flask server:

```bash
python app.py
```

### Step 5: Access the Dashboard
Once the console displays the message indicating the models have been loaded and the server is ready, open your web browser and navigate to:

**http://127.0.0.1:5000** 

Upload a PDF or input text directly to test the multi-model simplification engine.