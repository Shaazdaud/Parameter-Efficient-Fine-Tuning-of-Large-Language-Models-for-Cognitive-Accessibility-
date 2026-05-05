import os
os.environ["TRITON_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.config.disable = True # Completely disables the Linux compiler for Windows

# --- BLEEDING EDGE PYTORCH HOTFIX ---
if not hasattr(torch, 'int1'): torch.int1 = torch.int8
if not hasattr(torch, 'int2'): torch.int2 = torch.int8
if not hasattr(torch, 'int3'): torch.int3 = torch.int8
if not hasattr(torch, 'int4'): torch.int4 = torch.int8
if not hasattr(torch, 'int5'): torch.int5 = torch.int8
if not hasattr(torch, 'int6'): torch.int6 = torch.int8
if not hasattr(torch, 'int7'): torch.int7 = torch.int8
# ------------------------------------

import textstat
import ollama
import evaluate
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from unsloth import FastLanguageModel
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

print("=" * 80)
print("🚀 BOOTING ACCESSIBLE-TEXT MASTER THESIS DASHBOARD")
print("Loading 4 Models into VRAM... (Please wait)")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load V1 Model (Safe Seq2Seq)
print("1/4 Loading V1...")
v1_path = "./merged_accessible_text_model" 
v1_tokenizer = AutoTokenizer.from_pretrained(v1_path)
v1_model = AutoModelForSeq2SeqLM.from_pretrained(v1_path).to(device)

# 2. Load V2 Model (Aggressive Seq2Seq)
print("2/4 Loading V2...")
from peft import PeftModel
v2_base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
v2_model = PeftModel.from_pretrained(v2_base_model, "./final_edge_model_v2").merge_and_unload()
v2_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# 3. Load Custom Fine-Tuned Unsloth LLM (Thesis Model)
print("3/4 Loading Custom Fine-Tuned Llama...")
custom_model, custom_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth_llama_accessible", 
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(custom_model)

# 4. Load Evaluator
print("4/4 Loading BERTScore Metric...")
bertscore = evaluate.load("bertscore")
def calc_bert(pred, ref):
    try:
        return round(bertscore.compute(predictions=[pred], references=[ref], model_type="distilbert-base-uncased")['f1'][0], 4)
    except:
        return 0.0000

print("\n✅ System Ready! Running on http://127.0.0.1:5000")

# --- PROMPT TEMPLATES ---
custom_prompt_template = """### System Directive:
You are an academic text simplification engine designed for university students with dyslexia. 
Your primary objective is to break long, complex sentences into short, punchy sentences (maximum 12 words per sentence) while preserving all technical jargon and facts.
- SPLIT run-on sentences at commas and conjunctions.
- DO NOT add conversational filler (e.g., "Here is the simplified text:").
- DO NOT invent or hallucinate outside context.
Output ONLY the simplified text.

### Complex Text:
{}

### Simplified Text:
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'Empty filename'}), 400

    try:
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted_text = ""
        for page in doc:
            extracted_text += page.get_text("text") + "\n\n"
        doc.close()
        return jsonify({'text': extracted_text.strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    if not text: return jsonify({'error': 'No text provided'}), 400
    return jsonify({'flesch_kincaid': round(textstat.flesch_kincaid_grade(text), 1)})

@app.route('/simplify', methods=['POST'])
def simplify():
    data = request.json
    original_text = data.get('text', '')
    if not original_text: return jsonify({'error': 'No text provided'}), 400
    
    # 1. --- V1 Generation (Safe) ---
    in_v1 = v1_tokenizer("simplify this text for a person with dyslexia: " + original_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    out_v1 = v1_tokenizer.decode(v1_model.generate(**in_v1, max_length=128, num_beams=4, early_stopping=True)[0], skip_special_tokens=True)
    
    # 2. --- V2 Generation (Aggressive) ---
    in_v2 = v2_tokenizer("simplify this text for a person with dyslexia: " + original_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    out_v2 = v2_tokenizer.decode(v2_model.generate(**in_v2, max_length=128, num_beams=4, length_penalty=0.8, early_stopping=True)[0], skip_special_tokens=True)
    
    # 3. --- RAW LLM (Guide's Baseline via Ollama) ---
    try:
        raw_prompt = "Simplify this text for someone with dyslexia. Make it short and easy to read:\n"
        resp = ollama.chat(model='llama3.2:3b', messages=[
            {'role': 'user', 'content': raw_prompt + original_text}
        ])
        out_raw = resp['message']['content'].strip()
    except Exception as e:
        print(f"Ollama Error: {e}")
        out_raw = "[Ensure Ollama is running in the background!]"
        
    # 4. --- FINE-TUNED UNSLOTH LLM (Thesis Model) ---
    in_custom = custom_tokenizer([custom_prompt_template.format(original_text)], return_tensors="pt").to(device)
    outputs = custom_model.generate(
        **in_custom, 
        max_new_tokens=128, 
        use_cache=True, 
        temperature=0.6, 
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=custom_tokenizer.eos_token_id
    )
    decoded_custom = custom_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    out_custom = decoded_custom.split("### Simplified Text:\n")[-1].strip()

    # --- Calculate Metrics ---
    metrics = {
        'v1': {'text': out_v1, 'fk': round(textstat.flesch_kincaid_grade(out_v1), 1), 'bert': calc_bert(out_v1, original_text)},
        'v2': {'text': out_v2, 'fk': round(textstat.flesch_kincaid_grade(out_v2), 1), 'bert': calc_bert(out_v2, original_text)},
        'raw_llm': {'text': out_raw, 'fk': round(textstat.flesch_kincaid_grade(out_raw), 1), 'bert': calc_bert(out_raw, original_text)},
        'custom_llm': {'text': out_custom, 'fk': round(textstat.flesch_kincaid_grade(out_custom), 1), 'bert': calc_bert(out_custom, original_text)}
    }
    
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)