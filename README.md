# Parameter-Efficient-Fine-Tuning-of-Large-Language-Models-for-Cognitive-Accessibility-

## 🔬 The Research Problem

Standard Large Language Models (LLMs) and Sequence-to-Sequence (Seq2Seq) models fail at academic accessibility due to two primary vulnerabilities:

1. **The Seq2Seq Deletion Bottleneck:** Traditional models (like FLAN-T5) often panic when facing high Flesch-Kincaid (FK) complexity, opting to silently *delete* complex clauses rather than simplify them, resulting in massive semantic loss.
2. **LLM Generative Infantilization:** Raw, off-the-shelf LLMs (like Llama 3) frequently suffer from "Generative Semantic Drift." When asked to simplify text, they hallucinate conversational filler, incorrectly redefine technical terms, and treat the reader like a child (e.g., translating "Retrieval-Augmented Generation" to "special helpers").

**The Goal:** Engineer an architecture that forces a massive drop in reading difficulty (Flesch-Kincaid) while maintaining a near-perfect retention of academic meaning (BERTScore).

---

## 🏗️ Architectural Evolution (Ablation Study)

This project evolved through a rigorous ablation study of four distinct architectures:

1. **Baseline V1 (Safe LoRA Seq2Seq):** Anchored syntax safely but failed to adequately lower the reading grade of Post-Graduate text.
2. **Baseline V2 (Aggressive LoRA Seq2Seq):** Forced the reading grade down, but suffered from the "Deletion Bottleneck," destroying the BERTScore.
3. **Raw LLM (Llama-3.2-3B via Ollama):** Hallucinated facts (e.g., defining "zero-shot" as "asking zero questions") and utilized conversational filler, paradoxically raising the reading difficulty to Grade 15.3.
4. **🏆 Proposed Architecture (QLoRA Llama-3.2-3B via Unsloth):** The final, state-of-the-art solution. We fine-tuned Llama-3.2-3B on the **ASSET dataset** using strict, negative-constraint prompting (`DO NOT hallucinate`, `DO NOT add filler`). This mathematically burned structural translation rules into the model's cross-attention weights.

---

## 📊 Quantitative Results

The final Unsloth-optimized Llama-3.2-3B model was benchmarked against two highly rigorous, unseen datasets.

### Test 1: ASSET Test Split (100 Unseen Sentences)

Evaluated against 10 distinct human expert references per sentence.

* 📉 **Flesch-Kincaid Grade:** 11.4 (Original) ➡️ **9.8 (Model)**
* 🧠 **BERTScore (F1):** **0.9810** *(Near-perfect semantic retention)*
* 📝 **BLEU Score:** **93.38** *(Surgical, human-level precision)*
* 🌟 **SARI Score:** **50.87** *(World-class text simplification performance)*

### Test 2: Custom Academic Stress Test (20 Extreme Domain Sentences)

Tested on PhD-level jargon spanning Law, Medicine, NLP, and Quantum Physics.

* 📉 **Average FK Grade:** 20.9 (Original) ➡️ **12.4 (Simplified)** *(An 8.5-year drop in educational requirement)*
* 🧠 **Average BERTScore:** **0.8670**

---

## 💻 Tech Stack & Hardware Configuration

This entire pipeline was engineered to run locally on consumer-grade hardware (**8GB VRAM RTX 3070 Laptop GPU**).

* **Deep Learning Framework:** PyTorch (CUDA 12.1)
* **Optimization Engine:** Unsloth (For 2x faster, 60% less VRAM QLoRA training)
* **Quantization:** `bitsandbytes` (4-bit NF4 precision)
* **NLP Metrics:** Hugging Face `evaluate`, `textstat`, `sacrebleu`, `bertscore`
* **Web Dashboard:** Flask, HTML5/CSS3 (Styled with scotopic-safe contrast and Lexend font for dyslexia readability).

---

## 🚀 Installation & Usage

### 1. Environment Setup

Ensure you have Python 3.12+ and CUDA 12.1 installed.

```bash
git clone https://github.com/yourusername/AccessibleText.git
cd AccessibleText

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth & Dependencies
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" xformers trl peft accelerate bitsandbytes

# Install Web and Metric dependencies
pip install flask textstat evaluate sacrebleu PyMuPDF ollama

```

### 2. Running the Live Web Dashboard

The project includes a 4-panel live comparison dashboard (`app.py`) allowing users to upload PDFs or paste text, select sentences, and watch the 4 architectures process the text in real-time.

1. Ensure Ollama is running in the background for the baseline test: `ollama run llama3.2:3b`
2. Start the Flask server:

```bash
python app.py

```

3. Open `http://127.0.0.1:5000` in your browser.

### 3. Running the Benchmarks

To replicate the thesis data on the unseen ASSET test split or the custom 20-sentence stress test:

```bash
python unsloth_asset_eval.py
# OR
python custom_20_eval.py

```

*(Note: Windows users may see a `torch.int1` warning from `torchao` on boot. A runtime hotfix is included at the top of all executable scripts to automatically bypass this.)*

---

## 🎓 Academic Contribution & Future Scope

This research proves that while raw LLMs are unsuitable for high-stakes accessibility tasks due to "Generative Semantic Drift," applying parameter-efficient fine-tuning (QLoRA) with strict negative-constraint formatting yields a highly deterministic, safe, and academically rigorous simplification engine.

Future scope includes expanding the training dataset to include heavily formatted multi-modal text (e.g., tables and LaTeX equations) and deploying the model as a lightweight edge-device browser extension.
