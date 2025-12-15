# 🧠 NeuroNerd

A fine-tuned Llama 3.1 8B model specialized in **neuroscience and cognitive science**. Built with a custom data pipeline and trained using Unsloth.

## 🔗 Model

**HuggingFace:** [VickyK09/neuronerd-llama-8b](https://huggingface.co/VickyK09/neuronerd-llama-8b)

## 📁 Project Structure

```
neuronerd/
├── app.py                  # Streamlit chatbot for inference
├── config.py               # Pipeline configuration
├── run_pipeline.py         # Data generation orchestrator
├── requirements.txt        # Dependencies
├── scripts/                # Data pipeline scripts
│   ├── 01_extract_text.py  # PDF text extraction
│   ├── 02_preprocess.py    # Text cleaning & chunking
│   ├── 03_generate_qa.py   # Q&A generation (Gemini)
│   ├── 04_quality_control.py # Data filtering
│   └── 05_finalize.py      # Train/test split
├── training/               # Model training
│   └── train_runpod.py     # Unsloth training script (RunPod)
├── pdfs/                   # Source textbooks
└── output/                 # Generated data
    └── final/              # train.jsonl & test.jsonl
```

## 🚀 Quick Start

### Run the Chatbot

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Generate Dataset (from PDFs)

```bash
# Set your Gemini API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run the full pipeline
python run_pipeline.py --all
```

### Train on RunPod

1. Upload to GitHub
2. Create a RunPod pod with `unslothai/unsloth:latest` Docker image
3. Clone repo and run:
```bash
python training/train_runpod.py
```

## 📚 Training Data Sources

- *Computational Exploration in Cognitive Neuroscience*
- *Foundations of Neuroscience* (Casey Henley)
- *The Cognitive Neurosciences* (Gazzaniga et al.)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Data Generation | Gemini 2.5 Flash Lite |
| Training | Unsloth, QLoRA, HuggingFace TRL |
| Infrastructure | RunPod (A100 GPU) |
| Inference | Streamlit, Transformers |
| Model | Llama 3.1 8B |

## 📊 Stats

- **Dataset:** 8,200+ Q&A pairs
- **Training Time:** ~25 minutes (A100)
- **Total Cost:** < $3

## 📄 License

Model released under the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).

---

*Built with ❤️ using [Unsloth](https://github.com/unslothai/unsloth)*
