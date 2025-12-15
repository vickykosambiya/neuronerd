"""
NeuroNerd Chatbot - Gradio App
Run with: python gradio_app.py
Creates a public shareable link automatically!
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configuration
MODEL_ID = "VickyK09/neuronerd-llama-8b"
MAX_NEW_TOKENS = 512

print("🧠 Loading NeuroNerd model...")
print("This may take a few minutes on first run...")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("✅ Model loaded!")

def format_prompt(instruction: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def chat(message, history):
    """Generate a response."""
    prompt = format_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="🧠 NeuroNerd",
    description="Your AI Neuroscience Expert - Ask me anything about neuroscience!",
    examples=[
        "What is the role of dopamine in reward learning?",
        "How do mirror neurons contribute to action understanding?",
        "Explain the difference between LTP and LTD.",
        "What is the function of the prefrontal cortex?",
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # share=True creates a public URL!
    demo.launch(share=True, server_name="0.0.0.0")
