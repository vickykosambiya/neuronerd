"""
NeuroNerd Chatbot - Streamlit App
Inference using the fine-tuned neuroscience model from HuggingFace.
"""

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configuration
MODEL_ID = "VickyK09/neuronerd-llama-8b"
MAX_NEW_TOKENS = 512

# Page config
st.set_page_config(
    page_title="NeuroNerd 🧠",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🧠 NeuroNerd</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Your AI Neuroscience Expert</p>", unsafe_allow_html=True)

# Load model (cached)
@st.cache_resource
def load_model():
    """Load the model and tokenizer."""
    with st.spinner("Loading NeuroNerd model... (this may take a few minutes)"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        return model, tokenizer

# Prompt template (Alpaca format)
def format_prompt(instruction: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a response from the model."""
    formatted_prompt = format_prompt(prompt)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
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
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    **NeuroNerd** is a fine-tuned Llama 3.1 8B model specialized in neuroscience.
    
    **Training Data:**
    - Computational Exploration in Cognitive Neuroscience
    - Foundations of Neuroscience
    - The Cognitive Neurosciences (Gazzaniga)
    
    **Model:** [HuggingFace](https://huggingface.co/VickyK09/neuronerd-llama-8b)
    """)
    
    st.markdown("---")
    
    st.markdown("## 💡 Example Questions")
    examples = [
        "What is the role of dopamine in reward learning?",
        "How do mirror neurons contribute to action understanding?",
        "Explain the difference between LTP and LTD.",
        "What is the function of the prefrontal cortex?",
    ]
    
    for example in examples:
        if st.button(example, key=example):
            st.session_state.pending_question = example
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check for pending question from sidebar
if "pending_question" in st.session_state:
    pending = st.session_state.pending_question
    del st.session_state.pending_question
    st.session_state.messages.append({"role": "user", "content": pending})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about neuroscience..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                model, tokenizer = load_model()
                response = generate_response(model, tokenizer, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure you have enough GPU memory (~16GB VRAM) to run this model.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.8em;'>Built with ❤️ using Unsloth & Streamlit</p>", unsafe_allow_html=True)
