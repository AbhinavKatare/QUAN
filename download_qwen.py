import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen1.5-14B-Chat"

# Define the 4-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Downloading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer downloaded successfully.")

print(f"Downloading and loading model {model_name}...")
# Load the model with the 4-bit quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
print("Model loaded successfully. Ready for inference!")

"""
print(f"Downloading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer downloaded successfully.")

print(f"Downloading and loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Uses the most efficient data type for your hardware
    device_map="auto"    # Automatically uses GPU if available
)
print("Model loaded successfully. Ready for inference!")"""


# Example of how to use the model
prompt = "Tell me a short story about a brave knight and a wise owl."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True

)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

generated_ids = [
    output_id[len(model_inputs.input_ids[0]):] for output_id in generated_ids
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\nGenerated Response:")
print(response)


"""  
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen1.5-14B-Chat"

# 4-bit quantization configuration for bitsandbytes
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Download tokenizer
print(f"Downloading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer downloaded successfully.")

# Download and load model with quantization config
print(f"Downloading and loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
print("Model loaded successfully. Ready for inference!")

# Example prompt and messages formatted for Qwen chat
prompt = "Tell me a short story about a brave knight and a wise owl."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and move inputs to correct device
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate response with a maximum number of new tokens
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# Remove prompt tokens from output and decode response
generated_ids = [
    output_id[len(model_inputs.input_ids[0]):] for output_id in generated_ids
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\nGenerated Response:")
print(response)

"""   




import torch
import platform
import psutil
import GPUtil
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

# ==============================
# SETTINGS
# ==============================
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"   # ✅ 7B instead of 14B
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Enable detailed logging
logging.set_verbosity_info()

# ==============================
# SYSTEM CHECKS
# ==============================
print("="*60)
print("🔍 System Information")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"Torch: {torch.__version__}")
print("="*60)

# Check RAM
ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
print(f"🖥️  System RAM: {ram_gb} GB")

# Check GPU
gpus = GPUtil.getGPUs()
if gpus:
    gpu = gpus[0]
    gpu_name = gpu.name
    gpu_mem = round(gpu.memoryTotal, 2)
    print(f"🎮 GPU Detected: {gpu_name} ({gpu_mem} GB VRAM)")
else:
    gpu = None
    gpu_mem = 0
    print("⚠️  No GPU detected. Running on CPU.")

# ==============================
# MODEL LOADING STRATEGY
# ==============================
if gpu and gpu_mem >= 12:
    # Full 7B model can fit into VRAM
    load_dtype = torch.float16
    device_map = "auto"
    print("✅ Strategy: Load full model on GPU.")
elif gpu and 6 <= gpu_mem < 12:
    # Partial GPU + CPU offload
    load_dtype = torch.float16
    device_map = "auto"
    print("✅ Strategy: Partial GPU + CPU offload.")
else:
    # CPU only fallback
    load_dtype = torch.float32
    device_map = "cpu"
    print("✅ Strategy: Run fully on CPU (slower).")

# ==============================
# LOAD TOKENIZER & MODEL
# ==============================
print(f"\n📥 Downloading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("✅ Tokenizer ready.")

print(f"📥 Downloading & initializing model {MODEL_NAME}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=load_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    exit(1)

# ==============================
# INFERENCE FUNCTION
# ==============================
def chat(prompt: str, system_prompt: str = "You are a helpful assistant."):
    """Generate a response from Qwen-7B."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Prepare inputs
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )

    # Remove prompt tokens
    generated_ids = [
        output_id[len(model_inputs.input_ids[0]):] for output_id in generated_ids
    ]

    # Decode output
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ==============================
# TEST RUN
# ==============================
if __name__ == "__main__":
    test_prompt = "Write me a motivational message for someone starting their journey as an AI developer."
    print("\n📝 Running test inference...")
    response = chat(test_prompt)
    print("\n🤖 Generated Response:")
    print(response)
