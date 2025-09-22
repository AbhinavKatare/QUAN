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
