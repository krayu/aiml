from transformers import pipeline

# Load pre-trained GPT-2 model for text generation
generator = pipeline("text-generation", model="gpt2")

# Generate text
prompt = "Once upon a time"
output = generator(prompt, max_length=50, num_return_sequences=1)

# Print generated text
print(output[0]["generated_text"])