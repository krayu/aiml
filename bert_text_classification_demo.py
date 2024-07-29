from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # Pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Function to classify text using the BERT model
def classify_text(text):
    # Encode the input text (convert to token IDs)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Forward pass through the model
    with torch.no_grad():  # No gradient computation needed for inference
        outputs = model(**inputs)
    
    # Get predicted class (0 or 1 based on classification)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class

# Example usage
if __name__ == "__main__":
    text = "I love using BERT for text classification!"  # Example text
    prediction = classify_text(text)
    
    print(f"Predicted class: {prediction}")