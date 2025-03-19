from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Load Legal-BERT for classification
tokenizer_bert = AutoTokenizer.from_pretrained("lexpredict/legal-bert")
model_bert = AutoModelForSequenceClassification.from_pretrained("lexpredict/legal-bert")

# Load GPT-2 for text generation
tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt = GPT2LMHeadModel.from_pretrained("gpt2")

# Example retrieved chunk
retrieved_chunk = "The penalty for theft is imprisonment up to 3 years."

# Classify the relevance of the chunk (optional step)
inputs = tokenizer_bert(retrieved_chunk, return_tensors="pt")
outputs = model_bert(**inputs)
logits = outputs.logits
relevance_score = logits.softmax(dim=1)  # This is an optional score

# Generate a response using GPT-2 based on the retrieved chunk
input_ids = tokenizer_gpt.encode(retrieved_chunk, return_tensors="pt")
generated_text = model_gpt.generate(input_ids, max_length=150, num_return_sequences=1)

response = tokenizer_gpt.decode(generated_text[0], skip_special_tokens=True)
print(response)
