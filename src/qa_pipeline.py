from transformers import pipeline

def load_gpt2_pipeline(model_name='gpt2'):
    return pipeline('text-generation', model=model_name)

def generate_answer(question, context, gpt2_pipeline, max_length=100):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    response = gpt2_pipeline(prompt, max_length=max_length, num_return_sequences=1)
    return response[0]['generated_text']

if __name__ == "__main__":
    gpt2_pipeline = load_gpt2_pipeline()
    question = "What is the punishment for theft?"
    context = "In India, the punishment for theft is outlined in Section 379 of the Indian Penal Code. The section states that whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with a fine, or with both."
    answer = generate_answer(question, context, gpt2_pipeline)
    print(f"Generated Answer:\n{answer}")
