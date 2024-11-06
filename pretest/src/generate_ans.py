import pandas as pd
from transformers import pipeline

def generate_answers(model, dataset, output_path):
    generator = pipeline("text-generation", model=model[0], tokenizer=model[1])

    results = []

    for _, row in dataset.iterrows():
        question = row["Question"]
        disease = row["Disease"]
        
        generated = generator(question, max_length=100, num_return_sequences=1)[0]["generated_text"]
        
        results.append({
            "No": row["No"],
            "Disease": disease,
            "Question": question,
            "Generated_Answer": generated
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
