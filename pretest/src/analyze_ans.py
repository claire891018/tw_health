import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

def calculate_similarity(answer1, answer2, model, tokenizer):
    tokens1 = tokenizer(answer1, return_tensors="pt", padding=True, truncation=True)
    tokens2 = tokenizer(answer2, return_tensors="pt", padding=True, truncation=True)
    
    embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
    embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)
    
    return cosine_similarity(embeddings1.detach().numpy(), embeddings2.detach().numpy())[0][0]

def analyze_answers(generated_path, output_path, high_threshold=0.85, medium_threshold=0.7):
    generated_df = pd.read_csv(generated_path)
    
    analysis_results = []

    similarity_model = AutoModel.from_pretrained("bert-base-uncased")
    similarity_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for _, row in generated_df.iterrows():
        generated_answer = row["Generated_Answer"]
        original_answer = row.get("Answer", "")
        
        similarity_score = calculate_similarity(original_answer, generated_answer, similarity_model, similarity_tokenizer)
        
        analysis_results.append({
            "No": row["No"],
            "Disease": row["Disease"],
            "Question": row["Question"],
            "Generated_Answer": generated_answer,
            "Original_Answer": original_answer,
            "Similarity_Score": similarity_score
        })

    analysis_df = pd.DataFrame(analysis_results)
    analysis_df.to_csv(output_path, index=False)

    total_questions = analysis_df.shape[0]
    high_similarity_count = analysis_df[analysis_df['Similarity_Score'] >= high_threshold].shape[0]
    medium_similarity_count = analysis_df[(analysis_df['Similarity_Score'] >= medium_threshold) & (analysis_df['Similarity_Score'] < high_threshold)].shape[0]
    low_similarity_count = analysis_df[analysis_df['Similarity_Score'] < medium_threshold].shape[0]

    avg_similarity = analysis_df['Similarity_Score'].mean()
    std_similarity = analysis_df['Similarity_Score'].std()

    print(f"高相似度回答比例 (>={high_threshold}): {high_similarity_count / total_questions:.2%}")
    print(f"中等相似度回答比例 (>= {medium_threshold} and < {high_threshold}): {medium_similarity_count / total_questions:.2%}")
    print(f"低相似度回答比例 (< {medium_threshold}): {low_similarity_count / total_questions:.2%}")
    print(f"平均相似度: {avg_similarity:.2f}")
    print(f"相似度標準差: {std_similarity:.2f}")

    if high_similarity_count / total_questions > 0.5:
        print("結論：模型可能使用了該資料集作為訓練資料。")
    elif avg_similarity > 0.7:
        print("結論：模型可能接觸過類似知識，但未必直接使用該資料集。")
    else:
        print("結論：模型大概率未使用該資料集進行訓練。")


