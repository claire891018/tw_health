from src.load_datasets import load_and_shuffle_dataset
from src.load_models import load_model
from src.generate_ans import generate_answers
from src.analyze_ans import analyze_answers

def main(dataset_path, model_path, output_path):
    dataset = load_and_shuffle_dataset(dataset_path)

    model = load_model(model_path)

    generate_answers(model, dataset, output_path="output/generated_answers.csv")

    analyze_answers("output/generated_answers.csv", output_path="output/analysis_report.csv")

if __name__ == "__main__":
    main()
