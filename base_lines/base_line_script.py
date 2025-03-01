import os
import argparse
import nltk
import pandas as pd
import time
from minicheck.minicheck import MiniCheck

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nltk.download('punkt_tab')
nltk.download('punkt')

def predict(claim, claim_1, claim_2, doc, doc1, doc2):
    start_time = time.time()
    scorer = MiniCheck(model_name='roberta-large', cache_dir='./ckpts')
    pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc1, doc2], claims=[claim, claim_1, claim_2])
    end_time = time.time()
    total_time = end_time - start_time
    return str(pred_label), total_time

def predict_simple(claim_1, claim_2, doc):
    start_time = time.time()
    scorer = MiniCheck(model_name='roberta-large', cache_dir='./ckpts')
    pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])
    end_time = time.time()
    all_time = end_time - start_time
    return str(pred_label), all_time

def process_data(input_file, output_file):
    filtered_df = pd.read_csv(input_file)
    first_write = True

    for index, row in filtered_df.iterrows():
        pred_label, elapsed_time = predict_simple(row['processed_base_query'], row['processed_negation_query'], row['context'])
        row['prediction'] = pred_label
        row['time'] = elapsed_time
        row_df = pd.DataFrame([row])
        row_df.to_csv(output_file, mode='a', index=False, header=first_write)
        first_write = False
        print(f"processed row {index}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to process data and generate predictions")
    parser.add_argument('--input_file', type=str, required=True, help="path to the input CSV file")
    parser.add_argument('--output_file', type=str, required=True, help="path to the output CSV file")
    args = parser.parse_args()
    process_data(args.input_file, args.output_file)
    print("processing complete!")
