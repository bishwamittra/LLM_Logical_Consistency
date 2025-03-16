import os
import argparse
import nltk
import pandas as pd
import time
from minicheck.minicheck import MiniCheck

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nltk.download('punkt_tab')
nltk.download('punkt')

def separate_context_and_question(text):
    split_text = text.split("[NEWLINE]")
    question = f"{split_text[-3]} {split_text[-1]}"
    context = "[NEWLINE]".join(split_text[:-3]).strip()
    return question, context

def remove_new_line(text):
    return text[len("[NEWLINE][NEWLINE]"): ] if text.startswith("[NEWLINE][NEWLINE]") else text

def extract_triplets_from_questions(question):
    return question.split('.')[2].strip()
    
def replace_newlines(text):
    return text.replace('[NEWLINE]', '\n')

def clean_context(context):
  text = "Consider the context as a set of triplets where entries are separated by \'|\' symbol. Answer question according to the context."
  cleaned_text = context.replace(text, "")
  cleaned_text=replace_newlines(cleaned_text)
  return cleaned_text.strip()

def predict_complex(claim, claim_1, claim_2, doc, doc1, doc2):
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
    df = pd.read_csv(input_file)
    first_write = True
    # Exctract context and triplets 
    df['question_positive']=df['prompt_base_query'].apply(lambda row: separate_context_and_question(row)[0])
    df['question_negative']=df['prompt_negation_query'].apply(lambda row: separate_context_and_question(row)[0])
    df['context']=df['prompt_base_query'].apply(lambda row: separate_context_and_question(row)[1]).apply(remove_new_line)
    df['processed_base_query']=df['question_positive'].apply(extract_triplets_from_questions)
    df['processed_negation_query']=df['question_negative'].apply(extract_triplets_from_questions)
    
    #Preprocess context and triplets
    df['processed_base_query']=df['processed_base_query'].apply(replace_newlines)
    df['processed_negation_query']=df['processed_negation_query'].apply(replace_newlines)
    df['context']=df['context'].apply(clean_context)
    
    for index, row in df.iterrows():
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
