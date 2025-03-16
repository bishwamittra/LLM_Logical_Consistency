import argparse
import os
# This script is to generate, zero-shot, few-shot, chain of thoughts prompts
# First run : python prompt_generation_negation_query.py ../data_optimized/1c_data_final_${dataset}_test_2.csv --context_length 1000 --nrows $nrows
# Then use the previous output file as input to this script, for example in this case the input file will be called 1c_prompt_${dataset}_test_2.csv 

# Set up argument parsing
parser.add_argument("file_path", type=str, help="Path to the input file in csv format")
args = parser.parse_args()
file_path = args.file_path

def create_few_shot_prompt(input_text, prompt_type='zero_shot'):
    if prompt_type == 'few_shot':
        prefix = """<<SYS>>You are a helpful assistant. Always follow the instructions precisely and output the response exactly in the requested format.<</SYS>>[NEWLINE][NEWLINE][INST] Consider the context as a set of triplets where entries are separated by '|' symbol. Answer question according to the context.[NEWLINE][NEWLINE]
Hartlepool | administrative children reverse | County Durham[NEWLINE]
County Durham | administrative parent reverse | Darlington[NEWLINE]
County Durham | contains reverse | England[NEWLINE]
County Durham | containedby | England[NEWLINE]
Hartlepool | containedby | County Durham[NEWLINE]
County Durham | containedby reverse | Darlington[NEWLINE]
County Durham | second level division of | England[NEWLINE]
County Durham | contains | Darlington[NEWLINE]
Hartlepool | contains reverse | County Durham[NEWLINE]
County Durham | second level divisions reverse | England[NEWLINE]
County Durham | contains | Durham University[NEWLINE]
County Durham | administrative children | Darlington[NEWLINE]
Hartlepool | administrative parent | County Durham[NEWLINE]
County Durham | containedby reverse | Durham University[NEWLINE][NEWLINE][NEWLINE]
Do not add additional text. Is the following triplet FACTUALLY CORRECT? Answer with Yes or No.[NEWLINE][NEWLINE]
Hartlepool | contains reverse | County Durham [/INST][NEWLINE]Yes.[NEWLINE][NEWLINE][INST] Consider the context as a set of triplets where entries are separated by '|' symbol. Answer question according to the context.[NEWLINE][NEWLINE]
Hartlepool | administrative children reverse | County Durham[NEWLINE]
County Durham | administrative parent reverse | Darlington[NEWLINE]
County Durham | contains reverse | England[NEWLINE]
County Durham | containedby | England[NEWLINE]
Hartlepool | containedby | County Durham[NEWLINE]
County Durham | containedby reverse | Darlington[NEWLINE]
County Durham | second level division of | England[NEWLINE]
County Durham | contains | Darlington[NEWLINE]
Hartlepool | contains reverse | County Durham[NEWLINE]
County Durham | second level divisions reverse | England[NEWLINE]
County Durham | contains | Durham University[NEWLINE]
County Durham | administrative children | Darlington[NEWLINE]
Hartlepool | administrative parent | County Durham[NEWLINE]
County Durham | containedby reverse | Durham University[NEWLINE][NEWLINE][NEWLINE]
Do not add additional text. Is the following triplet FACTUALLY CORRECT? Answer with Yes or No.[NEWLINE][NEWLINE]
Hartlepool | contains reverse | Cottam, East Riding of Yorkshire [/INST][NEWLINE]No.[NEWLINE][NEWLINE]"""
    elif prompt_type == 'zero_shot':
        prefix = """<<SYS>>You are a helpful assistant. Always follow the instructions precisely and output the response exactly in the requested format. <</SYS>>[NEWLINE][NEWLINE]"""  
    output_text = prefix + '[INST] ' + input_text + ' [/INST]'
    return output_text.replace('[NEWLINE]', '\n')

def create_cot_prompt(input_text, query_type='simple'):
    prefix = """[NEWLINE][NEWLINE][INST] Consider the context as a set of triplets where entries are separated by '|' symbol. Answer question according to the context.[NEWLINE][NEWLINE]
Hartlepool | administrative children reverse | County Durham[NEWLINE]
County Durham | administrative parent reverse | Darlington[NEWLINE]
County Durham | contains reverse | England[NEWLINE]
County Durham | containedby | England[NEWLINE]
Hartlepool | containedby | County Durham[NEWLINE]
County Durham | containedby reverse | Darlington[NEWLINE]
County Durham | second level division of | England[NEWLINE]
County Durham | contains | Darlington[NEWLINE]
Hartlepool | contains reverse | County Durham[NEWLINE]
County Durham | second level divisions reverse | England[NEWLINE]
County Durham | contains | Durham University[NEWLINE]
County Durham | administrative children | Darlington[NEWLINE]
Hartlepool | administrative parent | County Durham[NEWLINE]
County Durham | containedby reverse | Durham University[NEWLINE][NEWLINE][NEWLINE]
Do not add additional text. Is the following triplet FACTUALLY CORRECT? Answer with Yes or No.[NEWLINE][NEWLINE]
Hartlepool | contains reverse | County Durham [/INST][NEWLINE]The provided triplet 'Hartlepool | contains reverse | County Durham' appears in the context on the ninth line. The answer is Yes.[NEWLINE][NEWLINE][INST] Consider the context as a set of triplets where entries are separated by '|' symbol. Answer question according to the context.[NEWLINE][NEWLINE]
Hartlepool | administrative children reverse | County Durham[NEWLINE]
County Durham | administrative parent reverse | Darlington[NEWLINE]
County Durham | contains reverse | England[NEWLINE]
County Durham | containedby | England[NEWLINE]
Hartlepool | containedby | County Durham[NEWLINE]
County Durham | containedby reverse | Darlington[NEWLINE]
County Durham | second level division of | England[NEWLINE]
County Durham | contains | Darlington[NEWLINE]
Hartlepool | contains reverse | County Durham[NEWLINE]
County Durham | second level divisions reverse | England[NEWLINE]
County Durham | contains | Durham University[NEWLINE]
County Durham | administrative children | Darlington[NEWLINE]
Hartlepool | administrative parent | County Durham[NEWLINE]
County Durham | containedby reverse | Durham University[NEWLINE][NEWLINE][NEWLINE]
Do not add additional text. Is the following triplet FACTUALLY CORRECT? Answer with Yes or No.[NEWLINE][NEWLINE]
Hartlepool | contains reverse | Cottam, East Riding of Yorkshire [/INST][NEWLINE]The provided triplet 'Hartlepool | contains reverse | Cottam, East Riding of Yorkshire' does not appear in any line of the context. The answer is No.[NEWLINE][NEWLINE]"""
    output_text = prefix + '[INST] ' + input_text + ' [/INST]'
    return output_text
    
# split the context from  the question
def separate_context_and_question(text):
    split_text = text.split('[NEWLINE]')
    question = split_text[-3] + ' ' + split_text[-1]
    context = '[NEWLINE]'.join(split_text[:-3])
    return question, context.strip()

# remove exactly '[NEWLINE][NEWLINE]' from the beginning
def remove_new_line(text):
  if text.startswith("[NEWLINE][NEWLINE]"):
      text = text[len("[NEWLINE][NEWLINE]"):]
  return text
    

"""# Reading the data"""
import pandas as pd
df=pd.read_csv(file_path)
df=df[df['use_context']==True]
df=df.head(5000)

"""# Applying the functions"""
#df['question_positive']=df['prompt_base_query'].apply(lambda row: separate_context_and_question(row)[0])
#df['question_negative']=df['prompt_negation_query'].apply(lambda row: separate_context_and_question(row)[0])
#df['context']=df['prompt_base_query'].apply(lambda row: separate_context_and_question(row)[1]).apply(remove_new_line)

# CoT call 
suffix='_CoT' you can change it according to the called function, it will be used later in the output file name
df['prompt_base_query'] = df.apply(lambda row: create_cot_prompt(row['prompt_base_query']), axis=1)
df['prompt_negation_query'] = df.apply(lambda row: create_cot_prompt(row['prompt_negation_query']), axis=1)

#print(df['prompt_negation_query'][1])
#print(df['prompt_base_query'][1])

df['prompt_base_query']=df['prompt_base_query'].apply(lambda row: row.replace("\n","[NEWLINE]"))
df['prompt_negation_query']=df['prompt_negation_query'].apply(lambda row: row.replace("\n","[NEWLINE]"))

# Save the file 
base_name = os.path.basename(file_path)  
input_file_name, ext = os.path.splitext(base_name)  
output_file_name = f"{input_file_name}{suffix}{ext}"  

df.to_csv(output_file_name)
