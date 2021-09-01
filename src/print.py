import torch
import pandas as pd
from rouge import Rouge
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

def load_model1():
    model = BartForConditionalGeneration.from_pretrained('rsc/Binary_model/aihub법률문서')
    return model

def load_model2():
    model = BartForConditionalGeneration.from_pretrained('rsc/Binary_model/aihub사설잡지')
    return model

model1 = load_model1()
model2 = load_model2()
tokenizer = get_kobart_tokenizer()

dev_v1 = pd.read_csv('rsc/data/Validation/AI_hub/사설잡지/dev_v1.csv', index_col = 0)
len=20

for i in range(len):
    text=dev_v1['text'][i]
    summarization = dev_v1['abstractive'][i]
    if text and summarization:
            input_ids = tokenizer.encode(text)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0)
            output1 = model1.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
            output1 = tokenizer.decode(output1[0], skip_special_tokens=True)
            rouge1=Rouge()
            rouge_score1 = rouge1.get_scores(output1, summarization)[0]
            output2 = model2.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
            output2 = tokenizer.decode(output2[0], skip_special_tokens=True)
            rouge2=Rouge()
            rouge_score2 = rouge2.get_scores(output2, summarization)[0]
    print('text: '+text)
    print('output1: '+output1)
    print(rouge_score1)
    print('output2: '+output2)
    print(rouge_score2)
    print()
    
