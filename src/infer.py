import torch
import pandas as pd
import streamlit as st
from rouge import Rouge
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration


@st.cache(allow_output_mutation=True)
def load_model():
    model = BartForConditionalGeneration.from_pretrained('kobart_summary')
    # rsc/BY_domain_ckpt/aihub한국어대화요약
    # tokenizer = get_kobart_tokenizer()
    return model


model = load_model()
tokenizer = get_kobart_tokenizer()
st.title("KoBART 요약 Test")
text = st.text_area("문서 입력:")
summarization = st.text_area("요약 입력:")

st.markdown("## 문서 원문")
st.write(text)

if text and summarization:
    text = text.replace('\n', '')
    st.markdown("## 요약 정답")
    st.write(summarization)
    st.markdown("## KoBART model 요약 결과")
    with st.spinner('processing..'):
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        rouge=Rouge()
        rouge_score = rouge.get_scores(output, summarization)[0]

    st.write(output)
    st.markdown("## KoBART model 요약 점수")
    df=pd.DataFrame(data=rouge_score)
    st.table(df)
