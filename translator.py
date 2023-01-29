# pip install kobert-transformers
import torch
from flask import Flask, request, jsonify 
from transformers import (
    PreTrainedTokenizerFast as BaseGPT2Tokenizer,
    EncoderDecoderModel
)
from kobert_transformers.tokenization_kobert import KoBertTokenizer
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from typing import Dict, List

from threading import Semaphore

class GPT2Tokenizer(BaseGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]

src_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
trg_tokenizer = GPT2Tokenizer.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

model = EncoderDecoderModel.from_pretrained('leadawon/jeju-ko-nmt-v6')
model.config.decoder_start_token_id = trg_tokenizer.bos_token_id
model.eval()

semaphore = Semaphore(5)

app = Flask(__name__)

@app.route("/")
def home():
    return "테스트 중 입니다."

@app.route("/translate", methods=['POST'])
def translate():
    input_json = request.get_json()
    print(input_json['text'])
    text = input_json['text']
    embeddings = src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
    with semaphore:
        embeddings = src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        output = model.generate(**embeddings)[0, 1:-1].cpu()
        del embeddings
    print(trg_tokenizer.decode(output))
    return jsonify({'answer':trg_tokenizer.decode(output)})
        
if __name__ == "__main__":
    # app.run(host='0.0.0.0', port='5000')
    app.run(host='0.0.0.0')

    