import transformers
from models.med import BertModel, BertLMHeadModel
from transformers import BertTokenizer, pipeline, AutoTokenizer, AutoModel, BertConfig, EsmModel, EsmConfig, EsmTokenizer
import torch

model_name_text = 'allenai/scibert_scivocab_uncased'
tokenizer = BertTokenizer.from_pretrained(model_name_text, do_lower_case=False)
encoder = BertModel.from_pretrained(model_name_text, add_pooling_layer=False)

torch.save(tokenizer, "./weights/scibert_tokenizer.pt")
torch.save(encoder, "./weights/scibert.pt")
print("saved scibert")

model_name_text = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = BertTokenizer.from_pretrained(model_name_text, do_lower_case=True)
encoder = BertModel.from_pretrained(model_name_text, add_pooling_layer=False)

torch.save(tokenizer, "./weights/pubmedbert_tokenizer.pt")
torch.save(encoder, "./weights/pubmedbert.pt")
print("saved pubmedbert")

from transformers.models.esm import EsmTokenizer
from models.med_esm import EsmModel, EsmConfig, EsmLMHeadModel

# Protbert bfd
model_name_seq="Rostlab/prot_bert_bfd"
seq_config = './configs/config_seq.json'
seq_encoder_config = BertConfig.from_json_file(seq_config)
decoder_config = './configs/config_light_seq.json'
seq_encoder_config.encoder_width = 768

tokenizer = BertTokenizer.from_pretrained(model_name_seq, do_lower_case=False)
encoder = BertModel.from_pretrained(model_name_seq, config=seq_encoder_config, add_pooling_layer=False)

decoder = BertLMHeadModel.from_pretrained(model_name_seq, config=decoder_config, is_decoder=True)

torch.save(tokenizer, "./weights/ProtBert_tokenizer.pt")
torch.save(encoder, "./weights/ProtBert.pt")
torch.save(decoder, "./weights/ProtDecoder.pt")
print("saved protbert_bfd")



# esm1b
model_name_seq = "facebook/esm1b_t33_650M_UR50S"
seq_encoder_config_file = "./configs/encoder.json"
seq_decoder_config_file = "./configs/decoder.json"

seq_encoder_config = EsmConfig.from_json_file(seq_encoder_config_file)
seq_encoder_config.encoder_width = 768


tokenizer = EsmTokenizer.from_pretrained(model_name_seq, do_lower_case=False)
encoder = EsmModel.from_pretrained(model_name_seq, config=seq_encoder_config, add_pooling_layer=False)
torch.save(tokenizer, "./weights/esm1b_tokenizer.pt")
torch.save(encoder, "./weights/esm1b.pt")

# esm2
model_name_seq = "facebook/esm2_t33_650M_UR50D"
seq_encoder_config_file = "./configs/encoder2.json"
seq_decoder_config_file = "./configs/decoder2.json"

seq_encoder_config = EsmConfig.from_json_file(seq_encoder_config_file)
seq_encoder_config.encoder_width = 768

tokenizer = EsmTokenizer.from_pretrained(model_name_seq, do_lower_case=False)
encoder = EsmModel.from_pretrained(model_name_seq, config=seq_encoder_config, add_pooling_layer=False)
torch.save(tokenizer, "./weights/esm2_tokenizer.pt")
torch.save(encoder, "./weights/esm2.pt")
