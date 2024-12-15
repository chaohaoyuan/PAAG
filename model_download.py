import transformers
from models.med import BertModel, BertLMHeadModel
from transformers import BertTokenizer, pipeline, AutoTokenizer, AutoModel, BertConfig, EsmModel, EsmConfig, EsmTokenizer
import torch

model_name_text = 'allenai/scibert_scivocab_uncased'
model_name_seq="Rostlab/prot_bert",
model_name_seq = "yarongef/DistilProtBert"
seq_config = 'config_distill_seq.json'
text_config = 'config_text.json'

seq_encoder_config = BertConfig.from_json_file(seq_config)
seq_encoder_config.encoder_width = 768
decoder_config = BertConfig.from_json_file(seq_config)
# decoder.add_cross_attention = True
decoder_config.encoder_width = 768
text_encoder_config = BertConfig.from_json_file(text_config)

tokenizer = BertTokenizer.from_pretrained(model_name_seq, do_lower_case=False)
encoder = BertModel.from_pretrained(model_name_seq, config=seq_encoder_config, add_pooling_layer=False)
decoder = BertLMHeadModel.from_pretrained(model_name_seq, config=decoder_config)

torch.save(tokenizer, "./protein_data/DistillProtBert_tokenizer.pt")
torch.save(encoder, "./protein_data/DistillProtBert.pt")
torch.save(decoder, "./protein_data/DistillProtDecoder.pt")
print("saved distillprotbert")
model_name_seq="Rostlab/prot_bert"
seq_config = 'config_seq.json'
seq_encoder_config = BertConfig.from_json_file(seq_config)
seq_encoder_config.encoder_width = 768

tokenizer = BertTokenizer.from_pretrained(model_name_seq, do_lower_case=False)
encoder = BertModel.from_pretrained(model_name_seq, config=seq_encoder_config, add_pooling_layer=False)
decoder = BertLMHeadModel.from_pretrained(model_name_seq, is_decoder=True)

torch.save(tokenizer, "./protein_data/ProtBert_tokenizer.pt")
torch.save(encoder, "./protein_data/ProtBert.pt")
print("saved protbert")


model_name_text = 'allenai/scibert_scivocab_uncased'
tokenizer = BertTokenizer.from_pretrained(model_name_text, do_lower_case=False)
encoder = BertModel.from_pretrained(model_name_text, config=text_encoder_config, add_pooling_layer=False)

torch.save(tokenizer, "./protein_data/scibert_tokenizer.pt")
torch.save(encoder, "./protein_data/scibert.pt")
print("saved scibert")

