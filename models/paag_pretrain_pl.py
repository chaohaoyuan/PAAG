from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer, pipeline, AutoTokenizer, AutoModel
import os
import transformers
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_tokenizer, build_paag_args
import ruamel.yaml as yaml
import pytorch_lightning as pl
import random
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
import re





def is_empty_arrays(arr):
    if isinstance(arr, list) and all(isinstance(i, list) and not i for i in arr):
        return True
    return False


class WarmupConstantLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super(WarmupConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class PAAG_Pretrain(pl.LightningModule):
    def __init__(self,
                 model_name_text='scibert',
                 model_name_seq="protbert",
                 seq_config='config_seq.json',
                 text_config='config_text.json',
                 embed_dim=256,
                 queue_size=16384,
                 momentum=0.995,
                 ):
        super().__init__()
        print(queue_size)
        self.embed_dim = embed_dim
        self.model_name_text = model_name_text
        args = build_paag_args()
        seq_encoder_config = BertConfig.from_json_file(seq_config)
        seq_encoder_config.encoder_width = 768
        text_encoder_config = BertConfig.from_json_file(text_config)
        self.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        # self.batch_size = self.config['batch_size']
        self.text_proj = nn.Linear(768, embed_dim)
        self.seq_proj = nn.Linear(1024, embed_dim)
        input_path = "../weights"
        if model_name_seq == "protbert":
            self.seq_tokenizer = torch.load(os.path.join(input_path, "ProtBert_tokenizer.pt"), map_location='cpu')
            self.seq_encoder = torch.load(os.path.join(input_path, "ProtBert.pt"), map_location='cpu')
            self.seq_decoder = torch.load(os.path.join(input_path, "ProtDecoder.pt"), map_location='cpu')
            self.seq_encoder_m = torch.load(os.path.join(input_path, "ProtBert.pt"), map_location='cpu')
            self.seq_dim = 1024
        elif model_name_seq == "esm1b":
            self.seq_tokenizer = torch.load(os.path.join(input_path, "esm1b_tokenizer.pt"), map_location='cpu')
            self.seq_encoder = torch.load(os.path.join(input_path, "esm1b.pt"), map_location='cpu')
            self.seq_encoder_m = torch.load(os.path.join(input_path, "esm1b.pt"), map_location='cpu')
            self.seq_dim = 1280
            print('esm1b')
        elif model_name_seq == "esm2":
            self.seq_tokenizer = torch.load(os.path.join(input_path, "esm2_tokenizer.pt"), map_location='cpu')
            self.seq_encoder = torch.load(os.path.join(input_path, "esm2.pt"), map_location='cpu')
            self.seq_encoder_m = torch.load(os.path.join(input_path, "esm2.pt"), map_location='cpu')
            self.seq_decoder = torch.load(os.path.join(input_path, "esm2_decoder.pt"), map_location='cpu')
            self.seq_dim = 1280
            print('esm')
        else:
            print("none sequence encoder found")
        if model_name_text == "scibert":
            self.text_tokenizer = torch.load(os.path.join(input_path, "scibert_tokenizer.pt"), map_location='cpu')
            self.text_encoder = torch.load(os.path.join(input_path, "scibert.pt"), map_location='cpu')
        elif model_name_text == "pub_abstract":
            self.text_tokenizer = torch.load(os.path.join(input_path, "pubmedbert_tokenizer.pt"), map_location='cpu')
            self.text_encoder = torch.load(os.path.join(input_path, "pubmedbert.pt"), map_location='cpu')
        else:
            print("none text encoder found")
        # add special token
        self.seq_tokenizer = init_tokenizer(self.seq_tokenizer)
        self.seq_encoder.resize_token_embeddings(len(self.seq_tokenizer))
        self.seq_proj = nn.Linear(self.seq_dim, embed_dim)
        self.target_token_idx = 0
        # create the queue
        self.register_buffer("seq_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.seq_queue = nn.functional.normalize(self.seq_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        # learnable temperature
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # create momentum encoders
        self.seq_encoder_m.resize_token_embeddings(len(self.seq_tokenizer))
        self.text_encoder_m = BertModel(config=text_encoder_config, add_pooling_layer=False)
        self.seq_proj_m = nn.Linear(self.seq_dim, embed_dim)
        self.text_proj_m = nn.Linear(768, embed_dim)

        self.model_pairs = [[self.seq_encoder, self.seq_encoder_m],
                            [self.seq_proj, self.seq_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.copy_params()

        self.itm_head = nn.Linear(self.seq_dim, 2)

        # create the decoder

        if model_name_seq == "protbert":
            self.seq_decoder.resize_token_embeddings(len(self.seq_tokenizer))
            tie_encoder_decoder_weights(self.seq_encoder, self.seq_decoder.bert, '', '/attention')
        else:
            tie_encoder_decoder_weights(self.seq_encoder, self.seq_decoder.esm, '', '/attention')
        # self.number = 0

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.config['init_lr'],
                                      weight_decay=self.config['weight_decay'])
        warmup_epochs = 3
        scheduler = {
            'scheduler': WarmupConstantLR(optimizer, warmup_epochs),
            'interval': 'epoch',  # The scheduler will update the learning rate every epoch
            'frequency': 1,
            'strict': True,
        }
        return [optimizer], [scheduler]


    def forward(self, seq, description, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        text = self.text_tokenizer(description, padding='max_length', truncation=True, max_length=50,
                              return_tensors="pt").to(self.text_encoder.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        seq = self.seq_tokenizer(seq, padding='max_length', truncation=True, max_length=450,
                              return_tensors="pt").to(self.seq_encoder.device)
        seq_output = self.seq_encoder(seq.input_ids, attention_mask=seq.attention_mask,
                                        return_dict=True, mode='text')
        seq_feat = F.normalize(self.seq_proj(seq_output.last_hidden_state[:, 0, :]), dim=-1)

        # get momemtum features
        with torch.no_grad():
            if self.training:
                self._momentum_update()
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            seq_output_m = self.seq_encoder_m(seq.input_ids, attention_mask=seq.attention_mask,
                                                return_dict=True, mode='text')
            seq_feat_m = F.normalize(self.seq_proj_m(seq_output_m.last_hidden_state[:, 0, :]), dim=-1)
            seq_feat_all = torch.cat([seq_feat_m.t(), self.seq_queue.clone().detach()], dim=1)

            sim_t2s_m = text_feat_m @ seq_feat_all / self.temp
            sim_s2t_m = seq_feat_m @ text_feat_all / self.temp

            sim_targets = torch.zeros(sim_t2s_m.size()).to(self.seq_encoder.device)
            sim_targets.fill_diagonal_(1)

            sim_t2s_targets = alpha * F.softmax(sim_t2s_m, dim=1) + (1 - alpha) * sim_targets
            sim_s2t_targets = alpha * F.softmax(sim_s2t_m, dim=1) + (1 - alpha) * sim_targets

        sim_t2s = text_feat @ seq_feat_all / self.temp
        sim_s2t = seq_feat @ text_feat_all / self.temp

        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1) * sim_t2s_targets, dim=1).mean()
        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1) * sim_s2t_targets, dim=1).mean()

        loss_ita = (loss_t2s + loss_s2t) / 2
        if self.training:
            self._dequeue_and_enqueue(text_feat_m, seq_feat_m)
        # ###============== Image-text Matching ===================###
        encoder_input_ids = seq.input_ids.clone()
        encoder_input_ids[:, 0] = self.seq_tokenizer.enc_token_id

        bs = encoder_input_ids.size(0)
        output_pos = self.seq_encoder(encoder_input_ids,
                                       attention_mask=seq.attention_mask,
                                       encoder_hidden_states=text_output.last_hidden_state,
                                       encoder_attention_mask=text.attention_mask,
                                       return_dict=True,
                                       )
        with torch.no_grad():
            weights_s2t = F.softmax(sim_s2t[:, :bs], dim=1) + 1e-4
            weights_s2t.fill_diagonal_(0)
            weights_t2s = F.softmax(sim_t2s[:, :bs], dim=1) + 1e-4
            weights_t2s.fill_diagonal_(0)

        text_embeds_neg = []
        text_atts_neg = []
        text_embeds = text_output.last_hidden_state
        for b in range(bs):
            neg_idx = torch.multinomial(weights_s2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # select a negative seq for each text
        seq_ids_neg = []
        seq_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2s[b], 1).item()
            seq_ids_neg.append(encoder_input_ids[neg_idx])
            seq_atts_neg.append(seq.attention_mask[neg_idx])

        seq_ids_neg = torch.stack(seq_ids_neg, dim=0)
        seq_atts_neg = torch.stack(seq_atts_neg, dim=0)



        seq_ids_all = torch.cat([encoder_input_ids, seq_ids_neg], dim=0)
        seq_atts_all = torch.cat([seq.attention_mask, seq_atts_neg], dim=0)

        text_embeds_all = torch.cat([text_embeds_neg, text_embeds], dim=0)
        text_atts_all = torch.cat([text_atts_neg, text.attention_mask], dim=0)

        output_neg = self.seq_encoder(seq_ids_all,
                                       attention_mask=seq_atts_all,
                                       encoder_hidden_states=text_embeds_all,
                                       encoder_attention_mask=text_atts_all,
                                       return_dict=True,
                                       )
        ts_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        ts_output = self.itm_head(ts_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(self.seq_encoder.device)
        loss_itm = F.cross_entropy(ts_output, itm_labels)
        #================= LM ========================##
        decoder_input_ids = seq.input_ids.clone()
        decoder_input_ids[:, 0] = self.seq_tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.seq_tokenizer.pad_token_id, -100)

        decoder_output = self.seq_decoder(decoder_input_ids,
                                           attention_mask=seq.attention_mask,
                                           encoder_hidden_states=text_output.last_hidden_state,
                                           encoder_attention_mask=text.attention_mask,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )
        loss_lm = decoder_output.loss
        return loss_ita, loss_itm, loss_lm


    def training_step(self, batch, batch_idx):
        sequence, text = batch
        batch_size = len(sequence)
        alpha = self.config['alpha'] * min(1, (self.trainer.current_epoch * len(self.trainer.train_dataloader)) + batch_idx) / (2 * len(self.trainer.train_dataloader))
        loss_ita, loss_itm, loss_lm = self(sequence, text, alpha)
        loss = loss_ita + loss_itm + loss_lm

        self.log("train_ita", loss_ita, batch_size=batch_size, sync_dist=True)
        self.log("train_itm", loss_itm, batch_size=batch_size, sync_dist=True)
        self.log("train_lm", loss_lm, batch_size=batch_size, sync_dist=True)
        self.log("train_total", loss, batch_size=batch_size, sync_dist=True)
        return loss



    def on_predict_epoch_start(self):
        tie_encoder_decoder_weights_initial(self.seq_encoder, self.seq_decoder.bert, '', '/attention')


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        task = 'generate'
        sequence, text = batch
        words = text[0].split(" ")
        length = int(words[words.index('contains') + 1])
        input_text = text
        batch_size = len(sequence)
        captions = self.generate(input_text, sequence, batch_size, sample=True, max_length=length, min_length=length)
        return [input_text[0], "".join(captions[0].split(" "))]


    def nucleus_sampling(self, input_ids, outputs, original_aa, p=0.9, repetition_penalty=1.1):
        logits = outputs.logits[:, -1, :]
        # print(logits)
        for token_id in input_ids[0].tolist():
            logits[0, token_id] *= repetition_penalty
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        num_tokens_to_keep = torch.where(cumulative_probs >= p)[0][0].item() + 1
        # if original_aa in sorted_indices[0, :num_tokens_to_keep]:
        candidate = sorted_indices[0, :num_tokens_to_keep]
        random_index = torch.randint(len(candidate), size=(1,)).item()
        selected_token_id = candidate[random_index]
        # selected_token_id = sorted_indices[0, :num_tokens_to_keep].multinomial(1).item()

        return selected_token_id


    def generate(self, text, seq, batch_size, sample=False, num_beams=3, max_length=200, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        text = self.text_tokenizer(text, padding='max_length', truncation=True, max_length=50,
                                   return_tensors="pt").to(self.text_encoder.device)
        text_output = self.text_encoder(text.input_ids.to(self.text_encoder.device), attention_mask=text.attention_mask.to(self.text_encoder.device),
                                        return_dict=True, mode='text')
        last_hidden_state = text_output.last_hidden_state
        text_atts = text.attention_mask.to(self.text_encoder.device)

        bad_words_ids = [[25], [26], [27], [28], [29]]

        if not sample:
            last_hidden_state = last_hidden_state.repeat_interleave(num_beams, dim=0)
            text_atts = text_atts.repeat_interleave(num_beams, dim=0)
        model_kwargs = {"encoder_hidden_states": last_hidden_state, "encoder_attention_mask": text_atts}

        input_ids = torch.zeros((batch_size, 1), dtype=torch.long).to(self.text_encoder.device)
        input_ids[:, 0] = self.seq_tokenizer.bos_token_id

        if sample:
            # nucleus sampling
            outputs = self.seq_decoder.generate(input_ids=input_ids.to(self.text_encoder.device),
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.seq_tokenizer.sep_token_id,
                                                 pad_token_id=self.seq_tokenizer.pad_token_id,
                                                 repetition_penalty=1,
                                                # repetition_penalty=3,
                                                bad_words_ids=bad_words_ids,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.seq_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.seq_tokenizer.sep_token_id,
                                                 pad_token_id=self.seq_tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                bad_words_ids=bad_words_ids,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.seq_tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        return captions

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, text_feat, seq_feat):
        seq_feats = concat_all_gather(seq_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = text_feats.shape[0]

        ptr = int(self.queue_ptr)

        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.seq_queue[:, ptr:ptr + batch_size] = seq_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


def paag_pretrain(**kwargs):
    model = PAAG_Pretrain(**kwargs)
    return model


class PAAG_Pretrain_Partial_Local(pl.LightningModule):
    def __init__(self,
                 model_name_text='scibert',
                 model_name_seq="protbert",
                 seq_config='config_seq.json',
                 text_config='config_text.json',
                 embed_dim=256,
                 queue_size=16384,
                 momentum=0.995,
                 ):
        super().__init__()
        print(model_name_seq)
        self.embed_dim = embed_dim
        self.model_name_text = model_name_text
        args = build_paag_args()
        seq_encoder_config = BertConfig.from_json_file(seq_config)
        seq_encoder_config.encoder_width = 768
        text_encoder_config = BertConfig.from_json_file(text_config)
        self.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

        input_path = "../weights"
        if model_name_seq == "protbert":
            self.seq_tokenizer = torch.load(os.path.join(input_path, "ProtBert_tokenizer.pt"), map_location='cpu')
            self.seq_encoder = torch.load(os.path.join(input_path, "ProtBert.pt"), map_location='cpu')
            self.seq_encoder_m = torch.load(os.path.join(input_path, "ProtBert.pt"), map_location='cpu')
            self.seq_decoder = torch.load(os.path.join(input_path, "ProtDecoder.pt"), map_location='cpu')
            self.seq_dim = 1024
        elif model_name_seq == "esm1b":
            self.seq_tokenizer = torch.load(os.path.join(input_path, "esm1b_tokenizer.pt"), map_location='cpu')
            self.seq_encoder = torch.load(os.path.join(input_path, "esm1b.pt"), map_location='cpu')
            self.seq_encoder_m = torch.load(os.path.join(input_path, "esm1b.pt"), map_location='cpu')
            self.seq_decoder = torch.load(os.path.join(input_path, "esm1b_decoder.pt"), map_location='cpu')
            self.seq_dim = 1280
            print('esm1b')
        elif model_name_seq == "esm2":
            self.seq_tokenizer = torch.load(os.path.join(input_path, "esm2_tokenizer.pt"), map_location='cpu')
            self.seq_encoder = torch.load(os.path.join(input_path, "esm2.pt"), map_location='cpu')
            self.seq_decoder = torch.load(os.path.join(input_path, "esm2_decoder.pt"), map_location='cpu')
            self.seq_dim = 1280
            self.seq_encoder_m = torch.load(os.path.join(input_path, "esm2.pt"), map_location='cpu')
            print('esm2')
        else:
            print("none sequence encoder found")
        if model_name_text == "scibert":
            self.text_tokenizer = torch.load(os.path.join(input_path, "scibert_tokenizer.pt"), map_location='cpu')
            self.text_encoder = torch.load(os.path.join(input_path, "scibert.pt"), map_location='cpu')
            self.text_encoder_m = torch.load(os.path.join(input_path, "scibert.pt"), map_location='cpu')
        elif model_name_text == "pub_abstract":
            self.text_tokenizer = torch.load(os.path.join(input_path, "pubmedbert_tokenizer.pt"), map_location='cpu')
            self.text_encoder = torch.load(os.path.join(input_path, "pubmedbert.pt"), map_location='cpu')
        else:
            print("none text encoder found")
        # add special token

        self.text_proj = nn.Linear(768, embed_dim)
        self.seq_proj = nn.Linear(self.seq_dim, embed_dim)

        self.seq_tokenizer = init_tokenizer(self.seq_tokenizer)
        self.seq_encoder.resize_token_embeddings(len(self.seq_tokenizer))

        self.target_token_idx = 0
        # create the queue
        self.register_buffer("seq_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.seq_queue = nn.functional.normalize(self.seq_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        # learnable temperature
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # create momentum encoders


        self.seq_encoder_m.resize_token_embeddings(len(self.seq_tokenizer))
        self.text_encoder_m = BertModel(config=text_encoder_config, add_pooling_layer=False)
        # self.text_encoder_m = self.text_encoder

        self.seq_proj_m = nn.Linear(self.seq_dim, embed_dim)
        self.text_proj_m = nn.Linear(768, embed_dim)

        self.model_pairs = [[self.seq_encoder, self.seq_encoder_m],
                            [self.seq_proj, self.seq_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.copy_params()

        self.itm_head = nn.Linear(self.seq_dim, 2)

        # create the decoder
        if model_name_seq == "protbert":
            self.seq_decoder.resize_token_embeddings(len(self.seq_tokenizer))
            tie_encoder_decoder_weights(self.seq_encoder, self.seq_decoder.bert, '', '/attention')
        else:
            tie_encoder_decoder_weights(self.seq_encoder, self.seq_decoder.esm, '', '/attention')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.config['init_lr'],
                                      weight_decay=self.config['weight_decay'])
        warmup_epochs = 3
        scheduler = {
            'scheduler': WarmupConstantLR(optimizer, warmup_epochs),
            'interval': 'epoch',  # The scheduler will update the learning rate every epoch
            'frequency': 1,
            'strict': True,
        }
        return [optimizer], [scheduler]

    def forward(self, seq, description, subsequences, noisy_subsequences, domains, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        text = self.text_tokenizer(description, padding='max_length', truncation=True, max_length=200,
                                   return_tensors="pt").to(self.text_encoder.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        seq = self.seq_tokenizer(seq, padding='max_length', truncation=True, max_length=450,
                                 return_tensors="pt").to(self.seq_encoder.device)
        seq_output = self.seq_encoder(seq.input_ids, attention_mask=seq.attention_mask,
                                      return_dict=True, mode='text')
        seq_feat = F.normalize(self.seq_proj(seq_output.last_hidden_state[:, 0, :]), dim=-1)

        # get momemtum features
        with torch.no_grad():
            if self.training:
                self._momentum_update()
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            seq_output_m = self.seq_encoder_m(seq.input_ids, attention_mask=seq.attention_mask,
                                              return_dict=True, mode='text')
            seq_feat_m = F.normalize(self.seq_proj_m(seq_output_m.last_hidden_state[:, 0, :]), dim=-1)
            seq_feat_all = torch.cat([seq_feat_m.t(), self.seq_queue.clone().detach()], dim=1)

            sim_t2s_m = text_feat_m @ seq_feat_all / self.temp
            sim_s2t_m = seq_feat_m @ text_feat_all / self.temp

            sim_targets = torch.zeros(sim_t2s_m.size()).to(self.seq_encoder.device)
            sim_targets.fill_diagonal_(1)

            sim_t2s_targets = alpha * F.softmax(sim_t2s_m, dim=1) + (1 - alpha) * sim_targets
            sim_s2t_targets = alpha * F.softmax(sim_s2t_m, dim=1) + (1 - alpha) * sim_targets

        sim_t2s = text_feat @ seq_feat_all / self.temp
        sim_s2t = seq_feat @ text_feat_all / self.temp
        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1) * sim_t2s_targets, dim=1).mean()
        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1) * sim_s2t_targets, dim=1).mean()

        loss_ita = (loss_t2s + loss_s2t) / 2
        if self.training:
            self._dequeue_and_enqueue(text_feat_m, seq_feat_m)
        # ###============== Local Contrastive ===================###
        if not is_empty_arrays(subsequences):
            result = []
            for i, subsequence in enumerate(subsequences):
                if len(subsequence) != 0:
                    subsequence = self.seq_tokenizer(subsequence, padding='max_length', truncation=True, max_length=120,
                                       return_tensors="pt").to(self.seq_encoder.device)
                    subseq_output = self.seq_encoder(subsequence.input_ids, attention_mask=subsequence.attention_mask,
                                                  return_dict=True, mode='text')
                    temp_result = self.seq_proj(subseq_output.last_hidden_state[:, 0, :])
                    result.append(temp_result)
            avg_pooling_domain = torch.cat(result, dim=0).to(self.seq_encoder.device) # batch_size* domains, embedding
            avg_pooling_domain = F.normalize(avg_pooling_domain, dim=-1)
            result = []
            for i, noisy_subsequence in enumerate(noisy_subsequences):
                if len(noisy_subsequence) != 0:
                    noisy_subsequence = self.seq_tokenizer(noisy_subsequence, padding='max_length', truncation=True, max_length=40,
                                                     return_tensors="pt").to(self.seq_encoder.device)
                    noisy_subseq_output = self.seq_encoder(noisy_subsequence.input_ids, attention_mask=noisy_subsequence.attention_mask,
                                                     return_dict=True, mode='text')
                    temp_result = self.seq_proj(noisy_subseq_output.last_hidden_state[:, 0, :])
                    result.append(temp_result)
            avg_pooling_undomain = torch.cat(result, dim=0).to(self.seq_encoder.device) # batch_size* domains, embedding
            avg_pooling_undomain = F.normalize(avg_pooling_undomain, dim=-1).reshape(-1, 3, self.embed_dim)
            cls_embeddings_list = []
            # print('cls')
            for i, domain_list in enumerate(domains):
                if len(domain_list) != 0:
                    temp_result = torch.zeros(len(domain_list), self.embed_dim)
                    for j, domain in enumerate(domain_list):
                        encoded = self.text_tokenizer(domain, padding='max_length', truncation=True, max_length=10,
                                            return_tensors="pt").to(self.text_encoder.device)
                        outputs = self.text_encoder(encoded.input_ids, attention_mask=encoded.attention_mask,
                                                        return_dict=True, mode='text')
                        temp_result[j] = self.text_proj(outputs.last_hidden_state[:, 0, :])
                    cls_embeddings_list.append(temp_result)
            cls_embeddings = torch.cat(cls_embeddings_list, dim=0).to(self.seq_encoder.device) # batch_size, domains, embedding
            cls_embeddings = F.normalize(cls_embeddings, dim=-1)
            targets = torch.zeros(cls_embeddings.shape[0], dtype=torch.long).to(self.text_encoder.device)
            positive_similarity = torch.sum(cls_embeddings * avg_pooling_domain / self.temp, dim=-1)
            negative_similarity = torch.sum(cls_embeddings.unsqueeze(1) * avg_pooling_undomain / self.temp, dim=-1)
            similarity_matrix = F.softmax(torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1), dim=1)

            loss_fn = nn.CrossEntropyLoss()
            loss_local = loss_fn(similarity_matrix, targets)
        else:
            loss_local = torch.tensor([float(0)]).to(self.seq_encoder.device)
        # ###============== Image-text Matching ===================###
        encoder_input_ids = seq.input_ids.clone()
        encoder_input_ids[:, 0] = self.seq_tokenizer.enc_token_id

        bs = encoder_input_ids.size(0)
        output_pos = self.seq_encoder(encoder_input_ids,
                                      attention_mask=seq.attention_mask,
                                      encoder_hidden_states=text_output.last_hidden_state,
                                      encoder_attention_mask=text.attention_mask,
                                      return_dict=True,
                                      )
        with torch.no_grad():
            weights_s2t = F.softmax(sim_s2t[:, :bs], dim=1) + 1e-4
            weights_s2t.fill_diagonal_(0)
            weights_t2s = F.softmax(sim_t2s[:, :bs], dim=1) + 1e-4
            weights_t2s.fill_diagonal_(0)

        text_embeds_neg = []
        text_atts_neg = []
        text_embeds = text_output.last_hidden_state
        for b in range(bs):
            neg_idx = torch.multinomial(weights_s2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # select a negative seq for each text
        seq_ids_neg = []
        seq_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2s[b], 1).item()
            seq_ids_neg.append(encoder_input_ids[neg_idx])
            seq_atts_neg.append(seq.attention_mask[neg_idx])

        seq_ids_neg = torch.stack(seq_ids_neg, dim=0)
        seq_atts_neg = torch.stack(seq_atts_neg, dim=0)

        seq_ids_all = torch.cat([encoder_input_ids, seq_ids_neg], dim=0)
        seq_atts_all = torch.cat([seq.attention_mask, seq_atts_neg], dim=0)

        text_embeds_all = torch.cat([text_embeds_neg, text_embeds], dim=0)
        text_atts_all = torch.cat([text_atts_neg, text.attention_mask], dim=0)

        output_neg = self.seq_encoder(seq_ids_all,
                                      attention_mask=seq_atts_all,
                                      encoder_hidden_states=text_embeds_all,
                                      encoder_attention_mask=text_atts_all,
                                      return_dict=True,
                                      )
        ts_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        ts_output = self.itm_head(ts_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(self.seq_encoder.device)
        loss_itm = F.cross_entropy(ts_output, itm_labels)
        # ================= LM ========================##
        decoder_input_ids = seq.input_ids.clone()
        decoder_input_ids[:, 0] = self.seq_tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.seq_tokenizer.pad_token_id, -100)
        decoder_output = self.seq_decoder(decoder_input_ids,
                                          attention_mask=seq.attention_mask,
                                          encoder_hidden_states=text_output.last_hidden_state,
                                          encoder_attention_mask=text.attention_mask,
                                          labels=decoder_targets,
                                          return_dict=True,
                                          )
        loss_lm = decoder_output.loss
        return loss_ita, loss_local, loss_itm, loss_lm

    def training_step(self, batch, batch_idx):
        sequence, text, locations, noisy_locations, domains = batch
        batch_size = len(sequence)
        alpha = self.config['alpha'] * min(1, (
                    self.trainer.current_epoch * len(self.trainer.train_dataloader)) + batch_idx) / (
                            2 * len(self.trainer.train_dataloader))
        loss_ita, loss_local, loss_itm, loss_lm = self(sequence, text, locations, noisy_locations, domains, alpha)
        loss = loss_ita + loss_local + loss_itm + loss_lm
        self.log("train_local", loss_local, batch_size=batch_size, sync_dist=True)
        self.log("train_ita", loss_ita, batch_size=batch_size, sync_dist=True)
        self.log("train_itm", loss_itm, batch_size=batch_size, sync_dist=True)
        self.log("train_lm", loss_lm, batch_size=batch_size, sync_dist=True)
        self.log("train_total", loss, batch_size=batch_size, sync_dist=True)
        return loss

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, text_feat, seq_feat):
        seq_feats = concat_all_gather(seq_feat)
        text_feats = concat_all_gather(text_feat)
        batch_size = text_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.seq_queue[:, ptr:ptr + batch_size] = seq_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

def paag_pretrain_partial_local(**kwargs):
    model = PAAG_Pretrain_Partial_Local(**kwargs)
    return model


class PAAG_Pretrain_Protst(pl.LightningModule):
    def __init__(self,
                 model_name_text='pub_abstract',
                 model_name_seq="protbert",
                 seq_config='config_seq.json',
                 text_config='config_pubmed_abstract.json',
                 embed_dim=512,
                 queue_size=16384,
                 momentum=0.995,
                 ):
        super().__init__()
        print(queue_size)
        self.embed_dim = embed_dim
        self.model_name_text = model_name_text
        args = build_blip_args()
        seq_encoder_config = BertConfig.from_json_file(seq_config)
        seq_encoder_config.encoder_width = 768
        self.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

        input_path = "../weights"
        if model_name_seq == "protbert":
            self.seq_tokenizer = torch.load(os.path.join(input_path, "ProtBert_tokenizer_bfd.pt"), map_location='cpu')
            self.seq_encoder = torch.load(os.path.join(input_path, "ProtBert_bfd.pt"), map_location='cpu')
            self.seq_encoder_m = torch.load(os.path.join(input_path, "ProtBert_bfd.pt"), map_location='cpu')
            self.seq_dim = 1024
        elif model_name_seq == "esm2":
            self.seq_tokenizer = torch.load(os.path.join(input_path, "esm2_tokenizer.pt"), map_location='cpu')
            self.seq_encoder = torch.load(os.path.join(input_path, "esm2.pt"), map_location='cpu')
            self.seq_encoder_m = torch.load(os.path.join(input_path, "esm2.pt"), map_location='cpu')
            self.seq_dim = 1280
            print('esm2')
        else:
            print("none sequence encoder found")
        self.text_tokenizer = torch.load(os.path.join(input_path, "pubmedbert_tokenizer.pt"), map_location='cpu')
        self.text_encoder = torch.load(os.path.join(input_path, "pubmedbert.pt"), map_location='cpu')
        # add special tokenws

        self.seq_tokenizer = init_tokenizer(self.seq_tokenizer)
        self.seq_encoder.resize_token_embeddings(len(self.seq_tokenizer))
        #
        self.target_token_idx = 0
        # create the queue
        self.register_buffer("seq_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.seq_queue = nn.functional.normalize(self.seq_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        # learnable temperature
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # create momentum encoders


        self.seq_encoder_m.resize_token_embeddings(len(self.seq_tokenizer))
        self.text_encoder_m = self.text_encoder
        self.text_proj = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(self.seq_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.text_proj_m = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.seq_proj_m = nn.Sequential(
            nn.Linear(self.seq_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # self.seq_proj2 = nn.Linear(embed_dim, embed_dim)
        self.model_pairs = [[self.seq_encoder, self.seq_encoder_m],
                            [self.seq_proj, self.seq_proj_m],
                            # [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.attribute = ["prot_name", "function", "subloc", "similarity"]
        self.copy_params()
        self.itm_head = nn.Linear(self.seq_dim, 2)


    def configure_optimizers(self):
        params = [
            {
                "params": self.seq_encoder.parameters(),
                "lr": 2e-6,
            },
            {
                "params": self.seq_proj.parameters(),
                "lr": 2e-5,
            },
            {
                "params": self.text_proj.parameters(),
                "lr": 2e-5,
            },
        ]

        optimizer = torch.optim.Adam(
            params,
            weight_decay=0
        )

        return optimizer

    def forward(self, seq, input_ids, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        # print(input_ids)
        text = self.text_tokenizer(input_ids, padding='max_length', truncation=True, max_length=100,
                              return_tensors="pt").to(self.text_encoder.device)
        input_ids = text.input_ids
        attention_mask = text.attention_mask
        text_output = self.text_encoder(input_ids, attention_mask=attention_mask,
                                        return_dict=True, mode='text')


        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        seq = self.seq_tokenizer(seq, padding='max_length', truncation=True, max_length=450,
                              return_tensors="pt").to(self.seq_encoder.device)
        seq_output = self.seq_encoder(seq.input_ids, attention_mask=seq.attention_mask,
                                        return_dict=True, mode='text')
        seq_feat = F.normalize(self.seq_proj(seq_output.last_hidden_state[:, 0, :]), dim=-1)

        # get momemtum features
        with torch.no_grad():
            if self.training:
                self._momentum_update()
            text_output_m = self.text_encoder_m(input_ids, attention_mask=attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            seq_output_m = self.seq_encoder_m(seq.input_ids, attention_mask=seq.attention_mask,
                                                return_dict=True, mode='text')
            seq_feat_m = F.normalize(self.seq_proj_m(seq_output_m.last_hidden_state[:, 0, :]), dim=-1)
            seq_feat_all = torch.cat([seq_feat_m.t(), self.seq_queue.clone().detach()], dim=1)

            sim_t2s_m = text_feat_m @ seq_feat_all / self.temp
            sim_s2t_m = seq_feat_m @ text_feat_all / self.temp

            sim_targets = torch.zeros(sim_t2s_m.size()).to(self.device)
            sim_targets.fill_diagonal_(1)

            sim_t2s_targets = alpha * F.softmax(sim_t2s_m, dim=1) + (1 - alpha) * sim_targets
            sim_s2t_targets = alpha * F.softmax(sim_s2t_m, dim=1) + (1 - alpha) * sim_targets

        sim_t2s = text_feat @ seq_feat_all / self.temp
        sim_s2t = seq_feat @ text_feat_all / self.temp

        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1) * sim_t2s_targets, dim=1).mean()
        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1) * sim_s2t_targets, dim=1).mean()

        loss_ita = (loss_t2s + loss_s2t) / 2
        if self.training:
            self._dequeue_and_enqueue(text_feat_m, seq_feat_m)
        # ###============== Image-text Matching ===================###
        encoder_input_ids = seq.input_ids.clone()
        encoder_input_ids[:, 0] = self.seq_tokenizer.enc_token_id

        bs = encoder_input_ids.size(0)
        output_pos = self.seq_encoder(encoder_input_ids,
                                       attention_mask=seq.attention_mask,
                                       encoder_hidden_states=text_output.last_hidden_state,
                                       encoder_attention_mask=attention_mask,
                                       return_dict=True,
                                        output_attentions=True
                                       )
        with torch.no_grad():
            weights_s2t = F.softmax(sim_s2t[:, :bs], dim=1) + 1e-4
            weights_s2t.fill_diagonal_(0)
            weights_t2s = F.softmax(sim_t2s[:, :bs], dim=1) + 1e-4
            weights_t2s.fill_diagonal_(0)

        text_embeds_neg = []
        text_atts_neg = []
        text_embeds = text_output.last_hidden_state
        for b in range(bs):
            neg_idx = torch.multinomial(weights_s2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # select a negative seq for each text
        seq_ids_neg = []
        seq_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2s[b], 1).item()
            seq_ids_neg.append(encoder_input_ids[neg_idx])
            seq_atts_neg.append(seq.attention_mask[neg_idx])

        seq_ids_neg = torch.stack(seq_ids_neg, dim=0)
        seq_atts_neg = torch.stack(seq_atts_neg, dim=0)

        seq_ids_all = torch.cat([encoder_input_ids, seq_ids_neg], dim=0)
        seq_atts_all = torch.cat([seq.attention_mask, seq_atts_neg], dim=0)

        text_embeds_all = torch.cat([text_embeds_neg, text_embeds], dim=0)
        text_atts_all = torch.cat([text_atts_neg, attention_mask], dim=0)

        output_neg = self.seq_encoder(seq_ids_all,
                                       attention_mask=seq_atts_all,
                                       encoder_hidden_states=text_embeds_all,
                                       encoder_attention_mask=text_atts_all,
                                       return_dict=True,
                                       )
        ts_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        ts_output = self.itm_head(ts_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(self.seq_encoder.device)
        loss_itm = F.cross_entropy(ts_output, itm_labels)

        return loss_ita, loss_itm


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        key, sequence, text = batch
        words = text[0].split(" ")
        length = int(words[words.index('contains') + 1])
        batch_size = len(sequence)
        captions = self.generate(text, batch_size, sample=True, max_length=length, min_length=length)
        print([text[0], "".join(captions[0].split(" "))])
        return [key[0], text[0], "".join(captions[0].split(" "))]

    def generate(self, text, batch_size, sample=False, num_beams=3, max_length=200, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        text = self.text_tokenizer(text, padding='max_length', truncation=True, max_length=50,
                                   return_tensors="pt")
        text_output = self.text_encoder(text.input_ids.to(self.text_encoder.device),
                                        attention_mask=text.attention_mask.to(self.text_encoder.device),
                                        return_dict=True, mode='text')
        last_hidden_state = text_output.last_hidden_state
        text_atts = text.attention_mask.to(self.text_encoder.device)
        if not sample:
            last_hidden_state = last_hidden_state.repeat_interleave(num_beams, dim=0)
            text_atts = text_atts.repeat_interleave(num_beams, dim=0)
        model_kwargs = {"encoder_hidden_states": last_hidden_state, "encoder_attention_mask": text_atts}

        input_ids = torch.zeros((batch_size, 1), dtype=torch.long).to(self.text_encoder.device)
        input_ids[:, 0] = self.seq_tokenizer.bos_token_id

        if sample:
            # nucleus sampling
            outputs = self.seq_decoder.generate(input_ids=input_ids.to(self.text_encoder.device),
                                                max_length=max_length,
                                                min_length=min_length,
                                                do_sample=True,
                                                top_p=top_p,
                                                num_return_sequences=1,
                                                eos_token_id=self.seq_tokenizer.sep_token_id,
                                                pad_token_id=self.seq_tokenizer.pad_token_id,
                                                repetition_penalty=1.1,
                                                **model_kwargs)
        else:
            # beam search
            outputs = self.seq_decoder.generate(input_ids=input_ids,
                                                max_length=max_length,
                                                min_length=min_length,
                                                num_beams=num_beams,
                                                eos_token_id=self.seq_tokenizer.sep_token_id,
                                                pad_token_id=self.seq_tokenizer.pad_token_id,
                                                repetition_penalty=repetition_penalty,
                                                **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.seq_tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        return captions


    def training_step(self, batch, batch_idx):
        sequence, texts = batch
        batch_size = len(sequence)
        alpha = self.config['alpha'] * min(1, (self.trainer.current_epoch * len(self.trainer.train_dataloader)) + batch_idx) / (2 * len(self.trainer.train_dataloader))
        loss_ita, loss_itm = self(sequence, texts, alpha)
        loss = loss_ita + loss_itm

        self.log("train_ita", loss_ita, batch_size=batch_size, sync_dist=True)
        self.log("train_itm", loss_itm, batch_size=batch_size, sync_dist=True)
        self.log("train_total", loss, batch_size=batch_size, sync_dist=True)
        return loss




    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, text_feat, seq_feat):
        seq_feats = concat_all_gather(seq_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = text_feats.shape[0]

        ptr = int(self.queue_ptr)

        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.seq_queue[:, ptr:ptr + batch_size] = seq_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


def paag_pretrain_protst(**kwargs):
    model = PAAG_Pretrain_Protst(**kwargs)
    return model




@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



def do_CL(X, Y, temp):

    criterion = nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, temp)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B


    return CL_loss


from typing import List

def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            # print(decoder_pointer.weight)
            # print(encoder_pointer.weight)
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name+' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)




def tie_encoder_decoder_weights_initial(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            # encoder_pointer.weight = decoder_pointer.weight
            decoder_pointer.weight = encoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                # encoder_pointer.bias = decoder_pointer.bias
                decoder_pointer.bias = encoder_pointer.bias
            print(module_name+' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)


