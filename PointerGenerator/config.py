# -*- coding: utf-8 -*-

import os

SENTENCE_STA = '<s>'
SENTENCE_END = '</s>'

UNK = 0
PAD = 1
BOS = 2
EOS = 3

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'

beam_size=1
emb_dim= 10
batch_size= 16
hidden_dim= 10
max_enc_steps=10
max_dec_steps=10
max_tes_steps=10
min_dec_steps=5
vocab_size=50

lr=0.15
cov_loss_wt = 1.0
pointer_gen = True
is_coverage = True

max_grad_norm=2.0
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4

tran = False

eps = 1e-12
use_gpu=True
lr_coverage=0.15
max_iterations = 5

# root_dir = os.path.expanduser("./")
log_root = "./log/"

#train_data_path = os.path.join(root_dir, "./finished_files/train.bin")
train_data_path = "./finished_files/chunked/train_*"
eval_data_path = "./finished_files/val.bin"
decode_data_path = "./finished_files/test.bin"
vocab_path = "./finished_files/vocab"
