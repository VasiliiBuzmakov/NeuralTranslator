import pathlib
import os
from dataset import CustomDataset
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import DataLoader
from scheduler import CosineWarmupScheduler
from tqdm import tqdm
from torch import optim
import torch
import torch.nn as nn
from train import train
from model import Seq2SeqTransformer
import wandb
import sys 

def load_checkpoint(model, scheduler, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, scheduler, optimizer, epoch


def main():

    root_dir = str(pathlib.Path(__file__).parent.resolve())
    en_file_train, de_file_train = f"{root_dir}/train.de-en.en",f"{root_dir}/train.de-en.de"
    en_file_val, de_file_val = f"{root_dir}/val.de-en.en",f"{root_dir}/val.de-en.de"
    en_model_prefix, de_model_prefix = "en", "de"
    normalization_rule_name = 'nmt_nfkc_cf'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_type_de = 'bpe'
    model_type_en = 'word'

    size_train = 1
    batch_size = 150
    n_epochs = 100
    en_vocab_size, de_vocab_size = 20000, 23000
    max_length = 90

    num_encoder_layers = 4
    num_decoder_layers = 3
    dim_feedforward = 1536
    dropout = 0.25
    label_smoothing = 0.05
    emb_size = 384
    nhead = 8

    if len(sys.argv) == 1:
        sp_model_en = SentencePieceProcessor(model_file=en_model_prefix + '.model')
        sp_model_de = SentencePieceProcessor(model_file=de_model_prefix + '.model')
        model = Seq2SeqTransformer(num_encoder_layers=num_encoder_layers,
                                   num_decoder_layers=num_decoder_layers,
                                   emb_size=emb_size,
                                   nhead=nhead,
                                   src_vocab_size=sp_model_de.vocab_size(),
                                   tgt_vocab_size=sp_model_en.vocab_size(),
                                   dim_feedforward=dim_feedforward,
                                   dropout=dropout,
                                   pad_idx_en=sp_model_de.pad_id(),
                                   pad_idx_de=sp_model_en.pad_id()).to(device)

        model.load_state_dict(torch.load(f"{root_dir}/model.pt", map_location="cpu")['model_state_dict'])
        model = model.to(device)
        model.eval()
        de_file_test = f"{root_dir}/test1.de-en.de"
        with open(de_file_test) as file:
            test = file.readlines()
                
        trans = []
        for text in tqdm(test):
            trans.append(model.translate(sp_model_de.encode(text), sp_model_en, device))
        
        with open("file.txt", 'w') as output:
            for row in trans:
                output.write(str(row) + '\n')
        return 0

    if not os.path.isfile(en_model_prefix + '.model'):
        SentencePieceTrainer.train(
            input=en_file_train, vocab_size=en_vocab_size,
            model_type=model_type_en, model_prefix=en_model_prefix,
            normalization_rule_name=normalization_rule_name,
            split_by_unicode_script=False,
            pad_id=0, unk_id=3, eos_id=2, bos_id=1)
    sp_model_en = SentencePieceProcessor(model_file=en_model_prefix + '.model')

    if not os.path.isfile(de_model_prefix + '.model'):
        SentencePieceTrainer.train(
            input=de_file_train, vocab_size=de_vocab_size,
            model_type=model_type_de, model_prefix=de_model_prefix,
            normalization_rule_name=normalization_rule_name,
            split_by_unicode_script=False,
            pad_id=0, unk_id=3, eos_id=2, bos_id=1)
    sp_model_de = SentencePieceProcessor(model_file=de_model_prefix + '.model')

    with open(en_file_train) as file:
        texts_en_train = file.readlines()
        texts_en_train = texts_en_train[: int(size_train*len(texts_en_train))]
    with open(de_file_train) as file:
        texts_de_train = file.readlines()
        texts_de_train = texts_de_train[: int(size_train*len(texts_de_train))]
    with open(en_file_val) as file:
        texts_en_val = file.readlines()
    with open(de_file_val) as file:
        texts_de_val = file.readlines()
    train_dataset = CustomDataset(sp_model_de.encode(texts_de_train), 
                                  sp_model_en, 
                                  sp_model_de,
                                  indices_en=sp_model_en.encode(texts_en_train), 
                                  max_length=max_length)
    
    val_dataset = CustomDataset(sp_model_de.encode(texts_de_val), 
                                sp_model_en, 
                                sp_model_de,
                                text_en=texts_en_val,
                                indices_en=sp_model_en.encode(texts_en_val), 
                                max_length=max_length)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              pin_memory=True)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size,
                            pin_memory=True)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=sp_model_en.pad_id(), label_smoothing=label_smoothing)
    model = Seq2SeqTransformer(num_encoder_layers=num_encoder_layers,
                               num_decoder_layers=num_decoder_layers,
                               emb_size=emb_size,
                               nhead=nhead,
                               src_vocab_size=sp_model_de.vocab_size(),
                               tgt_vocab_size=sp_model_en.vocab_size(),
                               dim_feedforward=dim_feedforward,
                               dropout=dropout,
                               pad_idx_en=sp_model_en.pad_id(),
                               pad_idx_de=sp_model_de.pad_id()
                               ).to(device)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9) # мб в лучшем трае был lr=0.0005
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, threshold=0.001, patience=4)
    scheduler = CosineWarmupScheduler(optimizer, 3, n_epochs)
    if sys.argv[1] != "new":
        model, scheduler, optimizer, cur_epoch = load_checkpoint(model, scheduler, optimizer, f"{root_dir}/model.pt")
        n_epochs -= cur_epoch
    
    wandb.login()
    wandb.init(
        project="bhw-2",
        config={
            "architecture": "Transformer",
            "epochs": n_epochs,
            "optimizer": "Adam",
            "alpha_1": 0.9,
            "alpha_2": 0.98,
            "loss": "CrossEntropyLoss",
            "size_train": size_train,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "en_vocab_size": en_vocab_size,
            "de_vocab_size": de_vocab_size,
            "max_length": max_length,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "emb_size": emb_size,
            "nhead": nhead,
            "scheduler": str(scheduler)
        }
    )

    train(model, 
          optimizer, 
          scheduler, 
          n_epochs,
          train_loader, 
          val_loader, 
          sp_model_de,
          sp_model_en,
          loss_fn,
          val_dataset, 
          device)


if __name__ == "__main__":
    main()
