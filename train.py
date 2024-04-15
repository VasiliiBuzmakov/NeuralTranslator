import torch.nn as nn
from tqdm import tqdm
import torch
import wandb
import torch
import pathlib
from sacrebleu.metrics import BLEU
import numpy as np


def save_checkpoint(model, scheduler, optimizer, epoch, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, save_path)


def create_mask(src, tgt, de_pad_idx=2, en_pad_idx=2, device="cuda:0"):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device)) == 1).transpose(0, 1)
    tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == de_pad_idx)
    tgt_padding_mask = (tgt == en_pad_idx)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



def train_epoch(model, optimizer, train_loader, sp_model_de, sp_model_en, tqdm_desc, loss_fn, device="cuda:0"):
    losses = 0

    for src, tgt, len_src, len_tgt in tqdm(train_loader, tqdm_desc):
        src = src[:, :torch.max(len_src)]
        tgt = tgt[:, :torch.max(len_tgt)]
        optimizer.zero_grad()
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, sp_model_de.pad_id(), sp_model_en.pad_id(), device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.transpose(1, 2), tgt_out)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(train_loader)


def train(model, 
          optimizer, 
          scheduler,
          n_epochs, 
          train_loader, 
          val_loader, 
          sp_model_de,
          sp_model_en,
          loss_fn,
          val_dataset=None,
          device="cuda:0"):
     
    model.train()
    model = model.to(device)
    highest_score = 1337228148
    bleu_score = 0
    swa_weights = []
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, 
                                 optimizer, 
                                 train_loader, 
                                 sp_model_de,
                                 sp_model_en, 
                                 f'Training {epoch}/{n_epochs}', 
                                 loss_fn,
                                 device)
        if 9 <= epoch:
            swa_weights.append([param.data.clone().cpu() for param in model.parameters()])
        
        if epoch == 14:
            for i, param in enumerate(model.parameters()):
                param.data.copy_(torch.stack([w[i] for w in swa_weights]).mean(dim=0))
            dir_ = str(pathlib.Path(__file__).parent.resolve()) + f"/checkpoints/model_final_epoch={epoch}.pt"
            save_checkpoint(model, scheduler, optimizer, epoch, dir_)

        val_loss = validate(model, 
                            val_loader, 
                            sp_model_de,
                            sp_model_en,
                            f'Validating {epoch}/{n_epochs}', 
                            loss_fn, 
                            device)
        
        bleu_metric = BLEU(lowercase=True)
        sys = []
        refs = []
        if epoch >= 6 and epoch % 2 == 0:
            for i in np.random.choice(a=len(val_dataset), size=len(val_dataset)//2, replace=False):
                src = val_dataset.indices_de[i]
                result = model.translate(src, sp_model_en, device)
                sys.append(result)
                refs.append(val_dataset.text_en[i])
            bleu_score = bleu_metric.corpus_score(sys, [refs]).score
        if type(scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        wandb.log({"Loss train": train_loss,
                   "Loss val": val_loss,
                   "BLEU": bleu_score})
    
        if highest_score > val_loss:
            highest_score = val_loss
            dir_ = str(pathlib.Path(__file__).parent.resolve()) + f"/checkpoints/model_epoch={epoch}.pt"
            save_checkpoint(model, scheduler, optimizer, epoch, dir_)
    
    

@torch.inference_mode()
def validate(model, val_loader, sp_model_de, sp_model_en, tqdm_decs, loss_fn, device="cuda:0"):
    model.eval()
    losses = 0
    
    for src, tgt, len_src, len_tgt in tqdm(val_loader, tqdm_decs):
        src = src[:, :torch.max(len_src)]
        tgt = tgt[:, :torch.max(len_tgt)]
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, sp_model_de.pad_id(), sp_model_en.pad_id(), device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.transpose(1, 2), tgt_out)
        losses += loss.item()

    return losses / len(val_loader)
