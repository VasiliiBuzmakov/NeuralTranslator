import os
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, indices_de, sp_model_en, sp_model_de, indices_en=None, text_en=None, max_length=128):
        
        self.indices_de = indices_de
        self.indices_en = indices_en
        self.sp_model_en = sp_model_en
        self.sp_model_de = sp_model_de
        self.text_en = text_en

        self.pad_id_en, self.unk_id_en, self.bos_id_en, self.eos_id_en = \
            self.sp_model_en.pad_id(), self.sp_model_en.unk_id(), \
            self.sp_model_en.bos_id(), self.sp_model_en.eos_id()

        self.pad_id_de, self.unk_id_de, self.bos_id_de, self.eos_id_de = \
            self.sp_model_de.pad_id(), self.sp_model_de.unk_id(), \
            self.sp_model_de.bos_id(), self.sp_model_de.eos_id()

        self.max_length = max_length
        self.vocab_size_en = self.sp_model_en.vocab_size()
        self.vocab_size_de = self.sp_model_de.vocab_size()

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.indices_de)

    def __getitem__(self, item):
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """

        encoded_de = self.indices_de[item][:self.max_length]
        padded_de = torch.full((self.max_length, ), self.pad_id_de, dtype=torch.int64)
        padded_de[:len(encoded_de)] = torch.tensor(encoded_de)
        
        if self.indices_en is not None:
            encoded_en = [self.bos_id_en] + self.indices_en[item][:self.max_length - 2] + [self.eos_id_en]
            padded_en = torch.full((self.max_length, ), self.pad_id_en, dtype=torch.int64)
            padded_en[:len(encoded_en)] = torch.tensor(encoded_en)
            return padded_de, padded_en, len(encoded_de), len(encoded_en)
        return padded_de, len(encoded_de)

