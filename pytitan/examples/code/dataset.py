from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [sample["input_ids"] for sample in batch]

    # Pad sequences to the longest in the batch
    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=batch[0]["padding_value"])

    return {"input_ids": padded_inputs}


class CodeDataset(IterableDataset):
    def __init__(self, max_length=1024):
        self.dataset = load_dataset("bigcode/the-stack", split="train", streaming=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        
    def get_vocab_size(self):
        return len(self.tokenizer.vocab)

    def __iter__(self):
        for sample in iter(self.dataset):  # Streaming dataset yields samples one by one
            yield self.preprocess(sample)

    def preprocess(self, sample):
        code = sample["content"] 

        tokens = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding=False  # Padding is handled in `collate_fn`
        )

        return {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "padding_value": self.tokenizer.pad_token_id
        }


def get_loader(dset, batch_size:int =8):
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

if __name__ == "__main__":
    loader = get_loader()
    i = 0
    for batch in loader:
        print(batch["input_ids"].shape)
        i += 1
        if i == 30:
            break
        