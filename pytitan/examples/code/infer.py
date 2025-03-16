import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytitan.examples.code.dataset import CodeDataset, get_loader
from pytitan.examples.code.model import CodeModel


def train():
    dset = CodeDataset(max_length=1025)
    model = CodeModel(
        ctx_length=1024,
        embed_dim=128,
        vocab_size=dset.get_vocab_size(),
        padding_value=dset.tokenizer.pad_token_id
    ).to("cuda")
    print("loading model")
    model.load_state_dict(torch.load("code_model2.pth"))
    inpoo = input("Query: ")
    sample = dset.preprocess({"content": inpoo})
    x = sample["input_ids"].unsqueeze(0).to("cuda")
    for i in range(100):
        y = model(x)
        y = torch.argmax(y, dim=-1)
        x = torch.cat([x, y.unsqueeze(0)], dim=-1)
    print(dset.tokenizer.decode(x[0].cpu().numpy()))
        
    
    
    
if __name__ == "__main__":
    train()
    # state = torch.load("finance_model.pth")
    # dset = FinanceDataset("/home/andrew/Documents/Titan/datasets/stock_details_5_years.csv", ctx_length=100)
    # model = FinanceRegressionModel(
    #     ctx_length=100,
    #     embed_dim=16,
    #     num_companies=len(dset.get_companies())
    # ).to("cuda")

    # print(model)
