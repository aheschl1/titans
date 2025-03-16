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
    
    loader = get_loader(dset, 20)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.001, mode="triangular2")
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(10):
        epoch_losses = []
        for i, batch in enumerate(tqdm(loader)):
            optimizer.zero_grad()
            model.transformer.zero_grad()
            model.transformer2.zero_grad()
            model.transformer3.zero_grad()
            
            
            inputs = batch["input_ids"][:, :-1].to("cuda")
            targets = batch["input_ids"][:, -1].to("cuda") # [B]
            
            outputs = model(inputs) # [B, V]
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            scheduler.step()
            
            if i % 1000 == 0:
                print(f"Epoch {epoch} Iter {i} Loss: {loss.item()} LR: {scheduler.get_last_lr()}")
                torch.save(model.state_dict(), "code_model2.pth")

        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f"Epoch {epoch} Loss: {losses[-1]}")
        
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
