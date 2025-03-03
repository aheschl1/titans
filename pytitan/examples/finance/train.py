import torch
from pytitan.examples.finance.finance_model import FinanceRegressionModel
from pytitan.examples.finance.dataset import FinanceDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train():
    dset = FinanceDataset("./stock_details_5_years.csv", ctx_length=100)
    model = FinanceRegressionModel(
        ctx_length=100,
        embed_dim=16,
        num_companies=len(dset.get_companies())
    ).to("cuda")
    model.load_state_dict(torch.load("finance_model.pth"))
    loader = DataLoader(dset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(10):
        epoch_losses = []
        for batch in tqdm(loader):
            optimizer.zero_grad()
            model.transformer.zero_grad()
            
            x = batch["x"].to("cuda", non_blocking=True)
            company = batch["company"].to("cuda", non_blocking=True)
            y = batch["y"].to("cuda", non_blocking=True)
            
            y_hat = model(x, company).squeeze(1)
            loss = nn.L1Loss()(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f"Epoch {epoch} Loss: {losses[-1]}")
    torch.save(model.state_dict(), "finance_model.pth")
        
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