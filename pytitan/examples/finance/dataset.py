from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import torch

class FinanceDataset(Dataset):
    def __init__(self, data_path, transform=None, ctx_length=100):
        self.data = pd.read_csv(data_path)
        self.data = self.data.groupby("Company").filter(lambda x: len(x) > ctx_length)
        self.ctx_length = ctx_length
        # filter companies with less than ctx_length data points
        # standardize the data
        self.companies = sorted(self.data["Company"].unique())
        self.mean, self.mu = self.data["Close"].mean(), self.data["Close"].std()
        
    def __len__(self):
        return len(self.data) // self.ctx_length - 1
    
    def scale(self, x):
        return (x - self.mean) / self.mu
    
    def inverse_scale(self, x):
        return x * self.mu + self.mean
    
    def get_companies(self):
        return self.companies

    def __getitem__(self, idx):
        cidx = random.randint(0, len(self.companies) - 1)
        company = self.companies[cidx]
        cdata = self.data[self.data["Company"] == company]
        
        idx = min(idx, len(cdata) // self.ctx_length - 1)
        
        sample = cdata.iloc[idx*self.ctx_length:(idx+1)*self.ctx_length]
        next_sample = cdata.iloc[(idx+1)*self.ctx_length]

        sample = sample["Close"].values
        next_sample = next_sample["Close"]
        point = {
            "x": self.scale(torch.tensor(sample, dtype=torch.float32).unsqueeze(1)),
            "company": torch.tensor(cidx, dtype=torch.long),
            "y": self.scale(torch.tensor(next_sample, dtype=torch.float32))
        }
        
        return point
    
if __name__ == "__main__":
    dataset = FinanceDataset("/home/andrew/Documents/Titan/datasets/stock_details_5_years.csv", ctx_length=100)
    print(dataset.get_companies())
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(loader))
    for batch in loader:
        print(dataset.inverse_scale(batch["x"]))
        break