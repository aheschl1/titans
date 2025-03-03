from pytitan.model.mac import MemoryAsContext
import torch.nn as nn
import torch
from pytitan.examples.finance.dataset import FinanceDataset

class FinanceRegressionModel(nn.Module):
    def __init__(
        self, 
        ctx_length=100,
        embed_dim=16,
        num_companies=10
    ):
        super(FinanceRegressionModel, self).__init__()
        # from [B, N, 1] to [B, N, E]
        self.x_emd = nn.Linear(1, embed_dim)
        self.company_embed = nn.Embedding(num_companies, embed_dim)
        
        self.transformer = MemoryAsContext(
            dim_in=embed_dim,
            long_term_memory_dim=embed_dim,
            persistent_memory_dim=16,
            long_term_update_chunk_size=ctx_length // 20
        )
        
        self.projection = nn.Sequential(
            nn.SELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x, company):
        x = self.x_emd(x)
        company = self.company_embed(company)
        # company is [B, E]. place it at the end of the sequence as context so x is [B, N+1, E]
        x = torch.cat([x, company.unsqueeze(1)], dim=1) 
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.projection(x)
    
if __name__ == "__main__":
    dset = FinanceDataset("./stock_details_5_years.csv", ctx_length=100)
    model = FinanceRegressionModel(
        ctx_length=100,
        embed_dim=16,
        num_companies=len(dset.get_companies())
    )
    loader = torch.utils.data.DataLoader(dset, batch_size=2, shuffle=True)
    for batch in loader:
        x = batch["x"]
        company = batch["company"]
        y = batch["y"]
        
        y_hat = model(x, company).squeeze(1)
        loss = nn.L1Loss()(y_hat, y)
        print(loss)
