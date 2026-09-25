import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# 1. Define genotype mapping to numerical indices for neural network inputs
GENOTYPE_MAP = {
    "--": 0,  # Missing call
    "AA": 1,
    "AC": 2,
    "AG": 3,
    "AT": 4,
    "CC": 5,
    "CG": 6,
    "CT": 7,
    "GG": 8,
    "GT": 9,
    "TT": 10,
}


# 2. Build PyTorch Dataset using Pandas
class GeneticDataset(Dataset):

    def __init__(self, file_path):
        # Load data with Pandas
        df = pd.read_csv(
            file_path,
            sep="\t",
            comment="#",
            header=None,
            names=["rsid", "chromosome", "position", "genotype"],
            dtype=str,
        )

        # Convert genomic position to LongTensor
        self.positions = torch.tensor(
            df["position"].astype(int).values, dtype=torch.long
        )

        # Map string genotypes (e.g., 'AG') to numeric IDs
        numeric_genotypes = df["genotype"].map(GENOTYPE_MAP).fillna(0).values
        self.genotypes = torch.tensor(numeric_genotypes, dtype=torch.long)

    def __len__(self):
        return len(self.genotypes)

    def __getitem__(self, idx):
        return {
            "position": self.positions[idx],
            "genotype_id": self.genotypes[idx],
        }


# 3. Usage with PyTorch DataLoader
dataset = GeneticDataset("/Users/dr/Documents/personal/f1.txt")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Inspect a single batch
for batch in dataloader:
    print("Positions tensor shape:", batch["position"].shape)
    print("Genotype IDs tensor shape:", batch["genotype_id"].shape)
    print("Sample Genotype Tensor:", batch["genotype_id"][:5])
    break