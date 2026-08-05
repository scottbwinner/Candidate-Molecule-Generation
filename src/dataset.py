from torch.utils.data import Dataset

class Zinc250kDataset(Dataset):
    def __init__(self, encoded_smiles):
        """
        Parameters:
            encoded_smiles: encoded and padded SMILES sequences as tensor of shape (N, max_len) dtype torch.long
        """
        self.encoded_smiles = encoded_smiles

    def __len__(self):
        return len(self.encoded_smiles)

    def __getitem__(self, idx):
        return self.encoded_smiles[idx]