from torch.utils.data import Dataset


class EnemDataset(Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.df = df
        self.data = df.to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        return sample
