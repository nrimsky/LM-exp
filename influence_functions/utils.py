from torch.utils.data import Dataset

class StringDataset(Dataset):
    def __init__(self, string_list):
        self.data = string_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def slice_chain_dataset(chain_dataset, limit):
    sliced_data = []

    for i, item in enumerate(chain_dataset):
        if i >= limit:
            break
        sliced_data.append(item)

    return StringDataset(sliced_data)