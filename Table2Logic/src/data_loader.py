import torch


def get_data_loader(data, batch_size, shuffle=False):
    data_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=data,
        shuffle=shuffle,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )
    return data_loader
