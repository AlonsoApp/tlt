# setting device on GPU if available, else CPU
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(device))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(device) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(device) / 1024 ** 3, 1), 'GB')
    print('Total:   ', round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1), 'GB')