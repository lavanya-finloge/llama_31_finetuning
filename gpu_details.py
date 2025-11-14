import torch

# GPU Name
print("GPU:", torch.cuda.get_device_name(0))

# Total and free memory
total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
reserved = torch.cuda.memory_reserved(0) / (1024**3)
allocated = torch.cuda.memory_allocated(0) / (1024**3)
free = total_memory - reserved

print(f"Total GPU memory: {total_memory:.2f} GB")
print(f"Allocated: {allocated:.2f} GB")
print(f"Reserved: {reserved:.2f} GB")
print(f"Free (estimated): {free:.2f} GB")
