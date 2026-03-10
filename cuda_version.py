import torch
print("CUDA disponibile:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU rilevata:", torch.cuda.get_device_name(0))
exit()