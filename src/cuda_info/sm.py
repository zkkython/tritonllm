import torch
p = torch.cuda.get_device_properties(0)
print("name:", p.name)
print("SM count:", p.multi_processor_count)
print("warp size:", p.warp_size)
print("max threads/SM:", p.max_threads_per_multi_processor)
print("max warps/SM:", p.max_threads_per_multi_processor // p.warp_size)