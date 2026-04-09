import torch
p = torch.cuda.get_device_properties(0)
print("name:", p.name)
print("SM count:", p.multi_processor_count)
print("warp size:", p.warp_size)
print("max threads/SM:", p.max_threads_per_multi_processor)
print("max warps/SM:", p.max_threads_per_multi_processor // p.warp_size)
import ctypes

# 按实际环境改名：libcudart.so / libcudart.so.12
cudart = ctypes.CDLL("libcudart.so")

cudaDeviceGetAttribute = cudart.cudaDeviceGetAttribute
cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]

def get_attr(attr, dev=0):
    v = ctypes.c_int()
    rc = cudaDeviceGetAttribute(ctypes.byref(v), attr, dev)
    if rc != 0:
        raise RuntimeError(f"cudaDeviceGetAttribute failed, rc={rc}")
    return v.value

print("maxThreadsPerBlock =", get_attr(1))  # cudaDevAttrMaxThreadsPerBlock