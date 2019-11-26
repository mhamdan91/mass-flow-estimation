from tensorflow.python.client import device_lib
def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        for x in local_device_protos:
            if x.device_type == 'GPU':
                return True
        return False

GPU_available =get_available_gpus()
print(GPU_available)