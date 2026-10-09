"""
Prints the input tensor name and shape of a serialized engine. A
dynamic-batch engine should report -1 for the batch dimension.
"""

import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)

with open("policy_fp16_dynamic.trt", "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

input_name = engine.get_tensor_name(0)

print("Input name:", input_name)
print("Input shape:", engine.get_tensor_shape(input_name))
