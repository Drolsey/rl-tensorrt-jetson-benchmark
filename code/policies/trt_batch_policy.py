import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TRTBatchPolicy:
    """
    Runs a dynamic-batch TensorRT policy engine and returns greedy
    (argmax) actions.

    The observation and action sizes are read from the engine itself,
    so the same class serves both CartPole (4 obs, 2 actions) and
    LunarLander (8 obs, 4 actions). Device buffers are allocated once,
    sized for max_batch, and reused across calls.

    With return_kernel_time=True a call returns the GPU execution time
    of the engine in ms (CUDA events) instead of the actions.
    """

    def __init__(self, engine_path, max_batch=32):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = "obs"
        self.output_name = "logits"

        self.max_batch = max_batch

        # The output shape is only fully defined once an input shape is
        # set, so set it to max_batch and read the action dimension.
        self.context.set_input_shape(self.input_name, (self.max_batch,) + self._obs_tail_shape())
        out_shape = self.context.get_tensor_shape(self.output_name)
        self.obs_dim = self._obs_tail_shape()[0]
        self.action_dim = int(out_shape[-1])

        self.d_in = None
        self.d_out = None

    def _obs_tail_shape(self):
        """
        Non-batch dimensions of the input tensor, e.g. (4,) for CartPole.
        The engine reports the dynamic batch axis as -1, so it is dropped.
        """
        binding_shape = self.engine.get_tensor_shape(self.input_name)
        tail = tuple(d for d in binding_shape if d != -1)
        return tail

    def __call__(self, obs, return_kernel_time=False):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        batch, obs_dim = obs.shape

        if obs_dim != self.obs_dim:
            raise ValueError(
                f"obs dim {obs_dim} does not match engine's expected obs dim {self.obs_dim}"
            )
        if batch > self.max_batch:
            raise ValueError(f"Batch {batch} exceeds max_batch {self.max_batch}")

        self.context.set_input_shape(self.input_name, obs.shape)

        # allocate once, sized for the largest batch the engine accepts
        if self.d_in is None:
            self.d_in = cuda.mem_alloc(self.max_batch * self.obs_dim * 4)  # float32
        if self.d_out is None:
            self.d_out = cuda.mem_alloc(self.max_batch * self.action_dim * 4)  # float32

        obs = np.ascontiguousarray(obs, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_in, obs, self.stream)

        self.context.set_tensor_address(self.input_name, int(self.d_in))
        self.context.set_tensor_address(self.output_name, int(self.d_out))

        # the events bracket engine execution only; the host<->device
        # copies before and after are not included in kernel_time_ms
        start_event = cuda.Event()
        end_event = cuda.Event()

        start_event.record(self.stream)
        self.context.execute_async_v3(self.stream.handle)
        end_event.record(self.stream)
        self.stream.synchronize()

        kernel_time_ms = start_event.time_till(end_event)

        logits = np.empty((batch, self.action_dim), dtype=np.float32)
        cuda.memcpy_dtoh(logits, self.d_out)

        actions = np.argmax(logits, axis=1)

        if return_kernel_time:
            return kernel_time_ms
        return actions
