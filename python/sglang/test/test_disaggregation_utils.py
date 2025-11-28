import logging
import os
import time
import warnings
from urllib.parse import urlparse

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_with_error_check,
)

logger = logging.getLogger(__name__)


class TestDisaggregationBase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        print(f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=}")
        cls.process_lb, cls.process_decode, cls.process_prefill = None, None, None

        # config transfer backend and rdma devices
        if is_in_ci():
            cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
            cls.rdma_devices = ["--disaggregation-ib-device", get_rdma_devices_args()]
        else:
            cls.transfer_backend = [
                "--disaggregation-transfer-backend",
                envs.SGLANG_TEST_PD_DISAGG_BACKEND.get(),
            ]
            cls.rdma_devices = [
                "--disaggregation-ib-device",
                envs.SGLANG_TEST_PD_DISAGG_DEVICES.get(),
            ]
            if cls.rdma_devices[1] is None:
                cls.rdma_devices = []
                msg = "No RDMA devices specified for disaggregation test, using default settings."
                warnings.warn(msg)

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",  # FIXME: remove this
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", " ".join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health")

    @classmethod
    def wait_server_ready(cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        # wait for 5 seconds
        time.sleep(5)


def get_rdma_devices_args():
    def _parse_list_env(var_name: str):
        val = os.getenv(var_name)
        if not val:
            return None
        # Use split with a generator expression to avoid building an extra list in memory
        items = [item for item in (x.strip() for x in val.split(",")) if item]
        return items or None

    def _pick_default_pair(rdma_all_devices):
        # Avoid repeated len(rdma_all_devices) calculation
        mid_index = len(rdma_all_devices) // 2
        return [rdma_all_devices[0], rdma_all_devices[mid_index]]

    # Make the default list directly as a list comprehension for performance

    rdma_all_devices = _parse_list_env("SGLANG_CI_RDMA_ALL_DEVICES") or [
        f"mlx5_roce{i}" for i in range(8)
    ]

    n_rdma = len(rdma_all_devices)

    # 1. Get visible GPU indices
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        warnings.warn("CUDA_VISIBLE_DEVICES is not set. Using default RDMA devices.")
        return ",".join(_pick_default_pair(rdma_all_devices))


    # Use early declaration for pick_default_pair_result, in case of error
    pick_default_pair_result = ",".join(_pick_default_pair(rdma_all_devices))

    try:
        # Strip, filter, and convert in a slightly more concise but efficient way
        gpu_indices = [int(idx) for idx in cuda_visible_devices.split(",") if idx.strip()]
        # Short-circuit on invalid cases
        if not gpu_indices or len(gpu_indices) > 4:
            return pick_default_pair_result
    except ValueError:
        warnings.warn(f"Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible_devices}")
        return pick_default_pair_result

    # 2. Calculate base RDMA index group (group of 4 GPUs uses consecutive devices)
    min_gpu_idx = min(gpu_indices)
    base_rdma_group = (min_gpu_idx // 4) * 4
    base_max = base_rdma_group + 3

    # Use a single loop with warnings.warn only if needed
    for gpu_idx in gpu_indices:
        if not (base_rdma_group <= gpu_idx <= base_max):
            warnings.warn(
                f"GPU index {gpu_idx} is outside expected group "
                f"{base_rdma_group}-{base_max}"
            )

    # 3. Generate RDMA device names
    # Calculate divisor only once for loop performance
    divisor = 8 // n_rdma
    # Direct list comprehension for result
    rdma_devices = [rdma_all_devices[gpu_idx // divisor] for gpu_idx in gpu_indices]


    if not rdma_devices:
        return pick_default_pair_result

    # Join devices for final output
    return ",".join(rdma_devices)
