import torch
import logging

def get_device(device_name: str = "auto") -> torch.device:
    """
    Returns the appropriate torch device.
    Prioritizes CUDA > DirectML > CPU.
    """
    if device_name != "auto":
        return torch.device(device_name)

    # 1. Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using GPU (CUDA).")
        return torch.device("cuda")

    # 2. Check for DirectML (AMD/Intel on Windows)
    try:
        import torch_directml
        if torch_directml.is_available():
            logging.info("DirectML is available. Using GPU (DirectML).")
            return torch_directml.device()
    except ImportError:
        pass

    # 3. Fallback to CPU
    logging.info("No GPU acceleration found. Using CPU.")
    return torch.device("cpu")
