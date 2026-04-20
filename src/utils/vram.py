import gc
import torch
import pynvml
from typing import Callable, Any, Dict
from src.exceptions import VRAMError
import logging

logger = logging.getLogger(__name__)

class VRAMManager:
    """Manages VRAM usage by tracking peak usage and handling model unloading."""
    
    def __init__(self, device_id: int = 0):
        """
        Initialize the VRAM manager for a specific GPU.
        
        Args:
            device_id: The index of the GPU to manage.
        """
        self.device_id = device_id
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        except pynvml.NVMLError as e:
            raise VRAMError(f"Failed to initialize NVML: {e}")
        
        self.current_model_name: str | None = None
        self.current_model: Any = None
        self.peak_usage: Dict[str, float] = {}  # In GB

    def get_free_vram_gb(self) -> float:
        """
        Get the amount of free VRAM in Gigabytes.
        
        Returns:
            Free VRAM in GB.
        """
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return info.free / (1024**3)
        except pynvml.NVMLError as e:
            raise VRAMError(f"Failed to get memory info: {e}")

    def load_model(self, name: str, loader_fn: Callable[[], Any]) -> Any:
        """
        Unload the previous model, clear cache, and load a new model.
        
        Args:
            name: Description/name of the model being loaded.
            loader_fn: A function that returns the loaded model.
            
        Returns:
            The loaded model.
        """
        if self.current_model is not None:
            logger.info(f"Unloading model: {self.current_model_name}")
            del self.current_model
            self.current_model = None
            self.current_model_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        logger.info(f"Loading model: {name}")
        try:
            self.current_model = loader_fn()
            self.current_model_name = name
            return self.current_model
        except Exception as e:
            raise VRAMError(f"Failed to load model '{name}': {e}")

    def log_peak_usage(self, phase_name: str):
        """
        Track and log peak VRAM usage for a specific phase.
        
        Args:
            phase_name: Name of the pipeline phase to log for.
        """
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used_gb = info.used / (1024**3)
            total_gb = info.total / (1024**3)
            
            if phase_name not in self.peak_usage or used_gb > self.peak_usage[phase_name]:
                self.peak_usage[phase_name] = used_gb
                
            logger.info(f"Peak VRAM for phase '{phase_name}': {used_gb:.2f} GB / {total_gb:.2f} GB")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to log peak usage: {e}")

    def __del__(self):
        """Shutdown NVML on deletion."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass
