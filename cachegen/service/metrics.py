import time
from typing import Dict, List
import psutil
import numpy as np
from fastapi import APIRouter

router = APIRouter()

class MetricsTracker:
    def __init__(self):
        self.latencies_with_cache: List[float] = []
        self.latencies_without_cache: List[float] = []
        self.memory_samples_with_cache: List[float] = []
        self.memory_samples_without_cache: List[float] = []
        self.window_size = 50  # Keep last 50 samples
        
    def record_latency(self, latency_ms: float, with_cache: bool):
        """Record a latency measurement."""
        if with_cache:
            self.latencies_with_cache.append(latency_ms)
            self.latencies_with_cache = self.latencies_with_cache[-self.window_size:]
        else:
            self.latencies_without_cache.append(latency_ms)
            self.latencies_without_cache = self.latencies_without_cache[-self.window_size:]
            
    def record_memory(self, with_cache: bool):
        """Record current memory usage."""
        memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        if with_cache:
            self.memory_samples_with_cache.append(memory_mb)
            self.memory_samples_with_cache = self.memory_samples_with_cache[-self.window_size:]
        else:
            self.memory_samples_without_cache.append(memory_mb)
            self.memory_samples_without_cache = self.memory_samples_without_cache[-self.window_size:]
            
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        return {
            "withCache": {
                "avgLatency": np.mean(self.latencies_with_cache) if self.latencies_with_cache else None,
                "memoryUsage": np.mean(self.memory_samples_with_cache) if self.memory_samples_with_cache else None
            },
            "withoutCache": {
                "avgLatency": np.mean(self.latencies_without_cache) if self.latencies_without_cache else None,
                "memoryUsage": np.mean(self.memory_samples_without_cache) if self.memory_samples_without_cache else None
            }
        }

# Global metrics tracker
metrics = MetricsTracker()

@router.get("/metrics")
async def get_metrics():
    """Get current performance metrics."""
    return metrics.get_metrics()
