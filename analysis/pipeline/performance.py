#!/usr/bin/env python3
"""
Performance optimization utilities for linguistic analysis pipeline
"""

import time
import psutil
import gc
from functools import wraps
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import threading
import sys


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    execution_time: float
    memory_usage: float
    peak_memory: float
    cpu_percent: float
    function_name: str
    args_size: Optional[int] = None


class PerformanceProfiler:
    """Advanced performance profiler for linguistic analysis operations"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.total_times: Dict[str, float] = defaultdict(float)
        self._monitoring = False
        self._monitor_thread = None
        self._peak_memory = 0
        
    def profile(self, include_memory: bool = True, include_cpu: bool = False):
        """
        Decorator to profile function performance
        
        Args:
            include_memory: Whether to monitor memory usage
            include_cpu: Whether to monitor CPU usage
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = f"{func.__module__}.{func.__name__}"
                self.call_counts[func_name] += 1
                
                # Start monitoring
                start_time = time.time()
                process = psutil.Process()
                
                if include_memory:
                    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                    self._peak_memory = initial_memory
                    if include_cpu:
                        self._start_monitoring()
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Calculate metrics
                    execution_time = time.time() - start_time
                    self.total_times[func_name] += execution_time
                    
                    memory_usage = 0
                    cpu_percent = 0
                    
                    if include_memory:
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_usage = current_memory - initial_memory
                        peak_memory = self._peak_memory
                        
                    if include_cpu:
                        self._stop_monitoring()
                        cpu_percent = process.cpu_percent()
                    
                    # Calculate argument size estimate
                    args_size = None
                    try:
                        args_size = sum(sys.getsizeof(arg) for arg in args) + sum(
                            sys.getsizeof(v) for v in kwargs.values()
                        )
                    except:
                        pass
                    
                    # Store metrics
                    self.metrics[f"{func_name}_{self.call_counts[func_name]}"] = PerformanceMetrics(
                        execution_time=execution_time,
                        memory_usage=memory_usage,
                        peak_memory=peak_memory if include_memory else 0,
                        cpu_percent=cpu_percent,
                        function_name=func_name,
                        args_size=args_size
                    )
                    
                    return result
                    
                except Exception as e:
                    if include_cpu and self._monitoring:
                        self._stop_monitoring()
                    raise e
                finally:
                    if include_memory:
                        gc.collect()  # Force garbage collection
                        
            return wrapper
        return decorator
    
    def _start_monitoring(self):
        """Start background memory/CPU monitoring"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def _stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Background resource monitoring thread"""
        process = psutil.Process()
        while self._monitoring:
            try:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                self._peak_memory = max(self._peak_memory, current_memory)
                time.sleep(0.1)  # Monitor every 100ms
            except:
                break
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Dict: Performance summary statistics
        """
        if not self.metrics:
            return {"message": "No performance data collected"}
        
        summary = {
            "total_functions_profiled": len(set(m.function_name for m in self.metrics.values())),
            "total_calls": sum(self.call_counts.values()),
            "total_execution_time": sum(self.total_times.values()),
            "function_stats": {}
        }
        
        # Aggregate by function name
        for func_name in set(m.function_name for m in self.metrics.values()):
            func_metrics = [m for m in self.metrics.values() if m.function_name == func_name]
            
            if func_metrics:
                avg_time = sum(m.execution_time for m in func_metrics) / len(func_metrics)
                max_time = max(m.execution_time for m in func_metrics)
                avg_memory = sum(m.memory_usage for m in func_metrics) / len(func_metrics)
                max_memory = max(m.memory_usage for m in func_metrics)
                
                summary["function_stats"][func_name] = {
                    "call_count": self.call_counts[func_name],
                    "total_time": self.total_times[func_name],
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "avg_memory_change": avg_memory,
                    "max_memory_change": max_memory
                }
        
        return summary
    
    def print_summary(self):
        """Print formatted performance summary"""
        summary = self.get_summary()
        
        if "message" in summary:
            print(summary["message"])
            return
        
        print("\n🚀 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"📊 Functions profiled: {summary['total_functions_profiled']}")
        print(f"📞 Total function calls: {summary['total_calls']}")
        print(f"⏱️  Total execution time: {summary['total_execution_time']:.2f}s")
        
        print(f"\n📈 Function Performance:")
        print("-" * 60)
        
        # Sort by total time
        sorted_funcs = sorted(
            summary["function_stats"].items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )
        
        for func_name, stats in sorted_funcs[:10]:  # Top 10
            short_name = func_name.split('.')[-1]  # Just function name
            print(f"🔧 {short_name:<30}")
            print(f"   ⏱️  Total: {stats['total_time']:.2f}s | Avg: {stats['avg_time']:.3f}s | Calls: {stats['call_count']}")
            print(f"   🧠 Memory: Avg {stats['avg_memory_change']:+.1f}MB | Max {stats['max_memory_change']:+.1f}MB")
            print()
    
    def clear(self):
        """Clear all collected metrics"""
        self.metrics.clear()
        self.call_counts.clear()
        self.total_times.clear()


# Global profiler instance
profiler = PerformanceProfiler()


def optimize_memory():
    """
    Memory optimization utility
    """
    import gc
    
    print("🧹 Running memory optimization...")
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Force garbage collection
    collected = gc.collect()
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_freed = initial_memory - final_memory
    
    print(f"✅ Memory optimization complete:")
    print(f"   - Objects collected: {collected}")
    print(f"   - Memory freed: {memory_freed:.1f}MB")
    print(f"   - Current memory: {final_memory:.1f}MB")


def benchmark_function(func: Callable, *args, iterations: int = 1, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function's performance
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        iterations: Number of iterations to run
        **kwargs: Function keyword arguments
    
    Returns:
        Dict: Benchmark results
    """
    print(f"🏃 Benchmarking {func.__name__} over {iterations} iterations...")
    
    times = []
    memory_usage = []
    
    for i in range(iterations):
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        final_memory = process.memory_info().rss
        
        times.append(end_time - start_time)
        memory_usage.append((final_memory - initial_memory) / 1024 / 1024)  # MB
        
        if i % max(1, iterations // 10) == 0:
            print(f"   Iteration {i+1}/{iterations}: {times[-1]:.3f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_memory = sum(memory_usage) / len(memory_usage)
    
    results = {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "avg_memory_delta": avg_memory,
        "iterations": iterations
    }
    
    print(f"📊 Benchmark results for {func.__name__}:")
    print(f"   ⏱️  Average time: {avg_time:.3f}s")
    print(f"   ⚡ Best time: {min_time:.3f}s")
    print(f"   🐌 Worst time: {max_time:.3f}s")
    print(f"   🧠 Average memory change: {avg_memory:+.1f}MB")
    
    return results