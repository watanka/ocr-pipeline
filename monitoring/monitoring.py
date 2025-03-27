import os
import time
import json
import logging
from datetime import datetime
import psutil

# GPU 사용 가능 여부에 따라 GPU 유틸리티 임포트
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

try:
    import GPUtil
    has_gputil = True
except ImportError:
    has_gputil = False

class ResourceMonitor:
    """
    GPU, CPU, 메모리 사용량을 모니터링하는 간단한 클래스
    """
    def __init__(self, service_name, log_dir='logs'):
        self.service_name = service_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger(f"{service_name}_monitor")
        log_file = os.path.join(log_dir, f"{service_name}_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.log_file = log_file
        
        # 상태 변수
        self.timers = {}
        self.counters = {}
        self.metrics = {}
        
    def get_timestamp(self):
        """현재 타임스탬프를 반환합니다."""
        return datetime.now().isoformat()
    
    def get_system_resources(self):
        """시스템 리소스 사용량(CPU, 메모리, GPU)을 반환합니다."""
        resources = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent
        }
        
        # GPU 정보 수집 (가능한 경우)
        if has_gputil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 첫 번째 GPU 사용
                    resources["gpu_load"] = gpu.load * 100
                    resources["gpu_memory_used"] = gpu.memoryUsed
                    resources["gpu_memory_total"] = gpu.memoryTotal
                    resources["gpu_memory_percent"] = gpu.memoryUtil * 100
                    resources["gpu_temperature"] = gpu.temperature
            except Exception as e:
                self.logger.warning(f"GPU 정보 수집 실패: {str(e)}")
        elif has_torch and torch.cuda.is_available():
            try:
                resources["gpu_memory_used"] = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                resources["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
                resources["gpu_memory_percent"] = (resources["gpu_memory_used"] / resources["gpu_memory_total"]) * 100
            except Exception as e:
                self.logger.warning(f"CUDA 메모리 정보 수집 실패: {str(e)}")
        
        return resources
    
    def start_timer(self, name):
        """타이머를 시작합니다."""
        self.timers[name] = {
            "start": time.time(),
            "resources_start": self.get_system_resources()
        }
        return self.timers[name]["start"]
    
    def stop_timer(self, name):
        """타이머를 중지하고 경과 시간을 반환합니다."""
        if name not in self.timers:
            return 0
        
        end_time = time.time()
        elapsed = end_time - self.timers[name]["start"]
        resources_end = self.get_system_resources()
        
        # 리소스 사용량 변화 기록
        resources = {
            "start": self.timers[name]["resources_start"],
            "end": resources_end,
            "elapsed": elapsed
        }
        
        # 로그 저장
        self.log_event("timer", {
            "name": name,
            "elapsed": elapsed,
            "resources": resources
        })
        
        # 메트릭스에 추가
        if "timers" not in self.metrics:
            self.metrics["timers"] = {}
        
        if name not in self.metrics["timers"]:
            self.metrics["timers"][name] = []
            
        self.metrics["timers"][name].append(elapsed)
        
        return elapsed
    
    def increment_counter(self, name, value=1):
        """카운터를 증가시킵니다."""
        if name not in self.counters:
            self.counters[name] = 0
        
        self.counters[name] += value
        
        # 로그 저장
        self.log_event("counter", {
            "name": name,
            "value": value,
            "total": self.counters[name]
        })
        
        return self.counters[name]
    
    def record_metric(self, name, value, extra_info=None):
        """메트릭을 기록합니다."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # 추가 정보가 있는 경우만 로깅
        if extra_info:
            self.log_event("metric", {
                "name": name,
                "value": value,
                "extra": extra_info
            })
        
        return value
    
    def log_image_processing(self, image_id, process_type, elapsed_time, boxes_count=0, extra_info=None):
        """이미지 처리 정보를 로깅합니다."""
        resources = self.get_system_resources()
        
        data = {
            "image_id": image_id,
            "process_type": process_type,  # 'detection' 또는 'recognition'
            "elapsed_time": elapsed_time,
            "boxes_count": boxes_count,
            "resources": resources
        }
        
        if extra_info:
            data["extra_info"] = extra_info
            
        self.log_event("image_processing", data)
        
        # 성능 지표 저장
        summary = {
            "latency_ms": elapsed_time * 1000,
            "boxes_count": boxes_count,
            "cpu_percent": resources.get("cpu_percent", 0),
            "memory_percent": resources.get("memory_percent", 0)
        }
        
        # GPU 정보가 있는 경우 추가
        if "gpu_load" in resources:
            summary["gpu_load"] = resources.get("gpu_load", 0)
            summary["gpu_memory_percent"] = resources.get("gpu_memory_percent", 0)
            
        return summary
        
    def log_event(self, event_type, data):
        """이벤트 로그를 파일에 저장합니다."""
        log_entry = {
            "timestamp": self.get_timestamp(),
            "service": self.service_name,
            "type": event_type,
            event_type: data
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"로그 저장 실패: {str(e)}")
    
    def get_summary(self):
        """수집된 메트릭 요약 정보를 반환합니다."""
        summary = {
            "service": self.service_name,
            "counters": self.counters,
            "current_resources": self.get_system_resources()
        }
        
        # 타이머 통계 계산
        if "timers" in self.metrics:
            timer_stats = {}
            for name, values in self.metrics["timers"].items():
                if values:
                    timer_stats[name] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
            summary["timer_stats"] = timer_stats
        
        return summary

# 전역 모니터 인스턴스
_monitors = {}

def get_monitor(service_name, log_dir='logs'):
    """서비스별 모니터 인스턴스를 반환합니다 (싱글톤 패턴)."""
    if service_name not in _monitors:
        _monitors[service_name] = ResourceMonitor(service_name, log_dir)
    
    return _monitors[service_name] 