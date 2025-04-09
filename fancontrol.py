#!/usr/bin/env python3
import time
import json
import subprocess
import argparse
import shutil
import sys

def load_config(config_file):
    """
    Load and validate configuration from a JSON file using a unified format.
    
    Required global keys:
      - gpu_vendor: "nvidia" or "amd"
      - interval: number of seconds between temperature checks (positive number)
      - temp_samples: integer (>= 1) specifying how many consecutive temperature readings to consider
      - gpus: a non-empty list of GPU configuration objects; each object must include:
          - id: an integer GPU identifier
          - fan_curve: a non-empty list of points [{"temp": <temperature>, "speed": <fan_speed>}, ...]
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file '{config_file}': {e}")
        sys.exit(1)
    
    # Validate global required keys
    for key in ["gpu_vendor", "interval", "temp_samples", "gpus"]:
        if key not in config:
            print(f"Configuration error: Missing required key '{key}'.")
            sys.exit(1)

    # Validate gpu_vendor value
    if config["gpu_vendor"].lower() not in ["nvidia", "amd"]:
        print("Configuration error: 'gpu_vendor' must be either 'nvidia' or 'amd'.")
        sys.exit(1)
    
    # Validate interval is a positive number
    try:
        interval = float(config["interval"])
        if interval <= 0:
            print("Configuration error: 'interval' must be a positive number.")
            sys.exit(1)
    except ValueError:
        print("Configuration error: 'interval' must be a number.")
        sys.exit(1)

    # Validate temp_samples is an integer >= 1 (global setting)
    try:
        ts = int(config["temp_samples"])
        if ts < 1:
            print("Configuration error: 'temp_samples' must be at least 1.")
            sys.exit(1)
    except ValueError:
        print("Configuration error: 'temp_samples' must be an integer.")
        sys.exit(1)

    # Helper to validate a fan_curve structure (per GPU)
    def validate_fan_curve(fc, location=""):
        if not isinstance(fc, list) or not fc:
            print(f"Configuration error: {location}'fan_curve' must be a non-empty list.")
            sys.exit(1)
        for point in fc:
            if not isinstance(point, dict) or "temp" not in point or "speed" not in point:
                print(f"Configuration error: Each point in {location}'fan_curve' must be a dict with 'temp' and 'speed' keys.")
                sys.exit(1)
    
    # Validate "gpus" is a non-empty list
    if not isinstance(config["gpus"], list) or not config["gpus"]:
        print("Configuration error: 'gpus' must be a non-empty list.")
        sys.exit(1)
    
    for gpu_conf in config["gpus"]:
        if not isinstance(gpu_conf, dict):
            print("Configuration error: Each entry in 'gpus' must be a dictionary.")
            sys.exit(1)
        if "id" not in gpu_conf:
            print("Configuration error: Each GPU configuration must include an 'id' key.")
            sys.exit(1)
        if "fan_curve" not in gpu_conf:
            print(f"Configuration error: GPU configuration for id {gpu_conf.get('id')} must include a 'fan_curve' key.")
            sys.exit(1)
        validate_fan_curve(gpu_conf["fan_curve"], location=f"GPU {gpu_conf.get('id')} ")
    
    return config

def interpolate_speed(temp, fan_curve):
    """
    Calculate the target fan speed by linearly interpolating on the provided fan_curve.
    If the temperature is below or above the provided range, the speed is clamped to the endpoints.
    """
    if temp <= fan_curve[0]['temp']:
        return fan_curve[0]['speed']
    if temp >= fan_curve[-1]['temp']:
        return fan_curve[-1]['speed']
    for i in range(1, len(fan_curve)):
        if fan_curve[i]['temp'] >= temp:
            t1, s1 = fan_curve[i - 1]['temp'], fan_curve[i - 1]['speed']
            t2, s2 = fan_curve[i]['temp'], fan_curve[i]['speed']
            return s1 + (s2 - s1) * ((temp - t1) / (t2 - t1))
    return fan_curve[-1]['speed']

# Abstract base class for GPU operations
class GPUControllerBase:
    def get_gpu_list(self):
        """Return a list of GPU identifiers/handles."""
        raise NotImplementedError
    def get_temperature(self, gpu):
        """Return the current temperature for the given GPU."""
        raise NotImplementedError
    def set_fan_speed(self, gpu, speed):
        """Set the fan speed (in percent) for the given GPU."""
        raise NotImplementedError

# NVIDIA-specific implementation
class NvidiaGPUController(GPUControllerBase):
    def __init__(self):
        try:
            import pynvml
        except ImportError:
            print("Error: nvidia-ml-py library is not installed. Please install it and try again.")
            sys.exit(1)
        self.nvml = pynvml
        try:
            self.nvml.nvmlInit()
        except self.nvml.NVMLError as e:
            print(f"Error: Failed to initialize NVML: {e}")
            sys.exit(1)
        self.num_gpus = self.nvml.nvmlDeviceGetCount()

    def get_gpu_list(self):
        gpu_handles = []
        for i in range(self.num_gpus):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
            gpu_handles.append(handle)
        return gpu_handles

    def get_temperature(self, gpu):
        return self.nvml.nvmlDeviceGetTemperature(gpu, self.nvml.NVML_TEMPERATURE_GPU)

    def set_fan_speed(self, gpu, speed):
        """
        Set the fan speed using nvmlDeviceSetFanSpeed_v2.
        WARNING: This function requires that fan control be set to manual.
        """
        try:
            index = self.nvml.nvmlDeviceGetIndex(gpu)
            result = self.nvml.nvmlDeviceSetFanSpeed_v2(gpu, 0, speed)
            if result != self.nvml.NVML_SUCCESS:
                print(f"Error: Failed to set fan speed for NVIDIA GPU {index}. NVML error: {result}")
        except self.nvml.NVMLError as e:
            print(f"Error setting fan speed for NVIDIA GPU: {e}")

# AMD-specific implementation
class AMDGPUController(GPUControllerBase):
    def __init__(self):
        if shutil.which("rocm-smi") is None:
            print("Error: rocm-smi tool is not installed or not found in PATH. Please install it and try again.")
            sys.exit(1)

    def get_gpu_list(self):
        """
        Uses 'rocm-smi --showid --json' to list AMD GPUs.
        Expects JSON with keys like "card0", "card1", etc., and returns a sorted list of GPU IDs.
        """
        cmd = ["rocm-smi", "--showid", "--json"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            gpu_ids = []
            for key in data.keys():
                if key.startswith("card"):
                    try:
                        idx = int(key.replace("card", ""))
                        gpu_ids.append(idx)
                    except ValueError:
                        continue
            gpu_ids.sort()
            if not gpu_ids:
                print("Error: No AMD GPUs found in 'rocm-smi --showid --json' output.")
                sys.exit(1)
            return gpu_ids
        except Exception as e:
            print(f"Error obtaining AMD GPU list: {e}")
            sys.exit(1)

    def get_temperature(self, gpu):
        """
        Uses rocm-smi with --json to get the temperature.
        Parses the JSON for the "Temperature (Sensor edge) (C)" value.
        """
        cmd = ["rocm-smi", "--showtemp", "--device", str(gpu), "--json"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            card_key = f"card{gpu}"
            if card_key in data:
                temp_str = data[card_key].get("Temperature (Sensor edge) (C)")
                if temp_str is not None:
                    return float(temp_str)
                else:
                    print(f"Error: 'Temperature (Sensor edge) (C)' not found for AMD GPU {gpu}.")
                    return None
            else:
                print(f"Error: Key '{card_key}' not found in rocm-smi output.")
                return None
        except Exception as e:
            print(f"Error reading temperature for AMD GPU {gpu}: {e}")
            return None

    def set_fan_speed(self, gpu, speed):
        """
        Uses rocm-smi to set the fan speed of the specified AMD GPU.
        The fan speed is provided as a percentage string (e.g., "75%").
        """
        cmd = ["rocm-smi", "--setfan", f"{speed}%", "--device", str(gpu)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error setting fan speed for AMD GPU {gpu}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Custom GPU Fan Curve Controller")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the configuration file (default: config.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    gpu_vendor = config["gpu_vendor"].lower()
    interval = float(config["interval"])
    temp_samples = int(config["temp_samples"])

    # Build mapping from GPU id to the per-GPU configuration (must be provided for every GPU)
    per_gpu_configs = {}
    for gpu_conf in config["gpus"]:
        per_gpu_configs[int(gpu_conf["id"])] = gpu_conf

    # Initialize the appropriate controller based on GPU vendor
    if gpu_vendor == "nvidia":
        controller = NvidiaGPUController()
    elif gpu_vendor == "amd":
        controller = AMDGPUController()
    else:
        print("Error: Unknown GPU vendor specified in config. Use 'nvidia' or 'amd'.")
        sys.exit(1)

    # Obtain list of GPUs from the controller
    gpus = controller.get_gpu_list()

    # Make sure that every detected GPU has a corresponding configuration
    for gpu in gpus:
        if gpu_vendor == "nvidia":
            gpu_id = controller.nvml.nvmlDeviceGetIndex(gpu)
        else:
            gpu_id = gpu
        if gpu_id not in per_gpu_configs:
            print(f"Error: No configuration provided for GPU {gpu_id}.")
            sys.exit(1)

    # Dictionary to hold temperature history per GPU id
    temp_history = {}

    print("Starting GPU fan control. Press Ctrl+C to exit.")
    try:
        while True:
            for gpu in gpus:
                if gpu_vendor == "nvidia":
                    gpu_id = controller.nvml.nvmlDeviceGetIndex(gpu)
                else:
                    gpu_id = gpu

                gpu_config = per_gpu_configs[gpu_id]
                # Use the per-GPU fan curve
                fan_curve = sorted(gpu_config["fan_curve"], key=lambda x: x["temp"])

                # Initialize temperature history for this GPU if needed
                if gpu_id not in temp_history:
                    temp_history[gpu_id] = []
                current_history = temp_history[gpu_id]

                temp_val = controller.get_temperature(gpu)
                if temp_val is None:
                    continue
                current_history.append(temp_val)
                if len(current_history) > temp_samples:
                    current_history.pop(0)
                measured_temp = max(current_history) if temp_samples > 1 else temp_val

                target_speed = interpolate_speed(measured_temp, fan_curve)
                target_speed = int(round(target_speed))
                controller.set_fan_speed(gpu, target_speed)
                print(f"GPU {gpu_id} | Temp: {measured_temp:.1f}°C -> Fan Speed: {target_speed}%")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Exiting fan control script.")
    finally:
        if gpu_vendor == "nvidia":
            controller.nvml.nvmlShutdown()

if __name__ == '__main__':
    main()
