import subprocess
import platform
import re
import json


class BluetoothAudioScanner:
    def __init__(self):
        self.os_type = platform.system()
    
    def scan_audio_devices(self):
        devices = []
        
        if self.os_type == "Windows":
            devices = self._scan_windows()
        elif self.os_type == "Linux":
            devices = self._scan_linux()
        elif self.os_type == "Darwin":
            devices = self._scan_macos()
        
        # Filter audio devices
        audio_devices = []
        for dev in devices:
            dev_type = str(dev.get('type', '')).lower()
            name = str(dev.get('name', '')).lower()
            if any(keyword in dev_type or keyword in name for keyword in 
                   ['audio', 'headphone', 'headset', 'speaker', 'earphone', 'bluetooth', 'a2dp', 'handsfree', 'buds', 'airpod']):
                audio_devices.append(dev)
            elif dev.get('connected', False):
                audio_devices.append(dev)
        
        # Remove duplicates by name
        unique = []
        seen = set()
        for dev in audio_devices if audio_devices else devices:
            key = dev.get('name', '').lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(dev)
        return unique
    
    def _scan_windows(self):
        devices = []
        try:
            # Use PowerShell to get Bluetooth devices
            ps_cmd = """
            Get-PnpDevice -Class Bluetooth | Where-Object { $_.Status -eq 'OK' -or $_.Status -eq 'Unknown' } | 
            ForEach-Object {
                [PSCustomObject]@{
                    Name = $_.FriendlyName
                    InstanceId = $_.InstanceId
                    Status = $_.Status
                    Class = $_.Class
                }
            } | ConvertTo-Json
            """
            result = subprocess.run(['powershell', '-Command', ps_cmd], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.stdout.strip():
                try:
                    data = json.loads(result.stdout)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        name = item.get('Name', 'Unknown')
                        if not name or name == 'Unknown':
                            continue
                        devices.append({
                            'name': name,
                            'address': item.get('InstanceId', 'N/A')[-17:],
                            'type': self._detect_type(name),
                            'status': 'Подключено' if item.get('Status') == 'OK' else item.get('Status'),
                            'source': 'Windows PnP'
                        })
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Error scanning Windows BT: {e}")
        
        # Also try to get audio devices
        try:
            ps_cmd2 = """
            Get-WmiObject -Class Win32_SoundDevice | 
            Where-Object { $_.Name -like '*Bluetooth*' -or $_.Name -like '*Headset*' -or $_.Name -like '*Headphone*' } |
            ForEach-Object {
                [PSCustomObject]@{
                    Name = $_.Name
                    Status = $_.Status
                }
            } | ConvertTo-Json
            """
            result = subprocess.run(['powershell', '-Command', ps_cmd2], 
                                  capture_output=True, text=True, timeout=10)
            if result.stdout.strip():
                try:
                    data = json.loads(result.stdout)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        name = item.get('Name', '')
                        if any(d['name'] == name for d in devices):
                            continue
                        devices.append({
                            'name': name,
                            'address': 'N/A',
                            'type': 'Аудиоустройство',
                            'status': item.get('Status', 'Неизвестно'),
                            'source': 'Windows Sound'
                        })
                except:
                    pass
        except:
            pass
        
        return devices
    
    def _scan_linux(self):
        devices = []
        try:
            result = subprocess.run(['bluetoothctl', 'devices'], 
                                  capture_output=True, text=True, timeout=10)
            for line in result.stdout.splitlines():
                if line.startswith('Device'):
                    parts = line.split()
                    if len(parts) >= 3:
                        addr = parts[1]
                        name = ' '.join(parts[2:])
                        devices.append({
                            'name': name,
                            'address': addr,
                            'type': self._detect_type(name),
                            'status': 'Доступно',
                            'source': 'bluetoothctl'
                        })
        except:
            pass
        return devices
    
    def _scan_macos(self):
        devices = []
        try:
            result = subprocess.run(['system_profiler', 'SPBluetoothDataType'], 
                                  capture_output=True, text=True, timeout=10)
            # Simple parsing
            current_name = None
            for line in result.stdout.splitlines():
                line = line.strip()
                if ':' in line and not line.startswith('-') and len(line) < 50:
                    if 'Connected' in line or 'Not Connected' in line:
                        continue
                    current_name = line.replace(':', '').strip()
                elif 'Address' in line and current_name:
                    addr = line.split(':')[-1].strip()
                    devices.append({
                        'name': current_name,
                        'address': addr,
                        'type': self._detect_type(current_name),
                        'status': 'Доступно',
                        'source': 'system_profiler'
                    })
                    current_name = None
        except:
            pass
        return devices
    
    def _detect_type(self, name):
        name_lower = name.lower()
        if 'headset' in name_lower or 'handsfree' in name_lower:
            return 'Гарнитура'
        elif 'headphone' in name_lower or 'earphone' in name_lower:
            return 'Наушники'
        elif 'speaker' in name_lower:
            return 'Колонки'
        elif 'buds' in name_lower or 'airpods' in name_lower:
            return 'TWS наушники'
        else:
            return 'Аудиоустройство'


import json
