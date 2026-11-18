# Piper для Bluetooth Audio Monitor

В этой папке установлен локальный синтезатор речи **Piper** — он работает
офлайн и используется приложением по умолчанию.

## Что установлено

- `piper.exe` — официальная Windows-сборка Piper `2023.11.14-2`
  (<https://github.com/rhasspy/piper/releases/tag/2023.11.14-2>);
- `espeak-ng.dll`, `espeak-ng-data`, `onnxruntime.dll` — компоненты синтеза;
- `voices/ru_RU-irina-medium.onnx` (+ `.onnx.json`) — русский женский голос;
- `voices/ru_RU-dmitri-medium.onnx` (+ `.onnx.json`) — русский мужской голос.

Голоса взяты из репозитория
<https://huggingface.co/rhasspy/piper-voices> (раздел `ru/ru_RU`).

## Как добавить свой голос

1. Скачайте модель Piper: файлы `голос.onnx` и `голос.onnx.json`.
2. Положите оба файла в папку `voices`.
3. Перезапустите приложение или откройте «➕ Добавить голос...» — новый голос
   появится в общем списке автоматически.

Можно использовать и модель, обученную на собственном голосе.

## Нестандартное расположение

Приложение ищет Piper в папке `piper` рядом с `main.py`. Путь можно изменить
переменными окружения:

- `BLUETOOTH_MONITOR_PIPER_DIR` — каталог с `piper.exe`;
- `BLUETOOTH_MONITOR_PIPER_EXE` — полный путь к `piper.exe`;
- `BLUETOOTH_MONITOR_PIPER_VOICES` — каталог с моделями `.onnx`.

## Проверка

```powershell
cd C:\Users\user\Desktop\BluetoothAudioMonitor1
python -c "from src.tts_engine import get_tts; t=get_tts(); print([v['name'] for v in t.get_available_voices() if v.get('backend')=='piper'])"
```
