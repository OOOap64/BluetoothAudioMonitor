import os
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QTextEdit, QMessageBox, QComboBox, QFileDialog, QCheckBox,
    QScrollArea, QPlainTextEdit, QLineEdit, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, QSettings, QObject
)
from PyQt6.QtGui import QAction, QKeySequence

from src.bt_scanner import BluetoothAudioScanner
from src.audio_monitor import AudioMonitor
from src.audio_devices import (
    list_all_sources, describe_source, soundcard_available, describe_audio_pair
)
from src.file_analyzer import analyze_audio_file, AUDIO_EXTENSIONS
from src.voice_profiles import VoiceProfileStore
from src.voice_dialog import VoiceImportDialog
from src.session_log import (
    SessionStore, build_session_record, default_session_dir,
    default_recordings_dir,
)
from src.location import cached_location, detect_location, manual_location
from src.read_aloud import ReadAloudService
from src.custom_sound_model import default_model_path
from src.speech_transcriber import default_phrases

try:
    from src.tts_engine import get_tts
except Exception:
    get_tts = lambda: None

try:
    from src.text_corrector import correct_speech_text
except Exception:
    def correct_speech_text(text, enabled=True, add_final_period=True):
        return '' if text is None else text


def _format_duration(seconds):
    try:
        seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return '--'
    if seconds <= 0:
        return '--'
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def _source_key(source):
    if not source:
        return None
    return (source.get('backend'), source.get('id'), source.get('device'))


# Горячие клавиши: (клавиша, название процедуры, метод окна).
# Сгруппированы по смыслу — из этого списка строятся меню и памятка.
HOTKEY_GROUPS = [
    ('Устройства и файлы', [
        ('F2', 'Сканировать Bluetooth-устройства', 'scan_bluetooth'),
        ('Ctrl+R', 'Обновить список аудиоустройств', '_populate_audio_devices'),
        ('Ctrl+O', 'Загрузить аудиофайл…', 'load_audio_file'),
    ]),
    ('Захват и анализ', [
        ('F3', 'Начать захват и анализ', 'start_monitoring'),
        ('F4', 'Остановить захват и анализ', 'stop_monitoring'),
        ('F11', 'Сохранить запись аудио в WAV', 'save_monitoring_audio'),
        ('Ctrl+Shift+C', 'Скопировать характеристики в буфер', 'copy_report'),
    ]),
    ('Озвучивание', [
        ('F5', 'Озвучить распознанный текст', 'read_aloud'),
        ('F7', 'Озвучить последний фрагмент', 'read_last_aloud'),
        ('F8', 'Проверить выбранный голос', 'preview_selected_voice'),
        ('Esc', 'Остановить озвучивание', 'stop_speech'),
    ]),
    ('Текст и голоса', [
        ('F6', 'Исправить распознанный текст', 'fix_recognized_text'),
        ('Ctrl+I', 'Добавить голос…', 'import_voice'),
        ('Ctrl+Shift+D', 'Удалить добавленный голос', 'delete_voice'),
    ]),
    ('Чтение произвольного текста и журнал', [
        ('F9', 'Прочитать текст из поля «Свой текст»', 'read_custom_aloud'),
        ('F10', 'Сохранить результат чтения', 'save_read_result'),
        ('Ctrl+L', 'Открыть журнал чтения', 'show_session_history'),
    ]),
]


class MonitorThread(QThread):
    update_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, source=None, language='ru', log_callback=None,
                 autocorrect=True, model_size=None, phrases=None):
        super().__init__()
        self._source = source
        self._language = language
        self._log_callback = log_callback
        self._autocorrect = bool(autocorrect)
        self._model_size = model_size
        self._phrases = phrases
        self.monitor = None
        self.running = False
        self._stop_requested = False

    def run(self):
        self.monitor = AudioMonitor(
            source=self._source,
            language=self._language,
            log_callback=self._log_callback,
            autocorrect=self._autocorrect,
            model_size=self._model_size,
            phrases=self._phrases,
        )
        if self._stop_requested:
            self.running = False
            return
        self.running = True
        try:
            self.monitor.start()
        except Exception as exc:
            self.running = False
            self.error_signal.emit(str(exc))
            return

        while self.running and self.monitor.running:
            data = self.monitor.get_analysis()
            if data:
                self.update_signal.emit(data)
            self.msleep(50)

        if self.running:
            data = self.monitor.get_analysis()
            if data and data.get('error'):
                self.error_signal.emit(str(data['error']))
        self.running = False

    def stop(self):
        self._stop_requested = True
        self.running = False
        if self.monitor is not None:
            self.monitor.stop()
        self.wait(2000)


class BluetoothScanThread(QThread):
    result_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, scanner):
        super().__init__()
        self.scanner = scanner

    def run(self):
        try:
            self.result_signal.emit(self.scanner.scan_audio_devices())
        except Exception as exc:
            self.error_signal.emit(str(exc))



class FileAnalysisThread(QThread):
    progress_signal = pyqtSignal(int, str)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, path, language='ru', autocorrect=True,
                 model_size=None, phrases=None):
        super().__init__()
        self.path = path
        self.language = language
        self.autocorrect = bool(autocorrect)
        self.model_size = model_size
        self.phrases = phrases

    def run(self):
        try:
            result = analyze_audio_file(
                self.path,
                language=self.language,
                autocorrect=self.autocorrect,
                model_size=self.model_size,
                phrases=self.phrases,
                progress=lambda fraction, message: self.progress_signal.emit(
                    int(round(fraction * 100)), message),
            )
            self.result_signal.emit(result)
        except Exception as exc:
            self.error_signal.emit(str(exc))


class ReadAloudThread(QThread):
    """Синтез, проигрывание и (необязательно) запись микрофона."""

    result_signal = pyqtSignal(dict)

    def __init__(self, tts, text, voice, language='ru', record=False,
                 save_tts=True):
        super().__init__()
        self.tts = tts
        self.text = text
        self.voice = voice
        self.language = language
        self.record = bool(record)
        self.save_tts = bool(save_tts)

    def run(self):
        service = ReadAloudService(self.tts, data_dir=default_session_dir())
        try:
            result = service.read(
                self.text,
                self.voice,
                language=self.language,
                record=self.record,
                save_tts=self.save_tts,
            )
        except Exception as exc:
            result = {
                'status': 'error',
                'error': str(exc),
                'text': self.text,
                'voice': self.voice,
                'language': self.language,
                'duration_seconds': 0.0,
                'tts_audio': None,
                'recording': None,
            }
        self.result_signal.emit(result)


class LocationThread(QThread):
    result_signal = pyqtSignal(dict)

    def run(self):
        try:
            location = detect_location()
        except Exception:
            location = None
        self.result_signal.emit(location or {})


class SessionHistoryDialog(QDialog):
    """Просмотр и экспорт журнала чтения вслух."""

    def __init__(self, parent, store):
        super().__init__(parent)
        self.store = store
        self.setWindowTitle('Журнал чтения вслух')
        self.resize(1180, 620)
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            'Дата', 'Источник', 'Текст', 'Голос', 'Устройства',
            'Язык', 'Координаты', 'Статус аудио', 'Запись', 'Ошибка',
        ])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        self.summary = QLabel('')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        buttons = QHBoxLayout()
        self.export_csv_btn = QPushButton('Экспорт CSV…')
        self.export_csv_btn.clicked.connect(self.export_csv)
        buttons.addWidget(self.export_csv_btn)
        self.export_json_btn = QPushButton('Экспорт JSON…')
        self.export_json_btn.clicked.connect(self.export_json)
        buttons.addWidget(self.export_json_btn)
        self.refresh_btn = QPushButton('Обновить')
        self.refresh_btn.clicked.connect(self.reload)
        buttons.addWidget(self.refresh_btn)
        buttons.addStretch(1)
        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        buttons.addWidget(close_box)
        layout.addLayout(buttons)

        self.reload()

    def reload(self):
        records = list(reversed(self.store.load()))
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            voice = record.get('voice') or {}
            devices = record.get('devices') or {}
            location = record.get('location') or {}
            audio = record.get('audio') or {}
            recording = record.get('recording') or {}
            text = str(record.get('text') or '').replace('\n', ' ')
            if len(text) > 120:
                text = text[:117] + '…'
            coords = ''
            if location.get('lat') is not None:
                coords = f"{location.get('lat')}, {location.get('lon')}"
            device_text = (
                f"мик: {devices.get('microphone') or '—'}; "
                f"дин: {devices.get('speaker') or '—'}")
            values = [
                str(record.get('created_at') or '')[:19].replace('T', ' '),
                record.get('source') or '',
                text,
                voice.get('name') or voice.get('id') or '',
                device_text,
                record.get('language') or '',
                coords,
                audio.get('status') or '',
                recording.get('path') or '—',
                audio.get('error') or '',
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column in (2, 4, 8, 9):
                    item.setToolTip(str(value))
                self.table.setItem(row, column, item)
        for column, width in enumerate(
                (145, 100, 260, 150, 260, 60, 130, 95, 220, 180)):
            self.table.setColumnWidth(column, width)
        self.summary.setText(
            f'Записей: {len(records)} · файл: {self.store.path}')

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Экспорт журнала в CSV', 'read_aloud_sessions.csv',
            'CSV (*.csv)')
        if not path:
            return
        try:
            self.store.export_csv(path)
            QMessageBox.information(self, 'Экспорт', f'Сохранено:\n{path}')
        except Exception as exc:
            QMessageBox.critical(self, 'Экспорт', f'Ошибка:\n{exc}')

    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Экспорт журнала в JSON', 'read_aloud_sessions.json',
            'JSON (*.json)')
        if not path:
            return
        try:
            self.store.export_json(path)
            QMessageBox.information(self, 'Экспорт', f'Сохранено:\n{path}')
        except Exception as exc:
            QMessageBox.critical(self, 'Экспорт', f'Ошибка:\n{exc}')


class _MainThreadDispatcher(QObject):
    """Выполняет callable в главном потоке GUI через queued-сигнал."""

    _execute = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._execute.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def __call__(self, func):
        self._execute.emit(func)

    def _run(self, func):
        func()


class MainWindow(QMainWindow):
    monitor_log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Bluetooth Audio Monitor - Определение и анализ аудио')
        self.setMinimumSize(1100, 820)

        self.bt_scanner = BluetoothAudioScanner()
        self.monitor_thread = None
        self.file_thread = None
        self.scan_thread = None
        self._retired_threads = []
        self._monitor_active = False
        self._file_active = False
        self._last_data = None
        self._last_progress_message = None
        self._auto_read_tts = False
        self._last_spoken_text = ''
        self.tts = get_tts()
        self._tts_dispatcher = _MainThreadDispatcher(self)
        if self.tts is not None:
            # Системные голоса SAPI работают только в главном потоке GUI.
            self.tts.set_main_thread_dispatcher(self._tts_dispatcher)
        self.voice_store = VoiceProfileStore()
        self.settings = QSettings('BluetoothAudioMonitor', 'Desktop')
        self.session_store = SessionStore()
        self._read_thread = None
        self._read_active = False
        self._pending_read_meta = None
        self._last_read_record = None
        self._last_read_saved = False
        self._location_thread = None

        self.monitor_log_signal.connect(self.log)
        self.init_ui()
        self._populate_audio_devices()
        self._build_menu()

        # Сканирование Bluetooth может занимать несколько секунд,
        # поэтому запускаем его после появления окна.
        QTimer.singleShot(100, self.scan_bluetooth)

    def _build_menu(self):
        """Меню «Процедуры»: те же действия, что и кнопки, с клавиатуры."""
        menu_bar = self.menuBar()

        procedures_menu = menu_bar.addMenu('&Процедуры')
        for group_index, (_group_title, items) in enumerate(HOTKEY_GROUPS):
            if group_index:
                procedures_menu.addSeparator()
            for key, title, handler_name in items:
                procedures_menu.addAction(
                    self._make_action(title, key, handler_name))

        help_menu = menu_bar.addMenu('&Справка')
        help_menu.addAction(
            self._make_action('Горячие клавиши', 'F1', 'show_hotkeys_help'))

    def _make_action(self, title, shortcut, handler_name):
        """Создаёт действие с горячей клавишей для меню и окна."""
        action = QAction(title, self)
        action.setShortcut(QKeySequence(shortcut))
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.setStatusTip(f'{title} ({shortcut})')
        action.triggered.connect(getattr(self, handler_name))
        return action

    def show_hotkeys_help(self):
        """Показывает памятку по горячим клавишам."""
        rows = []
        for group_title, items in HOTKEY_GROUPS:
            rows.append(
                "<tr><td colspan='2' style='padding-top:10px'>"
                f"<b>{group_title}</b></td></tr>")
            for key, title, _handler_name in items:
                rows.append(
                    "<tr><td style='padding-right:18px'>"
                    f"<b>{key}</b></td><td>{title}</td></tr>")
        rows.append(
            "<tr><td colspan='2' style='padding-top:10px'>"
            "<b>Справка</b></td></tr>"
            "<tr><td style='padding-right:18px'><b>F1</b></td>"
            "<td>Показать эту памятку</td></tr>")
        QMessageBox.information(
            self, 'Горячие клавиши',
            '<p>Процедуры можно запускать с клавиатуры:</p>'
            f"<table cellspacing='0' cellpadding='3'>{''.join(rows)}</table>")

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ---------------------------- Left panel ----------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(430)

        bt_group = QGroupBox('1. Bluetooth-устройства')
        bt_layout = QVBoxLayout(bt_group)

        self.scan_btn = QPushButton('Сканировать')
        self.scan_btn.setToolTip('Найти Bluetooth-аудиоустройства (F2).')
        self.scan_btn.clicked.connect(self.scan_bluetooth)
        bt_layout.addWidget(self.scan_btn)

        self.bt_list = QListWidget()
        self.bt_list.itemClicked.connect(self.on_device_selected)
        bt_layout.addWidget(self.bt_list)

        self.device_info = QLabel('Устройство не выбрано')
        self.device_info.setWordWrap(True)
        bt_layout.addWidget(self.device_info)

        left_layout.addWidget(bt_group)

        mic_group = QGroupBox('2. Захват и распознавание речи')
        mic_layout = QVBoxLayout(mic_group)

        mic_layout.addWidget(QLabel('Источник (loopback или микрофон):'))
        self.device_combo = QComboBox()
        mic_layout.addWidget(self.device_combo)

        self.refresh_btn = QPushButton('Обновить список устройств')
        self.refresh_btn.setToolTip('Обновить список источников звука (Ctrl+R).')
        self.refresh_btn.clicked.connect(self._populate_audio_devices)
        mic_layout.addWidget(self.refresh_btn)

        language_row = QHBoxLayout()
        language_row.addWidget(QLabel('Язык речи:'))
        self.language_combo = QComboBox()
        self.language_combo.addItem('Русский', 'ru')
        self.language_combo.addItem('English', 'en')
        self.language_combo.addItem('Автоопределение', 'auto')
        language_row.addWidget(self.language_combo, 1)
        mic_layout.addLayout(language_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('Модель распознавания:'))
        self.model_combo = QComboBox()
        self.model_combo.addItem('Tiny — минимальная задержка (рекомендуется)', 'tiny')
        self.model_combo.addItem('Base — быстрее, точнее Tiny', 'base')
        self.model_combo.addItem('Small — точнее', 'small')
        self.model_combo.addItem('Medium — максимальная точность', 'medium')
        env_model = os.environ.get('BLUETOOTH_MONITOR_WHISPER_MODEL')
        if env_model and env_model not in ('tiny', 'small', 'base', 'medium'):
            self.model_combo.addItem(
                f'Из настроек: {env_model}', env_model)
        self.model_combo.setToolTip(
            'Tiny даёт результат примерно за 0.3 с и укладывается в задержку '
            'до 1 с. Base точнее, Small и Medium ещё точнее, но медленнее '
            '(Live-распознавание может занимать 2 с и больше). '
            'Модель скачивается при первом использовании.')
        saved_model = str(self.settings.value('asr/model', '') or '')
        default_model = (
            saved_model or
            os.environ.get('BLUETOOTH_MONITOR_WHISPER_MODEL') or
            'tiny')
        saved_index = self.model_combo.findData(default_model)
        if saved_index >= 0:
            self.model_combo.setCurrentIndex(saved_index)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo, 1)
        mic_layout.addLayout(model_row)

        self.autocorrect_cb = QCheckBox(
            'Автокоррекция речи (исправлять ошибки распознавания)')
        self.autocorrect_cb.setChecked(True)
        self.autocorrect_cb.setToolTip(
            'Исправляет типичные ошибки распознавания, пробелы, '
            'пунктуацию и регистр в распознанном тексте.')
        mic_layout.addWidget(self.autocorrect_cb)

        mic_layout.addWidget(QLabel(
            'Частые фразы — подсказки распознавания (одна на строку):'))
        self.phrases_edit = QPlainTextEdit()
        self.phrases_edit.setMaximumHeight(72)
        self.phrases_edit.setPlaceholderText(
            'Например:\nОткрой окно и включи свет\nПривет, меня зовут')
        self.phrases_edit.setToolTip(
            'Эти фразы передаются Whisper как подсказки. Они повышают '
            'точность конкретных команд без увеличения задержки.')
        stored_phrases = self.settings.value('asr/phrases', '')
        if not stored_phrases:
            stored_phrases = '\n'.join(default_phrases())
        self.phrases_edit.setPlainText(str(stored_phrases))
        self._phrases_save_timer = QTimer(self)
        self._phrases_save_timer.setSingleShot(True)
        self._phrases_save_timer.setInterval(300)
        self._phrases_save_timer.timeout.connect(self._save_phrases)
        self.phrases_edit.textChanged.connect(
            lambda: self._phrases_save_timer.start())
        mic_layout.addWidget(self.phrases_edit)

        self.start_btn = QPushButton('Начать захват и анализ')
        self.start_btn.setToolTip('Запустить захват и анализ аудио (F3).')
        self.start_btn.clicked.connect(self.start_monitoring)
        mic_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton('Остановить')
        self.stop_btn.setToolTip('Остановить захват и анализ (F4).')
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        mic_layout.addWidget(self.stop_btn)

        self.save_audio_btn = QPushButton('💾 Сохранить запись аудио (F11)')
        self.save_audio_btn.setToolTip(
            'Сохранить проанализированное аудио в WAV-файл (F11). '
            'Доступно во время захвата.')
        self.save_audio_btn.clicked.connect(self.save_monitoring_audio)
        self.save_audio_btn.setEnabled(False)
        mic_layout.addWidget(self.save_audio_btn)

        self.load_file_btn = QPushButton('Загрузить аудиофайл...')
        self.load_file_btn.setToolTip('Открыть аудиофайл для анализа (Ctrl+O).')
        self.load_file_btn.clicked.connect(self.load_audio_file)
        mic_layout.addWidget(self.load_file_btn)

        self.mic_status = QLabel('Статус: Остановлен')
        self.mic_status.setWordWrap(True)
        mic_layout.addWidget(self.mic_status)

        if not soundcard_available():
            hint = QLabel(
                'Для захвата системного звука установите пакет soundcard:\n'
                'pip install soundcard')
            hint.setWordWrap(True)
            mic_layout.addWidget(hint)

        left_layout.addWidget(mic_group)

        tts_group = QGroupBox('3. Озвучивание речи (TTS)')
        tts_layout = QVBoxLayout(tts_group)

        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel('Голос:'))
        self.tts_voice_combo = QComboBox()
        self.tts_voice_combo.currentIndexChanged.connect(self._on_voice_changed)
        voice_row.addWidget(self.tts_voice_combo, 1)
        self.test_voice_btn = QPushButton('▶ Проверить')
        self.test_voice_btn.setToolTip(
            'Произнести пример выбранным голосом.')
        self.test_voice_btn.clicked.connect(self.preview_selected_voice)
        voice_row.addWidget(self.test_voice_btn)
        tts_layout.addLayout(voice_row)

        self.voice_status = QLabel('')
        self.voice_status.setWordWrap(True)
        tts_layout.addWidget(self.voice_status)

        actions_row = QHBoxLayout()
        self.read_all_btn = QPushButton('🔊 Озвучить текст')
        self.read_all_btn.setToolTip(
            'Прочитать весь распознанный текст (F5).')
        self.read_all_btn.clicked.connect(self.read_aloud)
        actions_row.addWidget(self.read_all_btn)
        self.stop_tts_btn = QPushButton('⏹ Стоп')
        self.stop_tts_btn.setToolTip('Остановить озвучивание (Esc).')
        self.stop_tts_btn.clicked.connect(self.stop_speech)
        actions_row.addWidget(self.stop_tts_btn)
        tts_layout.addLayout(actions_row)

        secondary_row = QHBoxLayout()
        self.read_last_btn = QPushButton('Последний фрагмент')
        self.read_last_btn.setToolTip(
            'Прочитать последнюю строку распознанного текста.')
        self.read_last_btn.clicked.connect(self.read_last_aloud)
        secondary_row.addWidget(self.read_last_btn)
        self.fix_text_btn = QPushButton('✍ Исправить текст')
        self.fix_text_btn.setToolTip(
            'Применить автокоррекцию к показанному тексту (F6).')
        self.fix_text_btn.clicked.connect(self.fix_recognized_text)
        secondary_row.addWidget(self.fix_text_btn)
        tts_layout.addLayout(secondary_row)

        self.auto_read_cb = QCheckBox('Авто-чтение новых распознаваний')
        self.auto_read_cb.stateChanged.connect(self.toggle_auto_read)
        tts_layout.addWidget(self.auto_read_cb)

        voice_buttons = QHBoxLayout()
        self.import_voice_btn = QPushButton('➕ Добавить голос...')
        self.import_voice_btn.setToolTip(
            'Добавить системный голос, голос Edge TTS или локальный '
            'голос Piper (Ctrl+I).')
        self.import_voice_btn.clicked.connect(self.import_voice)
        voice_buttons.addWidget(self.import_voice_btn)
        self.delete_voice_btn = QPushButton('Удалить голос')
        self.delete_voice_btn.setToolTip(
            'Удалить только добавленный пользователем голос '
            '(Ctrl+Shift+D).')
        self.delete_voice_btn.clicked.connect(self.delete_voice)
        voice_buttons.addWidget(self.delete_voice_btn)
        tts_layout.addLayout(voice_buttons)

        self._populate_tts_voices()

        left_layout.addWidget(tts_group)

        # --------------- 4. Произвольный текст и журнал ---------------
        read_group = QGroupBox('4. Свой текст: чтение и сохранение результата')
        read_layout = QVBoxLayout(read_group)

        read_layout.addWidget(QLabel(
            'Введите любой текст и нажмите «Прочитать» (F9).'))
        self.custom_text_edit = QPlainTextEdit()
        self.custom_text_edit.setPlaceholderText(
            'Например: «Внимание! Обнаружен крик и пение птиц. '
            'Проверьте координаты и запись».')
        self.custom_text_edit.setMinimumHeight(90)
        read_layout.addWidget(self.custom_text_edit)

        custom_buttons = QHBoxLayout()
        self.read_custom_btn = QPushButton('🔊 Прочитать (F9)')
        self.read_custom_btn.setToolTip(
            'Озвучить текст из поля выше и сохранить результат (F9).')
        self.read_custom_btn.clicked.connect(self.read_custom_aloud)
        custom_buttons.addWidget(self.read_custom_btn)
        self.save_read_btn = QPushButton('💾 Сохранить (F10)')
        self.save_read_btn.setToolTip(
            'Сохранить последний результат чтения в журнал (F10).')
        self.save_read_btn.clicked.connect(self.save_read_result)
        custom_buttons.addWidget(self.save_read_btn)
        read_layout.addLayout(custom_buttons)

        journal_row = QHBoxLayout()
        self.history_btn = QPushButton('📂 Журнал (Ctrl+L)')
        self.history_btn.setToolTip('Открыть журнал сеансов чтения (Ctrl+L).')
        self.history_btn.clicked.connect(self.show_session_history)
        journal_row.addWidget(self.history_btn)
        self.auto_save_cb = QCheckBox('Автосохранение')
        self.auto_save_cb.setChecked(True)
        self.auto_save_cb.setToolTip(
            'Автоматически записывать каждый сеанс в журнал.')
        journal_row.addWidget(self.auto_save_cb)
        read_layout.addLayout(journal_row)

        self.record_while_read_cb = QCheckBox(
            'Записывать микрофон во время чтения')
        self.record_while_read_cb.setToolTip(
            'Сохранить WAV-запись микрофона вместе с результатом.')
        read_layout.addWidget(self.record_while_read_cb)
        self.save_tts_audio_cb = QCheckBox('Сохранять аудио TTS')
        self.save_tts_audio_cb.setChecked(True)
        self.save_tts_audio_cb.setToolTip(
            'Копировать синтезированный WAV/MP3 в папку журнала.')
        read_layout.addWidget(self.save_tts_audio_cb)

        location_row = QHBoxLayout()
        location_row.addWidget(QLabel('Широта:'))
        self.latitude_edit = QLineEdit()
        self.latitude_edit.setPlaceholderText('55.7558')
        self.latitude_edit.setMaximumWidth(100)
        location_row.addWidget(self.latitude_edit)
        location_row.addWidget(QLabel('Долгота:'))
        self.longitude_edit = QLineEdit()
        self.longitude_edit.setPlaceholderText('37.6173')
        self.longitude_edit.setMaximumWidth(100)
        location_row.addWidget(self.longitude_edit)
        self.location_btn = QPushButton('📍 Определить')
        self.location_btn.setToolTip(
            'Определить координаты по IP (нужен интернет). '
            'Координаты можно ввести вручную.')
        self.location_btn.clicked.connect(self.detect_location_clicked)
        location_row.addWidget(self.location_btn)
        read_layout.addLayout(location_row)

        cached = cached_location()
        if cached:
            self.latitude_edit.setText(str(cached.get('lat', '')))
            self.longitude_edit.setText(str(cached.get('lon', '')))

        left_layout.addWidget(read_group)

        self.status_label = QLabel('Готово')
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        hint_row = QHBoxLayout()
        self.hotkeys_hint = QLabel(
            'F2 — скан · F3 — захват · F4 — стоп · F11 — сохранить запись · '
            'F5 — озвучить · F9 — свой текст · F10 — сохранить')
        self.hotkeys_hint.setWordWrap(True)
        self.hotkeys_hint.setStyleSheet(
            'color: palette(mid); font-size: 11px;')
        hint_row.addWidget(self.hotkeys_hint, 1)
        self.hotkeys_btn = QPushButton('⌨ F1')
        self.hotkeys_btn.setToolTip('Показать все горячие клавиши (F1).')
        self.hotkeys_btn.setFixedWidth(70)
        self.hotkeys_btn.clicked.connect(self.show_hotkeys_help)
        hint_row.addWidget(self.hotkeys_btn)
        left_layout.addLayout(hint_row)

        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(445)
        left_scroll.setMaximumWidth(465)
        left_scroll.setWidget(left_panel)

        main_layout.addWidget(left_scroll)

        # ---------------------------- Right panel ---------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        char_group = QGroupBox('Характеристики аудио')
        char_layout = QGridLayout(char_group)

        self.sample_rate_label = QLabel('--')
        self.channels_label = QLabel('--')
        self.format_label = QLabel('--')
        self.duration_label = QLabel('--')
        self.rms_label = QLabel('--')
        self.peak_label = QLabel('--')
        self.amp_label = QLabel('--')
        self.db_label = QLabel('--')
        self.centroid_label = QLabel('--')
        self.dominant_label = QLabel('--')
        self.crest_label = QLabel('--')

        rows = [
            ('Частота дискретизации:', self.sample_rate_label),
            ('Каналы:', self.channels_label),
            ('Формат:', self.format_label),
            ('Длительность:', self.duration_label),
            ('RMS (уровень громкости):', self.rms_label),
            ('Пиковый уровень:', self.peak_label),
            ('Средняя амплитуда:', self.amp_label),
            ('Уровень сигнала (dBFS):', self.db_label),
            ('Спектральный центр:', self.centroid_label),
            ('Доминирующая частота:', self.dominant_label),
            ('Пик-фактор (crest factor):', self.crest_label),
        ]
        for row, (title, value_label) in enumerate(rows):
            char_layout.addWidget(QLabel(title), row, 0)
            value_label.setWordWrap(True)
            char_layout.addWidget(value_label, row, 1)

        self.copy_report_btn = QPushButton('Скопировать характеристики в текст')
        self.copy_report_btn.setToolTip(
            'Скопировать характеристики и текст в буфер обмена '
            '(Ctrl+Shift+C).')
        self.copy_report_btn.clicked.connect(self.copy_report)
        char_layout.addWidget(self.copy_report_btn, len(rows), 0, 1, 2)

        right_layout.addWidget(char_group)

        level_group = QGroupBox('Уровень сигнала')
        level_layout = QVBoxLayout(level_group)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setFormat('Уровень: %p%')
        level_layout.addWidget(self.level_bar)
        right_layout.addWidget(level_group)

        freq_group = QGroupBox('Частотный анализ')
        freq_layout = QVBoxLayout(freq_group)
        self.freq_table = QTableWidget(5, 2)
        self.freq_table.setHorizontalHeaderLabels(['Диапазон', 'Доля энергии'])
        self.freq_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        freq_ranges = [
            'Суб-бас (20-60 Гц)',
            'Бас (60-250 Гц)',
            'Середина (250-2000 Гц)',
            'Верхняя середина (2-4 кГц)',
            'Высокие (4-20 кГц)',
        ]
        for i, freq_range in enumerate(freq_ranges):
            self.freq_table.setItem(i, 0, QTableWidgetItem(freq_range))
            self.freq_table.setItem(i, 1, QTableWidgetItem('--'))
        freq_layout.addWidget(self.freq_table)
        right_layout.addWidget(freq_group)

        src_group = QGroupBox('Определение источника звука (ИИ)')
        src_layout = QVBoxLayout(src_group)
        self.source_text = QTextEdit()
        self.source_text.setReadOnly(True)
        self.source_text.setPlainText('Ожидание анализа...')
        self.source_text.setMaximumHeight(110)
        src_layout.addWidget(self.source_text)

        self.sound_model_label = QLabel(self._sound_model_status_text())
        self.sound_model_label.setWordWrap(True)
        src_layout.addWidget(self.sound_model_label)
        right_layout.addWidget(src_group)

        speech_group = QGroupBox('Распознанный текст из аудио (ИИ)')
        speech_layout = QVBoxLayout(speech_group)
        self.speech_text = QTextEdit()
        self.speech_text.setReadOnly(True)
        self.speech_text.setPlainText('Ожидание речи...')
        self.speech_text.setMinimumHeight(130)
        speech_layout.addWidget(self.speech_text)
        right_layout.addWidget(speech_group)

        log_group = QGroupBox('Журнал событий')
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(110)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)

        main_layout.addWidget(right_panel, 1)

    # ------------------------------------------------------------- devices
    def _populate_audio_devices(self):
        previous = _source_key(self.device_combo.currentData())
        self.device_combo.clear()
        self.device_combo.addItem(
            'Авто: системный звук (loopback), иначе микрофон', None)

        loopback_count = 0
        input_count = 0
        try:
            sources = list_all_sources()
        except Exception as exc:
            sources = []
            self.log(f'Ошибка получения списка устройств: {exc}')

        selected_index = 0
        for source in sources:
            if source.get('is_loopback'):
                loopback_count += 1
                label = (f"[WASAPI LOOPBACK] {source['name']} "
                         f"({source['channels']} кан.)")
            else:
                input_count += 1
                label = f"[ВХОД] {source['name']} ({source['channels']} кан.)"
                if source.get('hostapi'):
                    label += f" · {source['hostapi']}"
            if source.get('is_default'):
                label += ' — по умолчанию'
            self.device_combo.addItem(label, source)
            if previous and _source_key(source) == previous:
                selected_index = self.device_combo.count() - 1

        self.device_combo.setCurrentIndex(selected_index)
        self.log(
            f'Источников аудио: loopback — {loopback_count}, '
            f'устройств ввода — {input_count}')
        if not sources:
            self.log(
                'Устройства не найдены. Проверьте подключение микрофона '
                'или установите soundcard для системного звука.')

    def _on_model_changed(self, _index):
        self.settings.setValue(
            'asr/model', self.model_combo.currentData() or 'tiny')

    def _current_phrases(self):
        return str(self.phrases_edit.toPlainText() or '')

    def _save_phrases(self):
        self.settings.setValue('asr/phrases', self._current_phrases())
        self.settings.sync()

    def _refresh_buttons(self):
        idle = not self._monitor_active and not self._file_active

        self.start_btn.setEnabled(idle)
        self.stop_btn.setEnabled(self._monitor_active)
        self.save_audio_btn.setEnabled(self._monitor_active)
        self.device_combo.setEnabled(idle)
        self.language_combo.setEnabled(idle)
        self.model_combo.setEnabled(idle)
        self.autocorrect_cb.setEnabled(idle)
        self.phrases_edit.setEnabled(idle)
        self.refresh_btn.setEnabled(idle)
        self.load_file_btn.setEnabled(idle)
        self.read_custom_btn.setEnabled(not self._read_active)
        self.save_read_btn.setEnabled(not self._read_active)
        self.location_btn.setEnabled(
            not self._read_active and
            not (self._location_thread and self._location_thread.isRunning()))

    def log(self, message):
        if not hasattr(self, 'log_text'):
            print(message)
            return
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f'[{timestamp}] {message}')

    # ---------------------------------------------------------- bluetooth
    def scan_bluetooth(self):
        if self.scan_thread and self.scan_thread.isRunning():
            return
        self.scan_btn.setEnabled(False)
        self.bt_list.clear()
        self.log('Сканирование Bluetooth аудио устройств...')
        self.status_label.setText('Сканирование...')

        self.scan_thread = BluetoothScanThread(self.bt_scanner)
        self.scan_thread.result_signal.connect(self._on_bt_result)
        self.scan_thread.error_signal.connect(self._on_bt_error)
        self.scan_thread.finished.connect(self._on_bt_finished)
        self.scan_thread.start()

    def _on_bt_result(self, devices):
        if not devices:
            self.log('Bluetooth аудио устройства не найдены')
            self.status_label.setText('Устройства не найдены')
            return

        for device in devices:
            item = QListWidgetItem(f"{device['name']} ({device['type']})")
            item.setData(Qt.ItemDataRole.UserRole, device)
            self.bt_list.addItem(item)
        self.log(f'Найдено Bluetooth устройств: {len(devices)}')
        self.status_label.setText(f'Найдено: {len(devices)}')

    def _on_bt_error(self, message):
        self.log(f'Ошибка сканирования Bluetooth: {message}')
        self.status_label.setText('Ошибка сканирования')

    def _on_bt_finished(self):
        thread = self.sender()
        self._retain_thread(thread)
        if thread is self.scan_thread:
            self.scan_thread = None
        self.scan_btn.setEnabled(True)

    def on_device_selected(self, item):
        device = item.data(Qt.ItemDataRole.UserRole)
        info = (f"Имя: {device['name']}\nАдрес: {device['address']}\n"
                f"Тип: {device['type']}\nСтатус: {device['status']}")
        self.device_info.setText(info)
        self.log(f"Выбрано устройство: {device['name']}")

    # -------------------------------------------------------------- live
    def start_monitoring(self):
        if self._file_active:
            self.status_label.setText('Идёт анализ файла')
            self.log('Дождитесь завершения анализа файла')
            return
        if self._monitor_active:
            return

        source = self.device_combo.currentData()
        language = self.language_combo.currentData() or 'ru'
        autocorrect = self.autocorrect_cb.isChecked()
        model_size = self.model_combo.currentData() or 'tiny'
        phrases = self._current_phrases()
        self.settings.setValue('asr/model', model_size)
        self.settings.setValue('asr/phrases', phrases)

        self.monitor_thread = MonitorThread(
            source=source,
            language=language,
            log_callback=self.monitor_log_signal.emit,
            autocorrect=autocorrect,
            model_size=model_size,
            phrases=phrases,
        )
        self.monitor_thread.update_signal.connect(self.update_analysis)
        self.monitor_thread.error_signal.connect(self._on_monitor_error)
        self.monitor_thread.finished.connect(self._on_monitor_finished)
        self.monitor_thread.start()

        self._monitor_active = True
        self._last_spoken_text = ''
        self._refresh_buttons()
        self.mic_status.setText('Статус: захват и анализ активен')
        self.status_label.setText('Мониторинг активен')
        self.log(f'Запуск захвата: {describe_source(source)}')
        self.log(f'Модель распознавания: {model_size}')
        self.log(
            'Автокоррекция речи: '
            + ('включена' if autocorrect else 'выключена'))
        phrase_count = len([
            line for line in phrases.splitlines() if line.strip()])
        self.log(f'Подсказки распознавания: {phrase_count} фраз')

    def stop_monitoring(self):
        # Горячая клавиша F4 не должна очищать уже показанные результаты,
        # если захват и так не запущен.
        if not self._monitor_active and self.monitor_thread is None:
            self.status_label.setText('Захват не запущен')
            self.log('Захват уже остановлен')
            return

        thread = self.monitor_thread
        self.monitor_thread = None
        self._monitor_active = False
        if thread is not None:
            thread.stop()
            self._retain_thread(thread)

        self._refresh_buttons()
        self.mic_status.setText('Статус: Остановлен')
        self.status_label.setText('Остановлен')
        self.log('Анализ остановлен')
        self._clear_displays()

    def _retain_thread(self, thread):
        # Держим ссылку на завершённый QThread до закрытия окна. Если Python
        # уничтожит объект, пока Qt ещё доставляет его сигналы, приложение
        # может аварийно завершиться (0xC0000409).
        if thread is None:
            return
        if thread not in self._retired_threads:
            self._retired_threads.append(thread)

    def _on_monitor_error(self, message):
        self.log(f'Ошибка аудиопотока: {message}')
        thread = self.monitor_thread
        self.monitor_thread = None
        self._monitor_active = False
        if thread is not None:
            thread.stop()
            self._retain_thread(thread)

        self._refresh_buttons()
        self.mic_status.setText('Статус: ошибка захвата')
        self.status_label.setText('Ошибка захвата')
        self.source_text.setPlainText(f'Ошибка: {message}')
        QMessageBox.warning(
            self, 'Ошибка аудиопотока',
            f'{message}\n\nПопробуйте выбрать другое устройство '
            'или нажать «Обновить список устройств».')

    def _on_monitor_finished(self):
        thread = self.sender()
        self._retain_thread(thread)
        if thread is self.monitor_thread and not thread.isRunning():
            self.monitor_thread = None
            self._monitor_active = False
            self._refresh_buttons()
            self.mic_status.setText('Статус: Остановлен')

    def save_monitoring_audio(self):
        """Сохраняет проанализированное аудио текущего захвата в WAV."""
        thread = self.monitor_thread
        monitor = thread.monitor if thread is not None else None
        if not self._monitor_active or monitor is None:
            self.log('Сохранение записи доступно во время захвата и анализа (F3)')
            self.status_label.setText('Захват не запущен')
            return

        default_dir = default_recordings_dir()
        default_path = default_dir / (
            f"capture-{datetime.now().strftime('%Y%m%d-%H%M%S')}.wav")
        path, _ = QFileDialog.getSaveFileName(
            self, 'Сохранить запись аудио', str(default_path), 'WAV (*.wav)')
        if not path:
            return
        if not path.lower().endswith('.wav'):
            path += '.wav'

        info = monitor.save_recording(path)
        if info.get('status') == 'ok':
            self.log(
                f"Запись сохранена: {info.get('path')} "
                f"({float(info.get('duration_seconds') or 0.0):.1f} с, "
                f"{int(info.get('size_bytes') or 0)} байт)")
            self.status_label.setText('Запись сохранена')
        else:
            message = info.get('error') or info.get('status') or 'неизвестная ошибка'
            self.log(f'Не удалось сохранить запись: {message}')
            self.status_label.setText('Запись не сохранена')
            QMessageBox.warning(
                self, 'Сохранение записи',
                f'Не удалось сохранить запись:\n{message}')

    # -------------------------------------------------------------- file
    def load_audio_file(self):
        if self._file_active:
            return
        if self._monitor_active:
            self.stop_monitoring()
            self.log('Мониторинг остановлен перед анализом файла')

        patterns = ' '.join(AUDIO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Выберите аудиозапись',
            '',
            f'Аудиофайлы ({patterns});;Все файлы (*.*)',
        )
        if not path:
            return

        self._last_progress_message = None
        self._file_active = True
        autocorrect = self.autocorrect_cb.isChecked()
        model_size = self.model_combo.currentData() or 'tiny'
        phrases = self._current_phrases()
        self.settings.setValue('asr/model', model_size)
        self.settings.setValue('asr/phrases', phrases)
        self.file_thread = FileAnalysisThread(
            path,
            language=self.language_combo.currentData() or 'ru',
            autocorrect=autocorrect,
            model_size=model_size,
            phrases=phrases,
        )
        self.file_thread.progress_signal.connect(self._on_file_progress)
        self.file_thread.result_signal.connect(self._on_file_result)
        self.file_thread.error_signal.connect(self._on_file_error)
        self.file_thread.finished.connect(self._on_file_finished)
        self.file_thread.start()

        self._refresh_buttons()
        self.status_label.setText('Анализ аудиофайла...')
        self.log(f'Анализ аудиофайла: {os.path.basename(path)}')
        self.log(f'Модель распознавания: {model_size}')
        self.log(
            'Автокоррекция речи: '
            + ('включена' if autocorrect else 'выключена'))

    def _on_file_progress(self, percent, message):
        self.status_label.setText(f'{message} ({percent}%)')
        if message != self._last_progress_message:
            self._last_progress_message = message
            self.log(message)

    def _on_file_result(self, result):
        self._last_data = result
        self._apply_characteristics(result)
        self._apply_classifications(result.get('classifications', []))

        text = result.get('transcribed_text') or ''
        if text:
            self.speech_text.setPlainText(text)
            self._maybe_auto_read(text)
        elif result.get('transcription_error'):
            self.speech_text.setPlainText(
                'Речь распознать не удалось: '
                + str(result['transcription_error']))
            self.log('Ошибка распознавания: ' + str(result['transcription_error']))
        else:
            self.speech_text.setPlainText(
                'Речь в файле не обнаружена (или она не распознана).')

        duration = _format_duration(result.get('duration', 0))
        self.log(
            f"Файл обработан: {result.get('file_name', '')}, "
            f'длительность {duration}, распознано символов: {len(text)}')
        if result.get('classification_error'):
            self.log('Классификация недоступна: '
                     + str(result['classification_error']))
        self.status_label.setText('Файл проанализирован')
        self.mic_status.setText('Статус: файл проанализирован')

    def _on_file_error(self, message):
        self.log(f'Ошибка анализа файла: {message}')
        self.status_label.setText('Ошибка анализа файла')
        self.speech_text.setPlainText(f'Ошибка анализа файла: {message}')
        QMessageBox.critical(self, 'Ошибка анализа файла', message)

    def _on_file_finished(self):
        thread = self.sender()
        self._retain_thread(thread)
        if thread is self.file_thread:
            self.file_thread = None
        self._file_active = False
        self._refresh_buttons()

    # ----------------------------------------------------------- display
    @staticmethod
    def _set_text(widget, text):
        if widget.text() != text:
            widget.setText(text)

    @staticmethod
    def _set_plain_text(widget, text):
        if widget.toPlainText() != text:
            widget.setPlainText(text)

    def _apply_characteristics(self, data):
        sample_rate = data.get('sample_rate') or 0
        channels = data.get('channels') or 0
        duration = data.get('duration') or 0

        self._set_text(
            self.sample_rate_label,
            f'{sample_rate} Гц' if sample_rate else '--')
        self._set_text(
            self.channels_label, str(channels) if channels else '--')
        self._set_text(
            self.format_label, str(data.get('format') or '--'))
        self._set_text(
            self.duration_label, _format_duration(duration))
        self._set_text(
            self.rms_label, f"{float(data.get('rms', 0.0)):.6f}")
        self._set_text(
            self.peak_label, f"{float(data.get('peak', 0.0)):.6f}")
        self._set_text(
            self.amp_label, f"{float(data.get('amplitude', 0.0)):.6f}")
        self._set_text(
            self.db_label, f"{float(data.get('db_level', -100.0)):.1f} dBFS")
        self._set_text(
            self.centroid_label,
            f"{float(data.get('spectral_centroid', 0.0)):.1f} Гц")
        self._set_text(
            self.dominant_label,
            f"{float(data.get('dominant_frequency', 0.0)):.1f} Гц")
        self._set_text(
            self.crest_label, f"{float(data.get('crest_factor', 0.0)):.2f}")

        self.level_bar.setValue(int(data.get('level_percent', 0)))

        for index, level in enumerate(data.get('frequency_bands', [])[:5]):
            text = f'{float(level):.1f}%'
            item = self.freq_table.item(index, 1)
            if item is None or item.text() != text:
                self.freq_table.setItem(index, 1, QTableWidgetItem(text))

    def _apply_classifications(self, classifications):
        if classifications:
            lines = []
            for item in classifications:
                label = str(item.get('label', '')).replace('_', ' ')
                score = float(item.get('score', 0.0)) * 100.0
                lines.append(f'{label}: {score:.1f}%')
            self._set_plain_text(self.source_text, '\n'.join(lines))
        else:
            self._set_plain_text(self.source_text, 'Анализ звука...')

    def update_analysis(self, data):
        self._last_data = data
        if data.get('error'):
            self._set_plain_text(
                self.source_text, f"Ошибка: {data['error']}")
            return

        self._apply_characteristics(data)
        self._apply_classifications(data.get('classifications', []))

        transcribed_text = data.get('transcribed_text', '')
        live_text = data.get('live_text', '')
        displayed_text = ' '.join(
            part for part in (transcribed_text, live_text) if part).strip()
        if displayed_text:
            if displayed_text != self.speech_text.toPlainText():
                self._set_plain_text(self.speech_text, displayed_text)
                self._maybe_auto_read(transcribed_text)
        elif (not self.speech_text.toPlainText() or
              self.speech_text.toPlainText() == 'Ожидание речи...'):
            if data.get('level_percent', 0) > 2:
                self._set_plain_text(
                    self.speech_text, 'Речь обнаружена, распознавание...')
            else:
                self._set_plain_text(self.speech_text, 'Ожидание речи...')

    def _clear_displays(self):
        for label in (
            self.sample_rate_label, self.channels_label, self.format_label,
            self.duration_label, self.rms_label, self.peak_label,
            self.amp_label, self.db_label, self.centroid_label,
            self.dominant_label, self.crest_label,
        ):
            label.setText('--')
        self.level_bar.setValue(0)
        for index in range(5):
            self.freq_table.setItem(index, 1, QTableWidgetItem('--'))
        self.source_text.setPlainText('Ожидание анализа...')
        self.speech_text.setPlainText('Ожидание речи...')
        self._last_spoken_text = ''
        self._last_data = None

    def copy_report(self):
        data = self._last_data
        if not data:
            self.log('Пока нет данных для отчёта')
            return

        def number(key, fmt='{:.6f}'):
            return fmt.format(float(data.get(key, 0.0)))

        bands = data.get('frequency_bands', [])
        band_names = [
            'суб-бас 20-60 Гц',
            'бас 60-250 Гц',
            'середина 250-2000 Гц',
            'верхняя середина 2-4 кГц',
            'высокие 4-20 кГц',
        ]
        lines = [
            'Характеристики аудио',
            f"Источник: {data.get('file_name') or 'поток'}",
            f"Частота дискретизации: {data.get('sample_rate', 0)} Гц",
            f"Каналы: {data.get('channels', 0)}",
            f"Формат: {data.get('format', '--')}",
            f"Длительность: {_format_duration(data.get('duration', 0))}",
            f"RMS: {number('rms')}",
            f"Пиковый уровень: {number('peak')}",
            f"Средняя амплитуда: {number('amplitude')}",
            f"Уровень сигнала: {number('db_level', '{:.1f}')} dBFS",
            f"Спектральный центр: {number('spectral_centroid', '{:.1f}')} Гц",
            f"Доминирующая частота: {number('dominant_frequency', '{:.1f}')} Гц",
            f"Пик-фактор: {number('crest_factor', '{:.2f}')}",
            '',
            'Частотные диапазоны:',
        ]
        for index, level in enumerate(bands[:5]):
            lines.append(f'  {band_names[index]}: {float(level):.1f}%')

        classifications = data.get('classifications', [])
        if classifications:
            lines.append('')
            lines.append('Источник звука (ИИ):')
            for item in classifications:
                label = str(item.get('label', '')).replace('_', ' ')
                lines.append(
                    f"  {label}: {float(item.get('score', 0.0)) * 100.0:.1f}%")

        text = data.get('transcribed_text', '')
        if text:
            lines.append('')
            lines.append('Распознанный текст:')
            lines.append(text)

        QApplication.clipboard().setText('\n'.join(lines))
        self.log('Характеристики скопированы в буфер обмена')

    # --------------------------------------------------------------- TTS
    def _voice_key(self, data):
        if isinstance(data, dict):
            return data.get('id')
        return data

    def _populate_tts_voices(self, select_key=None):
        """Заполняет список встроенными и пользовательскими голосами."""
        saved_key = None
        if select_key is None:
            try:
                select_key = self._voice_key(
                    self.tts_voice_combo.currentData())
            except Exception:
                select_key = None
        if select_key is None:
            try:
                saved_key = self.settings.value('tts/voice_key', None)
            except Exception:
                saved_key = None
            select_key = saved_key

        self.tts_voice_combo.blockSignals(True)
        self.tts_voice_combo.clear()
        try:
            builtins = self.tts.get_available_voices() if self.tts else []
        except Exception:
            builtins = []

        for voice in builtins:
            if not isinstance(voice, dict):
                voice = {'id': str(voice), 'name': str(voice)}
            self.tts_voice_combo.addItem(
                voice.get('name', voice.get('id', 'голос')), voice)

        custom = self.voice_store.profiles
        backend_names = {
            'pyttsx3': 'системный',
            'edge': 'Edge TTS',
            'piper': 'Piper',
        }
        if custom and self.tts_voice_combo.count():
            self.tts_voice_combo.insertSeparator(
                self.tts_voice_combo.count())
        for profile in custom:
            label = backend_names.get(profile.get('backend'), 'свой')
            self.tts_voice_combo.addItem(
                f"★ {profile.get('name', 'голос')} ({label})", profile)

        if not self.tts_voice_combo.count():
            self.tts_voice_combo.addItem('TTS недоступен', None)

        restored = False
        if select_key is not None:
            for index in range(self.tts_voice_combo.count()):
                data = self.tts_voice_combo.itemData(index)
                if self._voice_key(data) == select_key:
                    self.tts_voice_combo.setCurrentIndex(index)
                    restored = True
                    break
        if not restored:
            # Сохранённый голос мог быть удалён. В этом случае, как и при
            # первом запуске, предпочитаем локальный Piper: он работает
            # офлайн и не зависит от интернета/системного TTS.
            for index in range(self.tts_voice_combo.count()):
                data = self.tts_voice_combo.itemData(index)
                if isinstance(data, dict) and data.get('backend') == 'piper':
                    self.tts_voice_combo.setCurrentIndex(index)
                    restored = True
                    break
        if not restored:
            self.tts_voice_combo.setCurrentIndex(0)
        self.tts_voice_combo.blockSignals(False)
        self._on_voice_changed()

    def _on_voice_changed(self, _index=None):
        """Обновляет подпись активного голоса при выборе."""
        data = self.tts_voice_combo.currentData()
        if isinstance(data, dict):
            backend = str(data.get('backend') or '')
            labels = {
                'pyttsx3': 'системный',
                'sapi': 'SAPI',
                'edge': 'Edge TTS, интернет',
                'piper': 'Piper, локальный',
            }
            label = labels.get(backend, backend or 'голос')
            name = data.get('name') or 'голос'
            self.voice_status.setText(
                f'Активный голос: {name} — {label}')
            self.test_voice_btn.setEnabled(True)
            key = self._voice_key(data)
            if key:
                try:
                    self.settings.setValue('tts/voice_key', key)
                except Exception:
                    pass
        else:
            self.voice_status.setText('Голос не выбран')
            self.test_voice_btn.setEnabled(False)

    def _on_tts_done(self, result):
        """Показывает итог фонового озвучивания (вызывается из потока TTS)."""
        if not isinstance(result, dict):
            return
        status = result.get('status')
        if status == 'error':
            self.monitor_log_signal.emit(
                'Ошибка озвучивания: '
                + str(result.get('error') or 'неизвестная ошибка'))

    def preview_selected_voice(self):
        """Произносит пример выбранным голосом (проверка после подбора)."""
        if not self.tts:
            self.log('TTS недоступен')
            QMessageBox.information(
                self, 'TTS',
                'Движок озвучивания недоступен. Установите pyttsx3, '
                'edge-tts или Piper.')
            return
        data = self._get_tts_voice()
        if data is None:
            self.log('Голос не выбран')
            return
        text = self._preview_text_for_voice(data)
        voice = self._voice_for_text(text)
        self.log(
            f'Проверка выбранного голоса: {self._voice_label(voice)}')
        try:
            self.tts.speak_async(
                text, voice, on_done=self._on_tts_done)
        except Exception as exc:
            self.log(f'Ошибка проверки голоса: {exc}')

    def import_voice(self):
        current = self._get_tts_voice()
        dialog = VoiceImportDialog(
            self, self.tts,
            profile=current if isinstance(current, dict) else None)
        if not dialog.exec():
            return
        profile = dialog.result_profile()
        if not profile:
            return
        try:
            saved = self.voice_store.add(profile)
        except Exception as exc:
            self.log(f'Не удалось сохранить голос: {exc}')
            QMessageBox.warning(
                self, 'Добавление голоса',
                f'Не удалось сохранить голос:\n{exc}')
            return
        self._populate_tts_voices(select_key=saved.get('id'))
        self.log(
            f"Голос добавлен: {saved.get('name')} "
            f"({saved.get('backend')})")
        QMessageBox.information(
            self, 'Добавление голоса',
            f"Голос «{saved.get('name')}» добавлен и выбран.\n"
            'Сейчас прозвучит пример.')
        # Сразу проверяем озвучивание новым голосом.
        self.preview_selected_voice()

    def delete_voice(self):
        data = self.tts_voice_combo.currentData()
        if not isinstance(data, dict) or not data.get('id'):
            self.log('Встроенный голос удалить нельзя')
            QMessageBox.information(
                self, 'Удаление голоса',
                'Встроенный голос удалить нельзя. Удалять можно только '
                'голоса, добавленные через «Добавить голос».')
            return
        name = data.get('name', 'голос')
        answer = QMessageBox.question(
            self, 'Удаление голоса',
            f'Удалить голос «{name}» из списка?',
            QMessageBox.StandardButton.Yes |
            QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.voice_store.remove(data['id']):
            self.log(f'Голос удалён: {name}')
            self._populate_tts_voices()
        else:
            self.log('Голос не найден в списке пользовательских')

    def toggle_auto_read(self, state):
        self._auto_read_tts = (state == Qt.CheckState.Checked.value)
        self.log(f'Авто-озвучивание: {"включено" if self._auto_read_tts else "выключено"}')
        if not self._auto_read_tts:
            self._last_spoken_text = ''

    def _get_tts_voice(self):
        try:
            return self.tts_voice_combo.currentData()
        except Exception:
            return None

    @staticmethod
    def _voice_is_english(voice):
        if not isinstance(voice, dict):
            return False
        blob = ' '.join(
            str(voice.get(key) or '') for key in ('id', 'voice', 'name')
        ).lower()
        return any(marker in blob for marker in (
            'en-us', 'en_gb', 'en-gb', 'english',
            'zira', 'aria', 'guy',
        ))

    @staticmethod
    def _text_has_cyrillic(text):
        return any('\u0400' <= char <= '\u04FF' for char in str(text or ''))

    def _russian_voice(self):
        voices = []
        for index in range(self.tts_voice_combo.count()):
            voice = self.tts_voice_combo.itemData(index)
            if isinstance(voice, dict) and not self._voice_is_english(voice):
                voices.append(voice)
        for voice in voices:
            if str(voice.get('backend') or '') == 'piper':
                return voice
        return voices[0] if voices else None

    def _voice_for_text(self, text):
        voice = self._get_tts_voice()
        if self._voice_is_english(voice) and self._text_has_cyrillic(text):
            fallback = self._russian_voice()
            if fallback is not None:
                self.log(
                    f'Голос «{self._voice_label(voice)}» не поддерживает '
                    f'русский текст — используется '
                    f'«{self._voice_label(fallback)}»')
                return fallback
        return voice

    def _preview_text_for_voice(self, voice):
        custom = ''
        try:
            custom = (self.custom_text_edit.toPlainText() or '').strip()
        except Exception:
            pass
        if custom:
            return self._clean_text_for_tts(custom)
        if self._voice_is_english(voice):
            return 'Hello! This is a test of the selected voice.'
        return 'Привет! Это пример выбранного голоса.'

    def read_aloud(self):
        text = (self.speech_text.toPlainText() or '').strip()
        if not text or text in ('Ожидание речи...', 'Речь обнаружена, распознавание...', 'Речь в файле не обнаружена (или она не распознана).'):
            self.log('Нет распознанного текста для озвучивания')
            return
        # Clean timestamps for better reading
        cleaned = self._clean_text_for_tts(text)
        if not self.tts:
            self.log('TTS недоступен (нет Piper, pyttsx3 или edge-tts)')
            QMessageBox.information(
                self, 'TTS',
                'Движок озвучивания недоступен. Установите Piper в папку '
                'piper проекта, pyttsx3 или edge-tts.')
            return
        self.log('Озвучивание распознанного текста...')
        try:
            self.tts.speak_async(
                cleaned, self._voice_for_text(cleaned),
                on_done=self._on_tts_done)
        except Exception as exc:
            self.log(f'Ошибка TTS: {exc}')

    def read_last_aloud(self):
        text = (self.speech_text.toPlainText() or '').strip()
        if not text or text in ('Ожидание речи...', 'Речь обнаружена, распознавание...'):
            self.log('Нет текста для озвучивания')
            return
        # Extract last segment (after last timestamp marker)
        segments = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            segments.append(line)
        if not segments:
            segments = [text]
        last = segments[-1]
        # Remove leading timestamp like [12:34:56]
        if last.startswith('[') and ']' in last:
            last = last.split(']', 1)[1].strip()
        if not last:
            self.log('Нет текста для озвучивания')
            return
        if not self.tts:
            self.log('TTS недоступен')
            return
        self.log('Озвучивание последнего фрагмента...')
        try:
            self.tts.speak_async(
                last, self._voice_for_text(last),
                on_done=self._on_tts_done)
        except Exception as exc:
            self.log(f'Ошибка TTS: {exc}')

    def fix_recognized_text(self):
        text = (self.speech_text.toPlainText() or '').strip()
        if not text or text in (
                'Ожидание речи...',
                'Речь обнаружена, распознавание...',
                'Речь в файле не обнаружена (или она не распознана).'):
            self.log('Нет распознанного текста для исправления')
            return
        corrected = correct_speech_text(text, enabled=True)
        if corrected != text:
            self.speech_text.setPlainText(corrected)
            self._last_spoken_text = corrected
            self.log('Распознанный текст исправлен')
        else:
            self.log('Распознанный текст уже корректен')

    def stop_speech(self):
        if self.tts:
            try:
                self.tts.stop()
                self.log('Озвучивание остановлено')
            except Exception as exc:
                self.log(f'Ошибка остановки TTS: {exc}')
        else:
            self.log('TTS недоступен')

    def _clean_text_for_tts(self, text):
        """Remove timestamps like [HH:MM:SS] for cleaner speech."""
        import re
        cleaned = re.sub(r'\[\d{1,2}:\d{2}:\d{2}\]\s*', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned or text

    # ------------------------------------------- произвольный текст/журнал
    def _sound_model_status_text(self):
        path = default_model_path()
        if os.path.isfile(path):
            created = datetime.fromtimestamp(
                os.path.getmtime(path)).strftime('%d.%m.%Y %H:%M')
            return f'Своя модель звука: {os.path.basename(path)} ({created})'
        return ('Своя модель звука не обучена — используется AudioSet. '
                'Обучение: python tools/train_sound_model.py')

    def _parse_coordinates(self):
        lat_text = (self.latitude_edit.text() or '').strip().replace(',', '.')
        lon_text = (self.longitude_edit.text() or '').strip().replace(',', '.')
        if not lat_text or not lon_text:
            return None
        try:
            lat = float(lat_text)
            lon = float(lon_text)
        except ValueError:
            return None
        return manual_location(lat, lon, source='manual')

    def _collect_read_metadata(self):
        source = self.device_combo.currentData()
        pair = describe_audio_pair(source)
        location = self._parse_coordinates() or cached_location()
        recognized = (self.speech_text.toPlainText() or '').strip()
        if recognized in (
                'Ожидание речи...',
                'Речь обнаружена, распознавание...',
                'Речь в файле не обнаружена (или она не распознана).'):
            recognized = ''
        return {
            'microphone': pair.get('microphone') or '',
            'speaker': pair.get('speaker') or '',
            'location': location,
            'language': self.language_combo.currentData() or 'ru',
            'recognized_text': recognized,
        }

    def read_custom_aloud(self):
        text = (self.custom_text_edit.toPlainText() or '').strip()
        if not text:
            self.log('Введите текст для чтения в поле «Свой текст»')
            self.status_label.setText('Нет текста для чтения')
            return
        self._start_read_aloud(text, source='custom_text')

    def _start_read_aloud(self, text, source='custom_text'):
        if self._read_active:
            self.log('Озвучивание уже выполняется — дождитесь окончания')
            return
        if not self.tts:
            self.log('TTS недоступен (нет Piper, pyttsx3 или edge-tts)')
            QMessageBox.information(
                self, 'TTS',
                'Движок озвучивания недоступен. Установите Piper в папку '
                'piper проекта, pyttsx3 или edge-tts.')
            return
        voice = self._voice_for_text(text)
        meta = self._collect_read_metadata()
        meta['source'] = source
        self._pending_read_meta = meta
        self._read_active = True
        self._refresh_buttons()

        self._read_thread = ReadAloudThread(
            tts=self.tts,
            text=text,
            voice=voice,
            language=meta['language'],
            record=self.record_while_read_cb.isChecked(),
            save_tts=self.save_tts_audio_cb.isChecked(),
        )
        self._read_thread.result_signal.connect(self._on_read_finished)
        self._read_thread.finished.connect(self._on_read_thread_finished)
        self._read_thread.start()

        self.status_label.setText('Чтение вслух...')
        self.mic_status.setText('Статус: озвучивание текста')
        self.log(
            f'Озвучивание: {len(text)} символов; '
            f'голос: {self._voice_label(voice)}; язык: {meta["language"]}')
        if self.record_while_read_cb.isChecked():
            self.log('Запись микрофона во время чтения включена')

    @staticmethod
    def _voice_label(voice):
        if isinstance(voice, dict):
            return str(voice.get('name') or voice.get('id') or 'голос')
        return str(voice or 'голос')

    def _on_read_finished(self, result):
        meta = self._pending_read_meta or {}
        record = build_session_record(
            text=result.get('text', ''),
            recognized_text=meta.get('recognized_text', ''),
            voice=result.get('voice'),
            language=result.get('language') or meta.get('language') or 'ru',
            location=meta.get('location'),
            microphone=meta.get('microphone', ''),
            speaker=meta.get('speaker', ''),
            audio_status=result.get('status', 'error'),
            recording=result.get('recording'),
            tts_audio=result.get('tts_audio'),
            started_at=result.get('started_at'),
            duration_seconds=result.get('duration_seconds'),
            error=result.get('error', ''),
            source=meta.get('source', 'custom_text'),
        )
        self._last_read_record = record
        self._last_read_saved = False

        status = result.get('status')
        if status == 'ok':
            self.status_label.setText('Текст прочитан')
            self.log(
                f"Озвучивание завершено за "
                f"{float(result.get('duration_seconds') or 0.0):.1f} с")
        elif status == 'stopped':
            self.status_label.setText('Озвучивание остановлено')
            self.log('Озвучивание остановлено пользователем')
        else:
            self.status_label.setText('Ошибка озвучивания')
            self.log(f"Ошибка озвучивания: {result.get('error') or status}")

        recording = result.get('recording') or {}
        if recording:
            if recording.get('status') == 'ok':
                self.log(f"Запись микрофона: {recording.get('path')}")
            elif recording.get('error'):
                self.log(f"Запись микрофона: {recording.get('error')}")

        if self.auto_save_cb.isChecked():
            self._save_record(record, automatic=True)

    def _on_read_thread_finished(self):
        thread = self.sender()
        self._retain_thread(thread)
        if thread is self._read_thread:
            self._read_thread = None
        self._read_active = False
        self._pending_read_meta = None
        self._refresh_buttons()
        self.mic_status.setText('Статус: чтение завершено')

    def save_read_result(self):
        record = self._last_read_record
        if record is None:
            text = (self.custom_text_edit.toPlainText() or '').strip()
            if not text:
                self.log('Нет результата для сохранения')
                self.status_label.setText('Нет результата для сохранения')
                return
            meta = self._collect_read_metadata()
            record = build_session_record(
                text=text,
                recognized_text=meta.get('recognized_text', ''),
                voice=self._get_tts_voice(),
                language=meta.get('language', 'ru'),
                location=meta.get('location'),
                microphone=meta.get('microphone', ''),
                speaker=meta.get('speaker', ''),
                audio_status='not_read',
                source='custom_text',
            )
            self._last_read_record = record
            self._last_read_saved = False
        if self._last_read_saved:
            self.log('Этот результат уже сохранён в журнале')
            self.status_label.setText('Результат уже сохранён')
            return
        self._save_record(record, automatic=False)

    def _save_record(self, record, automatic=False):
        try:
            saved = self.session_store.append(record)
        except Exception as exc:
            self.log(f'Не удалось сохранить результат: {exc}')
            QMessageBox.warning(
                self, 'Сохранение результата',
                f'Не удалось записать журнал:\n{exc}')
            return None
        self._last_read_saved = True
        self.log(
            'Результат сохранён' + (' автоматически' if automatic else '')
            + f": {self.session_store.path} (id {saved['id'][:8]})")
        self.status_label.setText(
            'Результат сохранён' + (' автоматически' if automatic else ''))
        return saved

    def show_session_history(self):
        dialog = SessionHistoryDialog(self, self.session_store)
        dialog.exec()
        self._refresh_buttons()

    def detect_location_clicked(self):
        if self._location_thread and self._location_thread.isRunning():
            return
        self.location_btn.setEnabled(False)
        self.log('Определение координат по IP...')
        self._location_thread = LocationThread()
        self._location_thread.result_signal.connect(self._on_location_result)
        self._location_thread.finished.connect(self._on_location_finished)
        self._location_thread.start()

    def _on_location_result(self, location):
        if not location:
            self.log('Не удалось определить координаты. Введите их вручную '
                     'или задайте BLUETOOTH_MONITOR_LATITUDE/LONGITUDE.')
            self.status_label.setText('Координаты не определены')
            return
        self.latitude_edit.setText(str(location.get('lat', '')))
        self.longitude_edit.setText(str(location.get('lon', '')))
        details = ', '.join(
            str(location.get(key)) for key in ('city', 'country')
            if location.get(key))
        suffix = f' ({details})' if details else ''
        self.log(
            f"Координаты: {location.get('lat')}, {location.get('lon')}"
            f"{suffix}; источник: {location.get('source')}")
        self.status_label.setText('Координаты определены')

    def _on_location_finished(self):
        thread = self.sender()
        self._retain_thread(thread)
        if thread is self._location_thread:
            self._location_thread = None
        self._refresh_buttons()

    def _maybe_auto_read(self, new_text):
        if not self._auto_read_tts or not self.tts or not new_text:
            return
        text = (new_text or '').strip()
        if not text or text in ('Ожидание речи...', 'Речь обнаружена, распознавание...', 'Речь в файле не обнаружена (или она не распознана).'):
            return
        # Find new part
        if self._last_spoken_text and text.startswith(self._last_spoken_text):
            delta = text[len(self._last_spoken_text):].strip()
        elif self._last_spoken_text and self._last_spoken_text in text:
            # Try to find suffix
            idx = text.rfind(self._last_spoken_text) + len(self._last_spoken_text)
            delta = text[idx:].strip()
        else:
            # If completely different or first time, read all
            delta = text
        self._last_spoken_text = text
        if delta:
            cleaned = self._clean_text_for_tts(delta)
            if cleaned and len(cleaned) > 1:
                try:
                    self.tts.speak_async(
                        cleaned, self._voice_for_text(cleaned),
                        on_done=self._on_tts_done)
                except Exception:
                    pass

    def closeEvent(self, event):
        try:
            if self.tts:
                self.tts.stop()
        except Exception:
            pass
        try:
            self.settings.sync()
        except Exception:
            pass
        if self.monitor_thread and self.monitor_thread.isRunning():
            self.monitor_thread.stop()
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.wait(1000)
        if self._read_thread and self._read_thread.isRunning():
            try:
                self.tts.stop()
            except Exception:
                pass
            self._read_thread.wait(2000)
        if self._location_thread and self._location_thread.isRunning():
            self._location_thread.wait(1000)
        for thread in list(self._retired_threads):
            thread.wait(500)
        if self.file_thread and self.file_thread.isRunning():
            self.file_thread.wait(1000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
