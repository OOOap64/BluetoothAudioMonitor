"""Упрощённый диалог добавления голоса (PyQt6).

Пользователю не нужно вручную выбирать тип движка: все доступные голоса
собраны в одном списке.

* системные голоса Windows (pyttsx3/SAPI);
* нейроголоса Edge TTS;
* локальные голоса Piper, найденные в папке ``piper/voices`` проекта.

Для нестандартной модели Piper достаточно раскрыть «Дополнительно» и указать
файлы. Диалог только собирает профиль и просит ``TTSEngine`` проиграть
пример, поэтому его можно использовать и без звукового устройства.
"""

from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from src.tts_engine import find_piper_executable, find_piper_models
from src.voice_profiles import normalize_profile


EDGE_VOICES = (
    ('ru-RU-DmitryNeural', 'Дмитрий', 'мужской'),
    ('ru-RU-SvetlanaNeural', 'Светлана', 'женский'),
    ('en-US-AriaNeural', 'Aria', 'English'),
    ('en-US-GuyNeural', 'Guy', 'English'),
)


class VoiceImportDialog(QDialog):
    def __init__(self, parent=None, tts=None, profile=None):
        super().__init__(parent)
        self.setWindowTitle('Добавить голос')
        self.setMinimumWidth(580)
        self.tts = tts
        self._profile = None
        self._presets = []
        self._last_suggestion = ''
        self._manual_piper = False
        self._build_ui()
        self._load_presets()
        if profile:
            self._load_profile(profile)
        elif self.voice_combo.count():
            self.voice_combo.setCurrentIndex(0)
            self._on_preset_changed()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)

        intro = QLabel(
            'Выберите готовый голос из списка. Piper уже установлен — '
            'локальные голоса доступны без настройки.')
        intro.setWordWrap(True)
        layout.addWidget(intro)

        common = QGroupBox('Голос')
        common_form = QFormLayout(common)

        self.voice_combo = QComboBox()
        self.voice_combo.currentIndexChanged.connect(self._on_preset_changed)
        common_form.addRow('Доступные голоса:', self.voice_combo)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText('Название в списке голосов')
        common_form.addRow('Название:', self.name_edit)

        preview_row = QWidget()
        preview_layout = QHBoxLayout(preview_row)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_btn = QPushButton('▶ Проверить голос')
        self.preview_btn.clicked.connect(self._preview)
        preview_layout.addWidget(self.preview_btn)
        preview_layout.addStretch(1)
        common_form.addRow('', preview_row)
        layout.addWidget(common)

        # --- дополнительные параметры (свёрнуты) ---
        self.advanced_group = QGroupBox('Дополнительно')
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_form = QFormLayout(self.advanced_group)

        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(50, 400)
        self.rate_spin.setValue(165)
        self.rate_spin.setSuffix(' слов/мин')
        advanced_form.addRow('Скорость:', self.rate_spin)

        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0.0, 1.0)
        self.volume_spin.setSingleStep(0.1)
        self.volume_spin.setDecimals(2)
        self.volume_spin.setValue(1.0)
        advanced_form.addRow('Громкость:', self.volume_spin)

        self.piper_exec_edit = QLineEdit()
        self.piper_exec_edit.setPlaceholderText(
            'piper.exe (пусто — найти автоматически)')
        advanced_form.addRow(
            'Программа Piper:', self._browse_row(
                self.piper_exec_edit, self._browse_executable,
                'Найти'))

        self.piper_model_edit = QLineEdit()
        self.piper_model_edit.setPlaceholderText('голос.onnx')
        advanced_form.addRow(
            'Модель (.onnx):', self._browse_row(
                self.piper_model_edit, self._browse_model,
                'Выбрать'))

        self.piper_config_edit = QLineEdit()
        self.piper_config_edit.setPlaceholderText(
            'голос.onnx.json (необязательно)')
        advanced_form.addRow(
            'Конфигурация (.json):', self._browse_row(
                self.piper_config_edit, self._browse_config,
                'Выбрать'))

        layout.addWidget(self.advanced_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Добавить')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('Отмена')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _browse_row(self, edit, callback, title):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(edit, 1)
        button = QPushButton(title)
        button.clicked.connect(callback)
        row_layout.addWidget(button)
        return row

    # ------------------------------------------------------------- presets
    def _load_presets(self):
        self._presets = []
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()

        system = []
        if self.tts is not None:
            try:
                system = self.tts.get_system_voices()
            except Exception:
                system = []
        for voice in system:
            name = str(voice.get('name') or voice.get('id') or 'голос')
            profile = {
                'name': name,
                'backend': 'pyttsx3',
                'voice': voice.get('id') or '',
                'rate': 165,
                'volume': 1.0,
            }
            self._presets.append(profile)
            self.voice_combo.addItem('Системный: ' + name, profile)

        for voice_id, short_name, details in EDGE_VOICES:
            profile = {
                'name': short_name,
                'backend': 'edge',
                'voice': voice_id,
                'rate': 165,
                'volume': 1.0,
            }
            self._presets.append(profile)
            self.voice_combo.addItem(
                f'Edge: {short_name} ({details})', profile)

        executable = find_piper_executable()
        for model in find_piper_models():
            short = model.get('name', 'Piper')
            if short.endswith(')'):
                short = short.split(' (', 1)[0]
            profile = {
                'name': short + ' (Piper)',
                'backend': 'piper',
                'voice': '',
                'model_path': model['model_path'],
                'config_path': model['config_path'],
                'executable_path': executable,
                'rate': 165,
                'volume': 1.0,
            }
            self._presets.append(profile)
            self.voice_combo.addItem('Piper: ' + model['name'], profile)

        manual = {
            'name': '',
            'backend': 'piper',
            'voice': '',
            'model_path': '',
            'config_path': '',
            'executable_path': executable,
            'rate': 165,
            'volume': 1.0,
            'manual': True,
        }
        self._presets.append(manual)
        self.voice_combo.addItem('Другая модель Piper (выбрать файл)...', manual)

        self.voice_combo.blockSignals(False)

    def _suggestion_for(self, profile):
        backend = profile.get('backend')
        if backend == 'pyttsx3':
            return str(profile.get('name') or 'Системный голос')
        if backend == 'edge':
            return str(profile.get('name') or 'Edge')
        name = str(profile.get('name') or 'Piper')
        return name

    def _on_preset_changed(self, _index=None):
        profile = self.voice_combo.currentData()
        if not isinstance(profile, dict):
            return
        self._manual_piper = bool(profile.get('manual'))
        if self._manual_piper:
            self.advanced_group.setChecked(True)
            self.piper_exec_edit.setText(profile.get('executable_path') or '')
            self.piper_model_edit.clear()
            self.piper_config_edit.clear()
            self.name_edit.setText('Мой голос Piper')
            self._last_suggestion = 'Мой голос Piper'
            self.status_label.setText(
                'Укажите файл модели .onnx в разделе «Дополнительно».')
            return

        current = self.name_edit.text().strip()
        if not current or current == self._last_suggestion:
            suggestion = self._suggestion_for(profile)
            self.name_edit.setText(suggestion)
            self._last_suggestion = suggestion
        if profile.get('backend') == 'piper':
            self.piper_exec_edit.setText(
                profile.get('executable_path') or find_piper_executable())
            self.piper_model_edit.setText(profile.get('model_path') or '')
            self.piper_config_edit.setText(profile.get('config_path') or '')
        self.status_label.setText('')

    def _load_profile(self, profile):
        try:
            profile = normalize_profile(profile)
        except Exception:
            return
        self.name_edit.setText(profile['name'])
        self._last_suggestion = profile['name']
        self.rate_spin.setValue(profile['rate'])
        self.volume_spin.setValue(profile['volume'])
        backend = profile['backend']
        if backend == 'piper':
            self.advanced_group.setChecked(True)
            self.piper_exec_edit.setText(profile.get('executable_path', ''))
            self.piper_model_edit.setText(profile.get('model_path', ''))
            self.piper_config_edit.setText(profile.get('config_path', ''))
            for index in range(self.voice_combo.count()):
                data = self.voice_combo.itemData(index)
                if (isinstance(data, dict)
                        and data.get('model_path') == profile['model_path']):
                    self.voice_combo.setCurrentIndex(index)
                    return
            # Модель могла быть удалена — оставляем ручной режим.
            self.voice_combo.setCurrentIndex(self.voice_combo.count() - 1)
        elif backend == 'edge':
            for index in range(self.voice_combo.count()):
                data = self.voice_combo.itemData(index)
                if (isinstance(data, dict)
                        and data.get('voice') == profile['voice']):
                    self.voice_combo.setCurrentIndex(index)
                    return
            # Нет в списке — добавляем как ручной голос Edge.
            manual = dict(profile)
            self.voice_combo.addItem('Edge: ' + profile['name'], manual)
            self.voice_combo.setCurrentIndex(self.voice_combo.count() - 1)
        else:
            for index in range(self.voice_combo.count()):
                data = self.voice_combo.itemData(index)
                if (isinstance(data, dict)
                        and data.get('voice') == profile['voice']
                        and data.get('backend') == 'pyttsx3'):
                    self.voice_combo.setCurrentIndex(index)
                    return

    # -------------------------------------------------------------- browse
    def _browse_executable(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Найти piper.exe', '',
            'Программа (*.exe);;Все файлы (*)')
        if path:
            self.piper_exec_edit.setText(path)
            self.advanced_group.setChecked(True)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выбрать модель Piper', '',
            'Модель Piper (*.onnx);;Все файлы (*)')
        if path:
            self.piper_model_edit.setText(path)
            self.advanced_group.setChecked(True)
            config = path + '.json'
            if os.path.isfile(config):
                self.piper_config_edit.setText(config)

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выбрать конфигурацию Piper', '',
            'Конфигурация (*.json);;Все файлы (*)')
        if path:
            self.piper_config_edit.setText(path)
            self.advanced_group.setChecked(True)

    # ------------------------------------------------------------- profile
    def _profile_from_fields(self):
        base = self.voice_combo.currentData()
        profile = dict(base) if isinstance(base, dict) else {}
        profile.pop('manual', None)
        backend = profile.get('backend') or 'pyttsx3'
        if self._manual_piper:
            backend = 'piper'
        profile['backend'] = backend
        name = self.name_edit.text().strip()
        if not name:
            name = self._suggestion_for(profile)
        profile['name'] = name
        profile['rate'] = self.rate_spin.value()
        profile['volume'] = self.volume_spin.value()
        if backend == 'piper':
            profile['model_path'] = self.piper_model_edit.text().strip()
            profile['config_path'] = self.piper_config_edit.text().strip()
            profile['executable_path'] = (
                self.piper_exec_edit.text().strip()
                or find_piper_executable())
            if not profile['model_path']:
                raise ValueError(
                    'Выберите модель Piper (.onnx) в разделе «Дополнительно».')
            if not os.path.isfile(profile['model_path']):
                raise ValueError(
                    f'Файл модели не найден:\n{profile["model_path"]}')
            executable = profile['executable_path']
            if executable and not os.path.isfile(executable):
                raise ValueError(
                    f'Файл piper.exe не найден:\n{executable}')
            if not executable:
                raise ValueError(
                    'Piper не найден. Укажите путь к piper.exe.')
        return normalize_profile(profile)

    def _preview(self):
        try:
            profile = self._profile_from_fields()
        except Exception as exc:
            QMessageBox.warning(self, 'Голос', str(exc))
            return
        if self.tts is None:
            QMessageBox.warning(self, 'Голос', 'Движок озвучивания недоступен.')
            return
        self.status_label.setText('Проверка голоса...')
        try:
            blob = ' '.join(
                str(profile.get(key) or '')
                for key in ('id', 'voice', 'name')
            ).lower()
            english = any(marker in blob for marker in (
                'en-us', 'en_gb', 'en-gb', 'english',
                'zira', 'aria', 'guy',
            ))
            sample = ('Hello! This is a test of the selected voice.'
                      if english else
                      'Привет! Это пример выбранного голоса.')
            self.tts.speak_async(sample, profile)
            self.status_label.setText('Пример отправлен на озвучивание.')
        except Exception as exc:
            self.status_label.setText(f'Ошибка: {exc}')

    def accept(self):
        try:
            self._profile = self._profile_from_fields()
        except Exception as exc:
            QMessageBox.warning(self, 'Голос', str(exc))
            return
        super().accept()

    def result_profile(self):
        return self._profile
