# coding:utf-8
"""
    :module: deconvolve_UI.py
    :description: Impulse Response tool - Batch deconvolve / process (trim/fade end, normalize) impulse responses
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.04

    NOTE : Under windows, when experiencing pauses with video players in a web browser (YT for example)
    or any other interference with any software currently running
    replace the portaudio dll used by sounddevice found in
    .../Lib/site-packages/_sounddevice_data/portaudio-binaries
    by a non-asio dll from this repository :
    https://github.com/spatialaudio/portaudio-binaries

    Sound device is used to preview sounds by double-clicking on the item
"""

import ctypes
import os
import platform
import shutil
import sys
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
from dark_fusion_style import apply_dark_theme
import sounddevice as sd
import soundfile as sf
from PyQt5 import QtWidgets, QtGui, QtCore, Qt

from UI import ir_tool as gui
from deconvolve import deconvolve, generate_sweep, generate_impulse, db_to_lin, compensate_ir, trim_end
from worker import Worker

if getattr(sys, 'frozen', False):
    import pyi_splash

    pyi_splash.close()

from __init__ import __version__


class IrToolUi(gui.Ui_ir_tool_mw, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.setWindowTitle(f'IR Tool v{__version__}')

        self.current_dir = Path(__file__).parent

        app_icon = QtGui.QIcon()
        app_icon.addFile(resource_path('UI/ir_tool_64.png'), QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.file_types = ['.wav', '.flac', '.aif']
        self.output_path = ''

        # For auto-completion
        self.ref_tone_path_l = cast(FilePathLabel, self.ref_tone_path_l)
        self.output_path_l = cast(FilePathLabel, self.output_path_l)

        self.threadpool = QtCore.QThreadPool(parent=self)
        self.worker = None
        self.active_workers = []
        self.worker_result = None
        self.event_loop = QtCore.QEventLoop()

        self.buffer = None
        self.data = None

        self.setup_connections()
        self.last_file = self.output_path_l.fullPath() or self.ref_tone_path_l.fullPath()

        self.progress_pb.setTextVisible(True)
        self.update_message(f'Batch deconvolve / process (trim/fade end, normalize) impulse responses')

        self.show()

    def setup_connections(self):
        # Ref tone widgets
        self.ref_tone_path_l = replace_widget(self.ref_tone_path_l, FilePathLabel(file_mode=True, parent=self))
        add_ctx(self.ref_tone_path_l, values=[''], names=['Clear'])
        self.ref_tone_path_l.setFullPath(resource_path('sweep_tone.wav'))
        self.ref_tone_path_l.validatePath()

        self.set_ref_tone_tb.clicked.connect(self.ref_tone_path_l.browse_path)
        self.ref_tone_path_l.mouseDoubleClickEvent = self.play_sweep_tone

        self.gen_sweep_pb.clicked.connect(self.do_sweep_gen)
        self.gen_impulse_pb.clicked.connect(self.do_impulse_gen)

        # Files widgets
        self.set_files_tb.clicked.connect(self.browse_files)
        self.files_lw.setContextMenuPolicy(3)
        self.files_lw.customContextMenuRequested.connect(self.files_lw_ctx)
        self.files_lw.doubleClicked.connect(self.play_lw_item)

        # Output path widgets
        self.output_path_l = replace_widget(self.output_path_l, FilePathLabel(file_mode=False, parent=self))
        default_dir = get_user_directory()
        desktop_dir = get_user_directory('Desktop')
        add_ctx(self.output_path_l, values=['', default_dir, desktop_dir],
                names=['Clear', 'Default directory', 'Desktop'])
        self.set_output_path_tb.clicked.connect(self.output_path_l.browse_path)

        # Trim widgets
        self.trim_cb.stateChanged.connect(lambda state: self.trim_db_dsb.setEnabled(state == 2))
        self.trim_cb.stateChanged.connect(lambda state: self.fadeout_cb.setEnabled(state == 2))
        self.trim_cb.stateChanged.connect(
            lambda state: self.fadeout_db_dsb.setEnabled(state == 2 and self.fadeout_cb.isChecked()))
        self.fadeout_cb.stateChanged.connect(
            lambda state: self.fadeout_db_dsb.setEnabled(state == 2 and self.trim_cb.isChecked()))

        add_ctx(self.trim_db_dsb, values=[-60, -90, -120])
        add_ctx(self.fadeout_db_dsb, values=[-48, -60, -90, -120])

        # Normalize widgets
        self.normalize_cb.stateChanged.connect(lambda state: self.normalize_cmb.setEnabled(state == 2))
        self.normalize_cb.stateChanged.connect(
            lambda state: self.peak_db_dsb.setEnabled(state == 2 and self.normalize_cmb.currentText() == 'peak'))
        self.normalize_cmb.currentTextChanged.connect(
            lambda state: self.peak_db_dsb.setEnabled(state == 'peak' and self.normalize_cb.isChecked()))

        add_ctx(self.peak_db_dsb, values=[-0.5, -6, -12, -18])

        # Suffix widget
        self.add_suffix_cb.stateChanged.connect(lambda state: self.suffix_le.setEnabled(state == 2))

        add_ctx(self.suffix_le, values=['_result', '_dc', '_trim', '_norm'])

        # Process button
        self.process_pb.clicked.connect(partial(self.as_worker, self.do_deconvolve))
        self.stop_pb.clicked.connect(self.stop_process)

        # Custom events
        self.files_lw.keyPressEvent = self.key_del_lw_items_event
        self.files_lw.setAcceptDrops(True)
        self.files_lw.dragEnterEvent = self.drag_enter_event
        self.files_lw.dragMoveEvent = self.drag_move_event
        self.files_lw.dropEvent = self.lw_drop_event

    def do_sweep_gen(self):
        file_dialog = RefToneDialog(self)
        file_dialog.setWindowTitle('Generate sweep tone from 20 Hz to Nyquist frequency')
        file_dialog.w_cb.setHidden(False)

        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if files:
                fp = Path(files[0])
                if not fp.suffix:
                    fp = fp.with_suffix('.wav')

                kwargs = {
                    'filepath': fp,
                    'duration': file_dialog.length_dsb.value(),
                    'sr': file_dialog.sr_sb.value(),
                    'bit_depth': int(file_dialog.bd_cmb.currentText()),
                    'windowed': file_dialog.w_cb.isChecked()
                }

                self.as_worker(partial(self.sweep_gen_process, **kwargs))
                self.play_notification()

    @staticmethod
    def sweep_gen_process(worker, progress_callback, message_callback,
                          filepath, duration, sr, bit_depth, windowed):

        range_callback = worker.signals.progress_range

        range_callback.emit(0, 1)
        progress_callback.emit(0)
        message_callback.emit('%p%')

        ext = filepath.suffix[1:]

        subtypes = {16: 'PCM_16', 24: 'PCM_24'}
        if ext == 'flac':
            subtypes[32] = 'PCM_24'
        else:
            subtypes[32] = 'FLOAT'

        data = generate_sweep(duration, sr=sr, db=-6, window=windowed)

        # Soundfile only recognizes aiff and not aif when writing
        sf_path = (filepath, filepath.with_suffix('.aiff'))[ext == 'aif']
        sf.write(str(sf_path), data, sr, subtype=subtypes[bit_depth])
        if str(sf_path) != filepath:
            os.rename(sf_path, filepath)

        progress_callback.emit(1)
        message_callback.emit(f'{filepath.name} generated.')

        return filepath

    def do_impulse_gen(self):
        file_dialog = RefToneDialog(self)
        file_dialog.setWindowTitle('Generate impulse click')
        file_dialog.w_cb.setHidden(True)

        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if files:
                fp = Path(files[0])
                if not fp.suffix:
                    fp = fp.with_suffix('.wav')

                kwargs = {
                    'filepath': fp,
                    'duration': file_dialog.length_dsb.value(),
                    'sr': file_dialog.sr_sb.value(),
                    'bit_depth': int(file_dialog.bd_cmb.currentText()),
                }

                self.as_worker(partial(self.impulse_gen_process, **kwargs))
                self.play_notification()

    @staticmethod
    def impulse_gen_process(worker, progress_callback, message_callback,
                            filepath, duration, sr, bit_depth):

        range_callback = worker.signals.progress_range

        range_callback.emit(0, 1)
        progress_callback.emit(0)
        message_callback.emit('%p%')

        ext = filepath.suffix[1:]

        subtypes = {16: 'PCM_16', 24: 'PCM_24'}
        if ext == 'flac':
            subtypes[32] = 'PCM_24'
        else:
            subtypes[32] = 'FLOAT'

        data = generate_impulse(duration, sr=sr, db=-0.5)

        # Soundfile only recognizes aiff and not aif when writing
        sf_path = (filepath, filepath.with_suffix('.aiff'))[ext == 'aif']
        sf.write(str(sf_path), data, sr, subtype=subtypes[bit_depth])
        if str(sf_path) != filepath:
            os.rename(sf_path, filepath)

        progress_callback.emit(1)
        message_callback.emit(f'{filepath.name} generated.')

        return filepath

    def do_deconvolve(self, worker, progress_callback, message_callback):
        count = self.files_lw.count()
        deconv = self.deconv_cb.isChecked()

        range_callback = worker.signals.progress_range

        if count < 1:
            progress_callback.emit(0)
            message_callback.emit('No file(s) to process')
            return False

        if deconv and not self.ref_tone_path_l.fullPath():
            progress_callback.emit(0)
            message_callback.emit('No reference tone provided')
            return False

        self.process_pb.setEnabled(False)
        self.stop_pb.setEnabled(True)

        range_callback.emit(0, count)
        progress_callback.emit(0)
        message_callback.emit('%p%')

        orig, orig_sr = sf.read(self.ref_tone_path_l.fullPath())

        items = self.get_lw_items()
        files = items

        trim_db, fade_db = None, None
        if self.trim_cb.isChecked():
            trim_db = self.trim_db_dsb.value()
        if self.fadeout_cb.isChecked():
            fade_db = self.fadeout_db_dsb.value()

        bit_depth = int(self.bitdepth_cmb.currentText())
        ext = self.format_cmb.currentText()
        suffix = ''
        if self.add_suffix_cb.isChecked():
            suffix = self.suffix_le.text()

        subtypes = {16: 'PCM_16', 24: 'PCM_24'}
        if ext == 'flac':
            subtypes[32] = 'PCM_24'
        else:
            subtypes[32] = 'FLOAT'

        done = 0
        for i, f in enumerate(files):
            if worker.is_stopped():
                progress_callback.emit(0)
                message_callback.emit('Process Interrupted')
                return False

            conv, conv_sr = sf.read(f)

            if deconv and conv_sr != orig_sr:
                msg = f'{f}: sampling rate ({conv_sr}) does not match reference tone ({orig_sr}), skipped'
                print(msg)
                message_callback.emit(msg)
                continue

            p = Path(f)
            parent = self.output_path_l.fullPath() or p.parent
            stem = p.stem
            filepath = Path(parent) / f'{stem}{suffix}.{ext}'

            if self.no_overwriting_cb.isChecked() and str(filepath) == f:
                resolve_overwriting(f, mode='dir', dir_name='backup_', do_move=True)

            if deconv:
                ir = deconvolve(conv, orig, lambd=1e-3, mode='minmax_sum')
            else:
                ir = conv

            mn_s = int(conv_sr * .25)
            ir = trim_end(ir, trim_db=trim_db, fade_db=fade_db, min_silence=mn_s, min_length=None)

            if self.normalize_cb.isChecked():
                if self.normalize_cmb.currentText() == 'peak':
                    ir *= db_to_lin(self.peak_db_dsb.value()) / np.max(np.abs(ir))
                elif self.normalize_cmb.currentText() == 'compensate':
                    ir = compensate_ir(ir, mode='rms', sr=conv_sr)

            # Soundfile only recognizes aiff and not aif when writing
            sf_path = (filepath, filepath.with_suffix('.aiff'))[ext == 'aif']
            try:
                sf.write(str(sf_path), ir, conv_sr, subtype=subtypes[bit_depth])
                if sf_path != filepath:
                    os.rename(sf_path, filepath)
                done += 1
            except Exception as e:
                msg = f'{filepath} could not be written'
                print(msg)
                message_callback.emit(msg)

            progress_callback.emit(i + 1)

        message_callback.emit(f'{done} of {count} file(s) processed.')
        if done < count:
            message_callback.emit('Some file(s) could not be processed. Please check settings.')

        self.play_notification()

        self.process_pb.setEnabled(True)
        self.stop_pb.setEnabled(False)

        return True

    def stop_process(self):
        for worker in self.active_workers:
            if worker.running:
                worker.request_stop()

        self.process_pb.setEnabled(True)
        self.stop_pb.setEnabled(False)

    def browse_files(self):
        self.refresh_lw_items()
        if not self.last_file:
            items = self.files_lw.selectedItems() or self.get_lw_items()
            items = [s.data(Qt.Qt.UserRole) for s in items]
            if items:
                self.last_file = items[-1]

        if self.last_file:
            startdir = str(Path(self.last_file).parent)
        else:
            startdir = os.getcwd()

        fmts = [f'*{fmt}' for fmt in self.file_types]
        fltr = 'Audio Files ({});;All Files (*)'.format(' '.join(fmts))
        new_files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select audio files", startdir, fltr)

        if new_files:
            files = self.get_lw_items()
            files.extend(new_files)
            self.add_lw_items(files)

    def files_lw_ctx(self):
        menu = QtWidgets.QMenu(self)
        names = [menu.addAction(item) for item in ['Remove item(s) from list', 'Clear list']]
        cmds = [self.del_lw_items, self.files_lw.clear]
        action = menu.exec_(QtGui.QCursor.pos())
        for name, cmd in zip(names, cmds):
            if action == name:
                cmd()

    def get_lw_items(self):
        return [self.files_lw.item(i).data(Qt.Qt.UserRole) for i in range(self.files_lw.count())]

    def add_lw_items(self, files):
        files = [os.path.normpath(f) for f in files]
        files = list(dict.fromkeys(files))
        names = [shorten_path(f) for f in files]

        self.files_lw.clear()
        self.files_lw.addItems(names)

        for i, file_path in enumerate(files):
            self.files_lw.item(i).setData(Qt.Qt.UserRole, file_path)

        if files:
            self.last_file = files[-1]

    def refresh_lw_items(self):
        lw_items = [self.files_lw.item(i) for i in range(self.files_lw.count())]
        for item in lw_items:
            f = item.data(Qt.Qt.UserRole)
            if Path(f).is_file():
                item.setText(shorten_path(f))
            else:
                self.files_lw.takeItem(self.files_lw.row(item))
        self.files_lw.update()

    def del_lw_items(self):
        for item in self.files_lw.selectedItems():
            self.files_lw.takeItem(self.files_lw.row(item))

    @staticmethod
    def play_lw_item(*args):
        audio_file = args[0].data(Qt.Qt.UserRole)
        if os.path.isfile(audio_file):
            data, sr = sf.read(audio_file)
            sd.play(data, sr)

    def play_sweep_tone(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if os.path.isfile(self.ref_tone_path_l.fullPath()):
                data, sr = sf.read(self.ref_tone_path_l.fullPath())
                sd.play(data, sr)

    @staticmethod
    def play_notification():
        audio_file = resource_path('process_complete.flac')
        if os.path.isfile(audio_file):
            data, sr = sf.read(audio_file)
            sd.play(data, sr)

    def key_del_lw_items_event(self, event):
        if event.key() == Qt.Qt.Key_Delete:
            items = self.files_lw.selectedItems()
            for item in items:
                self.files_lw.takeItem(self.files_lw.row(item))

        if event.key() == Qt.Qt.Key_Down:
            mx = self.files_lw.count() - 1
            sel_indices = [a.row() + 1 if a.row() < mx else mx for a in self.files_lw.selectedIndexes()]
            self.files_lw.clearSelection()
            for idx in sel_indices:
                self.files_lw.item(idx).setSelected(True)
        elif event.key() == Qt.Qt.Key_Up:
            sel_indices = [a.row() - 1 if a.row() > 0 else 0 for a in self.files_lw.selectedIndexes()]
            self.files_lw.clearSelection()
            for idx in sel_indices:
                self.files_lw.item(idx).setSelected(True)

        elif event.modifiers() & Qt.Qt.ControlModifier:
            if event.key() == Qt.Qt.Key_A:
                self.files_lw.selectAll()
            elif event.key() == Qt.Qt.Key_I:
                items = self.files_lw.selectedItems()
                self.files_lw.selectAll()
                for item in items:
                    item.setSelected(False)
        else:
            super().keyPressEvent(event)

    @staticmethod
    def drag_enter_event(event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    @staticmethod
    def drag_move_event(event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def lw_drop_event(self, event):
        if event.mimeData().hasUrls():
            self.refresh_lw_items()
            items = event.mimeData().urls()
            items = [Path(item.toLocalFile()) for item in items]

            files = self.get_lw_items()
            files.extend([item for item in items if Path(item).suffix in self.file_types])

            dirs = [item for item in items if item.is_dir()]
            for d in dirs:
                for ext in self.file_types:
                    files.extend(Path(d).glob(f'*{ext}'))

            self.add_lw_items(files)
        else:
            event.ignore()

    def ref_tone_drop_event(self, event):
        if event.mimeData().hasUrls():
            items = event.mimeData().urls()
            items = [Path(item.toLocalFile()) for item in items]

            files = [item for item in items if item.suffix in self.file_types]
            if files:
                self.ref_tone = os.path.normpath(files[0])
                p = Path(self.ref_tone)
                if p.is_relative_to(self.current_dir):
                    self.ref_tone = str(p.relative_to(self.current_dir))
                self.ref_tone_path_l.setText(self.ref_tone)
        else:
            event.ignore()

    def output_path_drop_event(self, event):
        if event.mimeData().hasUrls():
            items = event.mimeData().urls()
            items = [Path(item.toLocalFile()) for item in items]

            dirs = [item for item in items if item.is_dir()]
            if dirs:
                self.output_path = os.path.normpath(dirs[0])
                self.output_path_l.setText(self.output_path)
        else:
            event.ignore()

    def as_worker(self, cmd):
        if not any(worker.running for worker in self.active_workers):
            self.worker = Worker(cmd)

            # Worker signals
            self.worker.signals.progress.connect(self.update_progress)
            self.worker.signals.progress_range.connect(self.update_range)
            self.worker.signals.message.connect(self.update_message)
            self.worker.signals.result.connect(self.handle_result)

            self.worker.signals.finished.connect(lambda: self.cleanup_worker(self.worker))

            self.active_workers.append(self.worker)
            self.threadpool.start(self.worker)
        else:
            print('Task is already running!')

    def update_progress(self, value):
        self.progress_pb.setValue(value)

    def update_message(self, message):
        self.progress_pb.setFormat(message)

    def update_range(self, mn, mx):
        self.progress_pb.setRange(mn, mx)
        self.progress_pb.update()

    def handle_result(self, value):
        self.worker_result = value
        self.event_loop.quit()

    def cleanup_worker(self, worker):
        if worker in self.active_workers:
            self.active_workers.remove(worker)


class RefToneDialog(QtWidgets.QFileDialog):
    def __init__(self, *args):
        super().__init__(*args)
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.setOptions(QtWidgets.QFileDialog.DontUseNativeDialog)
        self.setFileMode(QtWidgets.QFileDialog.AnyFile)
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.setNameFilters(['Audio Files (*.wav *.flac *.aif)'])
        self.setWindowTitle('Generate reference tone')
        self.init_custom_widgets()

    def init_custom_widgets(self):
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.length_l = QtWidgets.QLabel('Length(s) :', self)
        self.length_l.setSizePolicy(size_policy)
        self.length_dsb = QtWidgets.QDoubleSpinBox(self)
        self.length_dsb.setFrame(False)
        self.length_dsb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.length_dsb.setMaximum(60)
        self.length_dsb.setValue(4.0)
        self.length_dsb.setDecimals(1)

        add_ctx(self.length_dsb, values=[3.0, 4.0, 6.0, 8.0, 16.0])

        self.sr_l = QtWidgets.QLabel('Sample Rate :', self)
        self.sr_l.setSizePolicy(size_policy)
        self.sr_sb = QtWidgets.QSpinBox(self)
        self.sr_sb.setFrame(False)
        self.sr_sb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.sr_sb.setMaximum(192000)
        self.sr_sb.setValue(48000)

        add_ctx(self.sr_sb, values=[44100, 48000, 96000, 192000])

        self.bd_l = QtWidgets.QLabel('Bit Depth :', self)
        self.bd_l.setSizePolicy(size_policy)
        self.bd_cmb = QtWidgets.QComboBox(self)
        self.bd_cmb.setFrame(False)
        values = [16, 24, 32]
        self.bd_cmb.addItems([str(v) for v in values])
        self.bd_cmb.setCurrentIndex(2)
        self.bd_cmb.setToolTip('flac only allows integer format up to 24 bits\nUse wav or aif for 32 bits float')

        self.w_cb = QtWidgets.QCheckBox('Windowed', self)
        self.w_cb.setSizePolicy(size_policy)
        self.w_cb.setChecked(True)
        self.w_cb.setToolTip('Apply Tukey window to signal to avoid popping')
        self.w_cb.setHidden(True)

        custom_wid = QtWidgets.QWidget(self)
        self.h_lyt = QtWidgets.QHBoxLayout(custom_wid)

        self.h_lyt.addWidget(self.length_l)
        self.h_lyt.addWidget(self.length_dsb)
        self.h_lyt.addWidget(self.sr_l)
        self.h_lyt.addWidget(self.sr_sb)
        self.h_lyt.addWidget(self.bd_l)
        self.h_lyt.addWidget(self.bd_cmb)
        self.h_lyt.addWidget(self.w_cb)

        lyt = self.layout()
        lyt.addWidget(QtWidgets.QWidget(self))
        lyt.addWidget(custom_wid)


class FilePathLabel(QtWidgets.QLabel):
    def __init__(self, text='', file_mode=False, parent=None):
        super().__init__(text, parent)
        self._full_path = ''
        self._file_mode = file_mode
        self.display_length = 40
        self.start_dir = ''
        self.current_dir = os.path.dirname(sys.modules['__main__'].__file__)
        self.file_types = ['.wav', '.flac', '.aif']

        # self.setStyleSheet('QLabel{color: #808080}')

        self.setAcceptDrops(True)

    def fullPath(self):
        return self._full_path

    def setFullPath(self, path):
        """
        Set full path while updating display name
        :param path:
        :return:
        """
        if path:
            path = os.path.normpath(path)
            p = Path(path)
            if p.is_relative_to(self.current_dir):
                path = str(p.relative_to(self.current_dir))
            self.start_dir = (path, str(Path(path).parent))[self._file_mode]
        self._full_path = path
        self.setText(self.shorten_path(path))

    def validatePath(self):
        if not Path(self._full_path).exists():
            self.setFullPath('')

    def shorten_path(self, path):
        if len(path) > self.display_length:
            return f"...{path[-self.display_length:]}"
        return path

    def browse_path(self):
        if not self.start_dir or not Path(self.start_dir).is_dir():
            self.start_dir = os.getcwd()

        if self._file_mode:
            fmts = [f'*{fmt}' for fmt in self.file_types]
            fltr = 'Audio File ({})'.format(' '.join(fmts))
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", self.start_dir, fltr)
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", self.start_dir)

        if path:
            path = os.path.normpath(path)
            p = Path(path)
            if p.is_relative_to(self.current_dir):
                path = str(p.relative_to(self.current_dir))
            self.setFullPath(path)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            items = event.mimeData().urls()
            items = [item.toLocalFile() for item in items]
            if self._file_mode:
                items = [item for item in items if Path(item).is_file()]
            else:
                items = [item for item in items if Path(item).is_dir()]
            if items:
                path = os.path.normpath(items[0])
                p = Path(path)
                if p.is_relative_to(self.current_dir):
                    path = str(p.relative_to(self.current_dir))
                self.setFullPath(path)
        else:
            event.ignore()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()


def replace_widget(old_widget, new_widget):
    """
    Replace a placeholder widget with another widget (typically a customised version of this widget)
    :param QtWidgets.QWidget old_widget:
    :param QtWidgets.QWidget new_widget:
    """

    attrs = ['objectName', 'parent', 'toolTip']
    values = [getattr(old_widget, attr)() for attr in attrs]

    lyt = old_widget.parent().layout()
    lyt.replaceWidget(old_widget, new_widget)
    old_widget.close()

    set_attrs = [f'set{a[0].upper()}{a[1:]}' for a in attrs]
    for a, v in zip(set_attrs, values):
        getattr(new_widget, a)(v)

    return new_widget


def get_user_directory(subdir='Documents'):
    if sys.platform == 'win32':
        import winreg

        folders = {
            'Desktop': 'Desktop',
            'Documents': 'Personal',
            'Downloads': '{374DE290-123F-4565-9164-39C4925E467B}',
            'Music': 'My Music',
            'Pictures': 'My Pictures',
            'Videos': 'My Video',
        }
        folders_lowercase = {k.lower(): v for k, v in folders.items()}

        p = Path(subdir)
        subdir_root = p.parts[0].lower()
        key_name = folders_lowercase.get(subdir_root, 'desktop')

        # This works even if user profile has been moved to some other drive
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders') as key:
            result, _ = winreg.QueryValueEx(key, key_name)
            result = Path(result)

        if subdir_root in folders_lowercase:
            result = result / '/'.join(p.parts[1:])
        else:
            result = result.parent / subdir
    else:
        result = Path.home() / subdir

    return result


def resolve_overwriting(input_path, mode='dir', dir_name='backup_', do_move=True):
    """
    Move/rename given input file path if it already exists
    :param str input_path:
    :param str mode: 'file' or 'dir'
    :param str dir_name: Directory name when using 'dir' mode
    :param bool do_move: Execute move/rename operation otherwise only return new name
    :return:
    :rtype: str
    """
    p = Path(input_path)
    parent, stem, ext = p.parent, p.stem, p.suffix
    i = 0
    new_path = p
    while new_path.is_file():
        i += 1
        if mode == 'file':
            new_path = Path.joinpath(parent, f'{stem}_{i:03d}{ext}')
        else:
            new_path = Path.joinpath(parent, f'{dir_name}{i:03d}', f'{stem}{ext}')
    if do_move and os.path.normpath(input_path) != os.path.normpath(new_path):
        os.makedirs(new_path.parent, exist_ok=True)
        shutil.move(input_path, new_path)
    return new_path


def shorten_path(file_path, max_length=77):
    if len(file_path) <= max_length:
        return file_path
    return '...' + file_path[-max_length:]


def add_ctx(widget, values=(), names=None, trigger=None):
    """
    Add a simple context menu setting provided values to the given widget

    :param QtWidgets.QWidget widget: The widget to which the context menu will be added
    :param list values: A list of values to be added as actions in the context menu
    :param list or None names: A list of strings or values to be added as action names
    must match values length
    :param QWidget or None trigger: Optional widget triggering the context menu
    typically a QPushButton or QToolButton
    """
    if not names:
        names = values

    def show_context_menu(event):
        menu = QtWidgets.QMenu(widget)
        for name, value in zip(names, values):
            if value == '---':
                menu.addSeparator()
            else:
                action = QtWidgets.QAction(f"{name}", widget)
                if hasattr(widget, 'setValue'):
                    action.triggered.connect(lambda checked, v=value: widget.setValue(v))
                elif hasattr(widget, 'setFullPath'):
                    action.triggered.connect(lambda checked, v=value: widget.setFullPath(v))
                elif hasattr(widget, 'setText'):
                    action.triggered.connect(lambda checked, v=value: widget.setText(v))
                menu.addAction(action)
        pos = widget.mapToGlobal(widget.contentsRect().bottomLeft())
        menu.setMinimumWidth(widget.width())
        menu.exec_(pos)

    widget.setContextMenuPolicy(3)
    if trigger is None:
        widget.customContextMenuRequested.connect(show_context_menu)
    else:
        trigger.clicked.connect(show_context_menu)


def resource_path(relative_path, as_str=True):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    Modified from :
    https://stackoverflow.com/questions/31836104/pyinstaller-and-onefile-how-to-include-an-image-in-the-exe-file
    :param str or WindowsPath relative_path:
    :param bool as_str: Return result as a string
    :return:
    :rtype: str or Path
    """
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path().resolve()
    result = base_path / relative_path
    return (result, str(result))[bool(as_str)]


def run(parent=None):
    return IrToolUi(parent)


if __name__ == "__main__":
    myappid = f'mitch.ir_tool.{__version__}'

    with suppress(ModuleNotFoundError):
        import pyi_splash  # noqa

    if platform.system() == 'Windows':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app)

    font = app.font()
    font.setPointSize(12)
    app.setFont(font)

    window = IrToolUi()
    window.show()

    if getattr(sys, 'frozen', False):
        pyi_splash.close()

    sys.exit(app.exec_())
