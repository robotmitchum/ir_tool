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
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
import qdarkstyle
import sounddevice as sd
import soundfile as sf
from PyQt5 import QtWidgets, QtGui, QtCore

import UI.ir_tool_ui as gui
from deconvolve import deconvolve, generate_sweep, generate_impulse, db_to_lin, compensate_ir, trim_end

__version__ = '1.0.1'


class IrToolUi(gui.Ui_ir_tool_mw, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_dir = Path(__file__).parent

        app_icon = QtGui.QIcon()
        app_icon.addFile('UI/ir_tool_64.png', QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.file_types = ['.wav', '.flac', '.aif']
        self.ref_tone = 'sweep_tone.wav'
        self.output_path = ''

        self.setupUi(self)

        # Check paths
        if not os.path.isfile(self.ref_tone):
            self.ref_tone = ''
        self.ref_tone_path_l.setText(self.ref_tone)

        if not os.path.isdir(self.output_path):
            self.output_path = ''
        self.output_path_l.setText(self.output_path)

        self.last_file = self.output_path or self.ref_tone

        self.output = None
        self.buffer = None
        self.data = None

        self.setup_connections()

        self.show()

    def setup_connections(self):
        # Ref tone widgets
        self.set_ref_tone_tb.clicked.connect(self.set_ref_tone_path)
        self.gen_sweep_pb.clicked.connect(self.do_sweep_gen)
        self.gen_impulse_pb.clicked.connect(self.do_impulse_gen)

        # Files widgets
        self.set_files_tb.clicked.connect(self.browse_files)
        self.files_lw.setContextMenuPolicy(3)
        self.files_lw.customContextMenuRequested.connect(self.files_lw_ctx)
        self.files_lw.doubleClicked.connect(self.play_lw_item)

        # Output path widgets
        self.set_output_path_tb.clicked.connect(self.set_output_path)
        self.output_path_l.setContextMenuPolicy(3)
        self.output_path_l.customContextMenuRequested.connect(self.output_path_l_ctx)

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
        self.process_pb.clicked.connect(self.do_deconvolve)

        # Custom events
        self.ref_tone_path_l.mouseDoubleClickEvent = self.play_sweep_tone
        self.ref_tone_path_l.setAcceptDrops(True)
        self.ref_tone_path_l.dragEnterEvent = self.drag_enter_event
        self.ref_tone_path_l.dragMoveEvent = self.drag_move_event
        self.ref_tone_path_l.dropEvent = self.ref_tone_drop_event

        self.files_lw.keyPressEvent = self.key_del_lw_items_event
        self.files_lw.setAcceptDrops(True)
        self.files_lw.dragEnterEvent = self.drag_enter_event
        self.files_lw.dragMoveEvent = self.drag_move_event
        self.files_lw.dropEvent = self.lw_drop_event

        self.output_path_l.setAcceptDrops(True)
        self.output_path_l.dragEnterEvent = self.drag_enter_event
        self.output_path_l.dragMoveEvent = self.drag_move_event
        self.output_path_l.dropEvent = self.output_path_drop_event

    def do_sweep_gen(self):
        file_dialog = RefToneDialog(self)
        file_dialog.setWindowTitle('Generate sweep tone from 20 Hz to Nyquist frequency')
        file_dialog.w_cb.setHidden(False)

        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if files:
                self.progress_pb.setMaximum(1)
                self.progress_pb.setValue(0)
                self.progress_pb.setTextVisible(True)
                self.progress_pb.setFormat('%p%')

                filepath = files[0]
                ext = Path(filepath).suffix.lstrip('.')
                if not ext:
                    ext = 'wav'
                    filepath = f'{filepath}.{ext}'

                subtypes = {16: 'PCM_16', 24: 'PCM_24'}
                if ext == 'flac':
                    subtypes[32] = 'PCM_24'
                else:
                    subtypes[32] = 'FLOAT'

                duration = file_dialog.length_dsb.value()
                sr = file_dialog.sr_sb.value()
                bit_depth = int(file_dialog.bd_cmb.currentText())
                data = generate_sweep(duration, sr=sr, db=-6, window=file_dialog.w_cb.isChecked())

                # Soundfile only recognizes aiff and not aif when writing
                sf_path = (filepath, f'{filepath}f')[ext == 'aif']
                sf.write(str(sf_path), data, sr, subtype=subtypes[bit_depth])
                if str(sf_path) != filepath:
                    os.rename(sf_path, filepath)

                self.progress_pb.setValue(1)
                self.progress_pb.setFormat(f'{Path(filepath).name} generated.')

                self.play_notification()

    def do_impulse_gen(self):
        file_dialog = RefToneDialog(self)
        file_dialog.setWindowTitle('Generate impulse click')
        file_dialog.w_cb.setHidden(True)

        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if files:
                self.progress_pb.setMaximum(1)
                self.progress_pb.setValue(0)
                self.progress_pb.setTextVisible(True)
                self.progress_pb.setFormat('%p%')

                filepath = files[0]
                ext = Path(filepath).suffix.lstrip('.')
                if not ext:
                    ext = 'wav'
                    filepath = f'{filepath}.{ext}'

                subtypes = {16: 'PCM_16', 24: 'PCM_24'}
                if ext == 'flac':
                    subtypes[32] = 'PCM_24'
                else:
                    subtypes[32] = 'FLOAT'

                duration = file_dialog.length_dsb.value()
                sr = file_dialog.sr_sb.value()
                bit_depth = int(file_dialog.bd_cmb.currentText())
                data = generate_impulse(duration, sr=sr, db=-0.5)

                # Soundfile only recognizes 'aiff' and not 'aif' when writing
                sf_path = (filepath, f'{filepath}f')[ext == 'aif']
                sf.write(str(sf_path), data, sr, subtype=subtypes[bit_depth])
                if str(sf_path) != filepath:
                    os.rename(sf_path, filepath)

                self.progress_pb.setValue(1)
                self.progress_pb.setFormat(f'{Path(filepath).name} generated.')

                self.play_notification()

    def do_deconvolve(self):
        count = self.files_lw.count()
        if not self.ref_tone or count < 1:
            return False

        self.progress_pb.setMaximum(count)
        self.progress_pb.setValue(0)
        self.progress_pb.setTextVisible(True)
        self.progress_pb.setFormat('%p%')

        orig, orig_sr = sf.read(self.ref_tone)

        items = self.get_lw_items(self.files_lw)
        files = items

        deconv = self.deconv_cb.isChecked()

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
            conv, conv_sr = sf.read(f)

            if deconv and conv_sr != orig_sr:
                print(f"{f}: sampling rate do not match reference tone, skipped.")
                continue

            p = Path(f)
            parent = self.output_path or p.parent
            stem = p.stem
            filepath = Path.joinpath(Path(parent), f'{stem}{suffix}.{ext}')

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
            sf_path = (filepath, f'{filepath}f')[ext == 'aif']
            try:
                sf.write(str(sf_path), ir, conv_sr, subtype=subtypes[bit_depth])
                if sf_path != filepath:
                    os.rename(sf_path, filepath)
                done += 1
            except Exception as e:
                print(f'{filepath} could not be written')
                self.progress_pb.setFormat(f'{filepath} could not be written')
                pass

            self.progress_pb.setValue(i + 1)

        self.progress_pb.setFormat(f'{done} of {count} file(s) processed.')
        if done < count:
            self.progress_pb.setFormat('Some file(s) could not be processed. Please check settings.')
        self.play_notification()

        return True

    def set_ref_tone_path(self):
        startdir = self.ref_tone or os.getcwd()
        fmts = [f'*{fmt}' for fmt in self.file_types]
        fltr = 'Audio File ({})'.format(' '.join(fmts))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select reference tone", startdir, fltr)
        if path:
            self.ref_tone = os.path.normpath(path)
            p = Path(self.ref_tone)
            if p.is_relative_to(self.current_dir):
                self.ref_tone = str(p.relative_to(self.current_dir))
            self.ref_tone_path_l.setText(self.ref_tone)

    def set_output_path(self):
        startdir = self.output_path or os.getcwd()
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", startdir)
        if path:
            self.output_path = os.path.normpath(path)
            self.output_path_l.setText(self.output_path)

    def browse_files(self):
        items = self.files_lw.selectedItems()

        items = [s.text() for s in items] or ['']
        self.last_file = items[-1]

        startdir = self.last_file or os.getcwd()
        fmts = [f'*{fmt}' for fmt in self.file_types]
        fltr = 'Audio Files ({});;All Files (*)'.format(' '.join(fmts))
        new_files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select audio files", startdir, fltr)
        if new_files:
            files = self.get_lw_items(self.files_lw)
            new_files = [os.path.normpath(f) for f in new_files]
            files.extend(new_files)
            files = list(dict.fromkeys(files))
            self.files_lw.clear()
            self.files_lw.addItems(files)
            self.last_file = files[-1]

    def files_lw_ctx(self, *args):
        menu = QtWidgets.QMenu(self)
        names = [menu.addAction(item) for item in ['Remove item(s) from list', 'Clear list']]
        cmds = [partial(self.del_lw_items, self.files_lw), self.files_lw.clear]
        action = menu.exec_(QtGui.QCursor.pos())
        for name, cmd in zip(names, cmds):
            if action == name:
                cmd()

    def output_path_l_ctx(self):
        menu = QtWidgets.QMenu(self)
        names, paths = ['Clear path'], ['']
        names.append('Set to home directory')
        paths.append(get_documents_directory())
        actions = [menu.addAction(name) for name in names]
        action = menu.exec_(QtGui.QCursor.pos())
        for a, path in zip(actions, paths):
            if action == a:
                self.output_path_l.setText(path)

    def get_lw_items(self, ui_item):
        return [ui_item.item(i).text() for i in range(ui_item.count())]

    def del_lw_items(self, ui_item):
        items = ui_item.selectedItems()
        for item in items:
            ui_item.takeItem(ui_item.row(item))

    def play_lw_item(self, *args):
        data, sr = sf.read(args[0].data())
        sd.play(data, sr)

    def play_sweep_tone(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if os.path.isfile(self.ref_tone):
                data, sr = sf.read(self.ref_tone)
                sd.play(data, sr)

    def play_notification(self):
        audio_file = 'process_complete.flac'
        if os.path.isfile(audio_file):
            data, sr = sf.read(audio_file)
            sd.play(data, sr)

    def key_del_lw_items_event(self, event):
        if event.key() == 16777223:  # Key code for delete key
            items = self.files_lw.selectedItems()
            for item in items:
                self.files_lw.takeItem(self.files_lw.row(item))
        else:
            super().keyPressEvent(event)

    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def drag_move_event(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def lw_drop_event(self, event):
        if event.mimeData().hasUrls():
            items = event.mimeData().urls()
            items = [Path(item.toLocalFile()) for item in items]

            files = self.get_lw_items(self.files_lw)
            files.extend([item for item in items if item.suffix in self.file_types])

            dirs = [item for item in items if item.is_dir()]

            for d in dirs:
                for ext in self.file_types:
                    files.extend(Path(d).glob(f'*{ext}'))

            files = [os.path.normpath(f) for f in files]
            files = list(dict.fromkeys(files))

            self.files_lw.clear()
            self.files_lw.addItems(files)
            self.last_file = files[-1]
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
        self.w_cb.setToolTip('Apply raised cosine fade-in/fade-out to signal to avoid popping')
        self.w_cb.setHidden(True)

        lyt = self.layout()
        self.h_lyt = QtWidgets.QHBoxLayout()
        custom_lyt = self.h_lyt
        lyt.addWidget(QtWidgets.QWidget(self))
        lyt.addItem(self.h_lyt)

        custom_lyt.addWidget(self.length_l)
        custom_lyt.addWidget(self.length_dsb)
        custom_lyt.addWidget(self.sr_l)
        custom_lyt.addWidget(self.sr_sb)
        custom_lyt.addWidget(self.bd_l)
        custom_lyt.addWidget(self.bd_cmb)
        custom_lyt.addWidget(self.w_cb)


def get_documents_directory():
    if sys.platform == 'win32':
        import winreg
        # This works even if user profile has been moved to some other drive
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders') as key:
            path, _ = winreg.QueryValueEx(key, 'Personal')
    else:
        homepath = Path.home()
        path = homepath / 'Documents'
    return path


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


if __name__ == "__main__":
    myappid = f'mitch.ir_tool.{__version__}'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    font = app.font()
    font.setPointSize(12)
    app.setFont(font)

    window = IrToolUi()
    window.show()
    sys.exit(app.exec_())
