# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ir_tool.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox,
    QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QProgressBar, QPushButton,
    QSizePolicy, QSpacerItem, QToolButton, QVBoxLayout,
    QWidget)

class Ui_ir_tool_mw(object):
    def setupUi(self, ir_tool_mw):
        if not ir_tool_mw.objectName():
            ir_tool_mw.setObjectName(u"ir_tool_mw")
        ir_tool_mw.resize(640, 640)
        font = QFont()
        ir_tool_mw.setFont(font)
        ir_tool_mw.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.centralwidget = QWidget(ir_tool_mw)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.ref_tone_title_l = QLabel(self.centralwidget)
        self.ref_tone_title_l.setObjectName(u"ref_tone_title_l")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ref_tone_title_l.sizePolicy().hasHeightForWidth())
        self.ref_tone_title_l.setSizePolicy(sizePolicy)
        self.ref_tone_title_l.setMinimumSize(QSize(64, 0))
        self.ref_tone_title_l.setStyleSheet(u"background-color: rgb(63, 95, 127);color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.ref_tone_title_l)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.ref_tone_path_l = QLabel(self.centralwidget)
        self.ref_tone_path_l.setObjectName(u"ref_tone_path_l")
        sizePolicy.setHeightForWidth(self.ref_tone_path_l.sizePolicy().hasHeightForWidth())
        self.ref_tone_path_l.setSizePolicy(sizePolicy)

        self.horizontalLayout.addWidget(self.ref_tone_path_l)

        self.set_ref_tone_tb = QToolButton(self.centralwidget)
        self.set_ref_tone_tb.setObjectName(u"set_ref_tone_tb")
        font1 = QFont()
        font1.setBold(True)
        self.set_ref_tone_tb.setFont(font1)

        self.horizontalLayout.addWidget(self.set_ref_tone_tb)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(4)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.gen_sweep_pb = QPushButton(self.centralwidget)
        self.gen_sweep_pb.setObjectName(u"gen_sweep_pb")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.gen_sweep_pb.sizePolicy().hasHeightForWidth())
        self.gen_sweep_pb.setSizePolicy(sizePolicy1)
        self.gen_sweep_pb.setMinimumSize(QSize(160, 24))
        self.gen_sweep_pb.setStyleSheet(u"")

        self.horizontalLayout_7.addWidget(self.gen_sweep_pb)

        self.gen_impulse_pb = QPushButton(self.centralwidget)
        self.gen_impulse_pb.setObjectName(u"gen_impulse_pb")
        sizePolicy1.setHeightForWidth(self.gen_impulse_pb.sizePolicy().hasHeightForWidth())
        self.gen_impulse_pb.setSizePolicy(sizePolicy1)
        self.gen_impulse_pb.setMinimumSize(QSize(160, 24))
        self.gen_impulse_pb.setStyleSheet(u"")

        self.horizontalLayout_7.addWidget(self.gen_impulse_pb)


        self.verticalLayout.addLayout(self.horizontalLayout_7)

        self.files_title_l = QLabel(self.centralwidget)
        self.files_title_l.setObjectName(u"files_title_l")
        sizePolicy.setHeightForWidth(self.files_title_l.sizePolicy().hasHeightForWidth())
        self.files_title_l.setSizePolicy(sizePolicy)
        self.files_title_l.setMinimumSize(QSize(64, 0))
        self.files_title_l.setStyleSheet(u"background-color: rgb(63, 95, 127);color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.files_title_l)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.files_lw = QListWidget(self.centralwidget)
        self.files_lw.setObjectName(u"files_lw")
        self.files_lw.setFrameShape(QFrame.NoFrame)
        self.files_lw.setDragDropMode(QAbstractItemView.DropOnly)
        self.files_lw.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.horizontalLayout_2.addWidget(self.files_lw)

        self.set_files_tb = QToolButton(self.centralwidget)
        self.set_files_tb.setObjectName(u"set_files_tb")
        self.set_files_tb.setFont(font1)

        self.horizontalLayout_2.addWidget(self.set_files_tb)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.output_path_title_l = QLabel(self.centralwidget)
        self.output_path_title_l.setObjectName(u"output_path_title_l")
        sizePolicy.setHeightForWidth(self.output_path_title_l.sizePolicy().hasHeightForWidth())
        self.output_path_title_l.setSizePolicy(sizePolicy)
        self.output_path_title_l.setMinimumSize(QSize(64, 0))
        self.output_path_title_l.setStyleSheet(u"background-color: rgb(63, 95, 127);color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.output_path_title_l)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.output_path_l = QLabel(self.centralwidget)
        self.output_path_l.setObjectName(u"output_path_l")
        sizePolicy.setHeightForWidth(self.output_path_l.sizePolicy().hasHeightForWidth())
        self.output_path_l.setSizePolicy(sizePolicy)

        self.horizontalLayout_3.addWidget(self.output_path_l)

        self.set_output_path_tb = QToolButton(self.centralwidget)
        self.set_output_path_tb.setObjectName(u"set_output_path_tb")
        self.set_output_path_tb.setFont(font1)

        self.horizontalLayout_3.addWidget(self.set_output_path_tb)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.settings_title_l = QLabel(self.centralwidget)
        self.settings_title_l.setObjectName(u"settings_title_l")
        sizePolicy.setHeightForWidth(self.settings_title_l.sizePolicy().hasHeightForWidth())
        self.settings_title_l.setSizePolicy(sizePolicy)
        self.settings_title_l.setMinimumSize(QSize(64, 0))
        self.settings_title_l.setStyleSheet(u"background-color: rgb(63, 95, 127);color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.settings_title_l)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.deconv_cb = QCheckBox(self.centralwidget)
        self.deconv_cb.setObjectName(u"deconv_cb")
        self.deconv_cb.setChecked(True)

        self.horizontalLayout_5.addWidget(self.deconv_cb)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)

        self.add_suffix_cb = QCheckBox(self.centralwidget)
        self.add_suffix_cb.setObjectName(u"add_suffix_cb")
        sizePolicy1.setHeightForWidth(self.add_suffix_cb.sizePolicy().hasHeightForWidth())
        self.add_suffix_cb.setSizePolicy(sizePolicy1)
        self.add_suffix_cb.setChecked(True)

        self.horizontalLayout_5.addWidget(self.add_suffix_cb)

        self.suffix_le = QLineEdit(self.centralwidget)
        self.suffix_le.setObjectName(u"suffix_le")
        sizePolicy1.setHeightForWidth(self.suffix_le.sizePolicy().hasHeightForWidth())
        self.suffix_le.setSizePolicy(sizePolicy1)
        self.suffix_le.setMaxLength(16)
        self.suffix_le.setFrame(False)

        self.horizontalLayout_5.addWidget(self.suffix_le)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)

        self.no_overwriting_cb = QCheckBox(self.centralwidget)
        self.no_overwriting_cb.setObjectName(u"no_overwriting_cb")
        self.no_overwriting_cb.setChecked(True)

        self.horizontalLayout_5.addWidget(self.no_overwriting_cb)


        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.bitdepth_l = QLabel(self.centralwidget)
        self.bitdepth_l.setObjectName(u"bitdepth_l")
        sizePolicy1.setHeightForWidth(self.bitdepth_l.sizePolicy().hasHeightForWidth())
        self.bitdepth_l.setSizePolicy(sizePolicy1)

        self.horizontalLayout_4.addWidget(self.bitdepth_l)

        self.bitdepth_cmb = QComboBox(self.centralwidget)
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.setObjectName(u"bitdepth_cmb")
        sizePolicy1.setHeightForWidth(self.bitdepth_cmb.sizePolicy().hasHeightForWidth())
        self.bitdepth_cmb.setSizePolicy(sizePolicy1)
        self.bitdepth_cmb.setFrame(False)

        self.horizontalLayout_4.addWidget(self.bitdepth_cmb)

        self.format_l = QLabel(self.centralwidget)
        self.format_l.setObjectName(u"format_l")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.format_l.sizePolicy().hasHeightForWidth())
        self.format_l.setSizePolicy(sizePolicy2)

        self.horizontalLayout_4.addWidget(self.format_l)

        self.format_cmb = QComboBox(self.centralwidget)
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.setObjectName(u"format_cmb")
        sizePolicy1.setHeightForWidth(self.format_cmb.sizePolicy().hasHeightForWidth())
        self.format_cmb.setSizePolicy(sizePolicy1)
        self.format_cmb.setFrame(False)

        self.horizontalLayout_4.addWidget(self.format_cmb)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)

        self.normalize_cb = QCheckBox(self.centralwidget)
        self.normalize_cb.setObjectName(u"normalize_cb")

        self.horizontalLayout_4.addWidget(self.normalize_cb)

        self.normalize_wid = QWidget(self.centralwidget)
        self.normalize_wid.setObjectName(u"normalize_wid")
        self.normalize_wid.setEnabled(False)
        self.horizontalLayout_9 = QHBoxLayout(self.normalize_wid)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.normalize_cmb = QComboBox(self.normalize_wid)
        self.normalize_cmb.addItem("")
        self.normalize_cmb.addItem("")
        self.normalize_cmb.setObjectName(u"normalize_cmb")
        self.normalize_cmb.setMinimumSize(QSize(96, 0))
        self.normalize_cmb.setFrame(False)

        self.horizontalLayout_9.addWidget(self.normalize_cmb)

        self.normalize_amp_dsb = QDoubleSpinBox(self.normalize_wid)
        self.normalize_amp_dsb.setObjectName(u"normalize_amp_dsb")
        sizePolicy1.setHeightForWidth(self.normalize_amp_dsb.sizePolicy().hasHeightForWidth())
        self.normalize_amp_dsb.setSizePolicy(sizePolicy1)
        self.normalize_amp_dsb.setAcceptDrops(False)
        self.normalize_amp_dsb.setFrame(False)
        self.normalize_amp_dsb.setAlignment(Qt.AlignCenter)
        self.normalize_amp_dsb.setReadOnly(False)
        self.normalize_amp_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.normalize_amp_dsb.setMinimum(-96.000000000000000)
        self.normalize_amp_dsb.setMaximum(12.000000000000000)
        self.normalize_amp_dsb.setValue(0.000000000000000)

        self.horizontalLayout_9.addWidget(self.normalize_amp_dsb)


        self.horizontalLayout_4.addWidget(self.normalize_wid)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.trim_cb = QCheckBox(self.centralwidget)
        self.trim_cb.setObjectName(u"trim_cb")
        sizePolicy1.setHeightForWidth(self.trim_cb.sizePolicy().hasHeightForWidth())
        self.trim_cb.setSizePolicy(sizePolicy1)

        self.horizontalLayout_6.addWidget(self.trim_cb)

        self.trim_db_dsb = QDoubleSpinBox(self.centralwidget)
        self.trim_db_dsb.setObjectName(u"trim_db_dsb")
        self.trim_db_dsb.setEnabled(False)
        sizePolicy1.setHeightForWidth(self.trim_db_dsb.sizePolicy().hasHeightForWidth())
        self.trim_db_dsb.setSizePolicy(sizePolicy1)
        self.trim_db_dsb.setAcceptDrops(False)
        self.trim_db_dsb.setFrame(False)
        self.trim_db_dsb.setAlignment(Qt.AlignCenter)
        self.trim_db_dsb.setReadOnly(False)
        self.trim_db_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.trim_db_dsb.setMinimum(-138.000000000000000)
        self.trim_db_dsb.setMaximum(-48.000000000000000)
        self.trim_db_dsb.setValue(-120.000000000000000)

        self.horizontalLayout_6.addWidget(self.trim_db_dsb)

        self.fadeout_cb = QCheckBox(self.centralwidget)
        self.fadeout_cb.setObjectName(u"fadeout_cb")
        self.fadeout_cb.setEnabled(False)
        sizePolicy1.setHeightForWidth(self.fadeout_cb.sizePolicy().hasHeightForWidth())
        self.fadeout_cb.setSizePolicy(sizePolicy1)
        self.fadeout_cb.setChecked(True)

        self.horizontalLayout_6.addWidget(self.fadeout_cb)

        self.fadeout_db_dsb = QDoubleSpinBox(self.centralwidget)
        self.fadeout_db_dsb.setObjectName(u"fadeout_db_dsb")
        self.fadeout_db_dsb.setEnabled(False)
        sizePolicy1.setHeightForWidth(self.fadeout_db_dsb.sizePolicy().hasHeightForWidth())
        self.fadeout_db_dsb.setSizePolicy(sizePolicy1)
        self.fadeout_db_dsb.setAcceptDrops(False)
        self.fadeout_db_dsb.setFrame(False)
        self.fadeout_db_dsb.setAlignment(Qt.AlignCenter)
        self.fadeout_db_dsb.setReadOnly(False)
        self.fadeout_db_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fadeout_db_dsb.setMinimum(-138.000000000000000)
        self.fadeout_db_dsb.setMaximum(-48.000000000000000)
        self.fadeout_db_dsb.setValue(-90.000000000000000)

        self.horizontalLayout_6.addWidget(self.fadeout_db_dsb)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.buttons_lyt = QGridLayout()
        self.buttons_lyt.setSpacing(4)
        self.buttons_lyt.setObjectName(u"buttons_lyt")
        self.process_sel_pb = QPushButton(self.centralwidget)
        self.process_sel_pb.setObjectName(u"process_sel_pb")
        sizePolicy1.setHeightForWidth(self.process_sel_pb.sizePolicy().hasHeightForWidth())
        self.process_sel_pb.setSizePolicy(sizePolicy1)
        self.process_sel_pb.setMinimumSize(QSize(160, 32))
        self.process_sel_pb.setFont(font1)

        self.buttons_lyt.addWidget(self.process_sel_pb, 0, 0, 1, 1)

        self.stop_pb = QPushButton(self.centralwidget)
        self.stop_pb.setObjectName(u"stop_pb")
        self.stop_pb.setEnabled(False)
        sizePolicy1.setHeightForWidth(self.stop_pb.sizePolicy().hasHeightForWidth())
        self.stop_pb.setSizePolicy(sizePolicy1)
        self.stop_pb.setMinimumSize(QSize(160, 32))
        self.stop_pb.setFont(font1)

        self.buttons_lyt.addWidget(self.stop_pb, 0, 1, 1, 1)

        self.process_pb = QPushButton(self.centralwidget)
        self.process_pb.setObjectName(u"process_pb")
        sizePolicy1.setHeightForWidth(self.process_pb.sizePolicy().hasHeightForWidth())
        self.process_pb.setSizePolicy(sizePolicy1)
        self.process_pb.setMinimumSize(QSize(160, 32))
        self.process_pb.setFont(font1)
        self.process_pb.setStyleSheet(u"QPushButton{background-color: rgb(63, 95, 127);color: rgb(255, 255, 255);}")

        self.buttons_lyt.addWidget(self.process_pb, 1, 0, 1, 1)


        self.verticalLayout.addLayout(self.buttons_lyt)

        self.progress_pb = QProgressBar(self.centralwidget)
        self.progress_pb.setObjectName(u"progress_pb")
        self.progress_pb.setStyleSheet(u"QProgressBar{border: none;}")
        self.progress_pb.setValue(0)
        self.progress_pb.setAlignment(Qt.AlignCenter)
        self.progress_pb.setFormat(u"")

        self.verticalLayout.addWidget(self.progress_pb)

        ir_tool_mw.setCentralWidget(self.centralwidget)

        self.retranslateUi(ir_tool_mw)

        self.gen_sweep_pb.setDefault(False)
        self.gen_impulse_pb.setDefault(False)
        self.bitdepth_cmb.setCurrentIndex(1)
        self.format_cmb.setCurrentIndex(1)
        self.normalize_cmb.setCurrentIndex(1)
        self.process_sel_pb.setDefault(False)
        self.stop_pb.setDefault(False)
        self.process_pb.setDefault(False)


        QMetaObject.connectSlotsByName(ir_tool_mw)
    # setupUi

    def retranslateUi(self, ir_tool_mw):
        ir_tool_mw.setWindowTitle(QCoreApplication.translate("ir_tool_mw", u"IR Tool", None))
        self.ref_tone_title_l.setText(QCoreApplication.translate("ir_tool_mw", u"Reference Tone", None))
#if QT_CONFIG(tooltip)
        self.ref_tone_path_l.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Double-click to play", None))
#endif // QT_CONFIG(tooltip)
        self.ref_tone_path_l.setText("")
#if QT_CONFIG(tooltip)
        self.set_ref_tone_tb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Set test tone to use for deconvolution", None))
#endif // QT_CONFIG(tooltip)
        self.set_ref_tone_tb.setText(QCoreApplication.translate("ir_tool_mw", u"...", None))
#if QT_CONFIG(tooltip)
        self.gen_sweep_pb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Generate a sweep to use as reference for recording and deconvolution", None))
#endif // QT_CONFIG(tooltip)
        self.gen_sweep_pb.setText(QCoreApplication.translate("ir_tool_mw", u"Generate Sweep", None))
#if QT_CONFIG(tooltip)
        self.gen_impulse_pb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Generate a single impulse click to use as source to capture an effect", None))
#endif // QT_CONFIG(tooltip)
        self.gen_impulse_pb.setText(QCoreApplication.translate("ir_tool_mw", u"Generate Impulse", None))
        self.files_title_l.setText(QCoreApplication.translate("ir_tool_mw", u"Input Files", None))
#if QT_CONFIG(tooltip)
        self.files_lw.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Drag and drop files or directories\n"
"Right click for context actions\n"
"Double-click to play", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.set_files_tb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Set files to process", None))
#endif // QT_CONFIG(tooltip)
        self.set_files_tb.setText(QCoreApplication.translate("ir_tool_mw", u"...", None))
        self.output_path_title_l.setText(QCoreApplication.translate("ir_tool_mw", u"Output Path", None))
        self.output_path_l.setText("")
#if QT_CONFIG(tooltip)
        self.set_output_path_tb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Set output path\n"
"Process files in their respective directory if empty", None))
#endif // QT_CONFIG(tooltip)
        self.set_output_path_tb.setText(QCoreApplication.translate("ir_tool_mw", u"...", None))
        self.settings_title_l.setText(QCoreApplication.translate("ir_tool_mw", u"Settings", None))
#if QT_CONFIG(tooltip)
        self.deconv_cb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Enable deconvolution", None))
#endif // QT_CONFIG(tooltip)
        self.deconv_cb.setText(QCoreApplication.translate("ir_tool_mw", u"Deconvolve", None))
        self.add_suffix_cb.setText(QCoreApplication.translate("ir_tool_mw", u"Add suffix", None))
#if QT_CONFIG(tooltip)
        self.suffix_le.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Suffix added to the base name", None))
#endif // QT_CONFIG(tooltip)
        self.suffix_le.setText(QCoreApplication.translate("ir_tool_mw", u"_result", None))
#if QT_CONFIG(tooltip)
        self.no_overwriting_cb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Avoid overwriting original files by moving them to a backup directory", None))
#endif // QT_CONFIG(tooltip)
        self.no_overwriting_cb.setText(QCoreApplication.translate("ir_tool_mw", u"Avoid Overwriting", None))
        self.bitdepth_l.setText(QCoreApplication.translate("ir_tool_mw", u"Bit Depth", None))
        self.bitdepth_cmb.setItemText(0, QCoreApplication.translate("ir_tool_mw", u"16", None))
        self.bitdepth_cmb.setItemText(1, QCoreApplication.translate("ir_tool_mw", u"24", None))
        self.bitdepth_cmb.setItemText(2, QCoreApplication.translate("ir_tool_mw", u"32", None))

        self.format_l.setText(QCoreApplication.translate("ir_tool_mw", u"Format", None))
        self.format_cmb.setItemText(0, QCoreApplication.translate("ir_tool_mw", u"wav", None))
        self.format_cmb.setItemText(1, QCoreApplication.translate("ir_tool_mw", u"flac", None))
        self.format_cmb.setItemText(2, QCoreApplication.translate("ir_tool_mw", u"aif", None))

#if QT_CONFIG(tooltip)
        self.format_cmb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"flac only allows integer format up to 24 bits\n"
"Use wav or aif for 32 bits float", None))
#endif // QT_CONFIG(tooltip)
        self.normalize_cb.setText(QCoreApplication.translate("ir_tool_mw", u"Normalize", None))
        self.normalize_cmb.setItemText(0, QCoreApplication.translate("ir_tool_mw", u"peak", None))
        self.normalize_cmb.setItemText(1, QCoreApplication.translate("ir_tool_mw", u"compensate", None))

#if QT_CONFIG(tooltip)
        self.normalize_cmb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"peak	normalize to given dB\n"
"\n"
"compensate	volume compensation computed from the RMS volume of the IR applied to a frequency sweep\n"
"			An optional amplification can be applied for softwares using a non-editable attenuation\n"
"			24 bits at least strongly recommended", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.normalize_amp_dsb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"peak	Peak volume in dB\n"
"compensate	Amplification in dB", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.trim_cb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Enable trim end", None))
#endif // QT_CONFIG(tooltip)
        self.trim_cb.setText(QCoreApplication.translate("ir_tool_mw", u"Trim End", None))
#if QT_CONFIG(tooltip)
        self.trim_db_dsb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Volume treshold (in dB) under which the end is trimmed to minimize file length", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fadeout_cb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Enable fade out", None))
#endif // QT_CONFIG(tooltip)
        self.fadeout_cb.setText(QCoreApplication.translate("ir_tool_mw", u"Fade Out", None))
#if QT_CONFIG(tooltip)
        self.fadeout_db_dsb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Volume treshold (in dB) from which a log fade out is applied preventing abrut end", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.process_sel_pb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Process selected item(s) with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_sel_pb.setText(QCoreApplication.translate("ir_tool_mw", u"Process Selected", None))
#if QT_CONFIG(tooltip)
        self.stop_pb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Interrupt Process", None))
#endif // QT_CONFIG(tooltip)
        self.stop_pb.setText(QCoreApplication.translate("ir_tool_mw", u"STOP", None))
#if QT_CONFIG(tooltip)
        self.process_pb.setToolTip(QCoreApplication.translate("ir_tool_mw", u"Process all input files with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_pb.setText(QCoreApplication.translate("ir_tool_mw", u"Batch Process", None))
    # retranslateUi

