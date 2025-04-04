# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\mitch\Documents\PycharmProjects\github\ir_tool\UI\ir_tool.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ir_tool_mw(object):
    def setupUi(self, ir_tool_mw):
        ir_tool_mw.setObjectName("ir_tool_mw")
        ir_tool_mw.resize(600, 600)
        font = QtGui.QFont()
        ir_tool_mw.setFont(font)
        ir_tool_mw.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(ir_tool_mw)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ref_tone_title_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ref_tone_title_l.sizePolicy().hasHeightForWidth())
        self.ref_tone_title_l.setSizePolicy(sizePolicy)
        self.ref_tone_title_l.setMinimumSize(QtCore.QSize(64, 0))
        self.ref_tone_title_l.setStyleSheet("background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")
        self.ref_tone_title_l.setObjectName("ref_tone_title_l")
        self.verticalLayout.addWidget(self.ref_tone_title_l)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.ref_tone_path_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ref_tone_path_l.sizePolicy().hasHeightForWidth())
        self.ref_tone_path_l.setSizePolicy(sizePolicy)
        self.ref_tone_path_l.setText("")
        self.ref_tone_path_l.setObjectName("ref_tone_path_l")
        self.horizontalLayout.addWidget(self.ref_tone_path_l)
        self.set_ref_tone_tb = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.set_ref_tone_tb.setFont(font)
        self.set_ref_tone_tb.setObjectName("set_ref_tone_tb")
        self.horizontalLayout.addWidget(self.set_ref_tone_tb)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.gen_sweep_pb = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gen_sweep_pb.sizePolicy().hasHeightForWidth())
        self.gen_sweep_pb.setSizePolicy(sizePolicy)
        self.gen_sweep_pb.setMinimumSize(QtCore.QSize(120, 0))
        self.gen_sweep_pb.setStyleSheet("")
        self.gen_sweep_pb.setDefault(False)
        self.gen_sweep_pb.setObjectName("gen_sweep_pb")
        self.horizontalLayout_7.addWidget(self.gen_sweep_pb)
        self.gen_impulse_pb = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gen_impulse_pb.sizePolicy().hasHeightForWidth())
        self.gen_impulse_pb.setSizePolicy(sizePolicy)
        self.gen_impulse_pb.setMinimumSize(QtCore.QSize(120, 0))
        self.gen_impulse_pb.setStyleSheet("")
        self.gen_impulse_pb.setDefault(False)
        self.gen_impulse_pb.setObjectName("gen_impulse_pb")
        self.horizontalLayout_7.addWidget(self.gen_impulse_pb)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.files_title_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.files_title_l.sizePolicy().hasHeightForWidth())
        self.files_title_l.setSizePolicy(sizePolicy)
        self.files_title_l.setMinimumSize(QtCore.QSize(64, 0))
        self.files_title_l.setStyleSheet("background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")
        self.files_title_l.setObjectName("files_title_l")
        self.verticalLayout.addWidget(self.files_title_l)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.files_lw = QtWidgets.QListWidget(self.centralwidget)
        self.files_lw.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.files_lw.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.files_lw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.files_lw.setObjectName("files_lw")
        self.horizontalLayout_2.addWidget(self.files_lw)
        self.set_files_tb = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.set_files_tb.setFont(font)
        self.set_files_tb.setObjectName("set_files_tb")
        self.horizontalLayout_2.addWidget(self.set_files_tb)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.output_path_title_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_path_title_l.sizePolicy().hasHeightForWidth())
        self.output_path_title_l.setSizePolicy(sizePolicy)
        self.output_path_title_l.setMinimumSize(QtCore.QSize(64, 0))
        self.output_path_title_l.setStyleSheet("background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")
        self.output_path_title_l.setObjectName("output_path_title_l")
        self.verticalLayout.addWidget(self.output_path_title_l)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.output_path_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_path_l.sizePolicy().hasHeightForWidth())
        self.output_path_l.setSizePolicy(sizePolicy)
        self.output_path_l.setText("")
        self.output_path_l.setObjectName("output_path_l")
        self.horizontalLayout_3.addWidget(self.output_path_l)
        self.set_output_path_tb = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.set_output_path_tb.setFont(font)
        self.set_output_path_tb.setObjectName("set_output_path_tb")
        self.horizontalLayout_3.addWidget(self.set_output_path_tb)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.settings_title_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.settings_title_l.sizePolicy().hasHeightForWidth())
        self.settings_title_l.setSizePolicy(sizePolicy)
        self.settings_title_l.setMinimumSize(QtCore.QSize(64, 0))
        self.settings_title_l.setStyleSheet("background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")
        self.settings_title_l.setObjectName("settings_title_l")
        self.verticalLayout.addWidget(self.settings_title_l)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.deconv_cb = QtWidgets.QCheckBox(self.centralwidget)
        self.deconv_cb.setChecked(True)
        self.deconv_cb.setObjectName("deconv_cb")
        self.horizontalLayout_5.addWidget(self.deconv_cb)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.add_suffix_cb = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add_suffix_cb.sizePolicy().hasHeightForWidth())
        self.add_suffix_cb.setSizePolicy(sizePolicy)
        self.add_suffix_cb.setChecked(True)
        self.add_suffix_cb.setObjectName("add_suffix_cb")
        self.horizontalLayout_5.addWidget(self.add_suffix_cb)
        self.suffix_le = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.suffix_le.sizePolicy().hasHeightForWidth())
        self.suffix_le.setSizePolicy(sizePolicy)
        self.suffix_le.setMaxLength(16)
        self.suffix_le.setFrame(False)
        self.suffix_le.setObjectName("suffix_le")
        self.horizontalLayout_5.addWidget(self.suffix_le)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.no_overwriting_cb = QtWidgets.QCheckBox(self.centralwidget)
        self.no_overwriting_cb.setChecked(True)
        self.no_overwriting_cb.setObjectName("no_overwriting_cb")
        self.horizontalLayout_5.addWidget(self.no_overwriting_cb)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.bitdepth_cmb = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bitdepth_cmb.sizePolicy().hasHeightForWidth())
        self.bitdepth_cmb.setSizePolicy(sizePolicy)
        self.bitdepth_cmb.setFrame(False)
        self.bitdepth_cmb.setObjectName("bitdepth_cmb")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.horizontalLayout_4.addWidget(self.bitdepth_cmb)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.format_cmb = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.format_cmb.sizePolicy().hasHeightForWidth())
        self.format_cmb.setSizePolicy(sizePolicy)
        self.format_cmb.setFrame(False)
        self.format_cmb.setObjectName("format_cmb")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.horizontalLayout_4.addWidget(self.format_cmb)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.normalize_cb = QtWidgets.QCheckBox(self.centralwidget)
        self.normalize_cb.setObjectName("normalize_cb")
        self.horizontalLayout_4.addWidget(self.normalize_cb)
        self.normalize_cmb = QtWidgets.QComboBox(self.centralwidget)
        self.normalize_cmb.setEnabled(False)
        self.normalize_cmb.setMinimumSize(QtCore.QSize(96, 0))
        self.normalize_cmb.setFrame(False)
        self.normalize_cmb.setObjectName("normalize_cmb")
        self.normalize_cmb.addItem("")
        self.normalize_cmb.addItem("")
        self.horizontalLayout_4.addWidget(self.normalize_cmb)
        self.peak_db_dsb = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.peak_db_dsb.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.peak_db_dsb.sizePolicy().hasHeightForWidth())
        self.peak_db_dsb.setSizePolicy(sizePolicy)
        self.peak_db_dsb.setAcceptDrops(False)
        self.peak_db_dsb.setFrame(False)
        self.peak_db_dsb.setReadOnly(False)
        self.peak_db_dsb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.peak_db_dsb.setMinimum(-96.0)
        self.peak_db_dsb.setMaximum(0.0)
        self.peak_db_dsb.setProperty("value", -0.5)
        self.peak_db_dsb.setObjectName("peak_db_dsb")
        self.horizontalLayout_4.addWidget(self.peak_db_dsb)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.trim_cb = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trim_cb.sizePolicy().hasHeightForWidth())
        self.trim_cb.setSizePolicy(sizePolicy)
        self.trim_cb.setObjectName("trim_cb")
        self.horizontalLayout_6.addWidget(self.trim_cb)
        self.trim_db_dsb = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.trim_db_dsb.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trim_db_dsb.sizePolicy().hasHeightForWidth())
        self.trim_db_dsb.setSizePolicy(sizePolicy)
        self.trim_db_dsb.setAcceptDrops(False)
        self.trim_db_dsb.setFrame(False)
        self.trim_db_dsb.setReadOnly(False)
        self.trim_db_dsb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.trim_db_dsb.setMinimum(-138.0)
        self.trim_db_dsb.setMaximum(-48.0)
        self.trim_db_dsb.setProperty("value", -120.0)
        self.trim_db_dsb.setObjectName("trim_db_dsb")
        self.horizontalLayout_6.addWidget(self.trim_db_dsb)
        self.fadeout_cb = QtWidgets.QCheckBox(self.centralwidget)
        self.fadeout_cb.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fadeout_cb.sizePolicy().hasHeightForWidth())
        self.fadeout_cb.setSizePolicy(sizePolicy)
        self.fadeout_cb.setChecked(True)
        self.fadeout_cb.setObjectName("fadeout_cb")
        self.horizontalLayout_6.addWidget(self.fadeout_cb)
        self.fadeout_db_dsb = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.fadeout_db_dsb.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fadeout_db_dsb.sizePolicy().hasHeightForWidth())
        self.fadeout_db_dsb.setSizePolicy(sizePolicy)
        self.fadeout_db_dsb.setAcceptDrops(False)
        self.fadeout_db_dsb.setFrame(False)
        self.fadeout_db_dsb.setReadOnly(False)
        self.fadeout_db_dsb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.fadeout_db_dsb.setMinimum(-138.0)
        self.fadeout_db_dsb.setMaximum(-48.0)
        self.fadeout_db_dsb.setProperty("value", -90.0)
        self.fadeout_db_dsb.setObjectName("fadeout_db_dsb")
        self.horizontalLayout_6.addWidget(self.fadeout_db_dsb)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem4)
        self.process_pb = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.process_pb.sizePolicy().hasHeightForWidth())
        self.process_pb.setSizePolicy(sizePolicy)
        self.process_pb.setMinimumSize(QtCore.QSize(160, 0))
        font = QtGui.QFont()
        font.setBold(True)
        self.process_pb.setFont(font)
        self.process_pb.setDefault(False)
        self.process_pb.setObjectName("process_pb")
        self.horizontalLayout_8.addWidget(self.process_pb)
        self.stop_pb = QtWidgets.QPushButton(self.centralwidget)
        self.stop_pb.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stop_pb.sizePolicy().hasHeightForWidth())
        self.stop_pb.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        self.stop_pb.setFont(font)
        self.stop_pb.setDefault(False)
        self.stop_pb.setObjectName("stop_pb")
        self.horizontalLayout_8.addWidget(self.stop_pb)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem5)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.progress_pb = QtWidgets.QProgressBar(self.centralwidget)
        self.progress_pb.setStyleSheet("QProgressBar{border: none;}")
        self.progress_pb.setProperty("value", 0)
        self.progress_pb.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_pb.setFormat("")
        self.progress_pb.setObjectName("progress_pb")
        self.verticalLayout.addWidget(self.progress_pb)
        ir_tool_mw.setCentralWidget(self.centralwidget)

        self.retranslateUi(ir_tool_mw)
        self.bitdepth_cmb.setCurrentIndex(1)
        self.format_cmb.setCurrentIndex(1)
        self.normalize_cmb.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(ir_tool_mw)

    def retranslateUi(self, ir_tool_mw):
        _translate = QtCore.QCoreApplication.translate
        ir_tool_mw.setWindowTitle(_translate("ir_tool_mw", "IR Tool"))
        self.ref_tone_title_l.setText(_translate("ir_tool_mw", "Reference Tone"))
        self.ref_tone_path_l.setToolTip(_translate("ir_tool_mw", "Double-click to play"))
        self.set_ref_tone_tb.setToolTip(_translate("ir_tool_mw", "Set test tone to use for deconvolution"))
        self.set_ref_tone_tb.setText(_translate("ir_tool_mw", "..."))
        self.gen_sweep_pb.setToolTip(_translate("ir_tool_mw", "Generate a sweep to use as reference for recording and deconvolution"))
        self.gen_sweep_pb.setText(_translate("ir_tool_mw", "Generate Sweep"))
        self.gen_impulse_pb.setToolTip(_translate("ir_tool_mw", "Generate a single impulse click to use as source to capture an effect"))
        self.gen_impulse_pb.setText(_translate("ir_tool_mw", "Generate Impulse"))
        self.files_title_l.setText(_translate("ir_tool_mw", "Input Files"))
        self.files_lw.setToolTip(_translate("ir_tool_mw", "Drag and drop files or directories\n"
"Right click for context actions\n"
"Double-click to play"))
        self.set_files_tb.setToolTip(_translate("ir_tool_mw", "Set files to process"))
        self.set_files_tb.setText(_translate("ir_tool_mw", "..."))
        self.output_path_title_l.setText(_translate("ir_tool_mw", "Output Path"))
        self.set_output_path_tb.setToolTip(_translate("ir_tool_mw", "Set output path\n"
"Process files in their respective directory if empty"))
        self.set_output_path_tb.setText(_translate("ir_tool_mw", "..."))
        self.settings_title_l.setText(_translate("ir_tool_mw", "Settings"))
        self.deconv_cb.setToolTip(_translate("ir_tool_mw", "Enable deconvolution"))
        self.deconv_cb.setText(_translate("ir_tool_mw", "Deconvolve"))
        self.add_suffix_cb.setText(_translate("ir_tool_mw", "Add suffix"))
        self.suffix_le.setToolTip(_translate("ir_tool_mw", "Suffix added to the base name"))
        self.suffix_le.setText(_translate("ir_tool_mw", "_result"))
        self.no_overwriting_cb.setToolTip(_translate("ir_tool_mw", "Avoid overwriting original files by moving them to a backup directory"))
        self.no_overwriting_cb.setText(_translate("ir_tool_mw", "Avoid Overwriting"))
        self.label.setText(_translate("ir_tool_mw", "Bit Depth"))
        self.bitdepth_cmb.setItemText(0, _translate("ir_tool_mw", "16"))
        self.bitdepth_cmb.setItemText(1, _translate("ir_tool_mw", "24"))
        self.bitdepth_cmb.setItemText(2, _translate("ir_tool_mw", "32"))
        self.label_2.setText(_translate("ir_tool_mw", "Format"))
        self.format_cmb.setToolTip(_translate("ir_tool_mw", "flac only allows integer format up to 24 bits\n"
"Use wav or aif for 32 bits float"))
        self.format_cmb.setItemText(0, _translate("ir_tool_mw", "wav"))
        self.format_cmb.setItemText(1, _translate("ir_tool_mw", "flac"))
        self.format_cmb.setItemText(2, _translate("ir_tool_mw", "aif"))
        self.normalize_cb.setText(_translate("ir_tool_mw", "Normalize"))
        self.normalize_cmb.setToolTip(_translate("ir_tool_mw", "peak : Normalize to given dB\n"
"compensate : Modify IR volume so it does not change gain, 24 bits at least strongly recommanded"))
        self.normalize_cmb.setItemText(0, _translate("ir_tool_mw", "peak"))
        self.normalize_cmb.setItemText(1, _translate("ir_tool_mw", "compensate"))
        self.peak_db_dsb.setToolTip(_translate("ir_tool_mw", "Peak value in dB"))
        self.trim_cb.setToolTip(_translate("ir_tool_mw", "Enable trim end"))
        self.trim_cb.setText(_translate("ir_tool_mw", "Trim End"))
        self.trim_db_dsb.setToolTip(_translate("ir_tool_mw", "Volume treshold (in dB) under which the end is trimmed to minimize file lentgth"))
        self.fadeout_cb.setToolTip(_translate("ir_tool_mw", "Enable fade out"))
        self.fadeout_cb.setText(_translate("ir_tool_mw", "Fade Out"))
        self.fadeout_db_dsb.setToolTip(_translate("ir_tool_mw", "Volume treshold (in dB) from which a log fade out is applied preventing abrut end"))
        self.process_pb.setText(_translate("ir_tool_mw", "Process Files"))
        self.stop_pb.setText(_translate("ir_tool_mw", "Stop"))
