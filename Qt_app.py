import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QCheckBox, QRadioButton, QSlider, QMessageBox, QSplitter, QButtonGroup
from PyQt5.QtCore import Qt
import tempfile
from video_pre_and_post_processing import process_video

class TrafficAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ToFD Traffic Analyser")
        self.setGeometry(100, 100, 800, 600)  # Set the initial window size

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_widget.setLayout(self.sidebar_layout)
        self.splitter.addWidget(self.sidebar_widget)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_layout)
        self.splitter.addWidget(self.content_widget)

        self.splitter.setSizes([200, 600])  # Initial sizes for sidebar and content area

        self.setup_sidebar()

        self.input_video_path = None
        self.pred_save_dir = None
        self.output_video_path = None
        self.save_prediction_in = None

    def setup_sidebar(self):
        self.sidebar_layout.addWidget(QLabel("Traffic Analysis"))

        self.input_video_label = QLabel("Upload Video")
        self.sidebar_layout.addWidget(self.input_video_label)
        self.input_video_button = QPushButton("Browse")
        self.input_video_button.clicked.connect(self.browse_video)
        self.sidebar_layout.addWidget(self.input_video_button)

        self.sav_prediction_checkbox = QCheckBox("Save Prediction")
        self.sav_prediction_checkbox.stateChanged.connect(self.show_prediction_options)
        self.sidebar_layout.addWidget(self.sav_prediction_checkbox)

        self.save_prediction_group = QVBoxLayout()
        self.csv_radio = QRadioButton("CSV")
        self.xlsx_radio = QRadioButton("Excel")
        self.csv_radio.toggled.connect(self.show_pred_save_dir_button)
        self.xlsx_radio.toggled.connect(self.show_pred_save_dir_button)
        self.save_prediction_group.addWidget(self.csv_radio)
        self.save_prediction_group.addWidget(self.xlsx_radio)
        self.sidebar_layout.addLayout(self.save_prediction_group)

        self.pred_save_dir_button = QPushButton("Select Prediction Save Folder")
        self.pred_save_dir_button.hide()
        self.pred_save_dir_button.clicked.connect(self.select_pred_save_folder)
        self.sidebar_layout.addWidget(self.pred_save_dir_button)

        self.save_video_checkbox = QCheckBox("Save Processed Video")
        self.save_video_checkbox.stateChanged.connect(self.show_video_save_folder)
        self.sidebar_layout.addWidget(self.save_video_checkbox)

        self.sidebar_layout.addWidget(QLabel("Select number of lines to be drawn"))

        self.draw_line_options = QVBoxLayout()
        self.line_group = QButtonGroup(self)
        self.draw_line_1 = QRadioButton("1 line")
        self.draw_line_2 = QRadioButton("2 lines")
        self.draw_line_1.setChecked(True)  # Set 1 line as the default option
        self.line_group.addButton(self.draw_line_1)
        self.line_group.addButton(self.draw_line_2)
        self.draw_line_options.addWidget(self.draw_line_1)
        self.draw_line_options.addWidget(self.draw_line_2)
        self.sidebar_layout.addLayout(self.draw_line_options)

        self.sidebar_layout.addWidget(QLabel("Set Speed"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(5)
        self.speed_slider.setValue(1)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        self.sidebar_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("Speed: 3")
        self.sidebar_layout.addWidget(self.speed_label)

        # Add confidence level slider
        self.sidebar_layout.addWidget(QLabel("Set Confidence Level"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(70)
        self.confidence_slider.setValue(25)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(5)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        self.sidebar_layout.addWidget(self.confidence_slider)

        self.confidence_label = QLabel("Confidence Level: 0.25")
        self.sidebar_layout.addWidget(self.confidence_label)

        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.process_video)
        self.sidebar_layout.addWidget(self.process_button)

        self.sidebar_layout.addStretch()  # Add stretch to push elements to the top

    def browse_video(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4 *.avi *.mov)", options=options)
        if fileName:
            self.input_video_label.setText(f"Uploaded Video: {fileName}")
            self.input_video_path = fileName

    def show_prediction_options(self, state):
        if state == Qt.Checked:
            self.csv_radio.setEnabled(True)
            self.xlsx_radio.setEnabled(True)
        else:
            self.csv_radio.setEnabled(False)
            self.xlsx_radio.setEnabled(False)
            self.pred_save_dir_button.hide()

    def show_pred_save_dir_button(self):
        if self.csv_radio.isChecked() or self.xlsx_radio.isChecked():
            self.pred_save_dir_button.show()
        else:
            self.pred_save_dir_button.hide()

    def show_video_save_folder(self, state):
        if state == Qt.Checked:
            options = QFileDialog.Options()
            folderName = QFileDialog.getExistingDirectory(self, "Select Processed Video Save Folder", options=options)
            if folderName:
                self.output_video_path = folderName
                self.save_video_checkbox.setText(f"Save Processed Video: {folderName}")
        else:
            self.output_video_path = None
            self.save_video_checkbox.setText("Save Processed Video")

    def select_pred_save_folder(self):
        options = QFileDialog.Options()
        folderName = QFileDialog.getExistingDirectory(self, "Select Prediction Save Folder", options=options)
        if folderName:
            self.pred_save_dir = folderName
            self.pred_save_dir_button.setText(f"Selected Folder for Predictions: {folderName}")

    def update_speed_label(self, value):
        self.speed_label.setText(f"Speed: {value}")

    def update_confidence_label(self, value):
        confidence = value / 100
        self.confidence_label.setText(f"Confidence Level: {confidence:.2f}")

    def process_video(self):
        if self.input_video_path is None:
            QMessageBox.warning(self, "Warning", "Please upload a video file first.")
            return

        draw_line_option = 1 if self.draw_line_1.isChecked() else 2
        sav_prediction = self.sav_prediction_checkbox.isChecked()
        save_video = self.save_video_checkbox.isChecked()
        video_speed = self.speed_slider.value()
        confidence_level = self.confidence_slider.value() / 100

        save_prediction_in = None  # Default to None if no option is selected

        if sav_prediction:
            if self.csv_radio.isChecked():
                save_prediction_in = "csv"
            elif self.xlsx_radio.isChecked():
                save_prediction_in = "excel"
            
            if save_prediction_in is None:
                QMessageBox.warning(self, "Warning", "Please select a prediction save format (CSV or Excel).")
                return

            if self.pred_save_dir is None:
                QMessageBox.warning(self, "Warning", "Please select a folder to save predictions.")
                return

        if save_video and self.output_video_path is None:
            QMessageBox.warning(self, "Warning", "Please select a folder to save the processed video.")
            return

        # Implement video processing logic here
        # Replace this with your actual processing logic
        process_video(
            self.input_video_path, draw_line_option, sav_prediction, save_prediction_in, save_video,
            video_speed, self.pred_save_dir, self.output_video_path, confidence_level
        )

        QMessageBox.information(self, "Success", "Video processing started successfully.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficAnalysisApp()
    window.show()
    sys.exit(app.exec_())
