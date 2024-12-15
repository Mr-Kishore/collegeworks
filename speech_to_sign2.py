import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import pyttsx3
import speech_recognition as sr
import os

class SpeechRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Recognition App")
        self.setGeometry(100, 100, 800, 800)

        self.main_layout = QVBoxLayout()

        self.label = QLabel("Press the button and speak...", self)
        self.label.setFont(QFont("Arial", 12))
        self.main_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.button = QPushButton("Start", self)
        self.button.clicked.connect(self.start_recognition)
        self.main_layout.addWidget(self.button, alignment=Qt.AlignCenter)

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(500, 500)
        self.main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)

        # Directory containing the images
        self.image_directory = r"D:\code\Kishore\letters"

        self.video_frames = []
        self.current_frame_index = 0

    def start_recognition(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.label.setText("Listening...")
            self.label.repaint()
            audio = self.recognizer.listen(source)

            try:
                print("Recognizing...")
                self.label.setText("Recognizing...")
                self.label.repaint()
                text = self.recognizer.recognize_google(audio)
                print("Recognized text:", text)
                self.label.setText(f"Recognized text: {text}")
                self.label.repaint()
                self.engine.say("Recognized text: " + text)
                self.engine.runAndWait()
                self.create_video_from_text(text)
            except sr.UnknownValueError:
                print("Could not understand audio")
                self.label.setText("Could not understand audio")
                self.label.repaint()
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
                self.label.setText(f"Could not request results; {e}")
                self.label.repaint()

    def create_video_from_text(self, text):
        self.video_frames = []
        self.current_frame_index = 0

        # Convert text to lowercase and remove non-alphabetic characters
        text = ''.join(filter(str.isalpha, text.lower()))

        for char in text:
            image_path = os.path.join(self.image_directory, f"{char}.jpg")
            if os.path.exists(image_path):
                frame = cv2.imread(image_path)
                if frame is not None:
                    frame = cv2.resize(frame, (500, 500))
                    self.video_frames.append(frame)

        if self.video_frames:
            self.timer.start(500)  # Update frame every 500ms

    def update_video_frame(self):
        if self.video_frames:
            frame = self.video_frames[self.current_frame_index]
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
            self.current_frame_index = (self.current_frame_index + 1) % len(self.video_frames)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpeechRecognitionApp()
    window.show()
    sys.exit(app.exec_())
