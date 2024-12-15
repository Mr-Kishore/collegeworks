import sys
import cv2
import os
import numpy as np
import string
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QLineEdit
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import pyttsx3
import speech_recognition as sr

# RNN Model
class RNNPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        return self.fc(out[:, -1, :]), hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Train and save the RNN model
def train_rnn_model():
    input_size = len(string.ascii_lowercase)
    hidden_size = 128
    output_size = len(string.ascii_lowercase)
    model = RNNPredictor(input_size, hidden_size, output_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Training loop: Iterates over each character independently
    for _ in range(100):  # Short training for demonstration
        for idx, char in enumerate(string.ascii_lowercase):
            # Prepare one-hot encoded input for the character
            input_tensor = torch.zeros(1, 1, input_size)
            input_tensor[0, 0, idx] = 1.0

            # The target is the index of the character in the alphabet
            target_tensor = torch.tensor([idx])

            hidden = model.init_hidden()

            # Forward pass
            optimizer.zero_grad()
            output, hidden = model(input_tensor, hidden)
            loss = criterion(output, target_tensor)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), "rnn_model.pth")


# Train and save the model if it doesn't exist
if not os.path.exists("rnn_model.pth"):
    train_rnn_model()

# Main App
class SpeechRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech & Text to Sign Language App with RNN")
        self.setGeometry(100, 100, 800, 800)

        # Layout and Widgets
        layout = QVBoxLayout()
        self.label = QLabel("Choose input mode: Speech or Text", self)
        self.label.setFont(QFont("Arial", 12))
        layout.addWidget(self.label, alignment=Qt.AlignCenter)
        
        self.speech_button = QPushButton("Speech", self)
        self.speech_button.clicked.connect(self.start_speech_recognition)
        layout.addWidget(self.speech_button, alignment=Qt.AlignCenter)
        
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Enter text here...")
        layout.addWidget(self.text_input, alignment=Qt.AlignCenter)
        
        self.text_button = QPushButton("Process Text", self)
        self.text_button.clicked.connect(self.start_text_processing)
        layout.addWidget(self.text_button, alignment=Qt.AlignCenter)
        
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(500, 500)
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Init TTS, Speech Recognizer, Timer
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        
        # Load RNN Model
        self.model = RNNPredictor(input_size=len(string.ascii_lowercase), hidden_size=128, output_size=len(string.ascii_lowercase))
        self.model.load_state_dict(torch.load("rnn_model.pth", weights_only=True))

        self.model.eval()

        # Video Frame Vars
        self.video_frames = []
        self.current_frame_index = 0
        self.image_directory = r"D:\code\Kishore\letters"  # Update with actual path

    def start_speech_recognition(self):
        with sr.Microphone() as source:
            self.label.setText("Listening...")
            audio = self.recognizer.listen(source)

            try:
                text = self.recognizer.recognize_google(audio)
                predicted_text = self.predict_next_characters(text)
                self.label.setText(f"Recognized text: {text} | Predicted: {predicted_text}")
                self.engine.say("Recognized text: " + text)
                self.engine.runAndWait()
                self.create_video_from_text(predicted_text)
            except sr.UnknownValueError:
                self.label.setText("Could not understand audio")
            except sr.RequestError as e:
                self.label.setText(f"Request error: {e}")

    def start_text_processing(self):
        text = self.text_input.text()
        predicted_text = self.predict_next_characters(text)
        self.label.setText(f"Processing text: {text} | Predicted: {predicted_text}")
        self.create_video_from_text(predicted_text)

    def predict_next_characters(self, text, num_predictions=5):
        hidden = self.model.init_hidden()
        predictions = []
        input_text = text[-1].lower() if text else ''

        for _ in range(num_predictions):
            input_tensor = torch.zeros(1, 1, len(string.ascii_lowercase))
            if input_text in string.ascii_lowercase:
                input_tensor[0, 0, string.ascii_lowercase.index(input_text)] = 1.0

            output, hidden = self.model(input_tensor, hidden)
            pred_index = output.argmax().item()
            pred_char = string.ascii_lowercase[pred_index]
            predictions.append(pred_char)
            input_text = pred_char

        return text + ''.join(predictions)

    def create_video_from_text(self, text):
        self.video_frames = []
        self.current_frame_index = 0
        text = ''.join(filter(str.isalpha, text.lower()))

        char_count = {}
        for char in text:
            if char_count.get(char, 0) < 3:
                image_path = os.path.join(self.image_directory, f"{char}.jpg")
                if os.path.exists(image_path):
                    frame = cv2.imread(image_path)
                    if frame is not None:
                        frame = cv2.resize(frame, (500, 500))
                        self.video_frames.append(frame)
                        char_count[char] = char_count.get(char, 0) + 1

        if self.video_frames:
            self.timer.start(500)

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
