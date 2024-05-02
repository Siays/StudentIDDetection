import tkinter as tk
import threading
from tkinter import ttk

import DetectronPredict
import SSDModelDetection
import YoloV8Detection  # Import the YoloV8Detection module


class ModelRunner:
    def __init__(self, model_name, ocr_display):
        self.model_name = model_name
        self.ocr_display = ocr_display
        self.running = False
        self.thread = None

    def run_model(self):
        self.running = True
        while self.running:
            # Start the appropriate model based on self.model_name
            if self.model_name == "Yolo v8":
                YoloV8Detection.yoloV8_detect(self.ocr_display)
            elif self.model_name == "Detectron 2":
                DetectronPredict.detectronDetect(self.ocr_display)
            elif self.model_name == "SSD MobileNet V2 FPNLite 320x320":
                SSDModelDetection.ssdModel_detect(self.ocr_display)
            stop_model(self)

    def stop_model(self):
        self.running = False


def start_model(model_runner):
    if model_runner.thread is None or not model_runner.thread.is_alive():
        model_runner.ocr_display.set("")  # Clear OCR display when starting a model
        model_runner.thread = threading.Thread(target=model_runner.run_model)
        model_runner.thread.start()
        disable_buttons(exclude_button=model_buttons[model_runners.index(model_runner)])


def stop_model(model_runner):
    model_runner.stop_model()
    enable_buttons()


def disable_buttons(exclude_button):
    for button in model_buttons:
        if button != exclude_button:
            button.config(state=tk.DISABLED)


def enable_buttons():
    for button in model_buttons:
        button.config(state=tk.NORMAL)


def key_pressed(event):
    if event.char == "q":
        enable_buttons()  # Enable model buttons when "q" is pressed
        # Stop all models
        for model_runner in model_runners:
            stop_model(model_runner)


root = tk.Tk()
root.title("Model GUI")
root.geometry("850x650")  # Set the default window size

# Create a frame for the displaying the instructions
instructions_frame = tk.Frame(root)
instructions_frame.grid(row=0, column=0, pady=10)  # Add some vertical padding

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.grid(row=1, column=0, pady=10)  # Add some vertical padding

# Create a frame for the displaying the results
results_frame = tk.Frame(root)
results_frame.grid(row=2, column=0, padx=5, pady=5)  # Add some vertical and horizontal padding

# Create a frame for the exit button
exit_button_frame = tk.Frame(root)
exit_button_frame.grid(row=3, column=0, pady=10)  # Add some vertical padding

# Add a label for displaying the instructions
instructions_title = tk.Label(instructions_frame, font=('Helvetica', 12),
                              text="Instructions:\nClick on a button to run that model.\nPress "
                                   "'q' to stop the running model.\nPress 'esc' or click the "
                                   "\"Exit\" button to exit the "
                                   "program.")
instructions_title.pack()

# Set the weight of the row and column in the root window to allow the results_frame to expand
root.rowconfigure(2, weight=2)
root.columnconfigure(0, weight=1)

# Set the weight of the rows and columns in the results_frame to allow them to expand vertically and horizontally
results_frame.rowconfigure(1, weight=1)
results_frame.columnconfigure(0, weight=1)
results_frame.columnconfigure(1, weight=1)

# Create text box for displaying last validated results
validated_display = tk.StringVar()
validated_display.set("")

# Add a label above the validated results display
validated_display_title = tk.Label(results_frame, text="Last validated Student ID:", font=('Helvetica', 12))
validated_display_title.grid(row=0, column=0, padx=5, pady=5)  # Add some vertical and horizontal padding

# Create a box for the validated results display with an outline
validated_display_box = tk.Label(results_frame, textvariable=validated_display, relief="solid", width=55,
                                 height=120)  # Set the default size of the display box
validated_display_box.grid(row=1, column=0, padx=5, pady=5)  # Add some vertical and horizontal padding

# Create text box for displaying OCR results
ocr_display = tk.StringVar()
ocr_display.set("")

# Add a label above the OCR display
ocr_label_title = tk.Label(results_frame, text="OCR Display:", font=('Helvetica', 12))
ocr_label_title.grid(row=0, column=1, padx=5, pady=10)  # Add some vertical and horizontal padding

# Create a box for the OCR display with an outline
ocr_box = tk.Label(results_frame, textvariable=ocr_display, relief="solid", width=55,
                   height=120)  # Set the default size of the display box
ocr_box.grid(row=1, column=1, padx=5, pady=5)  # Add some vertical and horizontal padding

# Create model runners
model_runners = [
    ModelRunner("Yolo v8", ocr_display),
    ModelRunner("Detectron 2", ocr_display),
    ModelRunner("SSD MobileNet V2 FPNLite 320x320", ocr_display)
]

# Create buttons for each model
style = ttk.Style()
style.configure('Model.TButton', font=('Helvetica', 14), foreground='black', padding=10)

model_buttons = []
for model_runner in model_runners:
    start_button = ttk.Button(button_frame, style="Model.TButton", text="Start " + model_runner.model_name,
                              command=lambda model=model_runner: start_model(model))
    start_button.pack(side=tk.LEFT, padx=10)  # Pack buttons to the left with more space in between
    model_buttons.append(start_button)

# Create exit button
style = ttk.Style()
style.configure('Exit.TButton', font=('Helvetica', 12, 'bold'), foreground='red', padding=10)
exit_button = ttk.Button(exit_button_frame, text="Exit", style="Exit.TButton", command=root.destroy)
exit_button.pack()

root.bind('<Escape>', lambda event: root.destroy())

# Start GUI
root.mainloop()
