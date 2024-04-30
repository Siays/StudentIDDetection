import tkinter as tk
import threading
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
                YoloV8Detection.yoloV8_dectect(self.ocr_display)

            elif self.model_name == "Model 2":
                pass
            elif self.model_name == "Model 3":
                pass

    def stop_model(self):
        self.running = False

def start_model(model_runner):
    if model_runner.thread is None or not model_runner.thread.is_alive():
        model_runner.ocr_display.set("")  # Clear OCR display when starting a model
        model_runner.thread = threading.Thread(target=model_runner.run_model)
        model_runner.thread.start()

def stop_model(model_runner):
    model_runner.stop_model()

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

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)  # Add some vertical padding

# Create text box for displaying OCR results
ocr_display = tk.StringVar()
ocr_display.set("")

# Add a label above the OCR display
ocr_label_title = tk.Label(root, text="OCR Display:")
ocr_label_title.pack()

# Create a label for the OCR display with an outline
ocr_label = tk.Label(root, textvariable=ocr_display, relief="solid", width=120, height=120)  # Set the default size of the OCR display box
ocr_label.pack()

# Create model runners
model_runners = [
    ModelRunner("Yolo v8", ocr_display),
    ModelRunner("Model 2", ocr_display),
    ModelRunner("Model 3", ocr_display)
]

# Create buttons for each model
model_buttons = []
for model_runner in model_runners:
    start_button = tk.Button(button_frame, text="Start " + model_runner.model_name, command=lambda model=model_runner: start_model(model))
    start_button.pack(side=tk.LEFT, padx=10)  # Pack buttons to the left with more space in between
    model_buttons.append(start_button)


root.bind("<Key>", key_pressed)

# Start GUI
root.mainloop()
