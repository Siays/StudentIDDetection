import tkinter as tk
import threading
from tkinter import ttk

import DetectronPredict
import SSDModelDetection
import YoloV8Detection  # Import the YoloV8Detection module


class ModelRunner:
    def __init__(self, model_name, model_ocr_display, last_validated_display):
        self.model_name = model_name
        self.model_ocr_display = model_ocr_display
        self.last_validated_display = last_validated_display
        self.running = False
        self.thread = None

    def run_model(self):
        self.running = True
        while self.running:
            # Start the appropriate model based on self.model_name
            if self.model_name == "Yolo v8":
                YoloV8Detection.yolo_v8_detect(self.model_ocr_display, self.last_validated_display)
            elif self.model_name == "Detectron 2":
                DetectronPredict.detectron_detect(self.model_ocr_display, self.last_validated_display)
            elif self.model_name == "SSD MobileNet V2 FPNLite 320x320":
                SSDModelDetection.ssd_model_detect(self.model_ocr_display, self.last_validated_display)
            stop_model(self)

    def stop_model(self):
        self.running = False


def start_model(model):
    if model.thread is None or not model.thread.is_alive():
        model.model_ocr_display.set("")  # Clear OCR display when starting a model
        model.last_validated_display.set("")  # Clear validated display when starting a model
        model.thread = threading.Thread(target=model.run_model)
        model.thread.start()
        disable_buttons(exclude_button=model_buttons[model_runners.index(model)])


def stop_model(model):
    model.stop_model()
    enable_buttons()


def disable_buttons(exclude_button):
    for button in model_buttons:
        if button != exclude_button:
            button.config(state=tk.DISABLED)
    unbind_escape_button()
    disable_close_button()


def enable_buttons():
    for button in model_buttons:
        button.config(state=tk.NORMAL)
    bind_escape_button()
    enable_close_button()


def unbind_escape_button():
    root.unbind("<Escape>")


def bind_escape_button():
    root.bind('<Escape>', lambda event: end_program())


def disable_close_button():
    # Disable the close button
    root.protocol("WM_DELETE_WINDOW", lambda: None)


def enable_close_button():
    # Enable the close button
    root.protocol("WM_DELETE_WINDOW", root.destroy)


def end_program():
    for model in model_runners:
        stop_model(model)
    root.destroy()


root = tk.Tk()
root.title("TARUMT Student ID detector")
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
                                 height=120, font=('Helvetica', 12))  # Set the default size of the display box
validated_display_box.grid(row=1, column=0, padx=5, pady=5)  # Add some vertical and horizontal padding

# Create text box for displaying OCR results
ocr_display = tk.StringVar()
ocr_display.set("")

# Add a label above the OCR display
ocr_label_title = tk.Label(results_frame, text="OCR Display:", font=('Helvetica', 12))
ocr_label_title.grid(row=0, column=1, padx=5, pady=10)  # Add some vertical and horizontal padding

# Create a box for the OCR display with an outline
ocr_box = tk.Label(results_frame, textvariable=ocr_display, relief="solid", width=55,
                   height=120, font=('Helvetica', 12))  # Set the default size of the display box
ocr_box.grid(row=1, column=1, padx=5, pady=5)  # Add some vertical and horizontal padding

# Create model runners
model_runners = [
    ModelRunner("Yolo v8", ocr_display, validated_display),
    ModelRunner("Detectron 2", ocr_display, validated_display),
    ModelRunner("SSD MobileNet V2 FPNLite 320x320", ocr_display, validated_display)
]

# Create buttons for each model
style = ttk.Style()
style.configure('Model.TButton', font=('Helvetica', 14, 'bold'), foreground='blue', background='pink', padding=10)


def add_shadow(widget, x_offset, y_offset, color='gray'):
    # Get the widget's coordinates
    x, y = widget.winfo_x(), widget.winfo_y()

    # Get the widget's dimensions
    width, height = widget.winfo_width(), widget.winfo_height()

    # Create a canvas for the shadow
    canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0)
    canvas.place(x=x + x_offset, y=y + y_offset)

    # Draw a rectangle for the shadow
    canvas.create_rectangle(0, 0, width, height, fill=color, outline='')
model_buttons = []
for model_runner in model_runners:
    start_button = ttk.Button(button_frame, style="Model.TButton", text="Start " + model_runner.model_name,
                              command=lambda model=model_runner: start_model(model))
    start_button.pack(side=tk.LEFT, padx=10)  # Pack buttons to the left with more space in between
    model_buttons.append(start_button)

add_shadow(start_button, 3, 3)
# Create exit button
style = ttk.Style()
style.configure('Exit.TButton', font=('Helvetica', 12, 'bold'), foreground='red', padding=10)
exit_button = ttk.Button(exit_button_frame, text="Exit", style="Exit.TButton", command=lambda: end_program())
exit_button.pack()
model_buttons.append(exit_button)

bind_escape_button()

# Start GUI
root.mainloop()
