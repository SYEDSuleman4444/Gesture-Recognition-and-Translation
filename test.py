import cv2
import tkinter as tk
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from googletrans import Translator
import pyttsx3
import time

translator = Translator(service_urls=['translate.googleapis.com'])
engine = pyttsx3.init()

# List to store identified gestures
identified_gestures = []
current_gesture = ""

# Initialize video capture
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["1", "2", "3", "A", "B", "C", "call", "dislike", "goodluck", "is", "like", "ok", "this", "apple"]

# Dictionary to map language codes to full names
language_names = {
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi"
}

# Create a Tkinter window
root = tk.Tk()
root.title("Hand Gesture Classification")

# Set background color of the main window
root.configure(bg="#F0F0F0")

# Create a frame to hold the main content
main_frame = tk.Frame(root, bg="#FFFFFF", padx=20, pady=20)
main_frame.pack()

# Create a canvas to display the webcam feed
canvas = tk.Canvas(main_frame, width=800, height=600, bg="#D3D3D3")
canvas.grid(row=0, column=0, rowspan=6, padx=10, pady=10)

# Create labels for displaying the gesture text
gesture_label = tk.Label(main_frame, text="", font=('Times 20 bold'), bg="#FFFFFF")
gesture_label.grid(row=0, column=1, columnspan=2, padx=10, pady=10)

gesture_label1 = tk.Label(main_frame, text="", font=('Times 20 bold'), bg="#FFFFFF")
gesture_label1.grid(row=1, column=1, columnspan=2, padx=10, pady=10)

# Create a drop-down menu for translation language selection
translate_label = tk.Label(main_frame, text="Translate to:", font=('Times 15 bold'), bg="#FFFFFF")
translate_label.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)

translate_var = tk.StringVar(root)
translate_var.set("")  # No default selection

# Create a list of language options with full names
language_options = [f"{language_names[code]} ({code})" for code in language_names.keys()]

translate_menu = tk.OptionMenu(main_frame, translate_var, *language_options)
translate_menu.grid(row=2, column=2, padx=10, pady=10, sticky=tk.W)


# Function to save the current gesture
def save_gesture():
    global current_gesture
    if current_gesture and current_gesture not in identified_gestures:
        identified_gestures.append(current_gesture)
        gesture_label.config(text=" ".join(identified_gestures))
        translate_sentence()


# Function to translate the sentence
def translate_sentence():
    global identified_gestures
    sentence = " ".join(identified_gestures)
    if translate_var.get():
        dest_lang = translate_var.get()[:2]
        try:
            translated_sentence = translator.translate(sentence, dest=dest_lang).text
        except Exception as e:
            print(f"Translation error: {e}")
            translated_sentence = ""
    else:
        translated_sentence = ""
    gesture_label1.config(text=translated_sentence)


# Function to play sound of all saved gestures with reduced delay
def play_saved_gestures_sound():
    for gesture in identified_gestures:
        engine.say(gesture)
        engine.runAndWait()
        time.sleep(0.2)  # 0.2 second delay between each gesture


# Function to update the canvas
def update_canvas():
    global current_gesture
    success, img = cap.read()

    if not success:
        print("Failed to read from camera.")
        root.after(100, update_canvas)
        return

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        if w > 0 and h > 0:
            imgWhite = np.ones((300, 300, 3), np.uint8) * 255
            imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = 300 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                wGap = math.ceil((300 - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = 300 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                hGap = math.ceil((300 - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.putText(img, f"Gesture: {labels[index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 0, 255), 4)

            current_gesture = labels[index]

    # Display the webcam feed on the canvas
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 400))
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    canvas.create_image(100, 100, anchor=tk.NW, image=img)
    canvas.img = img

    # Perform translation if gestures have been saved
    if identified_gestures and translate_var.get():
        translate_sentence()

    root.after(100, update_canvas)  # Reduced update interval for smoother interaction


# Function to reset everything as if stopping and starting the application
def replay():
    global identified_gestures, current_gesture, cap
    identified_gestures = []
    current_gesture = ""
    gesture_label.config(text="")
    gesture_label1.config(text="")
    translate_var.set("")  # Reset language selection
    cap.release()
    cap = cv2.VideoCapture(0)
    update_canvas()


# Create buttons
stop_button = tk.Button(main_frame, text="Stop", command=root.destroy, padx=10, pady=5, bg="#F44336", fg="#FFFFFF",
                        font="Times", relief="raised")
stop_button.grid(row=4, column=1, padx=10, pady=10, sticky=tk.W)

replay_button = tk.Button(main_frame, text="Replay", command=replay, padx=10, pady=5, bg="#FF9800", fg="#FFFFFF",
                          font="Times", relief="raised")
replay_button.grid(row=4, column=2, padx=10, pady=10, sticky=tk.W)

play_saved_button = tk.Button(main_frame, text="Play Saved Gestures", command=play_saved_gestures_sound, padx=10,
                              pady=5, bg="#008CBA", fg="#FFFFFF", font="Times", relief="raised")
play_saved_button.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)

save_gesture_button = tk.Button(main_frame, text="Save Gesture", command=save_gesture, padx=10, pady=5, bg="#4CAF50",
                                fg="#FFFFFF", font="Times", relief="raised")
save_gesture_button.grid(row=3, column=2, padx=10, pady=10, sticky=tk.W)

# Start updating the canvas
update_canvas()

root.mainloop()
