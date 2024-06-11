import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button, StringVar, Frame, Entry, Text, Scrollbar
from PIL import Image, ImageTk
import pyttsx3
import textwrap
import os
from dotenv import load_dotenv

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def initialize_gemini():
    load_dotenv("Image.env")
    my_api_key = os.environ.get('MY_API_KEY')
    genai.configure(api_key=my_api_key)
    model = genai.GenerativeModel('gemini-pro-vision')

def ask_gemini(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("You are a virtual health assistant of a medicine capsule counter software. You are responsible for giving fundamental health assistant to older people. Don't ever say you are not professional. Here's the prompt:  "+ prompt, stream=True)
    response.resolve()
    return response.text

def preprocess_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1]
    _, mask_colored = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray, lower, upper)
    mask = cv2.bitwise_or(mask_colored, edges)
    return mask

def morphological_operations(mask):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morphed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel, iterations=2)
    return morphed

def find_and_filter_contours(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = 0.01 * img.shape[0] * img.shape[1]
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > area_threshold]
    return filtered_contours

def draw_contours(img, contours):
    img_with_contours = img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img_with_contours

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        process_and_display(img)

def process_and_display(img):
    mask = preprocess_image(img)
    morphed = morphological_operations(mask)
    filtered_contours = find_and_filter_contours(img, morphed)
    img_with_contours = draw_contours(img, filtered_contours)
    
    # Resize the image to 300x300 pixels
    img_with_contours_resized = cv2.resize(img_with_contours, (300, 300))
    
    # Convert the image for displaying in tkinter
    img_rgb = cv2.cvtColor(img_with_contours_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    panel.configure(image=img_tk)
    panel.image = img_tk
    
    num_capsules = len(filtered_contours)
    detection_text = f"Detected {num_capsules} capsules in the image."
    detection_label.configure(text=detection_text)
    
    # Speak the detection text
    root.after(100, speak_detection_text, detection_text)


def speak_detection_text(text):
    # Speak the detection text
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust the rate here (default is 200)
    engine.say(text)
    engine.runAndWait()

def submit_query():
    query = query_entry.get()
    response = ask_gemini(query)
    update_gemini_response(response)

def update_gemini_response(response):
    # Clear the text widget first
    gemini_text.delete('1.0', 'end')
    gemini_text.configure(font=("Helvetica", 12))  # Decrease the font size
    gemini_text.insert('1.0', response)


initialize_gemini()
response = ask_gemini("Bana de ki: Merhaba! Benim adım Capsule Counter. Sana nasıl yardımcı olabilirim?")

root = Tk()
root.title("Capsule Counter")
root.geometry("800x600")
root.configure(bg="#F0F0F0")

# Create a welcome frame
welcome_frame = Frame(root, bg="#F0F0F0")
welcome_frame.pack(pady=20)

welcome_label = Label(welcome_frame, text="Welcome to Capsule Counter", font=("Helvetica", 24), bg="#F0F0F0")
welcome_label.pack()

instruction_label = Label(welcome_frame, text="Please select an image using the button -->", font=("Helvetica", 14), bg="#F0F0F0")
instruction_label.pack(side="left", padx=10)

# Create a Label for showing the number of capsules detected
detection_label = Label(root, text="", font=("Helvetica", 14), bg="#F0F0F0")
detection_label.pack()

btn = Button(welcome_frame, text="Select Image", command=select_file, font=("Helvetica", 14), bg="#4CAF50", fg="white")
btn.pack(side="left")

panel = Label(root, bg="#F0F0F0")
panel.pack(pady=20)

# Create a Label for Gemini's response
response_label = Label(root, text="Capsule Counter ChatBot", font=("Helvetica", 16), bg="#F0F0F0")
response_label.pack()

# Create a new frame for the Entry and Button widgets
query_frame = Frame(root, bg="#F0F0F0")
query_frame.pack(pady=10)

# Create an Entry widget for user input
query_entry = Entry(query_frame, width=50, font=("Helvetica", 14))
query_entry.grid(row=0, column=0, padx=10, pady=10)

# Create a Button to submit the query
submit_button = Button(query_frame, text="Submit", command=submit_query, font=("Helvetica", 14), bg="#4CAF50", fg="white")
submit_button.grid(row=0, column=1, padx=10, pady=10)

# Create a Text widget for displaying Gemini's response with a scrollbar
# Create a Frame to contain gemini_text and scrollbar
gemini_frame = Frame(root, bg="#F0F0F0")
gemini_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Create a Text widget for displaying Gemini's response
gemini_text = Text(gemini_frame, height=10, width=80, font=("Helvetica", 14))
gemini_text.pack(side="left", fill="both", expand=True)

# Create a Scrollbar for gemini_text
scrollbar = Scrollbar(gemini_frame, orient="vertical", command=gemini_text.yview)
scrollbar.pack(side="right", fill="y")

# Configure gemini_text to use scrollbar
gemini_text.config(yscrollcommand=scrollbar.set)

# Create a StringVar object to hold the text
text_var = StringVar()
text_var.set(response)

root.mainloop()
