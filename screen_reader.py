import pytesseract
import re
from screeninfo import get_monitors
from PIL import ImageGrab
from difflib import SequenceMatcher
from langdetect import detect
import pyautogui
import cv2
import numpy as np

class ScreenReader:
    def __init__(self):
        self.pytesseract = pytesseract
        self.last_text = ""

    def is_english(self, text):
        # You can adjust the threshold according to your needs
        threshold = 0.7
        english_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
        text_length = len(text)
        english_chars_count = sum(1 for char in text if char in english_chars)
        
        if (text_length == 0):
            return False

        # If a sufficient proportion of the characters are English letters
        return (english_chars_count / text_length) >= threshold

    def clean_text(self, text):
        # Assume text contains the garbled OCR output
        cleaned_text = re.sub(r'[^\w\s.,!?;]', ' ', text) # Remove non-word characters
        # Split by spaces and newlines
        segments = re.split(r'[\s\n]+', cleaned_text)
        # Remove empty segments
        real_language_segments = [segment for segment in segments if self.is_english(segment)]

        # Combine the real language segments back into a single string
        real_language_text = ' '.join(real_language_segments)

        if real_language_text.__len__() < 10:
            return None

        print("Cleaned text: " + real_language_text)
        return real_language_text


    def read_screen(self, do_text_check = True, clean_text = True):
        monitors = get_monitors()  # Assuming the first monitor is the main one

        # find the widest aspect-ratio monitor (hack)
        main_monitor = monitors[0]
        for monitor in monitors:
            if monitor.width / monitor.height > main_monitor.width / main_monitor.height:
                main_monitor = monitor
        print("Main monitor: " + str(main_monitor))

        # Calculate the region: center +/- 1/3rd of x-axis, and lower 1/2 of y-axis
        width_third = main_monitor.width // 3
        height_half = main_monitor.height // 2
        left = main_monitor.x + width_third
        top = main_monitor.y + height_half
        right = main_monitor.x + (2 * width_third)  # Adjust right boundary to two-thirds across
        bottom = main_monitor.y + main_monitor.height
        print("Region: " + str(left) + ", " + str(top) + ", " + str(right) + ", " + str(bottom))

        # Grab the screenshot of the specified region
        # screenshot = ImageGrab.grab() # bbox=(left, top, right, bottom))
        screenshot = pyautogui.screenshot()
        
        # since the pyautogui takes as a 
        # PIL(pillow) and in RGB we need to 
        # convert it to numpy array and BGR 
        # so we can write it to the disk
        # screenshot = cv2.cvtColor(np.array(image),
        #                     cv2.COLOR_RGB2BGR)
        text = self.pytesseract.image_to_string(screenshot)
        cleaned_text = re.sub(r'[^\w\s.,!?;]', '', text) # Remove non-word characters

        if clean_text == True:
            text = self.clean_text(text)
            if(text == None):
                return None, None

        # Compare the similarity of the current and last text
        if do_text_check == False:

            is_english = self.is_english(text)
            if(is_english == False):
                return None, None
        
            similarity = SequenceMatcher(None, text, self.last_text).ratio()

            # Update the last text
            self.last_text = text

            # If the similarity is 80% or higher, return None
            if similarity >= 0.5:
                print ("Too similar to last text: ")
                return None, None

        # print("New dialogue found...\n")
        # print(text + "\n")
        return screenshot, cleaned_text

# if main just take a screenshot
if __name__ == "__main__":
    screen_reader = ScreenReader()
    screen_reader.read_screen()