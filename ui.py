import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
from tkinter import Canvas  # Import the standard Canvas
from PIL import Image, ImageDraw, ImageOps  # Import the Python Imaging Library
from model_utils import load_model, train_model, process_and_predict  # Import backend functions
from tkinter import filedialog
import numpy as np
import cv2
class DigitRecognizerApp:
    def __init__(self, root, model_path="mnist_model.pth"):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        # change from 450 to 500 for adding new button to upload image
        self.root.geometry("300x500")
        self.style = ttk.Style()

        # Define canvas size
        self.canvas_width = 200
        self.canvas_height = 200

        # Add a themed frame for better layout
        self.frame = ttk.Frame(root, padding=10, borderwidth=2, relief="solid")
        self.frame.pack(fill=BOTH, expand=True)

        # # Canvas (standard tkinter Canvas)
        # self.canvas = Canvas(self.frame, width=self.canvas_width, height=self.canvas_height, bg="white")
        # self.canvas.pack(pady=10)
        # self.canvas.bind("<B1-Motion>", self.paint)
        # Frame for border around the Canvas
        self.canvas_frame = ttk.Frame(
            self.frame, 
            borderwidth=2,  # Set the border width
            relief="solid"  # Border style: "solid", "groove", "ridge", etc.
        )
        self.canvas_frame.pack(pady=10, anchor=CENTER)

        # Canvas inside the Frame
        self.canvas = Canvas(
            self.canvas_frame, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg="light green",  # Background color
            highlightthickness=0  # Disable default canvas border
        )

        
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        # Add buttons using ttkbootstrap styling
        self.predict_button = ttk.Button(self.frame, text="Predict", bootstyle=SUCCESS, command=self.predict)
        self.predict_button.pack(fill=X, pady=5)

        self.clear_button = ttk.Button(self.frame, text="Clear", bootstyle=WARNING, command=self.clear_canvas)
        self.clear_button.pack(fill=X, pady=5)
        
        # Add upload button
        self.upload_button = ttk.Button(self.frame, text="Upload Image", bootstyle=INFO, command=self.upload_picture)
        self.upload_button.pack(fill=X, pady=5)

        self.predict_multiple_button = ttk.Button(self.frame, text="Predict Multiple Digits", bootstyle=INFO, command=self.predict_multiple_digits)
        self.predict_multiple_button.pack(fill=X, pady=5)

        # Adding the new button in __init__ method
        self.predict_multiple_image_button = ttk.Button(
            self.frame,
            text="Predict Multiple Digits from Image",
            bootstyle=INFO,
            command=self.predict_multiple_digits_from_upload
        )
        self.predict_multiple_image_button.pack(fill=X, pady=5)


        # Load model and initialize image for drawing
        self.model = load_model(model_path)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")

    def paint(self, event):
        x, y = event.x, event.y
        radius = 4  # Adjust pen thickness
        # Draw on Canvas
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black")
        # Draw on self.image
        draw = ImageDraw.Draw(self.image)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="black")

    def predict(self):
        try:
            # prediction = predict_image(self.image, self.model)
            prediction = process_and_predict(self.image, self.model)
            messagebox.showinfo("Prediction", f"The model predicts: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")

    def upload_picture(self):
        """
        Handles uploading and preprocessing an image.
        """
        

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )

        if not file_path:
            messagebox.showerror("Upload Failed", "No file selected")
            return

        try:
            # Preprocess the uploaded image
            image = Image.open(file_path).convert('L')
            self.image = image
            try:
                prediction = process_and_predict(self.image, self.model)
                messagebox.showinfo("Prediction", f"The model predicts: {prediction}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")

            # Save the preprocessed image for debugging
            # image.save("uploaded_processed_image.png")
            # messagebox.showinfo("Upload Successful", "Image uploaded and processed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def predict_multiple_digits(self):
        """
        Handles predicting multiple digits from the canvas or uploaded image.
        """
        try:
            # Convert the canvas drawing to a binary image for processing
            image_array = np.array(self.image)
            
            # Save the raw canvas image for debugging
            debug_image_path = "debug_canvas_image.png"
            self.image.save(debug_image_path)
            print(f"Raw canvas image saved for debugging: {debug_image_path}")
            
            _, thresh = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY_INV)
            
            # Save the thresholded image for debugging
            thresh_debug_path = "debug_thresh_image.png"
            Image.fromarray(thresh).save(thresh_debug_path)
            print(f"Thresholded image saved for debugging: {thresh_debug_path}")

            # Find contours of the digits
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours left-to-right based on bounding box x-coordinates
            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            sorted_contours = sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0])

            # Segment and predict digits
            predictions = []
            for i, (contour, box) in enumerate(sorted_contours):
                x, y, w, h = box
                digit_image = thresh[y:y+h, x:x+w]

                # Invert colors to ensure black digit on white background
                digit_image = cv2.bitwise_not(digit_image)
                
                # Save each segmented and inverted digit image for debugging
                segment_debug_path = f"debug_segment_{i}.png"
                Image.fromarray(digit_image).save(segment_debug_path)
                print(f"Segmented and inverted digit image saved for debugging: {segment_debug_path}")
                
                # Convert the digit to a PIL image
                digit_pil = Image.fromarray(digit_image)

                # Predict using the model
                pred = process_and_predict(digit_pil, self.model)
                if pred is not None:
                    predictions.append(pred)

            if predictions:
                prediction_text = ''.join(map(str, predictions))
                messagebox.showinfo("Prediction", f"The model predicts: {prediction_text}")
            else:
                messagebox.showwarning("No Digits Found", "Could not detect any digits in the image.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def predict_multiple_digits_from_upload(self):
        """
        Handles predicting multiple digits from an uploaded image.
        """
        try:
            # Open file dialog to upload an image
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
            )

            if not file_path:
                messagebox.showerror("Upload Failed", "No file selected")
                return

            # Preprocess the uploaded image
            image = Image.open(file_path).convert('L')  # Convert to grayscale
            image_array = np.array(image)
            
            # Save the uploaded raw image for debugging
            debug_uploaded_path = "debug_uploaded_image.png"
            image.save(debug_uploaded_path)
            print(f"Uploaded raw image saved for debugging: {debug_uploaded_path}")
            
            _, thresh = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY_INV)
            
            # Save the thresholded image for debugging
            thresh_debug_path = "debug_thresh_uploaded.png"
            Image.fromarray(thresh).save(thresh_debug_path)
            print(f"Thresholded uploaded image saved for debugging: {thresh_debug_path}")

            # Find contours of the digits
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours left-to-right based on bounding box x-coordinates
            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            sorted_contours = sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0])

            # Segment and predict digits
            predictions = []
            for i, (contour, box) in enumerate(sorted_contours):
                x, y, w, h = box
                digit_image = thresh[y:y+h, x:x+w]

                # Invert colors to ensure black digit on white background
                digit_image = cv2.bitwise_not(digit_image)
                
                # Save each segmented and inverted digit image for debugging
                segment_debug_path = f"debug_uploaded_segment_{i}.png"
                Image.fromarray(digit_image).save(segment_debug_path)
                print(f"Uploaded segmented and inverted digit image saved for debugging: {segment_debug_path}")
                
                # Convert the digit to a PIL image
                digit_pil = Image.fromarray(digit_image)

                # Predict using the model
                pred = process_and_predict(digit_pil, self.model)
                if pred is not None:
                    predictions.append(pred)

            if predictions:
                prediction_text = ''.join(map(str, predictions))
                messagebox.showinfo("Prediction", f"The model predicts: {prediction_text}")
            else:
                messagebox.showwarning("No Digits Found", "Could not detect any digits in the image.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")






if __name__ == "__main__":
    mode = input("Please select mode: Training (t) or Use (u): ").strip().lower()
    model_path = "mnist_model.pth"

    while mode not in ["t", "u"]:
        mode = input("Invalid mode. Please select Training (t) or Use (u): ").strip().lower()

    if mode == "t":
        print("Training the model... Please wait.")
        train_model(epochs=5, path=model_path)
        print("Training completed!")
    elif mode == "u":
        root = ttk.Window(themename="litera")  # Select a bootstrap theme
        app = DigitRecognizerApp(root, model_path=model_path)
        root.mainloop()
    else:
        print("Error: Invalid mode, please select: Training (t) or Use (u).")
