import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class SLAMVisualizer:
    def __init__(self, positions, images):
        self.positions = positions
        self.images = images
        self.index = 0

        # Setup the Tkinter GUI
        self.root = tk.Tk()
        self.root.title("SLAM Trajectory and Image Visualizer")

        # Frame for the plot and image
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Setup the matplotlib figure for the trajectory
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([min(p[0] for p in positions) - 1, max(p[0] for p in positions) + 1])
        self.ax.set_ylim([min(p[1] for p in positions) - 1, max(p[1] for p in positions) + 1])
        self.ax.grid(True)

        # Create a canvas for matplotlib figure and add to the Tkinter frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Setup image label for OpenCV image
        self.image_label = ttk.Label(self.frame)
        self.image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Add a button to the window
        self.paused = False
        self.button = tk.Button(self.root, text="Next Position", command=self.update_content)
        self.button.pack(side=tk.BOTTOM)

    def update_content(self):
        if self.index < len(self.positions):
            # Update plot
            self.ax.clear()
            self.ax.grid(True)
            x_vals = [p[0] for p in self.positions[:self.index + 1]]
            y_vals = [p[1] for p in self.positions[:self.index + 1]]
            self.ax.plot(x_vals, y_vals, marker='o')
            self.canvas.draw()

            # Update image
            img = cv2.cvtColor(self.images[self.index], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = img_tk  # anchor imgtk so it's not garbage-collected
            self.image_label.config(image=img_tk)  # show the image

            self.index += 1
        else:
            print("No more positions or images to show.")

    def run(self):
        tk.mainloop()
