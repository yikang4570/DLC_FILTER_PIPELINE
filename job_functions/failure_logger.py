import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from aggregate_sub_files import aggregate_sub_files

class PNGReviewer:
    def __init__(self, root, folder_path, log_file="manually_determined_failures.txt"):
        self.root = root
        self.folder_path = folder_path
        self.log_file = folder_path + os.sep + log_file
        self.png_files = [f for f in sorted(os.listdir(folder_path))
                          if f.lower().endswith(".png")
                          and os.path.isfile(self.folder_path + os.sep + f)
                          and os.access(self.folder_path + os.sep + f, os.R_OK)
                          and not f.startswith("._")]
        self.current_index = 0

        self.root.title("PNG Reviewer")
        self.root.bind("<y>", lambda e: self.next_image())
        self.root.bind("<n>", lambda e: self.log_failure())

        # Image Display
        self.status_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.status_label.pack(pady=5)

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        #self.progress_canvas = tk.Canvas(self.root, height=20, bg="lightgray")
        self.progress_canvas = tk.Canvas(self.root, height=40, highlightthickness=0, bd=0)
        self.progress_canvas.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.segment_ids = []  # Store segment rectangles
        self.failure_indices = set()  # Track failures

        self.root.after(100, self.init_progress_bar)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        yes_btn = tk.Button(btn_frame, text="Yes (Keep) [Y]", command=self.next_image, width=20)
        yes_btn.grid(row=0, column=0, padx=10)

        no_btn = tk.Button(btn_frame, text="No (Log Failure) [N]", command=self.log_failure, width=20)
        no_btn.grid(row=0, column=1, padx=10)

        self.show_image()

    def init_progress_bar(self):
        self.progress_canvas.delete("all")

        self.root.update_idletasks()  # Force layout update to get real canvas width

        canvas_width = self.progress_canvas.winfo_width()
        total = len(self.png_files)
        segment_width = canvas_width / total

        self.segment_ids.clear()
        for i in range(total):
            x0 = i * segment_width
            x1 = x0 + segment_width
            rect = self.progress_canvas.create_rectangle(x0, 0, x1, 20, fill="lightgray", outline="")
            self.segment_ids.append(rect)

        self.update_progress_color()

    def update_progress_color(self):
        for i, rect in enumerate(self.segment_ids):
            if i < self.current_index:
                color = "red" if i in self.failure_indices else "green"
                self.progress_canvas.itemconfig(rect, fill=color)
            else:
                self.progress_canvas.itemconfig(rect, fill="lightgray")

    def show_image(self):
        if self.current_index >= len(self.png_files):
            print("All images reviewed.")
            self.root.destroy()
            return

        img_path = os.path.join(self.folder_path, self.png_files[self.current_index])
        image = Image.open(img_path)
        image.thumbnail((800, 800))  # Resize if too large
        self.tk_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.tk_image)

        self.status_label.config(text=f"Image {self.current_index + 1} of {len(self.png_files)}")

    def next_image(self):
        self.current_index += 1
        self.show_image()
        self.update_progress_color()

    def log_failure(self):
        img_path = os.path.join(self.folder_path, self.png_files[self.current_index])
        with open(self.log_file, "a") as f:
            f.write(img_path + "\n")
        self.failure_indices.add(self.current_index)
        self.update_progress_color()
        self.next_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    folder = aggregate_sub_files(root)
    #manual_folder = filedialog.askdirectory(title="Select Folder Containing PNGs",initialdir= '/Volumes/lake.s/Active/Shawn P/D. DATA (PROCESSED)/A. ELASTIN PROJECT/GAIT')
    #folder = [manual_folder]
    root.destroy()
    if folder:
        for f in folder:
            root = tk.Tk()
            app = PNGReviewer(root, f)
            root.mainloop()
    else:
        print("No folder selected.")