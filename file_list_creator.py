import tkinter as tk
from tkinter import filedialog

def select_files_and_save_paths():
    # Hide the root window
    root = tk.Tk()
    root.withdraw()

    # Open file selection dialog
    file_paths = filedialog.askopenfilenames(title="Select files")

    if file_paths:
        with open("file_list.txt", "w") as f:
            for path in file_paths:
                f.write(f"{path}\n")
        print(f"{len(file_paths)} file paths written to selected_files.txt")
    else:
        print("No files selected.")

if __name__ == "__main__":
    select_files_and_save_paths()