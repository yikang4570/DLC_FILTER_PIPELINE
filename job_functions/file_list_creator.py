import tkinter as tk
from tkinter import filedialog
import os

select_individual_files = False

def select_files_and_save_paths():
    root = tk.Tk()
    root.withdraw()

    file_paths = list(filedialog.askopenfilenames(title="Select files"))

    file_paths.sort()
    process_paths(file_paths)

def select_folders_and_save_paths():
    root = tk.Tk()
    root.withdraw()

    file_paths = []
    def_dir = filedialog.askdirectory(title="Select DEFAULT",initialdir= os.getcwd())
    while True:
        folder = filedialog.askdirectory(title="Select a folder (Cancel to finish)",initialdir=def_dir or os.getcwd())
        if not folder:
            break
        else:
            for root_dir, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(".avi") and not file.startswith("._"):
                        file_paths.append(os.path.join(root_dir, file))

    file_paths.sort()
    process_paths(file_paths)

def process_paths(file_paths):

    if file_paths:
        with open("file_list.txt", "w") as f:
            for path in file_paths:
                split_path = path.split("Active/",1)[1]
                f.write(split_path + "\n")
        print(f"{len(file_paths)} file paths written to selected_files.txt")
    else:
        print("No files selected.")

def create_job_list():
    if select_individual_files: select_files_and_save_paths()
    else: select_folders_and_save_paths()

if __name__ == "__main__":
    create_job_list()