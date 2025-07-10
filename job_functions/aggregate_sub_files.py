import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox


def aggregate_sub_files(root):
    def_dir = filedialog.askdirectory(title="Select DEFAULT",initialdir= os.getcwd())
    count = 0
    subplots_dir_list = []
    while True:
        # Ask user to select a directory
        folder_selected = filedialog.askdirectory(title="Select a folder",initialdir=def_dir or os.getcwd())
        if not folder_selected:
            break

        # Prepare target directory
        subplots_dir = os.path.join(folder_selected, "SUBPLOTS")

        os.makedirs(subplots_dir, exist_ok=True)
        subplots_dir_list.append(subplots_dir)
        # Iterate through files and copy matching ones
        for filename in os.listdir(folder_selected):
            full_path = os.path.join(folder_selected, filename)

            # Only copy regular, readable files
            if (
                    os.path.isfile(full_path)
                    and os.access(full_path, os.R_OK)
                    and "SUB" in filename
                    and full_path.split("/")[-1] != "._"
            ):
                try:
                    shutil.copy2(full_path, subplots_dir)
                    count += 1
                except Exception as e:
                    print(f"Failed to copy {filename}: {e}")

    messagebox.showinfo("Done", f"{count} file(s) with 'SUB' copied to SUBPLOTS folder.")
    return subplots_dir_list


if __name__ == "__main__":
    # Set up Tkinter without opening a full GUI window
    root = tk.Tk()
    root.withdraw()
    aggregate_sub_files(root)