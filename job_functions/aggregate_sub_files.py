import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox


def aggregate_sub_files(root):

    # Ask user to select a directory
    folder_selected = filedialog.askdirectory(title="Select a folder")
    if not folder_selected:
        messagebox.showinfo("Cancelled", "No folder selected.")
        return

    # Prepare target directory
    subplots_dir = os.path.join(folder_selected, "SUBPLOTS")
    os.makedirs(subplots_dir, exist_ok=True)

    # Iterate through files and copy matching ones
    count = 0
    for filename in os.listdir(folder_selected):
        full_path = os.path.join(folder_selected, filename)

        # Only copy regular, readable files
        if (
                os.path.isfile(full_path)
                and os.access(full_path, os.R_OK)
                and "SUB" in filename
        ):
            try:
                shutil.copy2(full_path, subplots_dir)
                count += 1
            except Exception as e:
                print(f"Failed to copy {filename}: {e}")

    messagebox.showinfo("Done", f"{count} file(s) with 'SUB' copied to SUBPLOTS folder.")
    return subplots_dir


if __name__ == "__main__":
    # Set up Tkinter without opening a full GUI window
    root = tk.Tk()
    root.withdraw()
    aggregate_sub_files(root)