import tkinter as tk
from tkinter import filedialog
import subprocess
from tkinter import ttk
from table_extraction import TableExtraction
import threading

def update_result_text(message):
    result_label.config(text=message)

def extract_data():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if not file_path:
        return

    select_button.config(state=tk.DISABLED)
    progress_bar.pack(pady=10)
    progress_bar.start()

    def extract_in_thread():
        try:
            extractor = TableExtraction()
            extractor.extract_from_file(file_path)
            root.after(2000, update_result_text, "Data successfully extracted and processed.")
        except subprocess.CalledProcessError:
            root.after(1000, update_result_text, "Error executing the program")
        finally:
            root.after(1000, enable_button)
            progress_bar.pack_forget()
            progress_bar.stop()

    thread = threading.Thread(target=extract_in_thread)
    thread.start()

def enable_button():
    select_button.config(state=tk.NORMAL)
    progress_bar.stop()

root = tk.Tk()
root.title("Extract data from PDF tables")

font_size = 16

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = screen_width // 2
window_height = screen_height // 2
root.geometry(f"{window_width}x{window_height}")

root.configure(bg="gray15")

progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, mode='indeterminate', maximum=100, value=0)
progress_bar.pack(pady=10)
progress_bar.pack_forget()

select_button = tk.Button(root, text="Выбрать PDF файл", font=("Helvetica", font_size), fg="white", bg="gray35", activebackground="gray35", command=extract_data)
select_button.pack(pady=10)

select_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

result_label = tk.Label(root, text="", font=("Helvetica", font_size), fg="white", bg="gray15")
result_label.pack(pady=5)

root.mainloop()