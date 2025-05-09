import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

class MLGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Pipeline GUI")
        self.root.geometry("600x400")

        self.fasta_file = ""
        self.independent_fasta = ""

        tk.Button(root, text="Select Training FASTA", command=self.load_fasta).pack(pady=5)
        tk.Button(root, text="Select Independent FASTA", command=self.load_independent_fasta).pack(pady=5)
        tk.Button(root, text="Run ML Pipeline", command=self.run_pipeline).pack(pady=10)
        tk.Button(root, text="Generate Plots", command=self.generate_plots).pack(pady=10)

    def load_fasta(self):
        self.fasta_file = filedialog.askopenfilename(title="Select Training FASTA")
        messagebox.showinfo("File Selected", f"Training FASTA: {self.fasta_file}")
    
    def load_independent_fasta(self):
        self.independent_fasta = filedialog.askopenfilename(title="Select Independent FASTA")
        messagebox.showinfo("File Selected", f"Independent FASTA: {self.independent_fasta}")
    
    def run_pipeline(self):
        if not self.fasta_file or not self.independent_fasta:
            messagebox.showerror("Error", "Please select both FASTA files")
            return
        
        try:
            subprocess.run(["python3", "ML_model.py", self.fasta_file, self.independent_fasta], check=True)
            messagebox.showinfo("Success", "ML pipeline completed. Output files generated.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"ML pipeline failed: {e}")
    
    def generate_plots(self):
        try:
            subprocess.run(["python3", "ML_plotting.py"], check=True)
            messagebox.showinfo("Success", "Plots generated successfully.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Plotting failed: {e}")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = MLGUI(root)
    root.mainloop()
