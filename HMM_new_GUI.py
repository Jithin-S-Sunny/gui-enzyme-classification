import tkinter as tk
import time
from tkinter import simpledialog
from tkinter import filedialog, messagebox
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
root = tk.Tk()
root.title("HMM GUI")
root.geometry("500x400")
input_sequences = tk.StringVar()
target_sequences = tk.StringVar()
alignment_method = tk.StringVar(value="clustalo")
output_file = tk.StringVar()
def browse_file(var):
    file = filedialog.askopenfilename()
    var.set(file)
def create_shell_script():
    input_file = input_sequences.get()
    target_file = target_sequences.get()
    align_method = alignment_method.get()
    
    if not input_file or not target_file:
        messagebox.showerror("Error", "Please select input and target sequences.")
        return

    script_content = f"""#!/bin/bash
    input_sequences="{input_file}"
    target_sequences="{target_file}"
    {align_method} -i $input_sequences -o aligned_sequences.fasta --outfmt fasta
    hmmbuild model.hmm aligned_sequences.fasta
    hmmsearch --tblout output_table.txt model.hmm $target_sequences
    """
    

    with open("script.sh", "w") as f:
        f.write(script_content)


    with open("nohup_output.log", "w") as log_file:
        process = subprocess.Popen(
            ["bash", "script.sh"], stdout=log_file, stderr=log_file
        )
    
    if process:
        messagebox.showinfo("Success", "Shell script created and running!")

        monitor_output_file()
    else:
        messagebox.showerror("Error", "Failed to run the shell script.")

def monitor_output_file():
    """Monitor for the creation of the output file and notify the user."""
    output_file_path = "output_table.txt"
    for _ in range(180):  
        if os.path.exists(output_file_path):
            messagebox.showinfo("Notification", "Output file generated: output_table.txt. You can now analyze the output.")
            return
        time.sleep(1)  
    
    messagebox.showwarning("Timeout", "The output file was not generated within the expected time.")

def analyze_output():
    output_file_path = filedialog.askopenfilename()
    if not output_file_path:
        messagebox.showerror("Error", "Please select the output table.")
        return

    try:
       
        df = pd.read_csv(output_file_path, delim_whitespace=True, comment='#', header=None)
        df_clean = df[[0, 5, 4]].copy()
        df_clean.columns = ['Query', 'Score', 'E-value']
        df_clean['Score'] = pd.to_numeric(df_clean['Score'], errors='coerce')
        df_clean['E-value'] = pd.to_numeric(df_clean['E-value'], errors='coerce')

        
        top_n = simpledialog.askinteger(
            "Input Required", "Enter the number of top hits to plot:",
            minvalue=1, maxvalue=len(df_clean)
        )

        if not top_n:
            messagebox.showinfo("Cancelled", "Plotting cancelled.")
            return


        df_top_hits = df_clean.sort_values(by='Score', ascending=False).head(top_n)


        from matplotlib.colors import Normalize
        evalue_colors = -np.log10(df_top_hits['E-value'])
        norm = Normalize(vmin=min(evalue_colors), vmax=max(evalue_colors))

 
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df_top_hits['Query'], df_top_hits['Score'],
            s=40, c=evalue_colors, cmap='viridis', norm=norm, alpha=0.8
        )
        plt.plot(
            df_top_hits['Query'], df_top_hits['Score'],
            color='blue', linestyle='-', marker='o'
        )

        plt.xlabel('Query (Target Name)', fontweight='bold')
        plt.ylabel('Score (log scale)', fontweight='bold')
        plt.title(f'Top {top_n} HMM Match Scores vs Query (E-value shown as color)')
        plt.xticks(rotation=45, ha='right', fontsize=6, fontweight='bold')
        plt.yscale('log')

        cbar = plt.colorbar(scatter)
        cbar.set_label('-log₁₀(E-value)')
        cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process output: {str(e)}")

tk.Label(root, text="Input Sequences File").grid(row=0, column=0, padx=10, pady=10)
tk.Entry(root, textvariable=input_sequences).grid(row=0, column=1, padx=10)
tk.Button(root, text="Browse", command=lambda: browse_file(input_sequences)).grid(row=0, column=2, padx=10)

tk.Label(root, text="Target Sequences File").grid(row=1, column=0, padx=10, pady=10)
tk.Entry(root, textvariable=target_sequences).grid(row=1, column=1, padx=10)
tk.Button(root, text="Browse", command=lambda: browse_file(target_sequences)).grid(row=1, column=2, padx=10)

tk.Label(root, text="Alignment Method").grid(row=2, column=0, padx=10, pady=10)
tk.OptionMenu(root, alignment_method, "clustalo", "clustalw", "mafft").grid(row=2, column=1, padx=10)

tk.Button(root, text="Create Shell Script & Run", command=create_shell_script).grid(row=3, column=0, columnspan=3, pady=20)

tk.Button(root, text="Analyze Output", command=analyze_output).grid(row=4, column=0, columnspan=3, pady=20)
root.mainloop()

