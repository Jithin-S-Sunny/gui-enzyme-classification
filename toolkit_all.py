import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess

def open_hmm_gui():
    try:
        subprocess.Popen(["python3", "HMM_new_GUI.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open HMM tool: {e}")

def open_network_clustering_gui():
    try:
        subprocess.Popen(["python3", "GMM_GUI.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open Network and Clustering tool: {e}")

def open_machine_learning_gui():
    try:
        subprocess.Popen(["python3", "ML_gui.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open Machine Learning tool: {e}")

def open_neural_network_gui():
    try:
        subprocess.Popen(["python3", "NN_gui.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open Neural Network tool: {e}")


def open_struct_classification():
    try:
        subprocess.Popen(["python3", "struct_class_GUI.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open Structural Classification tool: {e}")


root = tk.Tk()
root.title("TOOLKIT")
root.geometry("900x450")

toolbar = tk.Frame(root, bg="lightgray", pady=5)
toolbar.pack(side=tk.TOP, fill=tk.X)

tool_hmm = tk.Button(toolbar, text="HMM", command=open_hmm_gui, padx=10, pady=5)
tool_hmm.pack(side=tk.LEFT, padx=5)

tool_network = tk.Button(toolbar, text="Network and Clustering", command=open_network_clustering_gui, padx=10, pady=5)
tool_network.pack(side=tk.LEFT, padx=5)

tool_ml = tk.Button(toolbar, text="Machine Learning", command=open_machine_learning_gui, padx=10, pady=5)
tool_ml.pack(side=tk.LEFT, padx=5)

tool_nn = tk.Button(toolbar, text="Neural Network", command=open_neural_network_gui, padx=10, pady=5)
tool_nn.pack(side=tk.LEFT, padx=5)

tool_struct = tk.Button(toolbar, text="Structural Classification", command=open_struct_classification, padx=10, pady=5)
tool_struct.pack(side=tk.LEFT, padx=5)

frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
hmm_image = ImageTk.PhotoImage(Image.open("hmm.png").resize((100, 100)))
network_image = ImageTk.PhotoImage(Image.open("cluster.png").resize((100, 100)))
ml_image = ImageTk.PhotoImage(Image.open("ml.png").resize((100, 100)))
nn_image = ImageTk.PhotoImage(Image.open("nn.png").resize((100, 100)))
struct_image = ImageTk.PhotoImage(Image.open("struct.png").resize((100, 100)))  

hmm_box = tk.LabelFrame(frame, text="HMM", padx=10, pady=10)
hmm_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
hmm_img_label = tk.Label(hmm_box, image=hmm_image)
hmm_img_label.pack()
hmm_label = tk.Label(hmm_box, text="Run HMM and produce top hits graph for your query data.", anchor="w", justify="left", wraplength=200)
hmm_label.pack(fill=tk.BOTH, expand=True)

network_box = tk.LabelFrame(frame, text="Network and Clustering", padx=10, pady=10)
network_box.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
network_img_label = tk.Label(network_box, image=network_image)
network_img_label.pack()
network_label = tk.Label(network_box, text="Construct networks and cluster data to reveal connectivity patterns.", anchor="w", justify="left", wraplength=200)
network_label.pack(fill=tk.BOTH, expand=True)

ml_box = tk.LabelFrame(frame, text="Machine Learning", padx=10, pady=10)
ml_box.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
ml_img_label = tk.Label(ml_box, image=ml_image)
ml_img_label.pack()
ml_label = tk.Label(ml_box, text="Apply ML techniques to classify and analyze data.", anchor="w", justify="left", wraplength=200)
ml_label.pack(fill=tk.BOTH, expand=True)

nn_box = tk.LabelFrame(frame, text="Neural Network", padx=10, pady=10)
nn_box.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")
nn_img_label = tk.Label(nn_box, image=nn_image)
nn_img_label.pack()
nn_label = tk.Label(nn_box, text="Run deep learning models for complex sequence analysis.", anchor="w", justify="left", wraplength=200)
nn_label.pack(fill=tk.BOTH, expand=True)

struct_box = tk.LabelFrame(frame, text="Structural Classification", padx=10, pady=10)
struct_box.grid(row=0, column=4, padx=10, pady=10, sticky="nsew")
struct_img_label = tk.Label(struct_box, image=struct_image)
struct_img_label.pack()
struct_label = tk.Label(struct_box, text="Analyze torsional angles and classify enzyme structures.", anchor="w", justify="left", wraplength=200)
struct_label.pack(fill=tk.BOTH, expand=True)

frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)
frame.columnconfigure(2, weight=1)
frame.columnconfigure(3, weight=1)
frame.columnconfigure(4, weight=1)
frame.rowconfigure(0, weight=1)

root.mainloop()
