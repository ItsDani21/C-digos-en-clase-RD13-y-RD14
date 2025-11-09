import tkinter as tk
from modules.visula_tkinter import App

def main():
    root = tk.Tk()
    root.title("GameSoft - Análisis de Videojuegos")
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
