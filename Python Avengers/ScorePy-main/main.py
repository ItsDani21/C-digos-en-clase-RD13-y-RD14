import sys
import os


try:
    from src.interfaz import ExportedApp
except Exception:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from interfaz import ExportedApp


if __name__ == "__main__":
    app = ExportedApp()
    app.mainloop()