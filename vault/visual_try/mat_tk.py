import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig = Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, .01)
fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)

# Create toolbar using grid layout instead of pack
toolbar_frame = tkinter.Frame(root)
toolbar_frame.grid(row=1, column=0, columnspan=2)
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()

def on_key_press(event):
    print(f"you pressed {event.key}")
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect("key_press_event", on_key_press)

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = tkinter.Button(master=root, text="Quit", command=_quit)
button.grid(row=2, column=0)

tkinter.mainloop()