import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np
from matplotlib.gridspec import GridSpec

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

# Create a figure
fig = Figure(figsize=(8, 6), dpi=100)
# Create a GridSpec layout: 2 rows, 6 columns
gs = GridSpec(2, 6, figure=fig)

# Add two elements in the first row, each spanning 3 columns
ax1 = fig.add_subplot(gs[0, :3])  # First subplot in the top-left (span 3 columns)
ax2 = fig.add_subplot(gs[0, 3:])  # Second subplot in the top-right (span 3 columns)

# Add three elements in the second row, each spanning 2 columns
ax3 = fig.add_subplot(gs[1, :2])  # Third subplot in the bottom-left (span 2 columns)
ax4 = fig.add_subplot(gs[1, 2:4])  # Fourth subplot in the bottom-center (span 2 columns)
ax5 = fig.add_subplot(gs[1, 4:])  # Fifth subplot in the bottom-right (span 2 columns)

# Create some sample plots
t = np.arange(0, 3, .01)
ax1.plot(t, 2 * np.sin(2 * np.pi * t))
ax2.plot(t, np.cos(2 * np.pi * t))
ax3.plot(t, np.sin(2 * np.pi * t))
ax4.plot(t, np.tan(2 * np.pi * t))
ax5.plot(t, 2 * np.cos(2 * np.pi * t))

# Embed the figure into the Tkinter canvas
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)

# Create toolbar
toolbar_frame = tkinter.Frame(root)
toolbar_frame.grid(row=1, column=0, columnspan=2)
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()

# Key press event handler
def on_key_press(event):
    print(f"you pressed {event.key}")

canvas.mpl_connect("key_press_event", on_key_press)

# Quit button
def _quit():
    root.quit()
    root.destroy()

button = tkinter.Button(master=root, text="Quit", command=_quit)
button.grid(row=2, column=0)

tkinter.mainloop()
