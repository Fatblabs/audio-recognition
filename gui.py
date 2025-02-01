from tkinter import *

window = Tk()
window.geometry("300x300")
window.title("Audio Recognizer")
window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)
screen = Label(master = window)


peach = "#ff7a7b"
white = "#ffffff"
black = "#000000"

def init_start_screen() -> None:

    for l in window.grid_slaves():
        l.destroy()

    screen = Label(master = window)
    for col in range(6):
        screen.columnconfigure(col, weight=1)
        for row in range(10):
            screen.rowconfigure(row, weight=1)
            Label(
                master = screen,
                background = peach if col < 3 else white, 
            ).grid(row = row, column = col, sticky = "nsew")
    screen.grid(column=0, row=0, sticky="nsew")

    Label(
        master = screen,
        text = "The Audio ",
        anchor = E,
        font = "Arial 50 bold",
        foreground = white, 
        background = peach, 
    ).grid(row = 3, column = 1, sticky = "nsew")

    Label(
        master = screen,
        text = "🔊",
        anchor = E,
        font = "Arial 300 bold",
        foreground = white, 
        background = peach, 
    ).grid(row = 5, column = 1, sticky = "nsew")

    Label(
        master = screen, 
        text = "Recognizer",
        anchor = W,
        font = "Arial 50 bold",
        foreground = peach,
        background = white, 
    ).grid(row = 3, column = 4, sticky = "nsew")

    Label(
        master = screen,
        text = "🔍",
        anchor = E,
        font = "Arial 300 bold",
        foreground = peach, 
        background = white, 
    ).grid(row = 5, column = 4, sticky = "nsew")


    Button(
        master = screen,
        text = "Try Me!",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white, 
        command=init_import_screen,
    ).grid(row = 8, column = 1, sticky = "nsew")

    Button(
        master = screen,
        text = "Teach Me!",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white,  
    ).grid(row = 8, column = 4, sticky = "nsew")

def init_import_screen() -> None:

    for l in window.grid_slaves():
        l.destroy()
    
    screen = Label(master = window)
    for col in range(5):
        screen.columnconfigure(col, weight=1)#, minsize=330)
        for row in range(10):
            screen.rowconfigure(row, weight=1)#, minsize=100)
            Label(
                master = screen,
                background = peach
            ).grid(row = row, column = col, sticky = "nsew")
    screen.grid(column=0, row=0, sticky="nsew")

    Label(
        master = screen,
        text = "🔊 Import Audio Path 🔊",
        font = "Arial 50 bold",
        foreground = white, 
        background = peach, 
    ).grid(row = 1, column = 2, sticky = "nsew")

    Button(
        master = screen,
        text = "Return Home",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white, 
        command=init_start_screen, 
    ).grid(row = 8, column = 2, sticky = "nsew")

    input = Entry(
        master=screen, 
        font= "Arial 26 bold", 
        foreground = peach, 
        background = white, 
    
    )
    input.grid(row = 3, column = 2, sticky = "nsew")
    input.insert(0, "Your audio file path here")

    def clear_input(_):
        input.delete(0, "end")
        
    input.bind("<Button-1>", clear_input)

    
    


init_start_screen()
window.mainloop()