from tkinter import *
import threading
import time
import training

window = Tk()
window.geometry("300x300")
window.title("Audio Recognizer")
window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)
screen = Label(master = window)


peach = "#ff7a7b"
white = "#ffffff"
black = "#000000"

def type_slow(output : Label, btn : Button = None, text: str = None):
    if text == None:
        text = output['text']
    if btn != None:
        btn['state'] = DISABLED
    output['text'] = ""
    for i in range(len(text)):
        time.sleep(0.05)
        output['text'] += text[i]
    if btn != None:
        btn['state'] = NORMAL

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

    audio = Label(
        master = screen,
        text = "The Audio ",
        anchor = E,
        font = "Arial 50 bold",
        foreground = white, 
        background = peach, 
    )
    audio.grid(row = 3, column = 1, sticky = "nsew")
    threading.Thread(target = type_slow, args=(audio,)).start()


    Label(
        master = screen,
        text = "🔊",
        anchor = E,
        font = "Arial 300 bold",
        foreground = white, 
        background = peach, 
    ).grid(row = 5, column = 1, sticky = "nsew")

    recognizer = Label(
        master = screen, 
        text = "Recognizer",
        anchor = W,
        font = "Arial 50 bold",
        foreground = peach,
        background = white, 
    )
    recognizer.grid(row = 3, column = 4, sticky = "nsew")
    threading.Thread(target = type_slow, args=(recognizer,)).start()

    Label(
        master = screen,
        text = "🔍",
        anchor = E,
        font = "Arial 300 bold",
        foreground = peach, 
        background = white, 
    ).grid(row = 5, column = 4, sticky = "nsew")


    tryme = Button(
        master = screen,
        text = "Try Me!",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white, 
        command=init_import_screen,
    )
    tryme.grid(row = 8, column = 1, sticky = "nsew")
    threading.Thread(target = type_slow, args=(tryme,)).start()

    teachme = Button(
        master = screen,
        text = "Teach Me!",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white,  
        command = init_teach_screen,
    )
    teachme.grid(row = 8, column = 4, sticky = "nsew")
    threading.Thread(target = type_slow, args=(teachme,)).start()

def init_import_screen() -> None:

    for l in window.grid_slaves():
        l.destroy()
    
    screen = Label(master = window)
    for col in range(5):
        screen.columnconfigure(col, weight=1)#, minsize=330)
        for row in range(11):
            screen.rowconfigure(row, weight=1)#, minsize=100)
            Label(
                master = screen,
                background = peach
            ).grid(row = row, column = col, sticky = "nsew")
    screen.grid(column=0, row=0, sticky="nsew")

    title = Label(
        master = screen,
        text = "🔊 Import Audio Path 🔊",
        font = "Arial 50 bold",
        foreground = white, 
        background = peach, 
    )
    title.grid(row = 1, column = 2, sticky = "nsew")
    #threading.Thread(target = type_slow, args=(title,)).start()

    home = Button(
        master = screen,
        text = "Return Home",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white, 
        command=init_start_screen, 
    )
    home.grid(row = 9, column = 2, sticky = "nsew")
    threading.Thread(target = type_slow, args=(home,)).start()

    output = Label(
        master = screen,
        text = "🤖: ...",
        font = "Arial 26 bold",
        foreground = white, 
        background = peach, 
    )
    output.grid(row = 6, column = 2, sticky = "nsew")
    threading.Thread(target = type_slow, args=(output,)).start()

    btn = None

    def who_am_i():
        path = input.get().strip()
        #Add recognition method here. 
        t = threading.Thread(target = type_slow, args=(output, btn, "🤖: ...I don't know."))
        t.start()
        
    btn = Button(
        master = screen,
        text = "Who do you think am I?",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white, 
        command=who_am_i, 
    )
    btn.grid(row = 4, column = 2, sticky = "nsew")
    threading.Thread(target = type_slow, args=(btn,)).start()

    input = Entry(
        master=screen, 
        font= "Arial 26 bold", 
        foreground = peach, 
        background = white, 
    
    )
    input.grid(row = 3, column = 2, sticky = "nsew")
    input.insert(0, "Your audio file path here")
    threading.Thread(target = type_slow, args=(input,)).start()

    def clear_input(_):
        input.delete(0, "end")

    input.bind("<Button-1>", clear_input)

    
def init_teach_screen() -> None:

    for l in window.grid_slaves():
        l.destroy()
    
    screen = Label(master = window)
    for col in range(5):
        screen.columnconfigure(col, weight=1)#, minsize=330)
        for row in range(11):
            screen.rowconfigure(row, weight=1)#, minsize=100)
            Label(
                master = screen,
                background = white
            ).grid(row = row, column = col, sticky = "nsew")
    screen.grid(column=0, row=0, sticky="nsew")

    title = Label(
        master = screen,
        text = "🔍 Teach Me! 🔍",
        font = "Arial 50 bold",
        foreground = peach, 
        background = white, 
    )
    title.grid(row = 1, column = 2, sticky = "nsew")
    #threading.Thread(target = type_slow, args=(title,)).start()

    home = Button(
        master = screen,
        text = "Return Home",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white, 
        command=init_start_screen, 
    )
    home.grid(row = 9, column = 2, sticky = "nsew")
    threading.Thread(target = type_slow, args=(home,)).start()

    output = Label(
        master = screen,
        text = "🤖: ...",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white, 
    )
    output.grid(row = 6, column = 2, sticky = "nsew")
    threading.Thread(target = type_slow, args=(output,)).start()

    btn = None

    def teach_me():
        #Add teach method call here. 
        path = input.get().strip()
        if training.setDataSetPath(path):
            t = threading.Thread(target = type_slow, args=(output, btn, f"🤖: ...I am learning."))
            #C:\Users\duozh\Data Structures\Competitive Programming\alphabet.py
            t.start()
            threading.Thread(target = training.train).start()

        t = threading.Thread(target = type_slow, args=(output, btn, f"🤖: ...I can't find the folder."))
        t.start()
        
    btn = Button(
        master = screen,
        text = "Teach!",
        font = "Arial 26 bold",
        foreground = peach, 
        background = white, 
        command=teach_me, 
    )
    btn.grid(row = 4, column = 2, sticky = "nsew")
    threading.Thread(target = type_slow, args=(btn,)).start()

    input = Entry(
        master=screen, 
        font= "Arial 26 bold", 
        foreground = peach, 
        background = white, 
    
    )
    input.grid(row = 3, column = 2, sticky = "nsew")
    input.insert(0, "Your audio file path here")
    threading.Thread(target = type_slow, args=(input,)).start()

    def clear_input(_):
        input.delete(0, "end")

    input.bind("<Button-1>", clear_input)
    


init_start_screen()
window.mainloop()