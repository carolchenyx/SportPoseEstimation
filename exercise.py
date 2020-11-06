import tkinter as tk
import pickle
import tkinter.messagebox
import cv2


window = tk.Tk()
window.title('Welcome to Exercise Camping')
window.geometry('1000x600')

canvas = tk.Canvas(window, height=200, width=500)
image_file = tk.PhotoImage(file='welcome.gif')
image = canvas.create_image(0,0, anchor='nw', image=image_file)
canvas.pack(side='top')

var_action = tk.StringVar()
var_action.set(('Free push-ups','Push-ups count for one minute','Free sit-ups','sit-ups count for one minute','Free squats','squats count for one minute'))
lb1 = tk.Listbox(window, selectmode=tk.SINGLE, listvariable=var_action, width =35, selectforeground = 'black')
lb1.place(x=50, y= 300)


usr_name = var_usr_name.get()
usr_pwd = var_usr_pwd.get()
try:
    with open('usrs_info.pickle', 'rb') as usr_file:
        usrs_info = pickle.load(usr_file)
except FileNotFoundError:
    with open('usrs_info.pickle', 'wb') as usr_file:
        usrs_info = {'admin': 'admin'}
        pickle.dump(usrs_info, usr_file)
if usr_name in usrs_info:
    if usr_pwd == usrs_info[usr_name]:
        tk.messagebox.showinfo(title='Welcome to our exercise camp', message='Are you ready? ' + usr_name)

    else:
        tk.messagebox.showerror(message='Error, your password is wrong, try again.')
else:

is_sign_up = tk.messagebox.askyesno('Welcome',
                       'You have not sign up yet. Sign up today?')
if is_sign_up:
    usr_sign_up()