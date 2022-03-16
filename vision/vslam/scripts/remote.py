from tkinter import *
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("localhost", 1883, 30)

win = Tk()

win.geometry("700x350")

def key_released(e):
  if e.char == 'w':
    print('Forward')
    client.publish('RCControl', 'FORWARD')
  elif e.char == 'a':
    print('Left')
    client.publish('RCControl', 'LEFT')
  elif e.char == 's':
    print('Back')
    client.publish('RCControl', 'BACK')
  elif e.char == 'd':
    print('Right')
    client.publish('RCControl', 'RIGHT')


win.bind('<KeyRelease>', key_released)
win.mainloop()