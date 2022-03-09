

import base64
import math
import os
from queue import Queue
from typing import Tuple
import numpy as np
import cv2 as cv
import paho.mqtt.client as mqtt

from vslam.config import CONFIG
from vslam.serial import OnOffCommand, RelayCommand, SerialInterface
from vslam.state import AppState, State
from vslam.utils import Y_AXIS, Z_AXIS, angle_between_about, normalize

class MQTTClient(mqtt.Client):
  def __init__(self, state: AppState, map_x1: float, map_x2: float, map_z1: float, map_z2: float) -> 'MQTTClient':
    super().__init__()
    self.app_state = state
    self.serial = None
    self.map_x1 = map_x1
    self.map_x2 = map_x2
    self.map_z1 = map_z1
    self.map_z2 = map_z2
    self.connect("localhost", 1883, 5)
    self.subscribe("StartStopTopic", 2)
    self.loop_start()
  
  def initialize(self, serial: SerialInterface):
    self.serial = serial
  
  def on_connect(self, mqttc, obj, flags, rc):
    print("rc: " + str(rc))

  def on_connect_fail(self, mqttc, obj):
    print("Connect failed")

  def on_message(self, mqttc, obj, msg):
    print("{} {} {}".format(msg.topic, str(msg.qos), str(msg.payload.decode("utf-8"))))
    
    if msg.topic == "StartStopTopic":
      incoming = str(msg.payload.decode("utf-8"))
      if incoming == "START":
        #ADD LAWNY STARTING CODE HERE
        self.serial.write_message(RelayCommand('ON'))
        self.app_state.active = True
        print("Lawny Started message received - Execution started")
      elif incoming == "STOP":
        #ADD LAWNY STOPPING CODE (BEFORE MQTT DISCONNECT, AND CHECK IF SENT BEFORE DISCONNECTION OCCURS)
        self.disconnect()
        print("Lawny Stop message received - Lawny execution shut down")
      else:
        print("INVALID STOP/START COMMAND")

  def on_publish(self, mqttc, obj, mid):
    print("Data Published")

  def on_subscribe(self, mqttc, obj, mid, granted_qos):
    print("Subscribed to new Topic")
                  
  def publish_temperature(self, probe: str, temp_reading: str):
    send_msg = probe + ":" + temp_reading
    self.publish("TemperatureTopic", send_msg, qos=2)

  def publish_ultrasonic(self, direction: str, distance: str):
    send_msg = direction + ":" + distance
    self.publish("UltrasonicTopic", send_msg, qos=2)
  
  def publish_gyro(self, tilt: str):
    send_msg = "tilt:" + tilt
    self.publish("GyroTopic", send_msg, qos=2)

  def publish_battery(self, battery: str):
    self.publish("BatteryTopic", battery, qos=2)
      
  def publish_image(self):
    image = cv.imread(os.path.join(CONFIG.databasePath, 'map.png'))
    jpg_img = cv.imencode('.jpg', image)
    encoded = base64.b64encode(jpg_img[1]).decode('utf-8')
    self.publish("ImageTopic", encoded, qos=2)
      
  def publish_position(self, position_x: int, position_y: int, angle: str):
    self.publish("PositionTopic", "{}:{}:{}".format(position_x, position_y, angle), qos=2)

  def update_map_state(self, state: State):
    forward = state.forward.copy()
    forward[1] = 0.0
    forward = normalize(forward)
    theta = -angle_between_about(forward, -Z_AXIS, Y_AXIS)
    delta_map_x = self.map_x2 - self.map_x1
    x = np.clip((state.position[0] - self.map_x1) / delta_map_x, 0.0, 1.0)
    x = math.floor(x * CONFIG.map_width)
    delta_map_z = self.map_z2 - self.map_z1
    y = 1.0 - np.clip((state.position[2] - self.map_z1) / delta_map_z, 0.0, 1.0)
    y = math.floor(y * CONFIG.map_height)
    return self.publish_position(x, y, int(np.rad2deg(theta)))
