from abc import ABC, abstractmethod
from logging import exception
from queue import Queue
import threading
from typing import Optional, Union
import serial

class SerialValueException(ValueError):
  pass

class WriteSerialCommandInterface(ABC):
  output_serial_string = ""
  module_name = ""
  
  @abstractmethod
  def build_serial_command(self):
    pass
  
  def invalid_command_error_msg(self, field):
    return "Invalid incoming " + field + " for " + self.module_name
  
  def check_num_valid_boundaries(self, num, left_bound, right_bound) -> bool:
    if num < left_bound or num > right_bound:
      return False
    return True
  
  def provide_command(self):
    return self.output_serial_string

#Requires Direction, bias, and optionally distance
class DriveMotorCommand(WriteSerialCommandInterface):
  def __init__(self, direction: str, bias: int, distance=200) -> str:
    self.module_name = "Drive Motor Command"
    self.build_serial_command(direction=direction, bias=bias, distance=distance)
    
  def check_return_direction(self, direction: str):
    valid_direction_list = [
      "FORWARD", "REVERSE", "STOP", "POINT_LEFT", "POINT_RIGHT", 
      "FWD_LEFT", "FWD_RIGHT", "BWD_LEFT", "BWD_RIGHT"
    ]
    if direction.upper() not in valid_direction_list:
      return "INVALID"
    else:
      return direction.upper()
    
  def check_return_bias(self, bias: int):
    if self.check_num_valid_boundaries(bias, 0, 100):
      return bias
    else:
      return 0
      
  def check_return_distance(self, distance: int):
    if self.check_num_valid_boundaries(distance, -10000, 10000):
      return distance
    else:
      return 1000
    
  def build_serial_command(self, direction, bias, distance):
    self.output_serial_string += "D:"
    self.output_serial_string += self.check_return_direction(direction) + ":"
    self.output_serial_string += str(self.check_return_bias(bias)) + ":"
    self.output_serial_string += str(self.check_return_distance(distance))

#parent class, don't instantiate
class OnOffCommand(WriteSerialCommandInterface):
  def __init__(self, on_off: str, module_name: str, module_id: str):
    self.module_name = module_name
    self.build_serial_command(on_off=on_off, module_id=module_id)
    
  def check_return_on_off(self, on_off: str):
    valid_direction_list = ["ON", "OFF"]
    if on_off.upper() not in valid_direction_list:
      self.invalid_command_error_msg("on/off ID")
    else:
      return on_off.upper()
    
  def build_serial_command(self, on_off, module_id):
    self.output_serial_string += module_id + ":"
    self.output_serial_string += self.check_return_on_off(on_off)
    

#Requires ON or OFF string
class RelayCommand(OnOffCommand):
  def __init__(self, on_off: str):
    super().__init__(on_off=on_off, module_name="Relay Command", module_id="R")

#Requires ON or OFF string     
class BladeMotorCommand(OnOffCommand):
   def __init__(self, on_off: str):
    super().__init__(on_off=on_off, module_name="Blade Motor Command", module_id="B")

class IncomingErrorMessage():
  def __init__(self, module: str, type: str, msg: str):
    self.incoming_desig = "ERROR"
    self.module = module
    self.type = type
    self.msg = msg
    pass
  
class IncomingSensorReading():
  def __init__(self, module: str, info_1: str, info_2: str):
    if module == "ULTRASONIC":
      self.generate_ultrasonic_struct(direction=info_1, distance=info_2)
    elif module == "TEMPERATURE":
      self.generate_temperature_struct(probe=info_1, temperature=info_2)
    elif module == "GYRO":
      self.generate_gyro_struct(tilt=info_2)
    else:
      raise ValueError("Invalid module for incoming sensor reading")
    
  def generate_ultrasonic_struct(self, direction: str, distance: str):
    self.incoming_desig = "SENSOR_DATA"
    self.module = "ULTRASONIC"
    self.direction = direction
    self.distance = distance
    
  def generate_temperature_struct(self, probe: str, temperature: str):
    self.incoming_desig = "SENSOR_DATA"
    self.module = "TEMPERATURE"
    self.probe = probe
    self.temperature = temperature
  
  def generate_gyro_struct(self, tilt: str):
    self.incoming_desig = "SENSOR_DATA"
    self.module = "GYRO"
    self.tilt = tilt

class ReadSerialCommandController():
  def return_incoming_message_struct(self, incoming_msg: str) -> Union[IncomingErrorMessage, IncomingSensorReading]:
    tokens = incoming_msg.split(":")
    if len(tokens) != 4:
      raise ValueError("Incorrect number of incoming message tokens")
    
    msg_purpose = tokens[0]
    if msg_purpose == "ERROR":
      return IncomingErrorMessage(module=tokens[1], type=tokens[2], msg=tokens[3])
    elif msg_purpose == "SENSOR_DATA":
      return IncomingSensorReading(module=tokens[1], info_1=tokens[2], info_2=tokens[3])
    else:
      raise SerialValueException("INVALID INCOMING MESSAGE PURPOSE (NOT ERROR OR SENSOR_DATA)")

class SerialInterface:  
  def __init__(self, client) -> 'SerialInterface':
    self.client = client
    self.read_controller = ReadSerialCommandController()
    try:
      self.device = serial.Serial(port='/dev/ttyUSB0', baudrate=115200)
      print('[Serial] Connection established...')
    except Exception as e:
      print("[Serial] ERROR: {}".format(e))
      self.device = None
    
    if self.device is not None:
      self.thread = threading.Thread(target=self._runner)
      self.thread.start()
  
  def _runner(self):
    while True:
      try:
        msg = self.recv_message()
        if msg is not None:
          if msg.incoming_desig == 'ERROR':
            pass
          elif msg.incoming_desig == 'SENSOR_DATA':
            if msg.module == 'ULTRASONIC':
              self.client.publish_ultrasonic(msg.direction, msg.distance)
            elif msg.module == 'TEMPERATURE':
              self.client.publish_temperature(msg.probe, msg.temperature)
            elif msg.module == 'GYRO':
              self.client.publish_gyro(msg.tilt)
      except SerialValueException as e:
        print("[Serial] ERROR: {}".format(e))

  def recv_message(self) -> Union[IncomingErrorMessage, IncomingSensorReading]:
    line = self.device.readline().decode('utf-8')
    print(line)
    return self.read_controller.return_incoming_message_struct(line)
  
  def write_message(self, command: WriteSerialCommandInterface):
    if self.device is not None:
      self.device.write(command.provide_command())
