from abc import ABC, abstractmethod
from logging import exception
from queue import Queue
import threading
from typing import Optional, Union
import usb.core
import usb.util

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
      raise ValueError("INVALID INCOMING MESSAGE PURPOSE (NOT ERROR OR SENSOR_DATA)")

class SerialInterface:  
  def __init__(self) -> 'SerialInterface':
    self.device: usb.core.Device = usb.core.find(idVendor=0x2341) # Vendor: Arduino SA
    self.endpoint: Optional[usb.core.Endpoint] = None
    self.queue = Queue(1000)
    self.read_controller = ReadSerialCommandController()
    if self.device is not None:
      self.device.set_configuration()
      cfg = self.device.get_configuration()
      intf = cfg[(0, 0)]
      self.endpoint = usb.util.find_descriptor(
        intf,
        # match the first OUT endpoint
        custom_match = lambda e:
          usb.util.endpoint_direction(e.bEndpointAddress) ==
          usb.util.ENDPOINT_OUT
      )
      self.thread = threading.Thread(target=self._runner)
    
  
  def _runner(self):
    while True:
      try:
        self.queue.put(self.recv_message())
      except SerialValueException as e:
        pass

  def recv_message(self) -> Union[IncomingErrorMessage, IncomingSensorReading]:
    payload = self.device.read(0x81, [])
    # Are we reading until the \n character and is this a blocking or non-blocking read?
    return self.read_controller.return_incoming_message_struct(''.join([chr(x) for x in payload]))
  
  def write_message(self, command: WriteSerialCommandInterface):
    self.endpoint.write(command.provide_command())
