import argparse
import os

from vslam.app import App
from vslam.config import CONFIG, DebugWindows

def main():
  parser = argparse.ArgumentParser(description='VSLAM runner.')
  parser.add_argument('-d', '--debug', dest='debug', type=str)
  parser.add_argument('-c', '--clear', dest='clear', action='store_true')
  parser.add_argument('-q', '--quit', dest='quit', action='store_true')
  args = parser.parse_args()

  CONFIG.windows = DebugWindows.parse(args.debug)

  app = App()    

  if args.clear:
    app.clear()

  if not args.quit:
    app.run()
  else:
    app.close()

if __name__ == '__main__':
  main()
