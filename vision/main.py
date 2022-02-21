import argparse

from vslam.app import App
from vslam.config import CONFIG, DebugWindows

def main():
  parser = argparse.ArgumentParser(description='VSLAM runner.')
  parser.add_argument('-d', '--debug', dest='debug', type=str)
  args = parser.parse_args()

  CONFIG.windows = DebugWindows.parse(args.debug)

  app = App()
  app.run()

if __name__ == '__main__':
  main()
