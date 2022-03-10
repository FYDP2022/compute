import argparse
import os
import matplotlib.image as mpimg

from vslam.app import App
from vslam.config import CONFIG, DebugWindows

def main():
  parser = argparse.ArgumentParser(description='VSLAM runner.')
  parser.add_argument('-d', '--debug', dest='debug', type=str)
  parser.add_argument('-c', '--clear', dest='clear', action='store_true')
  parser.add_argument('-v', '--visualize', dest='visualize', action='store_true')
  parser.add_argument('-q', '--quit', dest='quit', action='store_true')
  args = parser.parse_args()

  CONFIG.windows = DebugWindows.parse(args.debug)

  app = App()

  if args.visualize:
    mpimg.imsave(os.path.join(CONFIG.databasePath, 'map.png'), app.visualize())

  if args.clear:
    app.clear()

  if not args.quit:
    app.run()
  else:
    app.close()

if __name__ == '__main__':
  main()
