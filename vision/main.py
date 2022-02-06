import argparse
import cv2 as cv

from vslam.camera import StereoCamera
from vslam.depth import DepthEstimator
from vslam.config import CONFIG, DebugWindows

def main():
  parser = argparse.ArgumentParser(description='VSLAM runner.')
  parser.add_argument('-d', '--debug', dest='debug', type=str)
  args = parser.parse_args()

  CONFIG.windows = DebugWindows.parse(args.debug)

  camera = StereoCamera(960, 540)
  depth = DepthEstimator(960, 540)

  while True:
    left, right = camera.read()
    grayL = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
    depth_image = depth.process(grayL, grayR)
    cv.waitKey(30)

if __name__ == '__main__':
  main()
