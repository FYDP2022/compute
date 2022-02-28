# vision

## Install Instructions

```
$ sudo apt install libspatialindex-dev
$ pip3 install -r requirements.txt
```

## Unit tests

```
$ python3 -m unittest tests
```

## Calibration

```
$ python3 -m vslam.scripts.calibrate -cbx 9 -cby 6 -cbw 22 -cp "./data/calibration/"
```