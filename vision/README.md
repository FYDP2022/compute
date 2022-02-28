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

## Attribution

```
@InProceedings{tas:metzger2020icpr-dataset-semantic-segmentation,
  author    = {Kai A. Metzger AND Peter Mortimer AND Hans-Joachim Wuensche},
  title     = {{A Fine-Grained Dataset and its Efficient Semantic Segmentation for Unstructured Driving Scenarios}},
  booktitle = {International Conference on Pattern Recognition (ICPR2020)},
  year      = {2021},
  address   = {Milano, Italy (Virtual Conference)},
  month     = jan,
}
```