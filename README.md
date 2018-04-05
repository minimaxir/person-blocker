# Person Blocker

![img4](example_output/img4_blocked.gif)

A script to automatically "block" people in images (like the [Black Mirror](https://en.wikipedia.org/wiki/Black_Mirror) episode [White Christmas](https://en.wikipedia.org/wiki/White_Christmas_(Black_Mirror))) using [Mask R-CNN](https://github.com/matterport/Mask_RCNN) pretrained on the [MS COCO](https://arxiv.org/abs/1405.0312) dataset. No GPU required!

Capable of blocking more than just people: up to [80 different types](https://github.com/minimaxir/person-blocker/blob/master/classes.py) of objects can be blocked, including giraffes and busses!

## Setup

This project relies on some dependencies. Use the command below to install the dependencies:

```shell
pip3 install -r requirements.txt
```

_Note_: Depending on the environment, you may need to use `sudo` or virtualenv.

## Usage

Person Blocker is used from the command line:

```shell
python3 person_blocker.py -i images/img3.jpg -c '(128, 128, 128)' -o 'bus' 'truck'

* `-i/--image`: specifies the image file.
* `-m/--model`: path to the pretrained COCO model weights (default: current directory): If a path is not specified and the file doesn't exist in the current directory, the file containing the pretrained COCO model weights will be downloaded. (note: the weights are 258 MB!)
* `-c/--color`: color of the mask, in either quote-wrapped hexidecimal or 3-element RGB tuple format. (default: white)
* `-o/--object`: list of types of objects to block (or object IDs of specific objects). You can see the allowable choices of objects to block in `classes.py` or by using the `-names` flag. (default: person)
* `-l/--labeled`: saves a labeled image annotated with detected objects and their object ID.
* `-n/--names`: prints the class options for objects, then exits.

The script outputs two images: a static (pun intended) image `person_blocked.png` and an animated image `person_blocked.gif` like the one at the beginning of this README.

## Examples

```shell
python3 person_blocker.py -i images/img1.jpg
```

![img1](example_output/img1_blocked.png)

```shell
python3 person_blocker.py -i images/img2.jpg -c '#c0392b' -o 'giraffe'
```

![img2](example_output/img2_blocked.png)

```shell
python3 person_blocker.py -i images/img3.jpg -c '(128, 128, 128)' -o 'bus' 'truck'
```

![img3](example_output/img3_blocked.png)

Blocking specific object(s) requires 2 steps: running in inference mode to get the object IDs for each object, and then blocking those object IDs.

```shell
python3 person_blocker.py -i images/img4.jpg -l
```

![img4 labels](example_output/img4_labels.png)

```shell
python3 person_blocker.py -i images/img4.jpg -o 1
```

![img4](example_output/img4_blocked.png)

## Requirements

The same requirements as Mask R-CNN:
* Python 3.4+
* TensorFlow 1.3+
* Keras 2.0.8+
* Numpy, skimage, scipy, Pillow, cython, h5py

plus matplotlib and imageio

## Maintainer

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## License

MIT

Code used from Mask R-CNN by Matterport, Inc. (MIT-Licensed), with minor alterations and copyright notices retained.
