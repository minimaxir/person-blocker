import os
import sys
import argparse
import numpy as np
import coco
import utils
import model as modellib
from classes import get_class_names, InferenceConfig
from ast import literal_eval as make_tuple
import imageio
import visualize

# Creates a color layer and adds Gaussian noise.
# For each pixel, the same noise value is added to each channel
# to mitigate hue shfting.


def create_noisy_color(image, color):
    color_mask = np.full(shape=(image.shape[0], image.shape[1], 3),
                         fill_value=color)

    noise = np.random.normal(0, 25, (image.shape[0], image.shape[1]))
    noise = np.repeat(np.expand_dims(noise, axis=2), repeats=3, axis=2)
    mask_noise = np.clip(color_mask + noise, 0., 255.)
    return mask_noise


# Helper function to allow both RGB triplet + hex CL input

def string_to_rgb_triplet(triplet):

    if '#' in triplet:
        # http://stackoverflow.com/a/4296727
        triplet = triplet.lstrip('#')
        _NUMERALS = '0123456789abcdefABCDEF'
        _HEXDEC = {v: int(v, 16)
                   for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
        return (_HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]],
                _HEXDEC[triplet[4:6]])

    else:
        # https://stackoverflow.com/a/9763133
        triplet = make_tuple(triplet)
        return triplet


def person_blocker(args):
    COCO_MODEL_PATH = args.model

    # Required to load model, but otherwise unused
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Load model and config
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    image = imageio.imread(args.image)

    # Create masks for all objects
    results = model.detect([image], verbose=0)
    r = results[0]

    if args.labeled:
        position_ids = ['[{}]'.format(x)
                        for x in range(r['class_ids'].shape[0])]
        visualize.display_instances(image, r['rois'],
                                    r['masks'], r['class_ids'],
                                    get_class_names(), position_ids)
        sys.exit()

    # Filter masks to only the selected objects
    objects = np.array(args.objects)

    # Object IDs:
    if np.all(np.chararray.isnumeric(objects)):
        object_indices = objects.astype(int)
    # Types of objects:
    else:
        selected_class_ids = np.flatnonzero(np.in1d(get_class_names(),
                                                    objects))
        object_indices = np.flatnonzero(
            np.in1d(r['class_ids'], selected_class_ids))

    mask_selected = np.sum(r['masks'][:, :, object_indices], axis=2)

    # Replace object masks with noise
    mask_color = string_to_rgb_triplet(args.color)
    image_masked = image.copy()
    noisy_color = create_noisy_color(image, mask_color)
    image_masked[mask_selected == 1] = noisy_color[mask_selected == 1]

    imageio.imwrite('person_blocked.png', image_masked)

    # Create GIF. The noise will be random for each frame,
    # which creates a "static" effect

    images = [image_masked]
    num_images = 10   # should be a divisor of 30

    for _ in range(num_images - 1):
        new_image = image.copy()
        noisy_color = create_noisy_color(image, mask_color)
        new_image[mask_selected == 1] = noisy_color[mask_selected == 1]
        images.append(new_image)

    imageio.mimsave('person_blocked.gif', images, fps=30., subrectangles=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Person Blocker - Automatically "block" people '
                    'in images using a neural network.')
    parser.add_argument('-i', '--image',  help='Image file name.',
                        required=True)
    parser.add_argument(
        '-m', '--model',  help='Path to COCO model', default='/')
    parser.add_argument('-o',
                        '--objects', nargs='+',
                        help='Object(s)/Object ID(s) to block',
                        default='person')
    parser.add_argument('-c',
                        '--color', nargs='?', default='(255, 255, 255)',
                        help='Color of the "block"')
    parser.add_argument('-l',
                        '--labeled', dest='labeled',
                        action='store_true',
                        help='Generate labeled image instead')
    parser.set_defaults(labeled=False)
    args = parser.parse_args()
    person_blocker(args)
