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
import face_recognition
from pathlib import Path

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


def loadKnown(pathstr):
    known_face_encodings = []
    pathlist = Path(pathstr).glob('**/*.jpg')
    for p in pathlist:
        p_str = str(p)
        im = face_recognition.load_image_file(p_str)
        encoding = face_recognition.face_encodings(im)[0]
        known_face_encodings.append(encoding)
    return known_face_encodings

def faceRecog(orig_img, known_face_encodings):
    face_locations = []
    face_encodings = []
    face_names = []
    img = orig_img[:, :, ::-1]

    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            name = "Hit"
        face_names.append(name)

    res = []
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name == "Hit":
            x1 = left
            y1 = top
            x2 = right
            y2 = bottom
            res.append([x1, y1, x2, y2])
    return res


def person_blocker(args):

    # Required to load model, but otherwise unused
    ROOT_DIR = os.getcwd()
    COCO_MODEL_PATH = args.model or os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")  # Required to load model

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
    
    # Do Face Recognition and get the Object ID
    if args.facerecog:
        if not args.known_path:
            print('No known path specified')
            sys.exit()
        known_face_encodings = loadKnown(args.known_path)
        detected_face_locations = faceRecog(image, known_face_encodings)
        rois = r['rois']
        tmp_objects = []
        for face_loc in detected_face_locations:
            for i,ri in enumerate(rois):
                x1 = ri[1]
                y1 = ri[0]
                x2 = ri[3]
                y2 = ri[2]
                #check if the face boundaries are within the boundary of the rois
                if face_loc[0] >= x1 and face_loc[1] <= x2 and face_loc[2] >= x1 and face_loc[2] <= x2:
                    if face_loc[1] >= y1 and face_loc[1] <= y2 and face_loc[3] >= y1 and face_loc[3] <= y2:
                        tmp_objects.append(str(i))

    if args.labeled:
        position_ids = ['[{}]'.format(x)
                        for x in range(r['class_ids'].shape[0])]
        visualize.display_instances(image, r['rois'],
                                    r['masks'], r['class_ids'],
                                    get_class_names(), position_ids)
        sys.exit()

    # Filter masks to only the selected objects
    if args.facerecog:
        #that face was not found in the image so dont do anything
        if len(tmp_objects) == 0:
            imageio.imwrite('person_blocked.png', image)
            return
        objects = np.array(tmp_objects)
    else:
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
    image_masked[mask_selected > 0] = noisy_color[mask_selected > 0]

    imageio.imwrite('person_blocked.png', image_masked)

    # Create GIF. The noise will be random for each frame,
    # which creates a "static" effect

    images = [image_masked]
    num_images = 10   # should be a divisor of 30

    for _ in range(num_images - 1):
        new_image = image.copy()
        noisy_color = create_noisy_color(image, mask_color)
        new_image[mask_selected > 0] = noisy_color[mask_selected > 0]
        images.append(new_image)

    imageio.mimsave('person_blocked.gif', images, fps=30., subrectangles=True)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Person Blocker - Automatically "block" people '
                    'in images using a neural network.')
    parser.add_argument('-i', '--image',  help='Image file name.',
                        required=False)
    parser.add_argument(
        '-m', '--model',  help='path to COCO model', default=None)
    parser.add_argument('-o',
                        '--objects', nargs='+',
                        help='object(s)/object ID(s) to block. ' +
                        'Use the -names flag to print a list of ' +
                        'valid objects',
                        default='person')
    parser.add_argument('-c',
                        '--color', nargs='?', default='(255, 255, 255)',
                        help='color of the "block"')
    parser.add_argument('-l',
                        '--labeled', dest='labeled',
                        action='store_true',
                        help='generate labeled image instead')
    parser.add_argument('-n',
                        '--names', dest='names',
                        action='store_true',
                        help='prints class names and exits.')
    parser.add_argument('-f',
                        '--facerecog', dest='facerecog',
                        action='store_true',
                        help='Does face recognition')
    parser.add_argument('-k',
                        '--known', dest='known_path',
                        help='Path to known images', required=False)
    parser.set_defaults(labeled=False, names=False)
    args = parser.parse_args()

    if args.names:
        print(get_class_names())
        sys.exit()

    person_blocker(args)
