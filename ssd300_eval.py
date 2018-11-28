import os
import json
import shutil
from collections import defaultdict

from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 1 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
classes = ['background', 'person']

# TODO: Set the paths to the datasets here.
dataset_path = '/home/share/dataset/DarkData'

# The directories that contain the images.
DARK_val_day_images_dir = os.path.join(dataset_path, 'images/val_day')
DARK_val_night_images_dir = os.path.join(dataset_path, 'images/val_night')

# The directories that contain the annotations.
DARK_val_day_annotations_dir = os.path.join(dataset_path, 'labels/val_day')
DARK_val_night_annotations_dir = os.path.join(dataset_path, 'labels/val_night')

class_names = ['background', 'person']
tmp_pred_files_path = "tmp_pred_files"
if not os.path.exists(tmp_pred_files_path):
    os.mkdir(tmp_pred_files_path)

tmp_val_gt_day_files_path = "tmp_val_gt_day_files"
tmp_val_gt_night_files_path = "tmp_val_gt_night_files"
os.mkdir(tmp_val_gt_day_files_path)
os.mkdir(tmp_val_gt_night_files_path)


def read_txt_file(txt, tmp_val_gt_files_path):
    """
    :param txt: annotations file
    :param tmp_val_gt_files_path: data ground truth path
    :return:
        1) images list
        2) classes counter
    """
    # Read validation data
    with open(txt) as f:
        lines = f.readlines()

    # Val day data
    images = []
    gt_counter_per_class = defaultdict(int)
    for line in lines:
        save_bboxes = []
        image, *bboxes = line.split()
        file_id = os.path.split(image)[-1].split('.jpg')[0]
        images.append(image)
        for bbox in bboxes:
            left, top, right, bottom, class_id = bbox.split(',')
            class_name = class_names[int(class_id) + 1]
            bbox = "{} {} {} {}".format(left, top, right, bottom)
            save_bboxes.append({"class_name": class_name, "bbox": bbox, "used": False})
            gt_counter_per_class[class_name] += 1

        with open(tmp_val_gt_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(save_bboxes, outfile)

    return images, gt_counter_per_class


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def eval(images_dir, annotations_dir, ground_truth_path, gt_counter_per_class):
    """
    :param step:
    :param eval_images_path: image_path list
    :param ground_truth_path:
    :param gt_counter_per_class:
    :param tag:
    :return:
    """
    # Add the class predict temp dict
    class_pred_tmp = {}
    for class_name in class_names:
        class_pred_tmp[class_name] = []

    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset.parse_txt(images_dirs=[images_dir],
                          annotations_dirs=[annotations_dir],
                          classes=classes,
                          ret=False)

    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)
    val_dataset_size = val_dataset.get_dataset_size()
    print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

    # 1: Set the generator for the predictions.
    predict_generator = val_dataset.generate(batch_size=1,
                                             shuffle=False,
                                             transformations=[convert_to_3_channels,
                                                              resize],
                                             label_encoder=None,
                                             returns={'processed_images',
                                                      'filenames',
                                                      'inverse_transform',
                                                      'original_images',
                                                      'original_labels'},
                                             keep_images_without_gt=False)

    # Predict!!!
    for _ in range(val_dataset_size):
        batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)
        file_id = os.path.split(batch_filenames[0])[-1].split('.jpg')[0]

        y_pred = model.predict(batch_images)
        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.45,
                                           iou_threshold=0.45,
                                           top_k=200,
                                           normalize_coords=normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width)
        y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)
        if y_pred_decoded_inv[0].size != 0:
            for box in y_pred_decoded_inv[0]:
                predicted_class = class_names[int(box[0])]
                score = box[1]
                left = box[2]
                top = box[3]
                right = box[4]
                bottom = box[5]
                bbox = "{} {} {} {}".format(left, top, right, bottom)
                class_pred_tmp[predicted_class].append({"confidence": str(score), "file_id": file_id, "bbox": bbox, 'pred': 0})

    # Create predict temp
    predict_nums = {}
    for class_name in class_names:
        with open(tmp_pred_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
            predict_nums[class_name] = len(class_pred_tmp[class_name])
            json.dump(class_pred_tmp[class_name], outfile)

    # calculate the AP for each class
    sum_AP = 0.0
    count_true_positives = {}
    for class_index, class_name in enumerate(sorted(gt_counter_per_class.keys())):
        count_true_positives[class_name] = 0

        # load predictions of that class
        predictions_file = tmp_pred_files_path + "/" + class_name + "_predictions.json"
        predictions_data = json.load(open(predictions_file))
        # Assign predictions to ground truth objects
        nd = len(predictions_data)      # number of predict data
        tp = [0] * nd                   # true positive
        fp = [0] * nd                   # false positive
        for idx, prediction in enumerate(predictions_data):
            file_id = prediction["file_id"]
            gt_file = ground_truth_path + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load prediction bounding-box
            bb = [float(x) for x in prediction["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    # Area of Overlap
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    # compute overlap (IoU) = area of intersection / area of union
                    if iw > 0 and ih > 0:
                        # Area of Union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                             (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            if ovmax >= 0.5:
                if not gt_match['used']:
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_name] += 1
                    gt_match['pred'] = prediction
                    with open(gt_file, 'w') as f:
                        f.write(json.dumps(ground_truth_data))
                else:
                    fp[idx] = 1
                    prediction['pred'] = 1
            else:
                fp[idx] = 1
                prediction['pred'] = 2

        with open(predictions_file, 'w') as outfile:
            json.dump(predictions_data, outfile)

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += ap

    mAP = sum_AP / len(gt_counter_per_class)

    # remove the tmp_files directory
    shutil.rmtree(tmp_pred_files_path)
    shutil.rmtree(ground_truth_path)
    os.mkdir(tmp_pred_files_path)
    return mAP * 100

# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'models_save/ssd300_DarkDataset.h5'
# model_path = 'models_save/ssd300_coco.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
K.clear_session() # Clear previous models from memory.
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})


# val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
# val_dataset.parse_txt(images_dirs=[DARK_val_day_images_dir, DARK_val_night_images_dir],
#                       annotations_dirs=[DARK_val_day_annotations_dir, DARK_val_night_annotations_dir],
#                       classes=classes,
#                       ret=False)
#
# convert_to_3_channels = ConvertTo3Channels()
# resize = Resize(height=img_height, width=img_width)
#
# # 1: Set the generator for the predictions.
# predict_generator = val_dataset.generate(batch_size=1,
#                                          shuffle=False,
#                                          transformations=[convert_to_3_channels,
#                                                           resize],
#                                          label_encoder=None,
#                                          returns={'processed_images',
#                                                   'filenames',
#                                                   'inverse_transform',
#                                                   'original_images',
#                                                   'original_labels'},
#                                          keep_images_without_gt=False)
#
# # 2: Generate samples.
# batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)
# i = 0 # Which batch item to look at
# print("Image:", batch_filenames[i])
# print()
# print("Ground truth boxes:\n")
# print(np.array(batch_original_labels[i]))
#
# # 3: Make predictions.
# y_pred = model.predict(batch_images)
#
# # 4: Decode the raw predictions in `y_pred`.
# y_pred_decoded = decode_detections(y_pred,
#                                    confidence_thresh=0.5,
#                                    iou_threshold=0.4,
#                                    top_k=200,
#                                    normalize_coords=normalize_coords,
#                                    img_height=img_height,
#                                    img_width=img_width)
#
# # 5: Convert the predictions for the original image.
# y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)
# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('   class   conf xmin   ymin   xmax   ymax')
# print(y_pred_decoded_inv[i])
#
# # 5: Draw the predicted boxes onto the image
# # Set the colors for the bounding boxes
# colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
# plt.figure(figsize=(20,12))
# plt.imshow(batch_original_images[i])
#
# current_axis = plt.gca()
# for box in batch_original_labels[i]:
#     xmin = box[1]
#     ymin = box[2]
#     xmax = box[3]
#     ymax = box[4]
#     label = '{}'.format(classes[int(box[0])])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
#
# for box in y_pred_decoded_inv[i]:
#     xmin = box[2]
#     ymin = box[3]
#     xmax = box[4]
#     ymax = box[5]
#     color = colors[int(box[0])]
#     label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
#
# # 5: Draw the predicted boxes onto the image
# # Set the colors for the bounding boxes
# colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
# classes = ['background', 'person']
# plt.figure(figsize=(20,12))
# plt.imshow(batch_original_images[i])
# current_axis = plt.gca()
#
# for box in batch_original_labels[i]:
#     xmin = box[1]
#     ymin = box[2]
#     xmax = box[3]
#     ymax = box[4]
#     label = '{}'.format(classes[int(box[0])])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
# for box in y_pred_decoded_inv[i]:
#     xmin = box[2]
#     ymin = box[3]
#     xmax = box[4]
#     ymax = box[5]
#     color = colors[int(box[0])]
#     label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.show()
day_txt_path = os.path.join(dataset_path, 'val_day.txt')
night_txt_path = os.path.join(dataset_path, 'val_night.txt')
_, val_day_gt_counter_per_class = read_txt_file(day_txt_path, tmp_val_gt_day_files_path)
_, val_night_gt_counter_per_class = read_txt_file(night_txt_path, tmp_val_gt_night_files_path)

val_day_mAP = eval(DARK_val_day_images_dir, DARK_val_day_annotations_dir, tmp_val_gt_day_files_path, val_day_gt_counter_per_class)
print("Day mAP: {}".format(val_day_mAP))
val_night_mAP = eval(DARK_val_night_images_dir, DARK_val_night_annotations_dir, tmp_val_gt_night_files_path, val_night_gt_counter_per_class)
print("Night mAP: {}".format(val_night_mAP))
