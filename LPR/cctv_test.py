from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from unet import UNet
from LPR_Net import*
import time
import cv2
import argparse
import torch
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='cctv', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

dataset_name = 'cctv'
csvfile = open(dataset_name + '.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=',')

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

unet = UNet(n_channels=1, n_classes=1)
unet.load_state_dict(torch.load('./unet/ckpt/cctv_unet_model_epoch40.pth'))

cuda = torch.cuda.is_available() and opt.use_cuda
if cuda:
    model.cuda()
    unet.cuda()

model.eval() # Set in evaluation mode
unet.eval()

## dummy processing
dummy = model(torch.zeros(1,3,320,320).cuda())
dummy = unet(torch.zeros(1,1,320,320).cuda())

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

############## Character Recognition ##############
# Initialize the network
global_step = tf.Variable(0, trainable=False)
logits, input_plate, seq_len = get_train_model(num_channels, label_len, BATCH_SIZE, img_size)
logits = tf.transpose(logits, (1, 0, 2))
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# Load the checkpoint
saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
saver.restore(session, "model/model60000.ckpt")
# dummy process
plate_img = np.zeros((1, 94, 24, 1), dtype=np.uint8)
seq_length = np.ones(BATCH_SIZE) * 24
test_feed = {input_plate: plate_img, seq_len: seq_length}
decoded_word = session.run(decoded[0], test_feed)
###################################################

cnt = 0
sum_time = 0
threshold = 0.5
for batch_i, (img_path, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    tic = time.time()
    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)

    # read image
    img = cv2.imread(img_path[0])

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    bbox_list = []
    # Draw bounding boxes and labels of detections
    if detections[0] is not None:
        unique_labels = detections[0][:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = np.uint(round(max((((y1 - pad_y // 2) / unpad_h) * img.shape[0]).item(), 0)))
            x1 = np.uint(round(max((((x1 - pad_x // 2) / unpad_w) * img.shape[1]).item(), 0)))
            y2 = np.uint(round(min((y1 + box_h).item(), img.shape[0])))
            x2 = np.uint(round(min((x1 + box_w).item(), img.shape[1])))

            if cls_pred.item() == 2:
                bbox_list.append([x1, y1, x2, y2, (x2-x1)*(y2-y1)])

        bbox_list = np.asarray(bbox_list)
        if len(bbox_list) != 0:
            bbox_list = bbox_list[bbox_list[:, -1].argsort()][::-1, :]
        if len(bbox_list) >= 3:
            bbox_list = bbox_list[:3, :]

        plate_list = []
        for (x1, y1, x2, y2, area) in bbox_list:
            if (y2 - y1) * (x2 - x1) > 250 * 200:
                cctv_input = img[y1:y2, x1:x2, :]
                cctv_input = cv2.cvtColor(cv2.resize(cctv_input, (320, 320)), cv2.COLOR_BGR2GRAY)
                cctv_input = np.expand_dims(cctv_input, axis=2)
                cctv_input = transform(cctv_input).unsqueeze(0).cuda()

                cctv_output = unet(cctv_input)
                cctv_output = cctv_output.squeeze(0).squeeze(0)
                cctv_output = cctv_output.detach().cpu().numpy()

                cctv_output[cctv_output >= threshold] = 255
                cctv_output[cctv_output < threshold] = 0

                morph_kernel = np.ones((3, 3), np.uint8)
                cctv_output = cv2.erode(cctv_output, morph_kernel, iterations=1)

                num, components = cv2.connectedComponents(np.uint8(cctv_output))

                y, x = np.where(components == 1)

                if len(x) * len(y) != 0:
                    x_min = np.uint(round(np.min(x) * (x2-x1) / 320)) + x1
                    y_min = np.uint(round(np.min(y) * (y2-y1) / 320)) + y1
                    x_max = np.uint(round(np.max(x) * (x2-x1) / 320)) + x1
                    y_max = np.uint(round(np.max(y) * (y2-y1) / 320)) + y1

                    plate_list.append([x_min, y_min, x_max, y_max, (x_max-x_min)*(y_max-y_min)])

        plate_list = np.asarray(plate_list)
        if len(plate_list) != 0:
            plate_list = plate_list[plate_list[:, -1].argsort()][-1]
            x_min = plate_list[0]
            y_min = plate_list[1]
            x_max = plate_list[2]
            y_max = plate_list[3]
            plate_img = img[y_min:y_max, x_min:x_max, :]

            # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # cv2.imshow('output', img)
            # cv2.waitKey(0)

            ############## Character Recognition ##############
            plate_img = cv2.resize(cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY), (94, 24), interpolation=cv2.INTER_LINEAR)
            plate_img = np.expand_dims(plate_img, axis=2)
            plate_img = np.expand_dims(plate_img, axis=0)
            plate_img = np.transpose(plate_img, axes=[0, 2, 1, 3])
            seq_length = np.ones(BATCH_SIZE) * 24

            test_feed = {input_plate: plate_img, seq_len: seq_length}
            decoded_word = session.run(decoded[0], test_feed)
            detected_word = decode_sparse_tensor(decoded_word)
            label = ''
            for char_idx in range(len(detected_word[0])):
                label = label + detected_word[0][char_idx]

            # print(label)
            # cv2.imshow('win', img[y_min:y_max, x_min:x_max, :])
            # cv2.waitKey(0)
            ###################################################
        else:
            label = ''
            x_min, y_min, x_max, y_max = 0, 0, 0, 0

        toc = time.time() - tic
        print(toc)
        sum_time += toc
        cnt += 1
        csvwriter.writerow([img_path[0], label, x_min, y_min, x_max, y_max])

avg_time = sum_time/cnt
print('Average time: %.2f msec' % (avg_time*1000))