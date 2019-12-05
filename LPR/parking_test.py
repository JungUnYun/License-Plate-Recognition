import numpy as np
import os
import glob
import cv2
import time
import csv
import torch
import torchvision.transforms as transforms
from unet import UNet
from LPR_Net import*

class parking_data(torch.utils.data.Dataset):
    def __init__(self, root, l, is_train, transform):
        self.transform = transform

        img_pathes = glob.glob(os.path.join(root, '*/*.jpg'))
        img_pathes = sorted(img_pathes)

        train_idx = int(len(img_pathes) * l)

        if is_train == True:
            ## indexing to divide train dataset
            self.img_pathes = img_pathes[train_idx:]
        elif is_train == False:
            ## indexing to divide train dataset
            self.img_pathes = img_pathes

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, index):
        img = cv2.imread(self.img_pathes[index])
        img = cv2.cvtColor(cv2.resize(img, (320, 320)), cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        img = self.transform(img)

        return img, self.img_pathes[index]

dataset_name = 'parking'
csvfile = open(dataset_name + '.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=',')

dataset_dir = 'parking/'
batch_size = 1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = parking_data(root=dataset_dir, l=0.05, is_train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = UNet(n_channels=1, n_classes=1)
net = net.to(device)

net.load_state_dict(torch.load('./unet/ckpt/parking_unet_model_epoch40.pth'))

net.eval()
dummy = net(torch.zeros(1,1,320,320).to(device))

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
for i, data in enumerate(trainloader):
    tic = time.time()
    inputs, img_path = data
    inputs = inputs.to(device)

    outputs = net(inputs)

    outputs = outputs.squeeze(0).squeeze(0)
    outputs = outputs.detach().cpu().numpy()

    outputs[outputs >= threshold] = 255
    outputs[outputs < threshold] = 0

    morph_kernel = np.ones((3, 3), np.uint8)
    outputs = cv2.erode(outputs, morph_kernel, iterations=1)

    num, components = cv2.connectedComponents(np.uint8(outputs))

    y, x = np.where(components == 1)

    if len(x)*len(y) != 0:
        img = cv2.imread(img_path[0])

        x_min = int(round(np.min(x) * (img.shape[1])/320))
        y_min = int(round(np.min(y) * (img.shape[0])/320))
        x_max = int(round(np.max(x) * (img.shape[1])/320))
        y_max = int(round(np.max(y) * (img.shape[0])/320))
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