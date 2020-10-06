import numpy as np
from PIL import Image

import os
from facenet_pytorch import MTCNN, extract_face,\
                            InceptionResnetV1, fixed_image_standardization
from utils import compute_dist

pair_in = open('test/pair.txt', 'r')

mtcnn = MTCNN(image_size=160, select_largest=True,
              selection_method='largest')
resnet1 = InceptionResnetV1(pretrained='vggface2').eval()
resnet2 = InceptionResnetV1(pretrained='casia-webface').eval()

resnet1 = resnet1.cuda()
resnet2 = resnet2.cuda()


class Counter(object):
    def __init__(self):
        self.cnt9 = 0
        self.cnt8 = 0
        self.cnt7 = 0
        self.cnt6 = 0
        self.cnt5 = 0
        self.n = 0

    def update(self, sim1, sim2):
        self.n += 1

        if sim1 > 0.5 and sim2 > 0.5:
            self.cnt5 += 1

        if sim1 > 0.6 and sim2 > 0.6:
            self.cnt6 += 1

        if sim1 > 0.7 and sim2 > 0.7:
            self.cnt7 += 1

        if sim1 > 0.8 and sim2 > 0.8:
            self.cnt8 += 1

        if sim1 > 0.9 and sim2 > 0.9:
            self.cnt9 += 1


pretrained_datasets = ['vggface2', 'casia-webface']
for dataset in pretrained_datasets:
    print('Target model on ', dataset)
    counter = Counter()

    origin_sim1_list = []
    sim1_list = []
    origin_sim2_list = []
    sim2_list = []
    for idx, line in enumerate(pair_in):
        if idx % 20 == 1:
            print('====> Pair ', idx+1)
        src, tgt = line.strip().split(' ')

        src_adv = '{}_adv.png'.format(src[:4])
        # print('src ', src_adv)
        # print('tgt ', tgt)
        src_img = Image.open(os.path.join('logs/{}/test'.format(dataset), src_adv))  # adv
        src_origin = Image.open(os.path.join('test', src))

        dist = compute_dist(np.asarray(src_img), np.asarray(src_origin))
        # print('pixel dist {:.3f}'.format(dist))
        tgt_img = Image.open(os.path.join('test', tgt))

        src_cropped, _, _ = mtcnn(src_img)
        src_origin_cropped, _, _ = mtcnn(src_origin)
        tgt_cropped, _, _ = mtcnn(tgt_img)

        src_rep = resnet1(src_cropped.unsqueeze(0))
        src_origin_rep = resnet1(src_origin_cropped.unsqueeze(0))
        tgt_rep = resnet1(tgt_cropped.unsqueeze(0))

        origin_sim1 = (src_origin_rep * tgt_rep).sum().item()
        sim1 = (src_rep * tgt_rep).sum(dim=1).mean().item()
        # print('similarity 1, origin: {:.4f}, adv: {:.4f}'.format(origin_sim1, sim1))
        origin_sim1_list.append(origin_sim1)
        sim1_list.append(sim1)

        src_rep = resnet2(src_cropped.unsqueeze(0))
        src_origin_rep = resnet2(src_origin_cropped.unsqueeze(0))
        tgt_rep = resnet2(tgt_cropped.unsqueeze(0))

        origin_sim2 = (src_origin_rep * tgt_rep).sum().item()
        sim2 = (src_rep * tgt_rep).sum(dim=1).mean().item()

        origin_sim2_list.append(origin_sim2)
        sim2_list.append(sim2)
        # print('similarity 2, origin: {:.4f}, adv: {:.4f}'.format(origin_sim2, sim2))

        counter.update(sim1, sim2)
        print('vggface2 prediction, origin_sim: {:.3f}, adv_sim: {:.3f}'.format(np.mean(origin_sim1_list), np.mean(sim1_list)))
        print('webface prediction, origin_sim: {:.3f}, adv_sim: {:.3f}'.format(np.mean(origin_sim2_list), np.mean(sim2_list)))
