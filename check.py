import numpy as np
from PIL import Image

import os
from facenet_pytorch import MTCNN, extract_face,\
                            InceptionResnetV1, fixed_image_standardization
from utils import compute_dist

dir = 'logs/test'

pair_in = open('test/pair.txt', 'r')

mtcnn = MTCNN(image_size=160)
resnet1 = InceptionResnetV1(pretrained='vggface2').eval()
resnet2 = InceptionResnetV1(pretrained='casia-webface').eval()

for idx, line in enumerate(pair_in):
    print('====> Pair ', idx+1)
    src, tgt = line.strip().split(' ')

    src_adv = '{}_adv.png'.format(src[:4])
    print('src ', src_adv)
    print('tgt ', tgt)
    src_img = Image.open(os.path.join('logs/test', src_adv))  # adv
    src_origin = Image.open(os.path.join('test', src))

    dist = compute_dist(np.asarray(src_img), np.asarray(src_origin))
    print('pixel dist {:.3f}'.format(dist))
    tgt_img = Image.open(os.path.join('test', tgt))


    src_cropped, _, _ = mtcnn(src_img)
    src_origin_cropped, _, _ = mtcnn(src_origin)
    tgt_cropped, _, _ = mtcnn(tgt_img)

    src_rep = resnet1(src_cropped.unsqueeze(0))
    src_origin_rep = resnet1(src_origin_cropped.unsqueeze(0))
    tgt_rep = resnet1(tgt_cropped.unsqueeze(0))

    origin_sim1 = (src_origin_rep * tgt_rep).sum().item()
    sim1 = (src_rep * tgt_rep).sum().item()
    print('similarity 1, origin: {:.4f}, adv: {:.4f}'.format(origin_sim1, sim1))

    src_rep = resnet2(src_cropped.unsqueeze(0))
    src_origin_rep = resnet2(src_origin_cropped.unsqueeze(0))
    tgt_rep = resnet2(tgt_cropped.unsqueeze(0))

    origin_sim2 = (src_origin_rep * tgt_rep).sum().item()
    sim2 = (src_rep * tgt_rep).sum().item()
    print('similarity 2, origin: {:.4f}, adv: {:.4f}'.format(origin_sim2, sim2))

    if idx == 10:
        break
