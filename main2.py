import os
import argparse
import pickle
import glob
import numpy as np
from PIL import Image

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from utils import crop_resize_back, compute_dist, tensor_to_image

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)


def fixed_image_standardization_inverse(image_tensor):
    return image_tensor * 128.0 + 127.5


def get_pretrained_inception_model(dataset='vggface2'):
    if dataset == 'vggface2':
        # For a model pretrained on VGGFace2
        model = InceptionResnetV1(pretrained='vggface2')
    elif dataset == 'casia-webface':
        # For a model pretrained on CASIA-Webface
        model = InceptionResnetV1(pretrained='casia-webface')
    return model


def cal_source_grad(inception_model, source_img, target_rep):
    source_img.requires_grad = True
    source_rep = inception_model(source_img)

    similarity = (target_rep * source_rep).sum(dim=1).mean()

    inception_model.zero_grad()
    # cal gradient
    similarity.backward()

    return source_img.grad.data, similarity.cpu().item()


def iterative_grad_attack(inception_model, source_tensor, target_tensor,
                          n_steps=200, lr=0.01):
    with torch.no_grad():
        target_rep = inception_model(target_tensor).detach()

    perturbed_tensor = source_tensor.clone()
    for step in range(1, n_steps+1):
        grad, similarity = cal_source_grad(inception_model, perturbed_tensor, target_rep)
        perturbed_tensor = perturbed_tensor + lr * grad
        perturbed_tensor = torch.clamp(perturbed_tensor, -1.0, 1.0).detach_()
        if similarity > 0.99:
            break

    adv_rep = inception_model(perturbed_tensor)
    rep_dist = (target_rep * adv_rep).sum(dim=1).mean().cpu().item()

    adv_img = tensor_to_image(fixed_image_standardization_inverse(perturbed_tensor.cpu()).squeeze(0))
    tgt_img = tensor_to_image(fixed_image_standardization_inverse(tgt_tensor.cpu()).squeeze(0))

    pixel_dist = compute_dist(np.asarray(adv_img), np.asarray(tgt_img))
    return adv_img, pixel_dist, rep_dist


def attack(args, mode='val'):
    mtcnn = MTCNN(image_size=args.image_size, select_largest=True,
                  selection_method='largest')
    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained=args.pretrained_dataset).eval()
    resnet = resnet.cuda()

    np.random.seed(args.seed)
    if mode == 'val':
        logger.info('==> Attach on Val Set')
        val_path = hydra.utils.to_absolute_path('val')
        tgt_paths = glob.glob(val_path + '/*.png')
        src_paths = np.random.permutation(tgt_paths)

        if not os.path.isdir('val'):
            os.mkdir('val')
    else:
        logger.info('==> Attach on Test Set')
        log_dir = os.path.join(args.log_dir, 'test')

        test_path = hydra.utils.to_absolute_path('test')
        pair_in = open(os.path.join(test_path, 'pair.txt'), 'r')
        lines = pair_in.readlines()
        src_paths = [os.path.join(test_path, line.strip().split(' ')[0]) for line in lines]
        tgt_paths = [os.path.join(test_path, line.strip().split(' ')[1]) for line in lines]

        if not os.path.isdir('test'):
            os.mkdir('test')

    pixel_dist_list = []
    rep_dist_list = []
    original_pixel_dist_list = []

    if mode == 'val':
        pair_list = []

    for pair_idx, (src_path, tgt_path) in enumerate(zip(src_paths, tgt_paths)):
        src_img = Image.open(src_path)
        tgt_img = Image.open(tgt_path)

        # crop
        src_crop, box_size, box = mtcnn(src_img)
        tgt_crop, _, _ = mtcnn(tgt_img)

        src_crop = src_crop.unsqueeze(0).cuda()
        tgt_crop = tgt_crop.unsqueeze(0).cuda()
        assert src_crop.size() == (1, 3, 160, 160)

        # adv attack
        adv_crop, dist, rep_dist = iterative_grad_attack(
            resnet, src_crop, tgt_crop,
            lr=args.attack_lr, n_steps=args.attack_steps)

        pixel_dist_list.append(dist)
        rep_dist_list.append(rep_dist)
        logger.info('sample {}, rep_similarity: {:.3f}'.format(pair_idx + 1, rep_dist))

        src_id = src_path.split('/')[-1][:4]
        tgt_id = tgt_path.split('/')[-1][:4]
        source_name = '{}_adv.png'.format(src_id)
        target_name = '{}.png'.format(tgt_id)

        # crop resize back and save.
        src_adv_img = crop_resize_back(src_img, adv_crop, box, box_size)
        src_adv_img.save(os.path.join(mode, source_name)) # save to val or test directory

        original_pixel_dist_list.append(compute_dist(np.asarray(src_adv_img), np.asarray(src_img)))
        if mode == 'val':
            pair_list.append((source_name, target_name))

    if mode == 'val':
        pickle.dump(pair_list, open(os.path.join('val', 'pair.pickle'), 'wb'))

    return np.mean(original_pixel_dist_list), np.mean(pixel_dist_list), np.mean(rep_dist_list)


@hydra.main(config_name='config.yml')
def run(args: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(args))

    pixel_dist, pixel_crop_dist, rep_dist = attack(args, 'val')
    logger.info('=======>[Val]')
    logger.info('pixel dist: {:.3f}'.format(pixel_dist))
    logger.info('crop_dist: {:.3f}'.format(pixel_crop_dist))
    logger.info('rep_similarity: {:.3f}'.format(rep_dist))

    pixel_dist, pixel_crop_dist, rep_dist = attack(args, 'test')
    logger.info('=======>[Test]')
    logger.info('pixel dist: {:.3f}'.format(pixel_dist))
    logger.info('crop_dist: {:.3f}'.format(pixel_crop_dist))
    logger.info('rep_similarity: {:.3f}'.format(rep_dist))


if __name__ == "__main__":
    run()
