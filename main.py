import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, extract_face, InceptionResnetV1
from utils import crop_resize_back, compute_dist

image_size = 160 
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=image_size)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

files = os.listdir('val/')
dist_list = []
for file in files:
    file = os.path.join('val/', file)
    print(file)
    img_origin = Image.open(file)
    img_cropped, box_size, box = mtcnn(img_origin, save_path='img_cropped.png')
    print('box size ', box_size)
    print(box)
    print(img_cropped.max(), img_cropped.min())
    img_out = resnet(img_cropped.unsqueeze(0))
    print(img_cropped.size())
    print(img_out.size())

    cropped_img = Image.open('img_cropped.png')
    print('crop size', cropped_img.size)
    img_recovered = crop_resize_back(img_origin, cropped_img, box, box_size)
    print('dist', compute_dist(np.asarray(img_origin), np.asarray(img_recovered)))
    img_recovered.save('recovered_img.png') 
    exit(0)
