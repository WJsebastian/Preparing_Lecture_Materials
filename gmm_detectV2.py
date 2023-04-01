import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
# from sklearn.mixture import GaussianMixture
from TORCHGMM import GaussianMixture
import json

video_path = './1.mov'
transform = transforms.ToTensor()
capture = cv2.VideoCapture(video_path)
tensor_pre = None
ret, frame = capture.read()
h, w, _ = frame.shape
K = 4

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
output = cv2.VideoWriter('output_video.mp4', fourcc, 25, (2*w,h))
N = 15

tensor_current = transform(frame).flatten(start_dim=1).transpose(1,0)

model = GaussianMixture(K, 3, covariance_type='diag')
tensor_current_int = (tensor_current * 255).to(torch.uint8)
tensor_mem = []
tensor_mem.append(tensor_current_int)
# histogram_current = tensor_current_int.bincount()
torch.manual_seed(0)
while capture.isOpened():
    ret, frame = capture.read()
    if ret==True:
        tensor_current = transform(frame).reshape(-1, 3)
        tensor_current_int = (tensor_current * 255).to(torch.uint8)
        tensor_mem.append(tensor_current_int)
        # print('wa')
        if(len(tensor_mem) > N):
            break
        
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    else:
        break
# print('1:')
# print(model.mu)
# print(model.pi)
# print(model.var)
model.fit(torch.stack(tensor_mem).reshape(-1, 3))
# print(torch.stack(tensor_mem).reshape(-1, 3))
# print('2:')
# print(model.mu)
# print(model.pi)
# print(model.var)
# print(model.mu.shape)
# print(model.var.shape)
# cv2.waitKey(0)
# print('wa')
capture.release()


index = model.pi.squeeze(dim = 2) / model.var.mean(dim = 2)

# data = {
#     'mu' : model.mu,
#     'var' : model.var,
#     'pi' : model.pi,
#     'covariance_type' : model.covariance_type,
#     'eps' : model.eps,
#     'n_components' : model.n_components,
#     'n_features': model.n_features
# }
# with open('model.json', 'wb') as f:
#     f.write(data)
print(model.mu)
print(model.pi)
print(model.var)
torch.save(model, 'modelV2.pt')

