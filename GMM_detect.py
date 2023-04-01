import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
# from sklearn.mixture import GaussianMixture
from TORCHGMM import GaussianMixture

video_path = './1.mov'
transform = transforms.ToTensor()
capture = cv2.VideoCapture(video_path)
tensor_pre = None
ret, frame = capture.read()
h, w, _ = frame.shape
K = 3
histogram = torch.zeros(256, 3)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
output = cv2.VideoWriter('output_video.mp4', fourcc, 25, (2*w,h))
N = 2
tensor_current = transform(frame).reshape(-1, 3)
model = GaussianMixture(K, 3)
count = 0.
count_pre = count
count_current = tensor_current.shape[0]
count += tensor_current.shape[0]

tensor_current_int = torch.tensor(tensor_current * 255, dtype=torch.uint8)
# histogram_current = tensor_current_int.bincount()
histogram_current = torch.zeros(256, 3)
histogram_current[:, 0] = tensor_current_int[:, 0].bincount()
histogram_current[:, 1] = tensor_current_int[:, 1].bincount()
histogram_current[:, 2] = tensor_current_int[:, 2].bincount()
histogram_current /= count_current

print(histogram_current)
# print(tensor_current_int)
# histogram = 
print()

# while capture.isOpened():
#     ret, frame = capture.read()
#     if ret==True:
#         # X = np.array(frame_mem)
#         # print(tensor_current)
#         # BG = cv2.resize(frame, (w *2, h), interpolation= cv2.INTER_AREA)
#         # BG[0:h, 0:w] = frame
#         # print('wa')
#         if(len(tensor_mem) > N):
#             break
#         tensor_mem.append(transform(frame).reshape(-1, 3))
        
#         if cv2.waitKey(27) & 0xFF == ord('q'):
#             break
#     else:
#         break
# print('1:')
# print(model.mu)

# ret, frame = capture.read()
# print(transform(frame).reshape(-1, 3)[0].unsqueeze(0))
# model.fit(transform(frame).reshape(-1, 3)[0].unsqueeze(0))
# print(model.mu)


# print('wa')
# model.fit(torch.concat(tensor_mem, dim=0)[:, 0])
# print('Done')
# print(torch.concat(tensor_mem, dim=0).shape)
# print(tensor_mem)

# cv2.waitKey(0)
# print('wa')
capture.release()
output.release()
cv2.destroyAllWindows()
