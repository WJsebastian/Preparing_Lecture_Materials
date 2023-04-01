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
N = 6

tensor_current = transform(frame).flatten(start_dim=1).transpose(1,0)

# data = {
#     'mu' : model.mu,
#     'var' : model.var,
#     'pi' : model.pi,
#     'covariance_type' : model.covariance_type,
#     'eps' : model.eps,
#     'n_components' : model.n_components,
#     'n_features': model.n_features
# }

model = GaussianMixture(K, 3, covariance_type='diag')
with open('model.pt', 'rb') as f:
    data = torch.load(f)
    model.mu  = data.mu
    model.var = data.var
    model.pi = data.pi
print(model.mu)
print(model.pi)
print(model.var)





index = model.pi.squeeze(dim = 2) / model.var.mean(dim = 2)

# index_forground = index.argmin() 
_, index_forground = torch.topk(index, 4)
# print(index_forground.shape)
index_forground1 = index_forground[0,-1]
index_forground2 = index_forground[0,-2]
index_forground3 = index_forground[0,-3]
index_forground4 = index_forground[0,-4]
# index_forground5 = index_forground[0,-5]
frame_shown_pre  = None



while capture.isOpened():
    # break 
    ret, frame = capture.read()
    if ret==True:
        tensor_current = transform(frame).flatten(start_dim=1).transpose(1,0)
        tensor_current_int = (tensor_current * 255).to(torch.uint8)
        predict_tensor = model.predict(tensor_current_int).reshape(h, w)
        frame_shown = np.zeros((h, w, 3))
        # frame_shown[(predict_tensor == 2).numpy()] = 255
        # frame_shown[(predict_tensor ==1).numpy()] = 255
        # frame_shown[(predict_tensor ==0).numpy()] = 255
        # frame_shown[(predict_tensor == index_forground)] = 255
        frame1 = torch.eq(predict_tensor, index_forground1)
        frame2 = torch.eq(predict_tensor, index_forground2)
        frame3 = torch.eq(predict_tensor, index_forground3)
        frame4 = torch.eq(predict_tensor, index_forground4)
        # frame5 = torch.eq(predict_tensor, index_forground5)
        # frame_shown[frame1] = 255
        # frame_shown[frame2] = 255
        # frame_shown[frame3] = 255
        # frame_shown_pre = frame_shown
        
        frame_shown[frame4] = 255
        # frame_shown[frame5] = 255
        # print(frame_shown.shape)
        # print(predict_tensor)
        # frame_shown = (predict_tensor == index_forground).unsqueeze(2).numpy()
        BG = cv2.resize(frame, (w *2, h), interpolation= cv2.INTER_AREA)
        BG[0:h, 0:w] = frame
        BG[0:h, w:2*w] = frame_shown
        output.write(BG)
        # print(frame_shown)
        cv2.imshow('video', BG) 
        # print('wa')
        
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    else:
        break
output.release()
cv2.destroyAllWindows()
