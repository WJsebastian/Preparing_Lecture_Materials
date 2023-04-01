import cv2
import numpy as np
import torchvision.transforms as transforms

transform = transforms.ToTensor()
video_path = './1.mov'
capture = cv2.VideoCapture(video_path)
tensor_pre = None
ret, frame = capture.read()
# print(ret)
# cv2.imshow('video', frame)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
tensor_current = transform(frame_gray)
tensor_pre = tensor_current
# print(tensor_current.shape)
# print(tensor_current)
_, h, w = tensor_current.shape
# print(frame_gray)
# BG = cv2.resize(frame, (w *2, h), interpolation= cv2.INTER_AREA)
# BG[0:h, 0:w] = frame
# BG[0:h, w:2 *w, 0] = frame_gray
# BG[0:h, w:2 *w, 1] = frame_gray
# BG[0:h, w:2 *w, 2] = frame_gray
# cv2.imshow('frame', BG) 
# demo_frame = tensor_current.numpy().transpose(1, 2, 0)
# print(demo_frame.shape)
# cv2.imshow('kkk', demo_frame)
# print(w, h)
T = 15
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
output = cv2.VideoWriter('output_video.mp4', fourcc, 36, (2*w,h))
while capture.isOpened():
    ret, frame = capture.read()
    if ret==True:
        # print('hey')
        # print(ret)
        # cv2.imshow('video', frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tensor_current = transform(frame_gray)
        
        diff_tensor = tensor_current - tensor_pre
        # diff_tensor *= 255
        diff_frame = diff_tensor.numpy().transpose(1, 2, 0)
        # diff_frame[diff_frame <30] = 0
        # diff_frame[diff_frame>50] = 255
        diff_frame = np.abs(diff_frame)
        # print(diff_frame)
        # print(diff_frame)
        diff_frame *= 255
        # print(tensor_current)
        BG = cv2.resize(frame, (w *2, h), interpolation= cv2.INTER_AREA)
        BG[0:h, 0:w] = frame
        diff_frame[diff_frame > T] = 255
        diff_frame[diff_frame <= T] = 0
        # BG[0:h, w:2 *w, 0] = diff_frame
        # BG[0:h, w:2 *w, 1] = diff_frame
        # BG[0:h, w:2 *w, 2] = diff_frame
        BG[0:h, w:2*w] = diff_frame
        cv2.imshow('video', BG) 
        tensor_pre = tensor_current
        output.write(BG)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    else:
        break
        

capture.release()
output.release()
cv2.destroyAllWindows()