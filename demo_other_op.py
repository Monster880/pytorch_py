import torch
import numpy as np
import cv2

data = cv2.imread("topic.png")
cv2.imshow("test1",data)
# cv2.waitKey(0)

# a = np.zeros([2,2])
out = torch.from_numpy(data)
print(out)

out = torch.flip(out,dims=[0])
data = out.numpy()
cv2.imshow("test2",data)
cv2.waitKey(0)