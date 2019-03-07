import torch
import numpy as np
import ThinPlateSpline as TPS
from PIL import Image
import matplotlib.pyplot as plt

p = np.array([
    [-1, 1],
    [1, 1],
    [1, -1],
    [-1, -1],
    [-0.2, 0.2],
    [0.2, -0.2],
])

v = np.array([
    [0., 0],
    [0., 0],    
    [0, 0],
    [0, 0],
    [0.1, -0.1],
    [-0.1, 0.1],
])
num_batch = 1
p = torch.Tensor(p.reshape([num_batch, p.shape[0],  p.shape[1]]))
v = torch.Tensor(v.reshape([num_batch, v.shape[0],  v.shape[1]]))

# input image
img_src = Image.open("image.png")
img_src = img_src.resize((512, 512),Image.ANTIALIAS)
img_src.save("image-resized.png")
img = np.array(img_src)

out_size = list(img.shape)
print("out_size:",out_size)
shape = [1]+out_size

t_img = img.reshape(shape)
t_img = torch.tensor(t_img, dtype=torch.float32)

T = TPS.solve_system(p, v)
output = TPS._transform(T,p,t_img,out_size)

result = Image.fromarray(np.uint8(output[0].reshape( out_size)))
plt.imshow(result)
plt.show()

result.save("image-transformed.png")
  
