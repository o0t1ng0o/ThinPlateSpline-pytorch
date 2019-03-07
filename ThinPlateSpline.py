import torch
import numpy as np
import torch.nn.functional as F

def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def _repeat(x, n_repeats):
    rep = torch.unsqueeze( torch.ones(n_repeats), 1).transpose(0, 1)
    rep = torch.tensor(rep, dtype=torch.int32)
    x = torch.matmul(torch.tensor(x.reshape(-1,1), dtype=torch.int32), rep)
    return x.reshape(-1)
    
def _interpolate(im, x, y, out_size):
    # constants
    num_batch = im.shape[0]
    height = im.shape[1]
    width = im.shape[2]
    channels = im.shape[3]
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    height_f = torch.tensor(height, dtype=torch.float32)
    width_f  = torch.tensor(width,  dtype=torch.float32)
    out_height = out_size[0]
    out_width  = out_size[1]
    zero = torch.tensor(0, dtype=torch.int32)
    max_y = torch.tensor(im.shape[1] - 1, dtype=torch.int32)
    max_x = torch.tensor(im.shape[2] - 1, dtype=torch.int32)

    # scale indices from [-1, 1] to [0, width/height]
    x = (x + 1.0)*(width_f) / 2.0
    y = (y + 1.0)*(height_f) / 2.0
        
    # do sampling
    x0 = torch.tensor(torch.floor(x), dtype=torch.int32)
    x1 = x0 + 1
    y0 = torch.tensor(torch.floor(y), dtype=torch.int32)
    y1 = y0 + 1
    
    x0 = torch.clamp(x0, min=zero, max=max_x)
    x1 = torch.clamp(x1, min=zero, max=max_x)
    y0 = torch.clamp(y0, min=zero, max=max_y)
    y1 = torch.clamp(y1, min=zero, max=max_y)
    
    dim2 = width
    dim1 = width*height
    base = _repeat(torch.range(0, num_batch-1)*dim1, out_height*out_width)

    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1
    
    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = im.reshape(-1, channels)
    im_flat = torch.tensor(im_flat, dtype=torch.float32)

    tmp = torch.tensor(idx_a.unsqueeze(1), dtype=torch.long)
    idx_a = torch.tensor(idx_a.unsqueeze(1), dtype=torch.long)
    idx_b = torch.tensor(idx_b.unsqueeze(1), dtype=torch.long)
    idx_c = torch.tensor(idx_c.unsqueeze(1), dtype=torch.long)
    idx_d = torch.tensor(idx_d.unsqueeze(1), dtype=torch.long)
    if channels != 1:
        tmp_idx_a = torch.tensor(idx_a, dtype=torch.long)
        tmp_idx_b = torch.tensor(idx_b, dtype=torch.long)
        tmp_idx_c = torch.tensor(idx_c, dtype=torch.long)
        tmp_idx_d = torch.tensor(idx_d, dtype=torch.long)
        for i in range(channels-1):
            idx_a = torch.cat((idx_a,tmp_idx_a), 1)
            idx_b = torch.cat((idx_b,tmp_idx_b), 1)
            idx_c = torch.cat((idx_c,tmp_idx_c), 1)
            idx_d = torch.cat((idx_d,tmp_idx_d), 1)
    
    
    Ia = torch.gather(im_flat, 0, idx_a)
    Ib = torch.gather(im_flat, 0, idx_b)
    Ic = torch.gather(im_flat, 0, idx_c)
    Id = torch.gather(im_flat, 0, idx_d)
    
    # and finally calculate interpolated values
    x0_f = torch.tensor(x0, dtype=torch.float32)
    x1_f = torch.tensor(x1, dtype=torch.float32)
    y0_f = torch.tensor(y0, dtype=torch.float32)
    y1_f = torch.tensor(y1, dtype=torch.float32)
    wa = torch.unsqueeze(((x1_f-x) * (y1_f-y)), 1)
    wb = torch.unsqueeze(((x1_f-x) * (y-y0_f)), 1)
    wc = torch.unsqueeze(((x-x0_f) * (y1_f-y)), 1)
    wd = torch.unsqueeze(((x-x0_f) * (y-y0_f)), 1)
    
    output = torch.add(wa*Ia, 1, wb*Ib)
    output = torch.add(output, 1, wc*Ic)
    output = torch.add(output, 1, wd*Id)
    return output
    
def solve_system(coord, vec):
    """Thin Plate Spline Spatial Transformer layer
    TPS control points are arranged in arbitrary positions given by `coord`.
    coord : float Tensor [num_batch, num_point, 2]
        Relative coordinate of the control points.
    vec : float Tensor [num_batch, num_point, 2]
        The vector on the control points.
    """
    num_batch = coord.shape[0]
    num_point = coord.shape[1]

    ones = torch.ones([num_batch, num_point, 1])
    p = torch.cat([ones, coord], 2)     # [bn, pn, 3]
    
    p_1 = torch.reshape(p, [num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
    p_2 = torch.reshape(p, [num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
    d = p_1 - p_2                                  # [bn, pn, pn, 3]
    d2 = torch.sum(torch.pow(d, 2), 3)  # [bn, pn, pn]
    r = d2 * torch.log(d2 + 1e-6)       # [bn, pn, pn]

    zeros = torch.zeros([num_batch, 3, 3])
    W_0 = torch.cat([p, r], 2)          # [bn, pn, 3+pn]
    W_1 = torch.cat([zeros, torch.transpose(p, 2, 1)], 2)  # [bn, 3, pn+3]
    W = torch.cat([W_0, W_1], 1)        # [bn, pn+3, pn+3]
    W_inv = b_inv(W)

    tp = F.pad(vec+coord, (0, 0, 0, 3))

    tp = tp.squeeze(1)                  # [bn, pn+3, 2]
    T = torch.matmul(W_inv, tp)         # [bn, pn+3, 2]
    T = torch.transpose(T, 2, 1)        # [bn, 2, pn+3]

    return T

def _meshgrid(height, width, coord):    
    x_t = torch.linspace(-1.0, 1.0, steps=width).reshape(1, width).expand(height, width)
    y_t = torch.linspace(-1.0, 1.0, steps=height).reshape(height, 1).expand(height,width)
    x_t_flat = x_t.reshape(1, 1, -1)
    y_t_flat = y_t.reshape(1, 1, -1)    

    num_batch = coord.shape[0]
    px = torch.unsqueeze(coord[:, :, 0], 2) # [bn, pn, 1]
    py = torch.unsqueeze(coord[:, :, 1], 2) # [bn, pn, 1]
    
    d2 = torch.pow(x_t_flat - px, 2) + torch.pow(y_t_flat - py, 2)

    r = d2 * torch.log(d2 + 1e-6) # [bn, pn, h*w]
    x_t_flat_g = x_t_flat.expand(num_batch, x_t_flat.shape[1], x_t_flat.shape[2])
    y_t_flat_g = y_t_flat.expand(num_batch, y_t_flat.shape[1], y_t_flat.shape[2])
    ones = torch.ones(x_t_flat_g.shape)

    grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1)
    return grid

def _transform(T, coord, input_dim, out_size):

    num_batch = input_dim.shape[0]
    height = input_dim.shape[1]
    width = input_dim.shape[2]
    num_channels = input_dim.shape[3]
    
    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    height_f = torch.tensor(height, dtype=torch.float32)
    width_f  = torch.tensor(width, dtype=torch.float32) 
    out_height = out_size[0]
    out_width = out_size[1]
    grid = _meshgrid(out_height, out_width, coord) # [2, h*w]
    # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
    # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
    T_g = torch.matmul(T, grid) 
    x_s = torch.unsqueeze(T_g[:, 0, :], 1)
    y_s = torch.unsqueeze(T_g[:, 1, :], 1)
    x_s_flat = x_s.reshape(-1)
    y_s_flat = y_s.reshape(-1)    
    
    input_transformed = _interpolate(
      input_dim, x_s_flat, y_s_flat, out_size)
      
    output = input_transformed.reshape(num_batch, out_height, out_width, num_channels)

    return output
    
def point_transform(point, T, coord):
    point = torch.Tensor(point.reshape([1, 1, 2]))
    d2 = torch.sum(torch.pow(point - coord, 2), 2)
    r = d2 * torch.log(d2 + 1e-6)
    q = torch.Tensor(np.array([[1, point[0, 0, 0], point[0, 0, 1]]]))
    x = torch.cat([q, r], 1)
    point_T = torch.matmul(T, torch.transpose(x.unsqueeze(1), 2, 1))
    return point_T
