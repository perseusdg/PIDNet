import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F 
import torch
from PIL import Image
import struct 
from torchinfo import summary

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

outputs = {}

def hook(module,input,output):
    outputs[module] = output

def bin_write(f,data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt,*data)
    f.write(bin)

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    # msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    # print('Attention!!!')
    # print(msg)
    # print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model


def exporter():
    image_name = "dog.jpg"
    model = models.pidnet.get_pred_model('pidnet-s',19)
    model = load_pretrained(model, "PIDNet_S_Cityscapes_val.pt")
    model = model.eval()
    model = model.cuda()
    torch.no_grad()
    img_1 = cv2.imread(image_name,cv2.IMREAD_COLOR)
    img = cv2.resize(img_1,(640,640))
    img = input_transform(img)
    img = img.transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).unsqueeze(0).cuda()

    #make bin directory
    if not os.path.exists('tkdnn_bin'):
        os.makedirs('tkdnn_bin')  
    if not os.path.exists('tkdnn_bin/debug'):
        os.makedirs('tkdnn_bin/debug')
    if not os.path.exists('tkdnn_bin/layers'):
        os.makedirs('tkdnn_bin/layers')

    #save input
    img_input = img.cpu().data.numpy()
    img_input = np.array(img_input,dtype=np.float32)
    img_input.tofile("tkdnn_bin/debug/input.bin",format="f")  
    for n,m in model.named_modules():
        m.register_forward_hook(hook);

    #save output
    pred = model(img)
    pred = torch.nn.functional.interpolate(pred, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
    pred_np = pred.cpu().data.numpy()
    pred_np = np.array(pred_np,dtype=np.float32)
    pred_np.tofile("tkdnn_bin/debug/output.bin",format="f")
    
    for n,m in model.named_modules():
        t = '-'.join(n.split('.'))
        if m not in outputs:
            continue
        in_outputs = outputs[m]
        in_outputs_cpu = in_outputs.cpu().data.numpy()
        in_outputs_cpu = np.array(in_outputs_cpu,dtype=np.float32)
        in_outputs_cpu.tofile("tkdnn_bin/debug/"+t+".bin",format="f")
        if t == "conv1-1":
            print(in_outputs_cpu)
    
    f = None
    for n,m in model.named_modules():
        print(n)
        t = '-'.join(n.split('.'))

        if not(' of Conv2d' in str(m.type) or ' of BatchNorm2d' in str(m.type) or ' of Linear' in str(m.type)):
            continue

        if ' of Conv2d' in str(m.type) or ' of Linear' in str(m.type) or ' of Sigmoid' in str(m.type):
            f=None
            file_name = "tkdnn_bin/layers/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')
            w = np.array([])
            b = np.array([])
            if 'weight' in m._parameters and m._parameters['weight'] is not None:
                w = m._parameters['weight'].cpu().data.numpy()
                w = np.array(w, dtype=np.float32)
                print("    weights shape:", np.shape(w))

            if 'bias' in m._parameters and m._parameters['bias'] is not None:
                b = m._parameters['bias'].cpu().data.numpy()
                b = np.array(b, dtype=np.float32)
                print("    bias shape:", np.shape(b))
                
            bin_write(f, w)
            bias_shape = w.shape[0]
            if b.size > 0:
                bin_write(f, b)
            else:
                bin_write(f,np.zeros(bias_shape))
            f.close()
            f=None  
        if 'of BatchNorm2d' in str(m.type):
            f=None
            file_name = "tkdnn_bin/layers/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            s = m._parameters['weight'].cpu().data.numpy()
            s = np.array(s, dtype=np.float32)
            rm = m.running_mean.cpu().data.numpy()
            rm = np.array(rm, dtype=np.float32)
            rv = m.running_var.cpu().data.numpy()
            rv = np.array(rv, dtype=np.float32)
            bin_write(f, b)
            bin_write(f, s)
            bin_write(f, rm)
            bin_write(f, rv)

            print("    b shape:", np.shape(b))
            print("    s shape:", np.shape(s))
            print("    rm shape:", np.shape(rm))
            print("    rv shape:", np.shape(rv))
            f.close()
            f=None
      
           

    
    # summary(model,input_data=img)
    torch.onnx.export(model,img,"model1.onnx")
    print(model)



if __name__ == '__main__':
    exporter()
