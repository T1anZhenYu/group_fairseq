import torch

from torchvision.utils import save_image

def save_attn(m, file_path):
    if len(m.shape) == 3:
        for i in range(m.shape[0]):
            save_image(m[i],file_path+'/'+str(i)+'_'+str(m.shape[0])+'_'+\
                       str(m.shape[1])+'_'+str(m.shape[-1])+'.jpg')
    elif len(m.shape) == 2:
        save_image(m[i],file_path+'/'+str(i)+'_'+str(m.shape[0])+'_'+\
                   str(m.shape[1])+'_'+str(m.shape[-1])+'.jpg')    	

