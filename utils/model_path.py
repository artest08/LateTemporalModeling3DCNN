"""
Created on Sun Apr 19 01:04:42 2020

@author: esat
"""

def rgb_3d_model_path_selection(architecture_name):
    model_path=''
    if 'I3D' in architecture_name:
        if 'resnet' in architecture_name:
            if '50' in architecture_name:
                if '32f' in architecture_name:
                    if 'NL' in architecture_name:
                        model_path='./weights/i3d_r50_nl_kinetics.pth'
                    else:
                        model_path='./weights/i3d_r50_kinetics.pth'
                elif '64f' in architecture_name:
                    if 'NL' in architecture_name:
                        if '8x8' in architecture_name:
                            model_path='./weights/i3d_r50_nl_kinetics_8x8.pth'
                        else:
                            model_path='./weights/i3d_r50_nl_kinetics.pth'
                    else:
                        if '8x8' in architecture_name:
                            model_path='./weights/i3d_r50_kinetics_8x8.pth'
                        else:
                            model_path='./weights/i3d_r50_kinetics.pth'
        else:
            model_path='./weights/rgb_imagenet.pth' #model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    elif 'MFNET3D' in architecture_name:
        if '16f' in architecture_name:
            model_path='./weights/MFNet3D_Kinetics-400_72.8.pth'
        elif '64f' in architecture_name:
            model_path='./weights/MFNet3D_Kinetics-400_72.8.pth'
    elif "3D" in architecture_name:
        if 'resnet' in architecture_name:
            if '101' in architecture_name:
                if '64f' in architecture_name and not '16fweight' in architecture_name:
                    model_path='./weights/resnet-101-64f-kinetics.pth'
                else:
                    model_path='./weights/resnet-101-kinetics.pth'
            elif '18' in architecture_name:
                if '64f' in architecture_name and not '16fweight' in architecture_name:
                    model_path='./weights/resnet-18-64f-kinetics.pth'
                else:
                    model_path='./weights/resnet-18-kinetics.pth'
                
        elif 'resneXt' in architecture_name:
            if '101' in architecture_name:
                if '64f' in architecture_name and not '16fweight' in architecture_name:
                    if 'mars' in architecture_name:
                        model_path='./weights/MARS_Kinetics_64f.pth'
                    else:
                        model_path='./weights/resnext-101-64f-kinetics.pth'
                else:
                    model_path='./weights/resnext-101-kinetics.pth'
    elif "tsm" in architecture_name:
        #model_path='./weights/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth'
        model_path='./weights/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth'
    elif 'rep_flow' in architecture_name:
        model_path='./weights/rep_flow_kinetics.pth'
    elif 'slowfast' in architecture_name:
        model_path='./weights/SLOWFAST_8x8_R50_torch.pth'
    return model_path