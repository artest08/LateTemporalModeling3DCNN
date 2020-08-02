from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import cv2
import math
import os, sys
import collections

class Compose(object):
    """Composes several video_transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> video_transforms.Compose([
        >>>     video_transforms.CenterCrop(10),
        >>>     video_transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, video_transforms):
        self.video_transforms = video_transforms

    def __call__(self, clips):
        for t in self.video_transforms:
            clips = t(clips)
        return clips

class Lambda(object):
    """Applies a lambda as a transform"""
    def __init__(self, lambd):
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def __call__(self, clips):
        return self.lambd(clips)

class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            # handle numpy array
            clips = torch.from_numpy(clips.transpose((2, 0, 1)))
            # backward compatibility
            return clips.float().div(255.0)
        
class ToTensor3(object):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            # handle numpy array
            clips = torch.from_numpy(clips.transpose((3, 2, 0, 1)))
            # backward compatibility
            return clips.float().div(255.0)
        
class ToTensor2(object):

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            # handle numpy array
            clips = torch.from_numpy(clips.transpose((2, 0, 1)))
            # backward compatibility
            return clips.float().div(1.0)
        
class Reset(object):
    def __init__(self, mask_prob, num_seg):
        self.mask_prob = mask_prob
        self.num_seg =num_seg
        
    def __call__(self, clips):
        mask=np.random.binomial(1, self.mask_prob, self.num_seg).repeat(3)
        return clips*mask
    
class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Here, the input is a clip, not a single image. (multi-channel data)
    The dimension of mean and std depends on parameter: new_length
    If new_length = 1, it falls back to single image case (3 channel)
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        torch_mean = torch.tensor([[self.mean]]).view(-1,1,1).float()
        torch_std = torch.tensor([[self.std]]).view(-1,1,1).float()
        tensor2 = (tensor - torch_mean) / torch_std
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t.sub_(m).div_(s)
        return tensor2
class DeNormalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Here, the input is a clip, not a single image. (multi-channel data)
    The dimension of mean and std depends on parameter: new_length
    If new_length = 1, it falls back to single image case (3 channel)
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        torch_mean = torch.tensor([[self.mean]]).view(-1,1,1).float()
        torch_std = torch.tensor([[self.std]]).view(-1,1,1).float()
        tensor2 = (tensor * torch_std) + torch_mean
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t.sub_(m).div_(s)
        return tensor2
    
class Normalize3(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Here, the input is a clip, not a single image. (multi-channel data)
    The dimension of mean and std depends on parameter: new_length
    If new_length = 1, it falls back to single image case (3 channel)
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        torch_mean = torch.tensor([[self.mean]]).view(1,-1,1,1)
        torch_std = torch.tensor([[self.std]]).view(1,-1,1,1)
        tensor2 = (tensor - torch_mean) / torch_std
        return tensor2

class Normalize2(object):

    def __init__(self, mean, std, num_seg):
        self.mean = mean
        self.std = std
        self.num_seg = num_seg

    def __call__(self, tensor, num_seg):
        # TODO: make efficient
        mean = self.mean * self.num_seg
        std = self.std * self.num_seg
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
    
class Scale(object):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clips):

        h, w, c = clips.shape
        new_w = 0
        new_h = 0
        if isinstance(self.size, int):
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return clips
            if w < h:
                new_w = self.size
                new_h = int(self.size * h / w)
            else:
                new_w = int(self.size * w / h)
                new_h = self.size
        else:
            new_w = self.size[0]
            new_h = self.size[1]


        num_imgs = int(c / 1)
        scaled_clips = np.zeros((new_h,new_w,c))
        for frame_id in range(num_imgs):
            cur_img = clips[:,:,frame_id:frame_id+1]
            scaled_clips[:,:,frame_id:frame_id+1] = np.expand_dims(
                cv2.resize(cur_img, (new_w, new_h), self.interpolation), axis=2)          
        return scaled_clips


class CenterCrop(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clips):
        h, w, c = clips.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))


        num_imgs = int(c / 1)
        scaled_clips = np.zeros((th,tw,c))
        for frame_id in range(num_imgs):
            cur_img = clips[:,:,frame_id:frame_id+1]
            crop_img = cur_img[y1:y1+th, x1:x1+tw, :]
            assert(crop_img.shape == (th, tw, 1))
            scaled_clips[:,:,frame_id:frame_id+1] = crop_img
        return scaled_clips

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    """
    def __call__(self, clips):
        if random.random() < 0.5:
            clips = np.fliplr(clips)
            clips = np.ascontiguousarray(clips)
        return clips

class RandomVerticalFlip(object):
    """Randomly vertically flips the given numpy array with a probability of 0.5
    """
    def __call__(self, clips):
        if random.random() < 0.5:
            clips = np.flipud(clips)
            clips = np.ascontiguousarray(clips)
        return clips


class RandomSizedCrop(object):
    """Random crop the given numpy array to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clips):
        h, w, c = clips.shape
        is_color = False
        if c % 3 == 0:
            is_color = True

        for attempt in range(10):
            area = w * h
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            new_w = int(round(math.sqrt(target_area * aspect_ratio)))
            new_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                new_w, new_h = new_h, new_w

            if new_w <= w and new_h <= h:
                x1 = random.randint(0, w - new_w)
                y1 = random.randint(0, h - new_h)

                scaled_clips = np.zeros((self.size,self.size,c))
                if is_color:
                    num_imgs = int(c / 3)
                    for frame_id in range(num_imgs):
                        cur_img = clips[:,:,frame_id*3:frame_id*3+3]
                        crop_img = cur_img[y1:y1+new_h, x1:x1+new_w, :]
                        assert(crop_img.shape == (new_h, new_w, 3))
                        scaled_clips[:,:,frame_id*3:frame_id*3+3] = cv2.resize(crop_img, (self.size, self.size), self.interpolation)
                    return scaled_clips
                else:
                    num_imgs = int(c / 1)
                    for frame_id in range(num_imgs):
                        cur_img = clips[:,:,frame_id:frame_id+1]
                        crop_img = cur_img[y1:y1+new_h, x1:x1+new_w, :]
                        assert(crop_img.shape == (new_h, new_w, 1))
                        scaled_clips[:,:,frame_id:frame_id+1] = cv2.resize(crop_img, (self.size, self.size), self.interpolation)
                    return scaled_clips

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(clips))

class MultiScaleCrop(object):
    """
    Description: Corner cropping and multi-scale cropping. Two data augmentation techniques introduced in:
        Towards Good Practices for Very Deep Two-Stream ConvNets,
        http://arxiv.org/abs/1507.02159
        Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao

    Parameters:
        size: height and width required by network input, e.g., (224, 224)
        scale_ratios: efficient scale jittering, e.g., [1.0, 0.875, 0.75, 0.66]
        fix_crop: use corner cropping or not. Default: True
        more_fix_crop: use more corners or not. Default: True
        max_distort: maximum distortion. Default: 1
        interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, size, scale_ratios, fix_crop=True, more_fix_crop=True, max_distort=1, interpolation=cv2.INTER_LINEAR):
        self.height = size[0]
        self.width = size[1]
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.interpolation = interpolation

    def fillFixOffset(self, datum_height, datum_width):
        h_off = int((datum_height - self.height) / 4)
        w_off = int((datum_width - self.width) / 4)

        offsets = []
        offsets.append((0, 0))          # upper left
        offsets.append((0, 4*w_off))    # upper right
        offsets.append((4*h_off, 0))    # lower left
        offsets.append((4*h_off, 4*w_off))  # lower right
        offsets.append((2*h_off, 2*w_off))  # center

        if self.more_fix_crop:
            offsets.append((0, 2*w_off))        # top center
            offsets.append((4*h_off, 2*w_off))  # bottom center
            offsets.append((2*h_off, 0))        # left center
            offsets.append((2*h_off, 4*w_off))  # right center

            offsets.append((1*h_off, 1*w_off))  # upper left quarter
            offsets.append((1*h_off, 3*w_off))  # upper right quarter
            offsets.append((3*h_off, 1*w_off))  # lower left quarter
            offsets.append((3*h_off, 3*w_off))  # lower right quarter

        return offsets

    def fillCropSize(self, input_height, input_width):
        crop_sizes = []
        base_size = np.min((input_height, input_width))
        scale_rates = self.scale_ratios
        for h in range(len(scale_rates)):
            crop_h = int(base_size * scale_rates[h])
            for w in range(len(scale_rates)):
                crop_w = int(base_size * scale_rates[w])
                # append this cropping size into the list
                if (np.absolute(h-w) <= self.max_distort):
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def __call__(self, clips, selectedRegionOutput=False):
        h, w, c = clips.shape
        is_color = False
        if c % 3 == 0:
            is_color = True

        crop_size_pairs = self.fillCropSize(h, w)
        size_sel = random.randint(0, len(crop_size_pairs)-1)
        crop_height = crop_size_pairs[size_sel][0]
        crop_width = crop_size_pairs[size_sel][1]

        if self.fix_crop:
            offsets = self.fillFixOffset(h, w)
            off_sel = random.randint(0, len(offsets)-1)
            h_off = offsets[off_sel][0]
            w_off = offsets[off_sel][1]
        else:
            h_off = random.randint(0, h - self.height)
            w_off = random.randint(0, w - self.width)

        scaled_clips = np.zeros((self.height,self.width,c))
        if is_color:
            num_imgs = int(c / 3)
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id*3:frame_id*3+3]
                crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
                scaled_clips[:,:,frame_id*3:frame_id*3+3] = cv2.resize(crop_img, (self.width, self.height), self.interpolation)
            if not selectedRegionOutput:
                return scaled_clips
            else:
                return scaled_clips, off_sel
        else:
            num_imgs = int(c / 1)
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id:frame_id+1]
                crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
                scaled_clips[:,:,frame_id:frame_id+1] = np.expand_dims(cv2.resize(crop_img, (self.width, self.height), self.interpolation), axis=2)
            if not selectedRegionOutput:
                return scaled_clips
            else:
                return scaled_clips, off_sel


class MultiScaleFixedCrop(object):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.height = size[0]
        self.width = size[1]
        self.interpolation = interpolation

    def fillFixOffset(self, datum_height, datum_width):
        h_off = int((datum_height - self.height) / 4)
        w_off = int((datum_width - self.width) / 4)

        offsets = []
        offsets.append((0, 0))          # upper left
        offsets.append((0, 4*w_off))    # upper right
        offsets.append((4*h_off, 0))    # lower left
        offsets.append((4*h_off, 4*w_off))  # lower right
        offsets.append((2*h_off, 2*w_off))  # center

        return offsets


    def __call__(self, clips, selectedRegionOutput=False):
        h, w, c = clips.shape
        is_color = False
        if c % 3 == 0:
            is_color = True

        crop_height = 224
        crop_width = 224


        offsets = self.fillFixOffset(h, w)
        scaled_clips_list = []
        for offset in offsets:
            h_off = offset[0]
            w_off = offset[1]
    
    
            scaled_clips = np.zeros((self.height,self.width,c))
            scaled_clips_flips = np.zeros((self.height,self.width,c))
            if is_color:
                num_imgs = int(c / 3)
                for frame_id in range(num_imgs):
                    cur_img = clips[:,:,frame_id*3:frame_id*3+3]
                    crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
                    scaled_clips[:,:,frame_id*3:frame_id*3+3] = cv2.resize(crop_img, (self.width, self.height), self.interpolation)
                    scaled_clips_flips = scaled_clips[:,::-1,:].copy()
            else:
                num_imgs = int(c / 1)
                for frame_id in range(num_imgs):
                    cur_img = clips[:,:,frame_id:frame_id+1]
                    crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
                    scaled_clips[:,:,frame_id:frame_id+1] = np.expand_dims(cv2.resize(crop_img, (self.width, self.height), self.interpolation), axis=2)
                    scaled_clips_flips = scaled_clips[:,::-1,:].copy()
                    
            scaled_clips_list.append(np.expand_dims(scaled_clips,-1))
            scaled_clips_list.append(np.expand_dims(scaled_clips_flips,-1))
        return np.concatenate(scaled_clips_list,axis=-1)

class rawPoseAugmentation(object):
    def __init__(self, scale_ratios):
        self.possible_scale_tuples = []
        self.scale_ratios = scale_ratios
        for i in range(len(scale_ratios)):
            for j in range(len(scale_ratios)):
                if np.abs(i-j) < 2:
                    scale_ration_height = self.scale_ratios[i]
                    scale_ration_width = self.scale_ratios[j]
                    self.possible_scale_tuples.append((scale_ration_height, scale_ration_width))
        self.length_possible_scale_tuples = len(self.possible_scale_tuples)
    def __call__(self, poses):
        selected_random_scale_tuple_index = np.random.randint(self.length_possible_scale_tuples)
        selected_scale_height = self.possible_scale_tuples[selected_random_scale_tuple_index][0]
        selected_scale_width = self.possible_scale_tuples[selected_random_scale_tuple_index][1]
        random_crop_height_start = np.random.uniform(0,1-selected_scale_height)
        random_crop_width_start = np.random.uniform(0,1-selected_scale_width)
#        pos_not_touched = poses.copy()
        check_width = poses[:,:,0,:] > random_crop_width_start + selected_scale_width
        check_height = poses[:,:,1,:] > random_crop_height_start + selected_scale_height
        check = np.logical_or(check_width,check_height)
        check = np.expand_dims(check, 2)
        check = np.concatenate((check,check),2)
        poses[check] = 0
        poses[:,:,0,:] -= random_crop_width_start
        poses[:,:,1,:] -= random_crop_height_start
        poses[poses < 0] = None
        poses[:,:,0,:] /= selected_scale_width
        poses[:,:,1,:] /= selected_scale_height
        if len(poses[poses>1]) > 0:
            print('basdasd')
        return poses
    
class pose_one_hot_decoding(object):
    def __init__(self,length):
        self.space = 0.1
        self.number_of_people = 1
        self.total_bins = self.number_of_people * 25
        self.one_hot_vector_length_per_joint = (1/self.space ) ** 2 
        self.one_hot_vector_length = int(self.total_bins * self.one_hot_vector_length_per_joint + 1)
        self.one_hot = np.zeros(self.one_hot_vector_length)
        self.length = length
        self.onehot_multiplication = np.repeat(range(self.total_bins), length).reshape(self.total_bins,length)
    def __call__(self, poses):
        poses = poses.reshape(-1,2,self.length)
        dim1 = np.floor(poses[:,0,:] / self.space)
        dim2 = np.floor(poses[:,1,:] / self.space)
        one_hot_values = (1/self.space ) * dim1 + dim2
        one_hot_values[np.isnan(one_hot_values)] = self.one_hot_vector_length_per_joint
        one_hot_values = one_hot_values * self.onehot_multiplication + one_hot_values
        one_hot_values[np.isnan(one_hot_values)] = self.one_hot_vector_length + 1
        
        return poses
    
class pose_one_hot_decoding2(object):
    def __init__(self,length):
        self.space = 1/32
        self.bin_number = int((1/self.space))
        self.number_of_people = 1
        self.total_bins = self.number_of_people * 25
        self.one_hot_vector_length = self.bin_number ** 2
        self.one_hot = np.zeros(self.one_hot_vector_length)
        self.length = length
        self.position_matrix = np.zeros([self.bin_number + 1, self.bin_number + 1, self.length])
    def __call__(self, poses):
        poses = poses.reshape(-1,2,self.length)
        dim1 = np.floor(poses[:,0,:] / self.space)
        dim2 = np.floor(poses[:,1,:] / self.space)
        dim1[np.isnan(dim1)] = self.bin_number
        dim2[np.isnan(dim2)] = self.bin_number
        dim1 = dim1.astype(np.int)
        dim2 = dim2.astype(np.int)
        for i in range(self.length):
            try:
                self.position_matrix[dim1[:,i], dim2[:,i], i] = 1
            except:
                print('hasdasd')
        one_hot_encoding = self.position_matrix[:self.bin_number, :self.bin_number, :]
        one_hot_encoding = one_hot_encoding.reshape(-1,self.length)
        one_hot_encoding_torch = torch.from_numpy(one_hot_encoding.transpose((1,0))).float()
        
        
        return one_hot_encoding_torch
        
class ToTensorPose(object):

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            # handle numpy ar
            clips = clips - 0.5
            clips[np.isnan(clips)] = 0
            clips = torch.from_numpy(clips.transpose((3,0,1,2))).float()
            # backward compatibility
            return clips
        
                    