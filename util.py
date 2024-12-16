import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.nn import functional as F
from torchvision import transforms

import os
import cv2
import operator
from scipy.ndimage import zoom

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     print("image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy():",image.shape, label.shape)
#     #(148, 512, 512) (148, 512, 512)
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         print("prediction = np.zeros_like(label):", prediction.shape)
#         #(148, 512, 512)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]#(512, 512)
#             print("slice = image[ind, :, :]", slice.shape)
#             x, y = slice.shape[0], slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
#                 print("slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3):", slice.shape)
#                 #(224, 224)
#             x_transforms = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5])
#             ])
#             input = x_transforms(slice).unsqueeze(0).float().cuda()
#             #torch.Size([1, 1, 224, 224])
#             print("input = x_transforms(slice).unsqueeze(0).float().cuda()", input.shape)
#
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 #torch.Size([1, 9, 224, 224])
#                 print("outputs = net(input):", outputs.shape)
#                 # outputs = F.interpolate(outputs, size=slice.shape[:], mode='bilinear', align_corners=False)
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 # torch.Size([224, 224])
#                 print("out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0):", out.shape)
#                 out = out.cpu().detach().numpy()
#                 # (224, 224)
#                 print("out = out.cpu().detach().numpy():", out.shape)
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                     #(512, 512)
#                     print("pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0):", pred.shape)
#
#                 else:
#                     pred = out
#                     print("pred = out:", pred.shape)
#                 prediction[ind] = pred
#                 #prediction[0] = (512, 512)
#                 #prediction[1] = (512, 512)
#                 #......
#                 #prediction[147] = (512, 512)
#                 print("prediction[ind] = pred:,ind", pred.shape, ind)
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         print("else:input = torch.from_numpy(image).:", input.shape)
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             print("out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0):", out.shape)
#             prediction = out.cpu().detach().numpy()
#             print("prediction = out.cpu().detach().numpy():", prediction.shape)
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))
#         print("metric_list:", metric_list)
#         #[[dice1, hd951],[dice2, hd952]....]
#
#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     return metric_list

def vis_save(original_img, pred, save_path,save_path2):
    blue   = [30,144,255] # aorta
    green  = [0,255,0]    # gallbladder
    red    = [255,0,0]    # left kidney
    cyan   = [0,255,255]  # right kidney
    pink   = [255,0,255]  # liver
    yellow = [255,255,0]  # pancreas
    purple = [128,0,255]  # spleen
    orange = [255,128,0]  # stomach
    original_img = original_img * 255.0
    original_img = original_img.astype(np.uint8)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
    original_img0 = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path2, original_img0)
    pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)

    original_img = np.where(pred==1, np.full_like(original_img, blue  ), original_img)
    original_img = np.where(pred==2, np.full_like(original_img, green ), original_img)
    original_img = np.where(pred==3, np.full_like(original_img, red   ), original_img)
    original_img = np.where(pred==4, np.full_like(original_img, cyan  ), original_img)
    original_img = np.where(pred==5, np.full_like(original_img, pink  ), original_img)
    original_img = np.where(pred==6, np.full_like(original_img, yellow), original_img)
    original_img = np.where(pred==7, np.full_like(original_img, purple), original_img)
    original_img = np.where(pred==8, np.full_like(original_img, orange), original_img)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, original_img)

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # (148, 512, 512) (148, 512, 512)
    #-------------------------------------------------------------------------------------
    # data_cpu = image.clone().cpu()  # 取出图到cpu
    # my_label_cpu = label.clone().cpu()  # 取出预测的二值分割到cpu
    data_cpu = image  # 取出图到cpu
    my_label_cpu = label  # 取出预测的二值分割到cpu
    # save_path_true = os.path.join(test_save_path, "true_pic")
    save_path_true = test_save_path + '/true_pic'
    save_path00 = test_save_path + '/CT_pic'
    if not os.path.exists(save_path_true):  # 建立subset文件夹
        os.mkdir(save_path_true)
    if not os.path.exists(save_path00):  # 建立subset文件夹
        os.mkdir(save_path00)
    for i in range(len(data_cpu)):  # 取出改batch中的单张图
        #save_path = os.path.join(save_path_true, '/%d_%d.jpg' % (case, i))
        save_path0 = save_path00 + '/%s_%d.jpg' % (case, i)
        save_path = save_path_true + '/%s_%d.jpg'%(case, i)
        original_img = data_cpu[i]  # 取图得到张量tensor，注意这里的[0]是因为我们在dataset部分给图增加了一个维度

        pred = my_label_cpu[i]  # 取得预测的二值分割张量tensor
        vis_save(original_img, pred, save_path=save_path,save_path2=save_path0)
    #--------------------------------------------------------------------------------------

    if len(image.shape) == 3:
        prediction = np.zeros_like(label) #(148, 512, 512)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]#(512, 512)
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input = x_transforms(slice).unsqueeze(0).float().cuda()
#            ##torch.Size([1, 1, 224, 224])
            net.eval()

            with torch.no_grad():

                outputs = net(input)
                # torch.Size([1, 9, 224, 224])
                # outputs = F.interpolate(outputs, size=slice.shape[:], mode='bilinear', align_corners=False)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    # -------------------------------------------------------------------------------------
    # data_cpu = image.clone().cpu()  # 取出图到cpu
    # my_pred_cpu = prediction.clone().cpu()  # 取出预测的二值分割到cpu
    data_cpu = image  # 取出图到cpu
    my_pred_cpu = prediction  # 取出预测的二值分割到cpu
    save_path_pred = test_save_path + '/pred_pic'

    #save_path_pred = os.path.join(test_save_path, "pred_pic")
    if not os.path.exists(save_path_pred):  # 建立subset文件夹
        os.mkdir(save_path_pred)
    for i in range(len(data_cpu)):  # 取出改batch中的单张图
        save_path0 = save_path00 + '/%s_%d.jpg' % (case, i)
        save_path = save_path_pred + '/%s_%d.jpg'%(case, i)
        #save_path = os.path.join(save_path_pred, '/%d_%d.jpg' % (case, i))
        original_img = data_cpu[i]  # 取图得到张量tensor，注意这里的[0]是因为我们在dataset部分给图增加了一个维度
        pred = my_pred_cpu[i]  # 取得预测的二值分割张量tensor
        vis_save(original_img, pred, save_path=save_path,save_path2=save_path0)
    # --------------------------------------------------------------------------------------
    for i in range(1, classes):
        # prediction=(148, 512, 512)，label=(148, 512, 512)
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        #[[dice1, hd951],[dice2, hd952]....]

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
