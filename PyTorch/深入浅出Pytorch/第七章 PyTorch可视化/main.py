# import torchvision.models as models
# from torchinfo import summary
# resnet18 = models.resnet18()
# summary(resnet18, (1, 3, 224, 224))
# import matplotlib.pyplot as plt
# import torch
# from torchvision.models import vgg11
#
# model = vgg11(pretrained=True)
# print(dict(model.features.named_children()))
#
# #
# # conv1 = dict(model.features.named_children())['3']
# # kernel_set = conv1.weight.detach()
# # num = len(conv1.weight.detach())
# # print(kernel_set.shape)
# # for i in range(0, num):
# #     i_kernel = kernel_set[i]
# #     plt.figure(figsize=(20, 17))
# #     if (len(i_kernel))>1:
# #         for idx, filter in enumerate(i_kernel):
# #             plt.subplot(9,9, idx+1)
# #             plt.axis('off')
# #             # plt.imshow(filter[:,:].detach(),cmap='bwr')
# #             plt.imsave(f'{i}_{idx}.png',filter[:,:].detach(),cmap='bwr')
#
# import torch
# from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
#
# model = vgg11(pretrained=True)
# img_path = r"D:\Downloads\Snipaste_2026-05-13_17-06-53.png"
# # resize操作是为了和传入神经网络训练图片大小一致
# img = Image.open(img_path).resize((224,224))
# # 需要将原始图片转为np.float32格式并且在0-1之间
# rgb_img = np.float32(img)/255
# # plt.imshow(img)
# plt.imsave('1.png', rgb_img)
#
# from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
#
# # 将图片转为tensor
# img_tensor = torch.from_numpy(rgb_img).permute(2,0,1).unsqueeze(0)
#
# target_layers = [model.features[-1]]
# # 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
# cam = GradCAM(model=model,target_layers=target_layers)
# targets = [ClassifierOutputTarget(preds)]
# # 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
# grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
# grayscale_cam = grayscale_cam[0, :]
# cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# print(type(cam_img))
# img = Image.fromarray(cam_img)
# img.save('2.png')


# Download example images
# !mkdir -p images
# !wget -nv \
#    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/great_grey_owl.jpg \
#    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/peacock.jpg   \
#    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/toucan.jpg    \
#    -P /content/images

import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

model = models.alexnet(pretrained=True)
backprop = Backprop(model)

image = load_image('./great_grey_owl.jpg')
owl = apply_transforms(image)

target_class = 24
backprop.visualize(owl, target_class, guided=True, use_gpu=True)

import torchvision.models as models
from flashtorch.activmax import GradientAscent

model = models.vgg16(pretrained=True)
g_ascent = GradientAscent(model.features)

# specify layer and filter info
conv5_1 = model.features[24]
conv5_1_filters = [45, 271, 363, 489]

g_ascent.visualize(conv5_1, conv5_1_filters, title="VGG16: conv5_1")
