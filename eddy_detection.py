
from PIL import Image
import cv2
from torchvision.utils import save_image
import torch
import numpy as np
def eddy_vis(input_data,edge, net):

    input = input_data.cuda()

    edge_cuda = edge.type(torch.FloatTensor).cuda()

    with torch.no_grad():
        seg_out, edge_out = net(input, edge_cuda)
    seg_predictions = seg_out.argmax(1)
    save_image(seg_out.data, "./data/test/val_segout.png",
               nrow=1, normalize=True)

from torch.autograd import Variable
if __name__ == '__main__':
    # is_cuda = torch.cuda.is_available()
    is_cuda = True
    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    img_path = "./datasets/test/eddy.png"
    # edge_path = "/home/eddy/GSCNN-master/edge_test.png"
    img = cv2.imread(img_path,0)
    # edge = cv2.imread(edge_path)
    from torchvision import transforms
    x_size = img.shape
    # im_arr = img.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
    img_width, img_height, channels = 101, 120, 1
    transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5), (0.5)), ]
    transform = transforms.Compose(transforms_)
    inp_img = transform(Image.open(img_path).convert('L'))
    # inp_img1 = transform(img)
    inp_img = Variable(inp_img).type(Tensor)

    canny = np.zeros((1,x_size[0],  x_size[1]))
    canny = cv2.Canny(img, 10, 100)
    # canny_1 = Variable(canny).type(Tensor).unsqueeze(0)
    canny = torch.from_numpy(canny).cuda().float().unsqueeze(0)
    img = transforms.ToTensor()(img)

    from network import mfed
    model = mfed.MFED(num_classes=3)
    model.load_state_dict(torch.load("./checkpoints/best_model.pth"))

    if is_cuda: model.cuda()
    model.eval()
    print("Loaded model from %s" % ("best_model"))
    inp_img = inp_img.unsqueeze(0)
    canny = canny.unsqueeze(0)
    eddy_vis(inp_img,canny ,model)

