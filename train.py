
from __future__ import absolute_import
from __future__ import division
import argparse
import torch.nn.functional as F
from torchvision.transforms import transforms
import sys
from my_functionals.Dloss import dice_coef_loss
import torch
import network
import optimizer1
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
from torchvision import transforms
from datasets.dataset_npy import MyDataSet, valDataset
parser = argparse.ArgumentParser(description='MFNET')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--arch', type=str, default='network.mfed.MFED')
parser.add_argument('--sgd', action='store_true', default=False)
parser.add_argument('--adam', action='store_true', default=True)
parser.add_argument('--trunk', type=str, default='resnet101', help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=60)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=16)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
args = parser.parse_args()
#Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1
'''
Main Function
'''
def main():
    net = network.get_eddynet(args)
    optim, scheduler = optimizer1.get_optimizer(args, net)
    train_sampler = None
    val_sampler = None
    data_path = "home/eddy/GSCNN-master/datasets"

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    transforms_ = standard_transforms.Compose(transforms_)

    train_dataset = MyDataSet(
        data=data_path,
        transform=transforms_
        )
    #
    val_dataset = valDataset(
        data=data_path,
        transform=transforms_
        )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              num_workers=4, shuffle=(train_sampler is None), drop_last=True, sampler = train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                            num_workers=4 // 2 , shuffle=False, drop_last=False, sampler = val_sampler)


    torch.cuda.empty_cache()
    acc = []
    good_score = 0
    #Main Loop
    for epoch in range(args.start_epoch, args.max_epoch):
        acc_epoch = 0
        scheduler.step()
        precision_epoch =0
        recall_epoch = 0
        train(train_loader, net, optim, epoch)
        acc_epoch,recall_epoch,precision_epoch = validate(val_loader, net, optim, epoch,acc_epoch,recall_epoch,precision_epoch)
        result_score,result_recall,result_precision = acc_epoch / len(val_loader),recall_epoch/ len(val_loader),precision_epoch/ len(val_loader)
        acc.append(acc_epoch/len(val_loader))


        if result_score > good_score:
            torch.save(net.state_dict(), "/home/eddy/GSCNN-master/datasets/sample/model_%d.pth" % (epoch))
            good_score=result_score
            print("best F1 :"+str(good_score)+" best epoch:"+str(epoch)+" best result_recall:"+str(result_recall)+" best result_precision:"+str(result_precision))


# eddy--contour--loss
def bce_contour_loss(input, target):
    # binary_cross_entropy_with_logits
    bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='mean')

    # l1 loss
    l1_loss = F.l1_loss(torch.sigmoid(input), target, reduction='mean')
    # balance loss
    loss = bce_loss + l1_loss
    return loss


def train(train_loader, net, optimizer, curr_epoch):
    net.train()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):

        inputs, input2,mask, edge,edge2 = data

        mask = torch.transpose(torch.transpose(mask, 1, 3), 2, 3)
        edge = torch.transpose(torch.transpose(edge, 1, 3), 2, 3)
        edge2 = torch.transpose(torch.transpose(edge2, 1, 3), 2, 3)

        if torch.sum(torch.isnan(inputs)) > 0:
            import pdb; pdb.set_trace()

        inputs, input2,mask, edge, edge2 = inputs.cuda(), input2.cuda(), mask.cuda(), edge.cuda(),edge2.cuda()
        inputs = inputs.type(torch.FloatTensor).cuda()
        input2 = input2.type(torch.FloatTensor).cuda()
        edge = edge.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()

        main_loss = None
        loss_dice = None

        L1_G = torch.nn.L1Loss()

        if torch.cuda.is_available():
            L1_G = L1_G.cuda()
        if args.max_epoch !=0:
            # model per
            seg_out, edge_out = net(inputs,input2,edge)
            # dice loss
            loss_dice  = dice_coef_loss(mask,seg_out)
            #L1 loss
            loss_l1 = L1_G(mask,seg_out)
            # contour loss
            loss_contour_class = bce_contour_loss(edge,edge_out)
            #
            main_loss=loss_dice+loss_contour_class*5+loss_l1*2
        main_loss.backward()
        optimizer.step()
        curr_iter += 1
        sys.stdout.write(
            "\r[Epoch %d/%d: batch %d/%d] [main_loss: %.3f, L1_loss: %.3f, dice_coef_loss: %.3f, contour_loss: %.3f ]"
            % (args.max_epoch, curr_epoch, i, len(train_loader),
               main_loss.item(), loss_l1.item(),loss_dice.item(),loss_contour_class.item())
            )

def precision(y_true, y_pred):
    true_positive = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positive = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positive / (predicted_positive+0.00001)
    return precision
def recall(y_true, y_pred):
    true_positive = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positive = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positive / (possible_positive +0.00001)
    return recall
def fbeta_score(y_true, y_pred, beta = 1):
    y_true = y_true.type(torch.FloatTensor)
    y_pred = y_pred.type(torch.FloatTensor)
    if beta < 0:
        raise ValueError('the lowest choosable beta is zero')
    temp = torch.clip(y_true, 0, 1)
    temp = torch.round(temp)
    if torch.sum(torch.round(torch.clip(y_true, 0, 2))) == 0:
    # if torch.sum(torch.round(torch.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    b = beta ** 2
    fbeta_score = (1 + b) * (p * r) / (b * p + r)
    return fbeta_score,r,p
def validate(val_loader, net, optimizer, curr_epoch,acc_epoch,recall_epoch,precision_epoch):

    net.eval()

    for vi, data in enumerate(val_loader):
        input, input2,mask, edge = data
        assert len(input.size()) == 4 and len(mask.size()) == 4
        mask = torch.transpose(torch.transpose(mask, 1, 3), 2, 3)

        input, input2,mask_cuda, edge_cuda = input.cuda(),input2.cuda(), mask.cuda(), edge.cuda()
        edge_cuda = torch.unsqueeze(edge_cuda, 1)
        edge_cuda = edge_cuda.type(torch.FloatTensor).cuda()
        input = input.type(torch.FloatTensor).cuda()
        input2 = input2.type(torch.FloatTensor).cuda()

        with torch.no_grad():
            seg_out, edge_out = net(input,input2,edge_cuda)    # output = (1, 19, 713, 713)

        fbeta_score1,recall,precision  = fbeta_score(mask_cuda,seg_out)

        sys.stdout.write(
            "\r[Epoch %d/%d: batch %d/%d] [fbeta_score1: %.3f, recall: %.3f, precision: %.3f ]"
            % (args.max_epoch, curr_epoch, vi, len(val_loader),
                fbeta_score1.item(),recall.item(),precision.item() )
            )

        acc_epoch += fbeta_score1
        recall_epoch += recall
        precision_epoch+=precision

    return acc_epoch,recall_epoch,precision_epoch

if __name__ == '__main__':
    main()




