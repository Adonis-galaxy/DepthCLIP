# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from calculate_error import *
from datasets.datasets_list import MyDataset
import imageio
import imageio.core.util
from path import Path
from utils import *
from logger import AverageMeter
from model import *
from monoclip import *


parser = argparse.ArgumentParser(description='Transformer-based Monocular Depth Estimation with Attention Supervision',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting 
parser.add_argument('--models_list_dir', type=str, default='')
parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--model_dir',type=str)
parser.add_argument('--other_method',type=str) # default='MonoCLIP'
parser.add_argument('--trainfile_kitti', type=str, default = "./datasets/eigen_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_kitti', type=str, default = "./datasets/eigen_test_files_with_gt_dense.txt")
parser.add_argument('--trainfile_nyu', type=str, default = "/home/rrzhang/zengzy/code/clip_depth/datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str, default = "/home/rrzhang/zengzy/code/clip_depth/datasets/nyudepthv2_test_files_with_gt_dense.txt")
parser.add_argument('--data_path', type=str, default = "/home/rrzhang/zengzy/code/clip_depth/datasets/NYU_Depth_V2")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')

# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--epochs', default=0, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=24, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default = "KITTI")

# Logging setting
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--log-metric', default='_LRDN_evaluation.csv', metavar='PATH', help='csv where to save validation metric value')
parser.add_argument('--val_in_train', action='store_true', help='validation process in training')

# Model setting
parser.add_argument('--encoder', type=str, default = "ResNext101")
parser.add_argument('--norm', type=str, default = "BN")
parser.add_argument('--act', type=str, default = "ReLU")
parser.add_argument('--height', type=int, default = 352)
parser.add_argument('--width', type=int, default = 704)
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# Evaluation setting
parser.add_argument('--evaluate', action='store_true', help='evaluate score')
parser.add_argument('--multi_test', action='store_true', help='test all of model in the dir')
parser.add_argument('--img_save', action='store_true', help='will save test set image')
parser.add_argument('--cap', default=80.0, type=float, metavar='MaxVal', help='cap setting for kitti eval')

# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default = "0,1,2,3", help='force available gpu index')
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning


def validate(args, val_loader, model, dataset = 'KITTI'):
    ##global device
    if dataset == 'KITTI':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3','rmse','rmse_log']
    elif dataset == 'NYU':
        # error_names = ['abs_diff', 'abs_rel', 'log10', 'a1', 'a2', 'a3','rmse','rmse_log']
        error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel','log10', 'rmse']
    
    elif dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']

    errors = AverageMeter(i=len(error_names))
    length = len(val_loader)
    # switch to evaluate mode
    model.eval()
    count = 0
    # max_depth=0
    for i, (rgb_data, gt_data, dense) in enumerate(val_loader):
        if gt_data.ndim != 4 and gt_data[0] == False:
            continue
        rgb_data = rgb_data.cuda()
        gt_data = gt_data.cuda()


        input_img = rgb_data
        input_img_flip = torch.flip(input_img,[3])
        with torch.no_grad():
            
            output_depth = model(input_img)
            # print(output_depth.shape)
            output_depth_flip = model(input_img_flip)
            output_depth_flip = torch.flip(output_depth_flip,[3])
            output_depth = 0.5 * (output_depth + output_depth_flip)

            if args.other_method == 'Adabins':
                output_depth = nn.functional.interpolate(output_depth, size=[2 * output_depth.shape[2], 2 * output_depth.shape[3]], mode='bilinear', align_corners=True)
            elif args.other_method == 'MonoCLIP':
                output_depth = nn.functional.interpolate(output_depth, size=[416, 544], mode='bilinear', align_corners=True)

        if dataset == 'KITTI':
            err_result = compute_errors(gt_data, output_depth, crop=True, cap=args.cap)
        elif dataset == 'NYU':
            err_result = compute_errors_NYU(gt_data, output_depth, crop=True,idx=i)

        errors.update(err_result)
        # measure elapsed time
        if i % 50 == 0:
            print('valid: {}/{} Abs Error {:.4f} ({:.4f})'.format(i, length, errors.val[0], errors.avg[0]))

    return errors.avg,error_names


def main():
    args = parser.parse_args() 
    print("=> No Distributed Training")
    print('=> Index of using GPU: ', args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    torch.manual_seed(args.seed)

    if args.evaluate is True:
        save_path = save_path_formatter(args, parser)
        args.save_path = 'checkpoints'/save_path

    ######################   Data loading part    ##########################
    if args.dataset == 'KITTI':
        args.max_depth = 80.0
    elif args.dataset == 'NYU':
        args.max_depth = 10.0

    if args.result_dir == '':
        args.result_dir = './' + args.dataset + '_Eval_results'
    args.log_metric = args.dataset + '_' + args.encoder + args.log_metric
    
    test_set = MyDataset(args, train=False)
    print("=> Dataset: ",args.dataset)
    print("=> Data height: {}, width: {} ".format(args.height, args.width))
    print('=> test  samples_num: {}  '.format(len(test_set)))

    test_sampler = None

    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=test_sampler)
    
    cudnn.benchmark = True
    ###########################################################################
    
    ###################### setting model list #################################
    if args.multi_test is True:
        print("=> all of model tested")
        models_list_dir = Path(args.models_list_dir)
        models_list = sorted(models_list_dir.files('*.pkl'))
    else:
        print("=> just one model tested")
        models_list = [args.model_dir]


    ###################### setting Network part ###################
    print("=> creating model")
    if args.other_method == None:
        Model = LDRN(args)

        num_params_encoder = 0
        num_params_decoder = 0
        for p in Model.encoder.encoder.parameters():
            num_params_encoder += p.numel()
        for p in Model.decoder.parameters():
            num_params_decoder += p.numel()
        print("===============================================")
        print("model encoder parameters: ", num_params_encoder)
        print("model decoder parameters: ", num_params_decoder)
        print("Total parameters: {}".format(num_params_encoder + num_params_decoder))
        print("===============================================")
    else:
        if args.other_method == 'DPT-Large':
            from dpt.models import DPTDepthModel
            Model = DPTDepthModel(
            scale=0.000305,
            shift=0.1378,
            invert=False,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,)
        if args.other_method == 'Adabins':
            from Adabins import UnetAdaptiveBins
            if args.dataset == 'KITTI':
                Model = UnetAdaptiveBins.build(n_bins=256, min_val=1e-3, max_val=80,norm="linear")
            if args.dataset == 'NYU':
                Model = UnetAdaptiveBins.build(n_bins=256, min_val=1e-3, max_val=10,norm="linear")
        
        if args.other_method == 'MonoCLIP':
            if args.dataset == 'KITTI':
                Model = MonoCLIP()
            if args.dataset == 'NYU':
                Model = MonoCLIP()

        num_params = 0
        for p in Model.parameters():
            num_params += p.numel()
        print("===============================================")
        print("Total parameters: {}".format(num_params))
        print("===============================================")


    Model = Model.cuda()
    Model = torch.nn.DataParallel(Model)

    if args.evaluate is True:
        test_model = Model

        print("Model Initialized")


        test_len = len(models_list)
        print("=> Length of model list: ", test_len)

        
        for i in range(test_len):
            if args.other_method == 'MonoCLIP':
                pass
            else:
                filename = models_list[i].split('/')[-1]
                old_model_dict = torch.load(models_list[i],map_location='cuda:0')
                net_state_dict = test_model.state_dict()
                new_model_dict = {key: value for key, value in old_model_dict.items()
                                if key in net_state_dict and value.shape == net_state_dict[key].shape}
                for key, value in old_model_dict.items():
                    if key not in net_state_dict or value.shape != net_state_dict[key].shape:
                            print(key)
                net_state_dict.update(new_model_dict)
                test_model.load_state_dict(net_state_dict)
            test_model.eval()
            if args.dataset == 'KITTI':
                errors, error_names = validate(args, val_loader, test_model,'KITTI')
            elif args.dataset == 'NYU':
                errors, error_names = validate(args, val_loader, test_model,'NYU')
            print(' * model: {}'.format(models_list[i]))
            print("")
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:len(error_names)], errors[0:len(errors)]))
            print(' * Avg {}'.format(error_string))
            print("")
        print(args.dataset," valdiation finish")
        ##  Test
        
        if args.img_save is False :
            print("--only Test mode finish--")
            return
    else:
        test_model = Model
        test_model.load_state_dict(torch.load(models_list[0],map_location='cuda:0'))
        test_model.eval()
        print("=> No validation")


    test_set = MyDataset(args, train=False, return_filename=True)
    test_sampler = None
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

if __name__ == "__main__":
    main()



