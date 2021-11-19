import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import numpy as np
import torch
from torchvision import models

import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm


from utils_se import *
from spaevo_attack import SpaEvoAtt
import argparse

if __name__ == "__main__":
    
    # ========== M0: args input data ===========    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset", default='imagenet', type=str, help="cifar10, cifar100 or imagenet.")
    parser.add_argument("--arch", default='vit', type=str, help="resnet18, resnet50 or vit.")
    parser.add_argument("--model_path", default=None, type=str, help="Path to a pretrained model")
    parser.add_argument("--output_dir", default=None, type=str, help="Dir to an output file")
    parser.add_argument("--norm", action='store_true', help="if dataset equivelant to a pretrained model is normalized, this arg should be True")
    parser.add_argument("--attack_setting", default='untargeted', type=str, help="targeted or untargeted.")
    parser.add_argument("--n_start", default=0, type=int, help="first sample of eval_set")
    parser.add_argument("--n_end", default=100, type=int, help="last sample of eval_set")
    parser.add_argument("--query_limit", default=5000, type=int, help="query budget for the attack")
    parser.add_argument("--cr", default=0.9, type=int, help="crossover rate")
    parser.add_argument("--mu", default=0.004, type=float, help="mutation rate, imagenet: target/untarget: 0.001/0.004, cifar10:target/untarget: 0.01/0.04")
    parser.add_argument("--pop_size", default=10, type=int, help="population size")
    parser.add_argument("--defense", default='standard', type=str, help="Standard (undefended) or AT (defend mechanism")
    args = parser.parse_args()

    # ========== M1: Load data =========== 
    batch_size = 1
    testloader, testset = load_data(args.dataset,batch_size=batch_size)
    print('=> M1: Load data successfully !!!')

    # ========== M2: Load model and draft it =========== 
    net = load_model(args.arch,args.model_path)
    model = PretrainedModel(net,args.dataset,args.arch, args.norm)
    print('=> M2: Load model successfully !!!')

    # ========== M3: Get evaluation set =========== 
    if args.attack_setting == 'targeted':
        flag = True #False - untargeted setting, True - targeted setting
    else:
        flag = False 

    if args.dataset == 'imagenet':
        n_pix = 196 # 49, 196, 784, 3136 only required for uni_rand: 4/(32*32) = 196/(224*224) = 0.004

    elif args.dataset == 'cifar10' :
        n_pix = 4 # 4, 16, 64, 256 only required for uni_rand: 4/(32*32) = 196/(224*224) = 0.004

    seed = 999
    ID_set = get_evalset(model,args.dataset, args.arch,testset,seed,flag)
    print('=> M3: Generate eval_set successfully !!!')

    # ========== M4: attack setup =========== 

    if flag:
        init_mode = 'target' #'salt_pepper', 'rand'
        pop_init = 'uni_rand' #'uni_rand' , 'grid_rand' 
        n = n_pix
    else: 
        init_mode = 'salt_pepper_att' #'gauss_rand' #'salt_pepper'
        pop_init =  'uni_rand' #'grid_rand' '=> others; 'uni_rand' => salt_pepper_att
        n = n_pix

    attack = SpaEvoAtt(model,pop_init,n,args.pop_size,args.cr,args.mu,seed,flag)

    print('=> M4: Attack setup Done !!!')

    # ========== M5: output setup ===========
    n_point = 100

    output_path = args.defense + args.arch+'_SpaEvoAtt_'+args.dataset+'_'+init_mode+'_'+pop_init+str(n)+'_popsize'+str(args.pop_size)+'_cr'+str(args.cr)+'_mu'+str(args.mu) + '_seed' + str(seed) +'_Fr'+ str(args.n_start)+'_To'+str(args.n_end)+'.csv'

    if flag:
        head = ['#','ocla','o_ID','tcla','t_ID','alabel']

    else:
        head = ['#','ocla','o_ID','alabel']

    for k in range(n_point):
        head.append('q'+str(k))

    print('=> M5: Output setup Done !!!')

    # ========== M6: Evaluation attack =========== 

    print('=> M6: Evaluation in progress ...')
    for i in tqdm(range(args.n_start,args.n_end),desc='Sample'):                          # ori_class - 10
        D = np.zeros(args.query_limit+500).astype(int)
        nquery = 0
        o = ID_set[i,1] #oID
        # 0. select original image
        oimg, olabel = testset[o]
        oimg = torch.unsqueeze(oimg, 0).cuda()

        # 1. select starting image
        if flag:
            t = ID_set[i,3] #tID, 3 is index acrross dataset - 4 is sample index in a class (not accross dataset)
            timg, tlabel = testset[t]
            timg = torch.unsqueeze(timg, 0).cuda()
            print(olabel,tlabel,model.predict_label(oimg),model.predict_label(timg))

        else:       
            timg, nqry,D_tmp = gen_starting_point(model,oimg,olabel,seed,args.dataset,init_mode)
            D[nquery:nquery + nqry] = D_tmp
            tlabel = None
            nquery += nqry

        # 2. Run attack
        max_query = args.query_limit - nquery
        adv, nqry,D_tmp = attack.evo_perturb(oimg,timg,olabel,tlabel,max_query)
        D[nquery:nquery + nqry] = D_tmp
        nquery += nqry

        # 3. write it out
        alabel = model.predict_label(adv)
        print(i, alabel,nquery,D[nquery-1])

        if flag:
            key_info = [i,olabel,o,tlabel,t,alabel.item()]
        else:
            key_info = [i,olabel,o,alabel.item()]

        export_pd_csv(D[:nquery],head,key_info,output_path,n_point,args.query_limit)