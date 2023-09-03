import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import numpy
import copy
import random
from tqdm import tqdm
import numpy as np
import argparse
from datasets.DomainNet import get_domainnet_dloader
from datasets.TingImagenet import TinyImageNet_load
from datasets.openimage import get_openimage_dataset
from datasets.NICOPP import get_nicopp_dataset,get_nicou_dataset
import os
import logging
import copy
from collections import OrderedDict
from utils import partition,Truncated,evaluation
from client import Client
from server import Server,ServerData_read
from sdmodel import ClientImageEncoder
# logging.basicConfig()


# os.environ['CUDA_VISIBLE_DEVICES'] ='2'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', default="/home/share/DomainNet")#/home/share/DomainNet/home/share/tiny-imagenet-200
    parser.add_argument('--alpha', default=1,type=float,help='degree of non-iid, only used for tinyimagenet')
    parser.add_argument('--beta', default=0,type=float,help='degree of noise')
    parser.add_argument('--data', default='openimage',help='tinyimagenet or domainnet or openimage or nicopp or nicou')
    parser.add_argument('--seed', default=0,type=int,)
    parser.add_argument('--batch_size', default=256,type=int,)
    parser.add_argument('--serverbs', default=256,type=int,)
    parser.add_argument('--serverepoch', default=10,type=int,)
    parser.add_argument('--clientepoch', default=10,type=int,)
    parser.add_argument('--learningrate', default=0.01,type=float,)
    parser.add_argument('--fewnum', default=5,type=int,help='how many imgs in each class of the client')
    parser.add_argument('--num_clients', default=5,type=int,help='number of clinets, only used for tinyimagenet')
    parser.add_argument('--split-type', default='shard',help='dirichlet or shard')
    parser.add_argument('--repeat', default=10,type=int,help='how many imgs to be generated on the server')
    parser.add_argument('--inference-steps', default=20,type=int,)
    parser.add_argument('--path-genimg', default='oi_test',help='where to save the generated imgs')#cluster _oneshot
    return parser

drop_last = False

########################################################################################################################
parser = get_parser()
args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
np.random.seed(seed) 
random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True


# logging.basicConfig(
#     filename='./log/'+ args.data+'_'+args.train+'_'+args.traintime+'.log',
#     # filename='/home/qinbin/test.log',
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

#======================= prepare dataset AND clients AND server==========================================
if args.data  == 'tinyimagenet': 
    num_classes = 200
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    # transform_train = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])#, normalize
    train_data = datasets.ImageFolder(root=os.path.join(args.base_path, 'train'), transform=transform)
    test_data = TinyImageNet_load(args.base_path, train=False, transform=transform)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8,shuffle=False, pin_memory=True)
    localdatas,traindata_cls_counts = partition(args.alpha,train_data,args.num_clients,ptype=args.split_type)
    clients = []
    for i,dataidxs in enumerate(localdatas):
        clientdataset = Truncated(train_data,dataidxs,transform)
        trainloader = torch.utils.data.DataLoader(clientdataset, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
        clients.append(Client(trainloader,num_classes,beta=args.beta))
        
    
elif args.data  == 'domainnet': 
    num_classes = 345
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch']#['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    clients,test_data = [],[]
    server_labeled = {ddd:None for ddd in domains}
    for domain in domains:
        # print(domain)
        train_dataset_server,train_dataset_client,testdataset = get_domainnet_dloader(args.base_path,domain,args.batch_size,transform,shotnum=args.fewnum)
        print(len(train_dataset_server),len(train_dataset_client),len(testdataset))
        server_labeled[domain] = train_dataset_server
        test_data.append(torch.utils.data.DataLoader(testdataset, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
        trainloader = torch.utils.data.DataLoader(train_dataset_client, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
        clients.append(Client(trainloader,num_classes,beta=args.beta))
        
    train_dataset_server,train_dataset_client,testdataset = get_domainnet_dloader(args.base_path,'real',args.batch_size,transform,shotnum=args.fewnum)
    server_labeled['real'] = train_dataset_server   
    print(len(train_dataset_server),len(train_dataset_client),len(testdataset))
    server_labeled_loader = torch.utils.data.DataLoader(server_labeled['real'], batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
    
elif args.data =='openimage':
    num_classes = 20
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    print('getting server data')
    server_train_data,server_test_data = get_openimage_dataset(transform,divide = 0,max_num = 400)
    print('server data num:',len(server_train_data),len(server_test_data))
    num_clients = 5
    clients,test_data = [],[]
    test_data.append(torch.utils.data.DataLoader(server_test_data, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
    server_labeled_loader = torch.utils.data.DataLoader(server_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
    for i in range(num_clients):
        print(f'getting client {i+1} data')
        client_train_data,client_test_data = get_openimage_dataset(transform,divide = i+1,max_num = 400)
        
        print(f'client {i+1} data num:',len(client_train_data),len(client_test_data))
        
        trainloader = torch.utils.data.DataLoader(client_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
        clients.append(Client(trainloader,num_classes,beta=args.beta))
        test_data.append(torch.utils.data.DataLoader(client_test_data, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
elif args.data =='nicopp':
    num_classes = 60
    nico_domains = ['autumn', 'dim', 'grass', 'outdoor', 'rock','water']
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    print('getting server data')
    server_train_data,server_test_data = get_nicopp_dataset(transform,divide = nico_domains[0])
    print('server data num:',len(server_train_data),len(server_test_data))
    num_clients = 5
    clients,test_data = [],[]
    test_data.append(torch.utils.data.DataLoader(server_test_data, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
    server_labeled_loader = torch.utils.data.DataLoader(server_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
    for i in range(num_clients):
        print(f'getting client {i+1} data')
        client_train_data,client_test_data = get_nicopp_dataset(transform,divide =nico_domains[i+1])
        print(f'client {i+1} data num:',len(client_train_data),len(client_test_data))
        trainloader = torch.utils.data.DataLoader(client_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
        clients.append(Client(trainloader,num_classes,beta=args.beta))
        test_data.append(torch.utils.data.DataLoader(client_test_data, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
elif args.data =='nicou':
    num_classes = 60
    nico_domains = [0,1,2,3,4,5]
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    print('getting server data')
    server_train_data,server_test_data = get_nicou_dataset(transform,divide = nico_domains[0])
    print('server data num:',len(server_train_data),len(server_test_data))
    num_clients = 5
    clients,test_data = [],[]
    test_data.append(torch.utils.data.DataLoader(server_test_data, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
    server_labeled_loader = torch.utils.data.DataLoader(server_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
    for i in range(num_clients):
        print(f'getting client {i+1} data')
        client_train_data,client_test_data = get_nicou_dataset(transform,divide =nico_domains[i+1])
        print(f'client {i+1} data num:',len(client_train_data),len(client_test_data))
        trainloader = torch.utils.data.DataLoader(client_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
        clients.append(Client(trainloader,num_classes,beta=args.beta))
        test_data.append(torch.utils.data.DataLoader(client_test_data, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
    
     
server = Server(server_labeled_loader,transform,args.serverbs,num_classes,beta=0,repeat = args.repeat,steps = args.inference_steps,imgpath=args.path_genimg)

print('getting protos') 
proto = server.get_class_proto()
print('getting features')
feas = []

for i,client in enumerate(tqdm(clients)):
    client.get_ori_features()
    fea=client.get_features(proto)
    feas.append(fea)

print('generating images')
#do_generate=False if images are already generated
server.update_features(feas,do_generate = True,directtrain=False)

print('server training')
server.train(lr=args.learningrate,epochs = args.serverepoch,test_data = test_data)


print('server testing')
print(domains)
top1, topk = evaluation(server.model,test_data)
print(f'final server model: top1 {top1}, top5 {topk}')



server.directtrain(lr=args.learningrate,epochs = args.serverepoch,test_data = test_data)
print('direct training')
print(domains)
top1, topk = evaluation(server.tempmodel,test_data)
print(f'final direct training model: top1 {top1}, top5 {topk}')

# clientaccuracy = []
# for i,client in enumerate(clients):
#     client.train(lr=args.learningrate,epochs = args.clientepoch)
#     top1,_ = evaluation(client.model,test_data[i])
#     print(top1)
#     clientaccuracy.append(top1)
# print(clientaccuracy)

# clientaccuracy = []
# for client in clients:
#     client.train(lr=args.learningrate,epochs = args.clientepoch)
#     top1,_ = evaluation(client.model,test_data)
#     clientaccuracy.append(top1)
# print(clientaccuracy)
    