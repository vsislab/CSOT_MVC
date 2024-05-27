import torch
from network import Network
from metric import valid
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


Dataname = 'Caltech-5V'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=0.5)
parser.add_argument("--alpha", default=0.3)
parser.add_argument("--beta", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=1)
parser.add_argument("--full_epochs", default=100)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if args.dataset == "Caltech-5V":
    args.full_epochs = 100
    seed = 3


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)


dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))



def full_train(epoch, pro, alpha, beta):
    tot_loss = 0.
    rec_loss = 0.
    sem_loss = 0.
    wcl_loss = 0.
    mes = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    center = pro
    for batch_idx, (xs, label, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        d_list = []
        s_list = []
        k_list = []
        wd_list = []
        e_list = []
        for v in range(view):
            cos_d = consine_similarity(hs[v], center)

            # using optimal transport to align samples with joint clusters
            T = sinkhorn(cos_d.detach(), epsilon=0.05, sinkhorn_iterations=100)
            s = T / T.sum(dim=1, keepdim=True)
            k, index = torch.max(s, dim=1)
            
            with torch.no_grad():
                wd, e = w_dist(cos_d.cuda(), T.cuda(), center.shape[0])
                s_list.append(s)
                wd_list.append(wd)
                e_list.append(e)
            k_list.append(k.cuda())
            d_list.append(cos_d)
            

        loss_list = []
        rec_loss_list = []
        sem_loss_list = []
        wcl_loss_list = []
        if batch_idx == 0:
            a1 = xrs[0]
            a2 = xrs[1]
        if batch_idx > 0:
            a1 = torch.cat((a1,xrs[0]),dim=0)
            a2 = torch.cat((a2,xrs[1]),dim=0)
        for v in range(view):
            rec_loss_list.append(mes(xs[v], xrs[v]))
            sem_loss_list.append(alpha * F.kl_div(qs[v], s_list[v]))
            sem_loss_list.append(alpha * (-wd_list[v] - e_list[v]))
            for w in range(v+1, view):
                sem_loss_list.append(alpha * criterion.forward_label(qs[v], qs[w]))
                wcl_loss_list.append(beta * criterion.forward_feature(hs[v], hs[w], k_list[v], k_list[w]))

 
        r_loss = sum(rec_loss_list)
        s_loss = sum(sem_loss_list)
        w_loss = sum(wcl_loss_list)
        loss = r_loss+ s_loss + w_loss 
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        rec_loss += r_loss
        sem_loss += s_loss
        wcl_loss += w_loss

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    return tot_loss/len(data_loader), rec_loss/len(data_loader), sem_loss/len(data_loader), wcl_loss/len(data_loader)


def w_dist(cos_dist, T, m, eps=1):
    temp_1 = torch.mm(cos_dist.t(), T)
    temp_2 = eps * torch.mm(T.t(), torch.log(T))
    a = torch.eye(m).cuda()
    b = a * temp_1
    c = a * temp_2
    distance = torch.sum(b)
    entropy = torch.sum(c)
    return distance, entropy


def prototype(model, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _ = model.forward(xs)
        hs_cat = torch.stack(hs, dim=0)
        hs_fusion = torch.sum(hs_cat, dim=0) / view
        kmeans = KMeans(n_clusters=class_num, n_init=100).fit(hs_fusion.cpu().detach().numpy())

    return kmeans.cluster_centers_

def consine_similarity(Z, center):
    similarity = torch.mm(Z.to('cpu').detach(),(torch.from_numpy(center).T))
    return similarity



def sinkhorn(out, epsilon, sinkhorn_iterations):
    """
    from https://github.com/facebookresearch/swav
    """
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    # Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()



accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    best_acc = 0
    best_nmi = 0
    best_ari = 0
    best_pur = 0
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    
    while epoch <= args.mse_epochs + args.full_epochs:
        pro = prototype(model, device)

        loss_tot, loss_rec, loss_sem, loss_wcl = full_train(epoch, pro, args.alpha, args.beta)
        if epoch == args.mse_epochs + args.full_epochs:
            acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num)
            
            accs.append(acc)
            nmis.append(nmi)
            purs.append(pur)
            
        epoch += 1 
    
    state = model.state_dict()
    torch.save(state, './models/' + args.dataset + '.pth')
