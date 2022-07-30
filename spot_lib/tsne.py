import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import cm
import torch
import torch.nn.functional as F
import seaborn as sns


def viusalize(features1,gt,count,modes):
    cmap = cm.get_cmap('tab20')
    # labels = np.zeros
    # print(features1.size())
    # print(features2.size())
    # print(gt.size()[0])
    len_gt = gt.size()
    temp_feat = torch.zeros_like(features1)
    temp_feat_bg = torch.zeros_like(features1)
    temp_feat[:,:,gt[0] == 0] = features1[:,:,gt[0] == 0]
    temp_feat_bg[:,:,gt[0] == 1] = features1[:,:,gt[0] == 1]
    fg_feat = temp_feat.detach().cpu().numpy()
    bg_feat = temp_feat_bg.detach().cpu().numpy()
    # print(fg_feat.shape)
    # print(bg_feat.shape)
    features1 = features1.detach().cpu().numpy()
    # features2 = features2.detach().cpu().numpy()

    gt = gt.detach().cpu().numpy()
    gt_bkg = 1-gt
    labels=np.concatenate((gt,gt_bkg))
    # print(features2)
    features = np.concatenate((fg_feat[0],bg_feat[0]))
    tsne = TSNE(n_components=2).fit_transform(features)
    # print(tsne)
    # extract x and y coordinates representing the positions of the images on T-SNE plot

    tx = tsne[:, 0]
    ty = tsne[:, 1]
    print(tx.shape)
    new_gt = np.ones(len_gt[0]+len_gt[0]).astype(int)                                                                                                         
    new_gt[0:len_gt[0]] = 0
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # print(tx)

    # new plotting

    # plt.figure(figsize=(16,10))
    # sns.scatterplot(
    # x="pca-one", y="pca-two",
    # hue="y",
    # palette=sns.color_palette("hls", 10),
    # data=df.loc[rndperm,:],
    # legend="full",
    # alpha=0.3
    # )







    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
          
    label_colors = ['b','g','r','c','m']
    colors = [label_colors[i+1] for i in new_gt]

    for lab in range(2):
        # print(lab)
        indices = (1-gt) == lab
        # print(tx)
        current_tx = np.take(tx,indices)
        # print(current_tx)
        current_ty = np.take(ty,indices)

    # ax.scatter(tx,ty, c=np.array(cmap(gt)).reshape(100,4),alpha=0.5)
    # print(np.shape(np.array(cmap(new_gt[0]))))
    
    # ax.scatter(tsne[0, 0],tsne[0, 1], np.array(cmap(new_gt[0])).reshape(1,4) ,alpha=0.5)
    ax.scatter(tx[0:400],ty[0:400], c = colors[0] ,alpha=0.5, marker = 'o', label = "foreground")
    ax.scatter(tx[401:],ty[401:], c = colors[1] ,alpha=0.5, marker = 'o', label = "background")
    
    # indices = 
    # current_tx = np.take(tx,indices)
    # ax.scatter(tx,ty, c=np.array(cmap(gt)), cmap=plt.cm.get_cmap("jet", 10) ,alpha=0.5)
    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    # plt.show()
    if modes=="before":
        plt.savefig('/home/phd/Desktop/sauradip_research/TAL/Semi-TAL/SSTAL-FixMatch/GSM/tsne/tSNE_before_'+str(count)+'.png')
    else:
        plt.savefig('/home/phd/Desktop/sauradip_research/TAL/Semi-TAL/SSTAL-FixMatch/GSM/tsne/tSNE_after_'+str(count)+'.png')

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range



def feat_sim(x,video_name,mode="before"):
    sns.set_theme()
    norm_x = F.normalize(x,dim=1)
    x_trans = torch.transpose(norm_x,1,2)

    sim = torch.matmul(x_trans,norm_x)
    sim_np = sim.detach().cpu().numpy()[0]
    plt.figure()
    # ax = plt.subplot()
    hm = sns.heatmap(sim_np, vmin=0,vmax=1, xticklabels=10, yticklabels=10)
    hm.set( xlabel = "Temporal Location", ylabel = "Temporal Location")
    # plt.imshow(sim_np, cmap='hot', interpolation='nearest')
    # plt.colorbar(im)
    # plt.yaxis.set_ticks_position('none') 
    # plt.show()
    if mode =="before":
        hm.figure.savefig('/media/phd/SAURADIP5TB/GSM-ABLATION/heatmap/feat_before/'+video_name+'_heatmap_before.png')
    else:
        hm.figure.savefig('/media/phd/SAURADIP5TB/GSM-ABLATION/heatmap/feat_after/'+video_name+'_heatmap_after.png')

def hmap(x,vid_name,mode="bottom"):
    sns.set_theme()
    # norm_x = F.normalize(x,dim=1)
    if mode == "bottom":
        norm_x = x
        norm_x1 = norm_x.detach().cpu().numpy()[0]
        # print(norm_x.shape)
        # ax = plt.subplot()
        plt.figure()
        hm = sns.heatmap(norm_x1, vmin=0,vmax=1, xticklabels=10, yticklabels=10)
        hm.set( xlabel = "Temporal Location", ylabel = "Global Mask Length")
        # plt.imshow(norm_x, cmap='hot', interpolation='nearest')
        # plt.colorbar(im)
        # plt.yaxis.set_ticks_position('none') 
        # plt.show()
        
        hm.figure.savefig("/media/phd/SAURADIP5TB/GSM-ABLATION/heatmap/bottom/"+vid_name+'_heatmap_bottom.png')
    elif mode =="top":
        # print(x[0].size())
        norm_x = torch.softmax(x[0],dim=0)
        norm_x1 = norm_x.detach().cpu().numpy()
        plt.figure()
        hm = sns.heatmap(norm_x1, vmin=0,vmax=1, xticklabels=10, yticklabels=20)
        hm.set( ylabel = "Class", xlabel = "Temporal Location")
        hm.figure.savefig("/media/phd/SAURADIP5TB/GSM-ABLATION/heatmap/top/"+vid_name+'_heatmap_top.png')

    elif mode == "gt":

        norm_x = x[0]
        norm_x1 = norm_x.detach().cpu().numpy()
        plt.figure()
        hm = sns.heatmap(norm_x1, vmin=0,vmax=1, xticklabels=10, yticklabels=20)
        hm.set( ylabel = "Class", xlabel = "Temporal Location")
        hm.figure.savefig("/media/phd/SAURADIP5TB/GSM-ABLATION/heatmap/gt/"+vid_name+'_heatmap_gt.png')



def error_viz(x):

    x = np.linspace(0, 10, 50)
    dy = 0.8
    y = np.sin(x) + dy * np.random.randn(50)
    plt.errorbar(x, y, yerr=dy, fmt='.k')


def grouped_bar():
    # create data
    x = np.arange(5)
    y1 = [5.95, 24.25, 51.60, 63.86, 31.38]
    y2 = [14.52, 28.08, 51.60, 63.86, 34.36]
    y3 = [13.21, 29, 54.89, 58.31, 36.49]
    width = 0.4
    
    # plot data in grouped manner of bar type
    plt.bar(x-0.4, y1, width, color='cyan')
    plt.bar(x, y2, width, color='orange')
    plt.bar(x+0.4, y3, width, color='green')
    plt.xticks(x, ['Extra Small', 'Small', 'Medium', 'Long', 'Extra Long'])
    plt.xlabel("Snippet Length")
    plt.ylabel("Localization Error Impact %")
    plt.legend(["R-C3D (Anchor Based)", "BMN (Anchor-Free)", "Ours (Proposal-Free)"])
    plt.show()




