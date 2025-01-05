import pickle
import json
import torch
import mmcv
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import imageio.v2 as imageio
from nuscenes.nuscenes import NuScenes
from tools.analysis_tools.visual import *
import shutil
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from projects.mmdet3d_plugin.bevformer.modules import guide_pts_gen as gpg
from mmcv.utils import Registry
from mmcv import Config

def pkl_read():
    train_path = 'data/nus_extend/nuscenes_infos_temporal_train.pkl'
    val_path = 'data/nus_extend/nuscenes_infos_temporal_val.pkl'
    test_path = 'data/nus_extend/nuscenes_infos_temporal_test.pkl'
    val_path_mini = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'

    # with open(train_path,'rb') as f1:
    #     a1 = pickle.load(f1)
    # with open(val_path,'rb') as f2:
    #     a2 = pickle.load(f2)
    # with open(test_path,'rb') as f3:
    #     a3 = pickle.load(f3)
    t1 = time.time()
    f1 = open(train_path, 'rb')
    a1 = pickle.load(f1)  # 28130 (/2=14065)
    t2 = time.time()
    f2 = open(val_path, 'rb')
    a2 = pickle.load(f2)  # 6019
    t3 = time.time()
    print('Now is orignal nuscenes_infos load time consume:')
    print(f'nuscenes train infos load:{t2-t1}s')
    print(f'nuscenes val infos load:{t3-t2}s')
    
    
    f3 = open(test_path, 'rb')
    a3 = pickle.load(f3)  # 6008         
    
    print('Read finish')

def pth_read():
    pretrain_pth_path = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
    posttrain_pth_path = 'work_dirs/bevformer_small_fusion/epoch_6.pth'
    
    m1 = torch.load(pretrain_pth_path, map_location=torch.device('cpu'))
    m2 = torch.load(posttrain_pth_path, map_location=torch.device('cpu'))
    
    print('load finish')

def json_read():
    path_1 = 'test/bevformer_small_fusion/Sat_Nov_16_08_48_33_2024/pts_bbox/results_nusc.json'
    path_2 = 'test/bevformer_small_fusion/Sat_Nov_16_08_48_33_2024/pts_bbox/metrics_details.json'
    f = open(path_2,'r')
    str = f.read()
    result = json.loads(str)
    print('load finish')


def vis_gif_gen():
    # path_sample = 'data/nus_extend/v1.0-trainval/sample.json'
    path_result = 'result_vis/rend_orig_png_500'
    # f = open(path_sample,'r')
    # sample = json.load(f)

    # 将乱序的rend_png根据sample.json排好序
    # result = os.listdir(path_result)  
    # result.sort(key=lambda x: x[-5:-4]) # 区分bev和cam
    # l = len(result)
    # result = result[:int(l/2)]
    # r_index = []
    # print('sorting token:')
    # for x in tqdm(result):
    #     for ind, y in enumerate(sample):
    #         tmp = {}
    #         if x[:-11] == y['token']:
    #             tmp['idx'] = ind
    #             tmp['token'] = y['token']
    #             r_index.append(tmp)
    # r_index_2 = sorted(r_index, key=lambda r_index:r_index['idx'])      
    
    # b = json.dumps(r_index_2)
    # f = open(f'result_vis/rend_sample_divide/sample_all.json', 'w')
    # f.write(b)
    # f.close()
    
    
    # 划分sample
    f = open(f'result_vis/rend_sample_divide/sample_all.json', 'r')
    r_index_2 = json.load(f)
    type = 'bev'
    # type = 'camera'
    # r_index_2 = r_index_2[460:500]
    
    # ind =  [[0,25], [25,64], [64,103], [103,143], [143,183], [183,261], [261,300], [300,380], [380,420], [420,460], [460,500]]
    # 
    # samp_div_0 = []
    # for a in ind:
    #     tmp = {}
    #     tmp['start_token']=r_index_2[a[0]]['token']
    #     tmp['samp_len']=a[1]-a[0]
    #     tmp['idx'] = r_index_2[a[0]]['idx']
    #     samp_div_0.append(tmp)
    # samp_div = sorted(samp_div_0, key=lambda samp_div_0:samp_div_0['idx'])
    # b = json.dumps(samp_div)
    # f = open(f'result_vis/rend_sample_divide/sample_div.json', 'w')
    # f.write(b)
    # f.close()

    # gif生成
    f2 = open(f'result_vis/rend_sample_divide/sample_div_cons-veh_val-set.json', 'r')
    r_div = json.load(f2)
    print('gif gen:')
    for i, a in enumerate(r_div):
        print(f"progress:{i+1}/{len(r_div)}")
        gif_images = []
        start = 0
        # t1=time.time()
        for j, x in enumerate(r_index_2):
            if a['start_token'] == x['token']:
                start=j
                break
            else:
                assert False, f"Couldn't find token-{a['start_token']} in sample_all"    
        tmp = r_index_2[start:start+a['samp_len']]
        # t2=time.time()
        for x in tmp:
            token = x['token']
            path = path_result+'/'+ token + '_' + type +'.png'
            gif_images.append(imageio.imread(path))
        imageio.mimsave(f"result_vis/rend_gif_constr-veh/sample{i+1}_{type}.gif", gif_images, fps=2)
        # t3=time.time()
        # print(f"start token search:{t2-t1}s    gif gen&save:{t3-t2}s\n")
    # 保存划分的sample
    # b = json.dumps(r_index_2)
    # x = 1
    # f = open(f'result_vis/rend_sample_divide/sample_{x}.json', 'w')
    # f.write(b)
    # f.close()

    print('finish')
                
    
def PR_anay():
    # 载入json文件
    # [0] path_details = 'test/bevformer_small_fusion/173_result/metrics_details_e6d6_epo8.json' 
    # [1] path_summary = 'test/bevformer_small_fusion/173_result/metrics_summary_e6d6_epo8.json'
    # [2] path_result = 'test/bevformer_small_fusion/173_result/results_nusc__e6d6_epo8.json'
    r_path = 'test/bevformer_small_fusion/173_result'
    j_path = ['metrics_details_e6d6_epo8.json',
              'metrics_summary_e6d6_epo8.json',
              'results_nusc__e6d6_epo8.json']
    file = 0
    f = open(os.path.join(r_path, j_path[file]), 'r')
    result = json.load(f)

        # 挑选某一cls&thr的P-R-C，返回三个list
    cls = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
    thr = ['0.5', '1.0', '2.0', '4.0']
    i = 0

    mp_cls = {}
    mr_cls = {}
    for i in range(len(cls)):
        p = {}
        r = {}
        c = {}
        mp = [0 for x in range(101)]
        mr = [0 for x in range(101)]
        for j in range(4):
            k = cls[i] + ':' + thr[j]
            p[j] = result[k]['precision']
            r[j] = result[k]['recall']
            c[j] = result[k]['confidence']
            mp = [a+b for a,b in zip(mp, p[j])]
            mr = [a+b for a,b in zip(mr, r[j])]

        # 画出曲线图，左边是4个thr，右边是mean并标注（cls&thr的标题、坐标轴、每间隔一定距离的置信水平）
        # 一个类别一张(2,1)plot
        # mp = [x/4 for x in mp]
        # mr = [x/4 for x in mr]  
        # fig = plt.figure(figsize=(6,4))
        # ax1 = fig.add_subplot(111)
        # # ax2 = fig.add_subplot(122)
        # cmp = ['orangered', 'green', 'fuchsia', 'darkorange']
        # ls= ['dashed', 'solid']
        
        # ax1.set_title(f'cls={cls[i]} 4-AP & mAP', fontsize=14)
        # for j in range(4):    
        #     ax1.plot(r[j], p[j], color=cmp[j], linestyle=ls[0], label='AP@'+thr[j])
        # ax1.plot(mr, mp, color='navy', linestyle=ls[1], label='mAP')

        # plt.legend(loc='best')
        # plt.savefig(os.path.join(r_path, 'PR_Cruve', cls[i]+'.jpg'))
        # plt.show() # 显示图片(别放在savefig前面，不然显示完会创建一个新的空白图片保存起来)
        
    # 把所有类别的P-R曲线画一起
        # 这个得套在第一层循环内
        mp_cls[cls[i]] = [x/4 for x in mp]
        mr_cls[cls[i]] = [x/4 for x in mr]
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.set_title('all_cls_combin P-R', fontsize=14)
    cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i, c in enumerate(cls):
        ax.plot(mr_cls[c], mp_cls[c], color=cmap[i], label=c)
    plt.legend(loc='best')
    plt.savefig(os.path.join(r_path, 'PR_Cruve', 'all_cls_combin.jpg'))
    

def cls_distr_ana():
    # nusc = NuScenes(version='v1.0-trainval', dataroot='data/nus_extend', verbose=True)
    divide = 'train'   # 28130个sample
    # divide = 'val'       # 6019个sample
    # path = 'data/nus_extend/nuscenes_infos_temporal_'+divide+'.pkl'  
         
    # f2 = open(path, 'rb')
    # a2 = pickle.load(f2)  
    
    # cls_num = {"car":0, 
    #            "truck":0, 
    #            "bus":0, 
    #            "trailer":0, 
    #            "vehicle.construction":0, 
    #            "pedestrian":0, 
    #            "motorcycle":0,
    #            "bicycle":0, 
    #            "trafficcone":0, 
    #            "barrier":0}
    # cons_samp = []
    # trail_samp = []
    # print('cate_ana:')
    # for i,x in enumerate(tqdm(a2['infos'])):
    #     tmp1 = x['token']
    #     sample = nusc.get('sample', tmp1)
    #     for y in sample['anns']:
    #         ann = nusc.get('sample_annotation', y)
    #         cate = ann['category_name']
    #         for key in cls_num.keys():
    #             ret = cate.find(key)
    #             if ret != -1:
    #                 cls_num[key] += 1 
    #                 if key == 'vehicle.construction':
    #                     tmp = {}
    #                     tmp['idx'] = i
    #                     tmp['samp_tk'] = sample['token']
    #                     if tmp not in cons_samp:
    #                         cons_samp.append(tmp)
    #                 if key == 'trailer':
    #                     tmp = {}
    #                     tmp['idx'] = i
    #                     tmp['samp_tk'] = sample['token']
    #                     if tmp not in trail_samp:
    #                         trail_samp.append(tmp)
    # print('cate_ana finish')
        
    save_path = 'test/bevformer_small_fusion/173_result/cons&trail_anay'
        
    # print('dumping')
    # a = json.dumps(cls_num)
    # b = json.dumps(cons_samp)
    # c = json.dumps(trail_samp)
    
    # print('saving')
    # a_name = 'cls_num_anaylsis_'+divide
    # b_name = 'construction_vehicle_sample_'+divide
    # c_name = 'trailer_sample_'+divide
    # f = open(os.path.join(save_path, a_name + '.json'), 'w')
    # f.write(a)
    # f.close()
    # f = open(os.path.join(save_path, b_name + '.json'), 'w')
    # f.write(b)
    # f.close()
    # f = open(os.path.join(save_path, c_name + '.json'), 'w')
    # f.write(c)
    # f.close()
    # print('save finish')
    
    f = open(os.path.join(save_path, a_name + '.json'), 'br')
    a = f.read()
    cls_num = json.loads(a)
    
    print('drawing:')
    fig = plt.figure(figsize=(12,8))
    fig.suptitle(f'all cls distribution in {divide}-set', fontsize=20)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    x = [i for i in range(10)]
    y = list(cls_num.values())
    ax1.bar(x,y)
    ax1.set_xlim(-0.5, 10)
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [r'$car$', r'$truck$', r'$bus$', r'$trailer$', r'$cons\_veh$', r'$pedst$', r'$mot\_cycle$', r'$bicycle$', r'$tfc\_cone$', r'$barrier$'],
                   rotation=25)
    ax1.tick_params(labelsize=13)
    cls = ['car', 'truck', 'bus', 'trailer', 'cons_veh', 'pedst', 'mot_cycle', 'bicycle', 'tfc_cone', 'barrier']
    ax2.pie(y, autopct='%1.1f%%', textprops={'fontsize':13})
    ax2.legend(ncol=5, labels=cls, loc=8, fontsize=14)
        # ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    print('drawing finish!')
    plt.savefig(os.path.join(save_path,f'all_cls_distb_in_{divide}.png'))
    plt.show()


def samp_div():
    # version = 'train'
    version = 'val'
    path = 'test/bevformer_small_fusion/173_result/cons&trail_anay/construction_vehicle_sample_'+version+'.json'
    f = open(path,'r')
    a = f.read()
    cons_samp = json.loads(a)
    
    cons_samp_div = []
    start_idx = cons_samp[0]['idx']
    start_tk = cons_samp[0]['samp_tk']
    i = 0
    while (i+1 < len(cons_samp)):
        # print(f"{i+1}/{len(cons_samp)}")
        if cons_samp[i+1]['idx'] - cons_samp[i]['idx'] > 1:
            tmp = {}
            tmp['start_token'] = start_tk
            tmp['samp_len'] = cons_samp[i]['idx'] - start_idx + 1
            tmp['idx'] = start_idx
            cons_samp_div.append(tmp)
            # 归零
            start_idx = cons_samp[i+1]['idx']
            start_tk = cons_samp[i+1]['samp_tk']
        i+=1
    b = json.dumps(cons_samp_div)
    f = open(f'result_vis/rend_sample_divide/sample_div_cons-veh_{version}-set.json', 'w')
    f.write(b)
    f.close()
    print("save file")

def rend_png():
    bevformer_results = mmcv.load('test/bevformer_small_fusion/173_result/results_nusc__e6d6_epo8.json')
    path_sample = 'result_vis/rend_sample_divide/sample_trainval.json'
    f = open(path_sample,'r')
    trainval_sample = json.load(f)
    
    # version = 'train'
    version = 'val'
    path = f'result_vis/rend_sample_divide/sample_div_cons-veh_{version}-set.json'
    f = open(path,'r')
    a = f.read()
    cons_samp = json.loads(a)
        
    for c, i in enumerate(cons_samp[:10]):
        print(f"png rend progress: {c+1}/10")
        st_tk=i['start_token']
        l = i['samp_len']
        idx = trainval_sample.index(st_tk)
        tk_list = trainval_sample[idx:idx+l]
        for x in tqdm(tk_list):
            render_sample_data(x, pred_data=bevformer_results, out_path=x)

def origin_data_obt():
    
    # 从cons_veh中挑选了scenes10的 70~75 sample，但是只有sample token，没有原始数据的路径&文件名
    # 需要根据sample token，在trainval_sample.json中，获取所有相关sample的原始数据路径
    # 这一步是根据sample挑出全部的数据（包括sample和sweep），并保存在data-path_cons-veh_6.json中
    path_sample = 'data/nus_extend/v1.0-trainval/sample_data.json'
    f = open(path_sample,'r')
    sample = json.load(f)
    data_path = []
    token_list = ['9ee4020153674b9e9943d395ff8cfdf3']
    for c,i in enumerate(token_list):
        print(f'sample process: {c+1}/{len(token_list)}')
        for j in tqdm(sample):
            if j['sample_token']==i:
                tmp={}
                tmp['sample_num']=c+1
                tmp['sample_token']=i
                tmp['data_format']=j['fileformat']
                tmp['file_name']=j['filename']    
                data_path.append(tmp)            
    # f2 = open('test/bevformer_small_fusion/173_result/cons&trail_anay/data-path_cons-veh_6.json','w')
    # b = json.dumps(data_path)
    # f2.write(b)
    # f2.close()
    
    
    # 根据sample挑出来的数据太多了（包含sample和sweep），需要将sample相关数据挑选出来
    # 同时，将不同模态的数据合并到一个sample中，并将结果保存在sample-only.json中
    # f2 = open('test/bevformer_small_fusion/173_result/cons&trail_anay/data-path_cons-veh_6.json','r')
    # a = f2.read()
    # data_path = json.loads(a)
    data_list = []
    for i in data_path:
        if ('samples' in i['file_name']) and ('CAM' in i['file_name'] or 'LIDAR' in i['file_name']):
            data_list.append(i)
    data_list_2 = []
    for j in range(len(token_list)):
        samp_div = data_list[j*7:(j+1)*7]
        tmp = {}
        for i in samp_div:
            tmp['num'] = i['sample_num']
            tmp['sample_tk'] = i['sample_token']
            if 'LIDAR' in i['file_name']:
                tmp['LIDAR'] = i['file_name']
            if 'CAM_FRONT/' in i['file_name']:
                tmp['CAM_FC'] = i['file_name']
            if 'CAM_FRONT_LEFT' in i['file_name']:
                tmp['CAM_FL'] = i['file_name']                
            if 'CAM_FRONT_RIGHT' in i['file_name']:
                tmp['CAM_FR'] = i['file_name']
            if 'CAM_BACK/' in i['file_name']:
                tmp['CAM_RC'] = i['file_name']
            if 'CAM_BACK_LEFT' in i['file_name']:
                tmp['CAM_RL'] = i['file_name']
            if 'CAM_BACK_RIGHT' in i['file_name']:
                tmp['CAM_RR'] = i['file_name']               
        data_list_2.append(tmp)
    # f2 = open('test/bevformer_small_fusion/173_result/cons&trail_anay/data-path_cons-veh_6_sample-only.json','w')
    # b = json.dumps(data_list_2)
    # f2.write(b)
    # f2.close()
    
    # 这里将相关文件从extend_nus中拷贝到当前项目目录中        
    orig_data_root = 'data/nus_extend'
    dest_data_root = 'cons_veh_orig_data'        
    # f2 = open('test/bevformer_small_fusion/173_result/cons&trail_anay/data-path_cons-veh_6_sample-only.json','r')
    # a = f2.read()
    # combin_data = json.loads(a)
    combin_data = data_list_2
    sensor_list = ['LIDAR','CAM_FC','CAM_FL','CAM_FR','CAM_RC','CAM_RL','CAM_RR']
    
    for i, data in enumerate(combin_data):
        print(f"progress:{i+1}/{len(token_list)}")
        dest_path = os.path.join(dest_data_root,f'scenes-mini-sample{i}')
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)
        for s in sensor_list:
            s_file = os.path.join(orig_data_root, data[s])
            shutil.copy(s_file, os.path.join(dest_path, s[8:]))    
        

# 原始数据是.pcd.bin的，并不方便查看，因此需要转换成.pcd格式的
def bin_pcd():
    path = 'cons_veh_orig_data/scenes2-sample24/n015-2018-07-18-11-41-49+0800__LIDAR_TOP__1531885781898979.pcd.bin'
    pcd = np.fromfile(path, dtype=np.float32).reshape([-1,5])
    pcd = pcd[:,:3]
    
    print('finish')
            
def test():
    scene_num = 2
    sample_num = 15
    data_root='data/nus_extend'

    sample_path='data/nus_extend/v1.0-trainval/sample.json'
    sample_data_info='data/nus_extend/v1.0-trainval/sample_data.json'
    f1=open(sample_path,'r')
    f1 = f1.read()
    sample=json.loads(f1)
    f2=open(sample_data_info,'r')
    f2 = f2.read()
    sample_data=json.loads(f2)
    
    # 所有场景
    scene_list = []
    for x in sample: 
        if x['scene_token'] not in scene_list:
            scene_list.append(x['scene_token'])
    
    print('finish')
   

def mm_data_vis(data_root, data_set):
    
    root_path = data_root
    ego_pos_file = 'v1.0-trainval/ego_pose.json'
    ssr_cal_file = 'v1.0-trainval/calibrated_sensor.json'
    
    f1 = open(os.path.join(root_path, ego_pos_file))
    a1 = f1.read()
    t_ego_pos = json.loads(a1)
    f2 = open(os.path.join(root_path, ssr_cal_file))
    a2 = f2.read()
    t_sensor_calib = json.loads(a2)
               
    path_lidar = os.path.join(root_path,data_set['LIDAR']['filename'])
    path_radar_fc = os.path.join(root_path,data_set['Radar_FC']['filename'])
    path_radar_fl = os.path.join(root_path,data_set['Radar_FL']['filename'])
    path_radar_fr = os.path.join(root_path,data_set['Radar_FR']['filename'])
    path_radar_rl = os.path.join(root_path,data_set['Radar_RL']['filename'])
    path_radar_rr = os.path.join(root_path,data_set['Radar_RR']['filename'])
    
    scan = np.fromfile(path_lidar, dtype=np.float32).reshape((-1,5))
    r_fc = o3d.io.read_point_cloud(path_radar_fc)
    r_fl = o3d.io.read_point_cloud(path_radar_fl)
    r_fr = o3d.io.read_point_cloud(path_radar_fr)
    r_rl = o3d.io.read_point_cloud(path_radar_rl)
    r_rr = o3d.io.read_point_cloud(path_radar_rr)
    
    # 构建 Open3D 中提供的 gemoetry 点云类 
    pcd_lidar = o3d.geometry.PointCloud() 
    pcd_lidar.points = o3d.utility.Vector3dVector(scan[:, :3]) 
    
    # 读入反射强度
    points_intensity = scan[:, 3]  # intensity
    # 强度正则化
    max = np.max(points_intensity)
    min = np.min(points_intensity)
    norm_intensity = (points_intensity-min)/(max-min)
    norm_intensity = norm_intensity ** 0.1
    norm_intensity = np.maximum(0, norm_intensity-0.5)
    # 强度分布直方图  
    fig,ax = plt.subplots()
    ax.hist(norm_intensity, bins=100)
    ax.set_title('hist of intensity')
    plt.show()
    
    # LiDAR & Radar 可视化
    # 自定义一个 colormap
    # 方案1:
    # lidar_colors = [np.array([54,79,107])/255,
    #                 np.array([63,193,201])/255,
    #                 np.array([245,245,245])/255,
    #                 np.array([252,81,133])/255,
    #                 np.array([193,81,252])/255]
    # 方案2:
    lidar_colors = [np.array([21,82,99])/255,
                    np.array([17,157,236])/255,
                    np.array([8,142,0])/255,
                    np.array([244,209,5])/255,
                    np.array([252,87,17])/255]
    
    points_colors = [lidar_colors[int(norm_intensity[i]*(len(lidar_colors)-1))] for i in range(norm_intensity.shape[0])] 
    pcd_lidar.colors = o3d.utility.Vector3dVector(points_colors)  # 根据 intensity 为点云着色
    
    # radar 颜色区分
    # 对比色
    radar_colors=[np.array([255,0,0])/255,
                  np.array([0,255,0])/255,
                  np.array([0,255,255])/255,
                  np.array([255,255,0])/255,
                  np.array([255,0,255])/255,
                  np.array([0,0,0])/255]  # 充数用的
    # 方案1: 统一为黄色
    # radar_colors=[np.array([255,134,53])/255 for x in range(5)]
    # 方案2: 统一为粉色
    # radar_colors=[np.array([255,168,252])/255 for x in range(5)]
            
    r_fc = o3d.io.read_point_cloud(path_radar_fc)
    r_fl = o3d.io.read_point_cloud(path_radar_fl)
    r_fr = o3d.io.read_point_cloud(path_radar_fr)
    r_rl = o3d.io.read_point_cloud(path_radar_rl)
    r_rr = o3d.io.read_point_cloud(path_radar_rr)
    radar_list = [r_fc, r_fl, r_fr, r_rl, r_rr]
    pts_list = [r_fc, r_fl, r_fr, r_rl, r_rr, pcd_lidar]
    sensor_list = ['Radar_FC', 'Radar_FL', 'Radar_FR', 'Radar_RL', 'Radar_RR', 'LIDAR',
                   'CAM_FC', 'CAM_FL', 'CAM_FR', 'CAM_RC', 'CAM_RL', 'CAM_RR']
    
    # 所有pts(lidar&radar)都转到ego vehicle下，并着色
    # for c, i in enumerate(pts_list):
    #     orig_pts = i
    #     source = c
    #     source_calib_token=data_set[sensor_list[source]]['calibrated_sensor_token']
    #     source_ego_token=data_set[sensor_list[source]]['ego_pose_token']
    #     source_calib = {}
    #     for i in t_sensor_calib:
    #         if i['token']==source_calib_token:
    #             source_calib['translation']=i['translation']
    #             source_calib['rotation']=i['rotation']
    #     source_ego = {}
    #     for j in t_ego_pos:
    #         if j['token']==source_ego_token:
    #             source_ego['translation']=j['translation']
    #             source_ego['rotation']=j['rotation']
    #     pts_list[c] = sensor_trans_to_ego(orig_pts, source_calib, source_ego)
    #     pts_list[c].colors=o3d.utility.Vector3dVector([np.array([255,168,252])/255 for x in range(len(pts_list[c].points))])
    # pts_list[5].colors = o3d.utility.Vector3dVector(points_colors)
    
    # 所有pts(lidar&radar)转到ego vehicle下
    target=6
    for c, i in enumerate(pts_list):
        orig_pts = i
        source = c
        source_calib_token=data_set[sensor_list[source]]['calibrated_sensor_token']
        source_ego_token=data_set[sensor_list[source]]['ego_pose_token']
        target_calib_token=data_set[sensor_list[target]]['calibrated_sensor_token']
        target_ego_token=data_set[sensor_list[target]]['ego_pose_token'] 
        source_calib = {}
        target_calib = {}
        for i in t_sensor_calib:
            if i['token']==source_calib_token:
                source_calib['translation']=i['translation']
                source_calib['rotation']=i['rotation']
            if i['token']==target_calib_token:
                target_calib['translation']=i['translation']
                target_calib['rotation']=i['rotation']
        source_ego = {}
        target_ego = {}
        for j in t_ego_pos:
            if j['token']==source_ego_token:
                source_ego['translation']=j['translation']
                source_ego['rotation']=j['rotation']
            if j['token']==target_ego_token:
                target_ego['translation']=j['translation']
                target_ego['rotation']=j['rotation']
        pts_list[c] = sensor_trans(orig_pts, source_calib, source_ego, target_calib, target_ego)
        pts_list[c].colors=o3d.utility.Vector3dVector([radar_colors[c]  for x in range(len(pts_list[c].points))])
    pts_list[5].colors = o3d.utility.Vector3dVector(points_colors)  
      
    # o3d.io.write_point_cloud('05_new_struct_dev/multi-md_anay/test.pcd', pcd) , [r_fc], [r_fl], [r_fr], [r_rl], [r_rr]
    # o3d.visualization.draw_geometries([pts[5]])
    
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 创建窗口,设置窗口名称
    vis.create_window(window_name="point_cloud")
    # 设置点云渲染参数
    opt = vis.get_render_option()
    # 设置背景色（这里为黑色）
    opt.background_color = np.array([50, 50, 50])/255
    # 设置渲染点的大小
    opt.point_size = 4.0
    # 添加点云
    vis.add_geometry(pts_list[0])
    vis.add_geometry(pts_list[1])
    vis.add_geometry(pts_list[2])
    vis.add_geometry(pts_list[3])
    vis.add_geometry(pts_list[4])
    vis.add_geometry(pts_list[5])
    
    # vis.add_geometry(pcd_lidar)
    # vis.add_geometry(radar_list[0])
    # vis.add_geometry(radar_list[1])
    # vis.add_geometry(radar_list[2])
    # vis.add_geometry(radar_list[3])
    # vis.add_geometry(radar_list[4])
    vis.run()
     

def q_transpose(q):
    return [q[3],q[0],q[1],q[2]]
    # tmp_q = np.transpose(tmp_q[3,0,1,2])
    #  tmp_q.to_list() 

def sensor_trans_to_ego(o3d_pts, calib, ego):
    
    r0 = R.from_quat(q_transpose(calib['rotation']))
    calib_rotation=r0.as_matrix()
    calib_trans=calib['translation']
    r1 = R.from_quat(q_transpose(ego['rotation']))
    source_ego_rotation=r1.as_matrix()
    source_ego_trans=ego['translation']

    pts = np.array(o3d_pts.points) 
    pts = (np.dot(calib_rotation, pts.T)).T
    pts = pts+calib_trans
    pts = (np.dot(source_ego_rotation, pts.T)).T
    pts = pts+source_ego_trans
    o3d_pts.points = o3d.utility.Vector3dVector(pts)
    
    return o3d_pts

def sensor_trans(o3d_pts,source_calib,source_ego,target_calib,target_ego):
    
    r0 = R.from_quat(q_transpose(source_calib['rotation']))
    source_calib_rotation=r0.as_matrix()
    source_calib_trans=source_calib['translation']
    r1 = R.from_quat(q_transpose(source_ego['rotation']))
    source_ego_rotation=r1.as_matrix()
    source_ego_trans=source_ego['translation']
    r2 = R.from_quat(q_transpose(target_ego['rotation']))
    target_ego_rotation=r2.as_matrix()
    target_ego_trans=target_ego['translation']
    r3 = R.from_quat(q_transpose(target_calib['rotation']))
    target_calib_rotation=r3 .as_matrix()
    target_calib_trans=target_calib['translation']

    pts = np.array(o3d_pts.points) 
    pts = (np.dot(source_calib_rotation, pts.T)).T
    pts = pts+source_calib_trans
    pts = (np.dot(source_ego_rotation, pts.T)).T
    pts = pts+source_ego_trans
    pts = pts-target_ego_trans
    pts = (np.dot(target_ego_rotation.T, pts.T)).T
    pts = pts-target_calib_trans
    pts = (np.dot(target_calib_rotation.T, pts.T)).T
    o3d_pts.points = o3d.utility.Vector3dVector(pts)
    
    return o3d_pts


def multi_modality_anay():
    
    data_root='data/nus_extend'
    sample_version='v1.0-trainval/sample.json'
    sample_data_version='v1.0-trainval/sample_data.json'
    
    sample_path=os.path.join(data_root, sample_version)
    sample_data_info=os.path.join(data_root, sample_data_version)
    f1=open(sample_path,'r')
    f1 = f1.read()
    sample=json.loads(f1)
    f2=open(sample_data_info,'r')
    f2 = f2.read()
    sample_data=json.loads(f2)
    
    # 所有场景
    scene_list = []
    for x in sample: 
        if x['scene_token'] not in scene_list:
            scene_list.append(x['scene_token'])
    print(f"***************************************************************")
    print(f" Now we have {len(scene_list)} scenes overall.")
    print(f" Select the start number you want to 9 grid visual from {1} to {len(scene_list)-8}")
    print(f"***************************************************************")
    scene_select = int(input('Start scene num:'))
    # scene_select = 1
    scene_select = scene_select-1
    assert type(scene_select)==int, "Error: only int input is accepeted."
    # scenes可视化
    # 筛选9个scenes的首帧sample
    tmp_sce=[]
    tmp_samp=[]
    for i in range(9):
        scene_token = scene_list[scene_select+i]
        for j in sample:
            if scene_token not in tmp_sce and j['scene_token']==scene_token:
                tmp_samp.append(j['token'])  
                tmp_sce.append(scene_token)
    tmp_jpg_path=[]
    for i in tmp_samp:
        for j in sample_data:
            if i == j['sample_token'] and j['fileformat']=='jpg' and j['filename'][:18]=='samples/CAM_FRONT/':     
                tmp_jpg_path.append(j['filename'])
    
    fig,axs=plt.subplots(3,3)
    fig.suptitle(f"scene{scene_select+1} to scene{scene_select+9}")
    for c,i in enumerate(tmp_jpg_path):
        file = os.path.join(data_root, i)
        assert os.path.exists(file), f"File path wrong: {file}"    
        img = plt.imread(file)
        axs[int(c/3),c%3].imshow(img)
        axs[int(c/3),c%3].set_title(f"scene{scene_select+c+1}")
    plt.show()

    # 某一场景中的全部sample
    scene_sec = int(input("Input the scene you want to chooise:"))
    # scene_sec = 3
    scene_sec = scene_sec-1
    scene_token = scene_list[scene_sec]        
    sample_list = []
    for x in sample:
        if scene_token == x['scene_token']:
            sample_list.append(x['token'])
    jpg_ind = [0, int(len(sample_list)/3), int(len(sample_list)*2/3), len(sample_list)-1]
    tmp_samp_file_list=[]
    for i in jpg_ind:
        samp = sample_list[i]
        for j in sample_data:
            if j['sample_token']==samp and j['filename'][:18]=='samples/CAM_FRONT/':
                tmp_samp_file_list.append(j['filename'])
    
    fig,axs=plt.subplots(2,2)
    fig.suptitle(f"4 sample of scene{scene_sec+1} ({len(sample_list)} in total)")
    for c, i in enumerate(tmp_samp_file_list):
        file = os.path.join(data_root, i)
        assert os.path.exists(file), f"File path wrong: {file}"    
        img = plt.imread(file)
        axs[int(c/2),c%2].imshow(img)
        axs[int(c/2),c%2].set_title(f"sample{jpg_ind[c]}")
    plt.show()
    
    # 选一个sample可视化
    sample_num = int(input("Input the sample you want to chooise:"))
    # sample_num = 2
    # 选中sample中的全部模态的数据        
    # data_list = {}
    sample_token = sample_list[sample_num]
    data_set = {}
    for i in sample_data:
        if i['sample_token'] == sample_token and i['filename'][:6]=='sample':
            if 'LIDAR' in i['filename']:
                data_set['LIDAR'] = i
            if 'CAM_FRONT/' in i['filename']:
                data_set['CAM_FC'] = i
            if 'CAM_FRONT_LEFT' in i['filename']:
                data_set['CAM_FL'] = i               
            if 'CAM_FRONT_RIGHT' in i['filename']:
                data_set['CAM_FR'] = i
            if 'CAM_BACK/' in i['filename']:
                data_set['CAM_RC'] = i
            if 'CAM_BACK_LEFT' in i['filename']:
                data_set['CAM_RL'] = i
            if 'CAM_BACK_RIGHT' in i['filename']:
                data_set['CAM_RR'] = i
            if 'RADAR_FRONT/' in i['filename']:
                data_set['Radar_FC'] = i
            if 'RADAR_FRONT_LEFT' in i['filename']:
                data_set['Radar_FL'] = i               
            if 'RADAR_FRONT_RIGHT' in i['filename']:
                data_set['Radar_FR'] = i
            if 'RADAR_BACK_LEFT' in i['filename']:
                data_set['Radar_RL'] = i
            if 'RADAR_BACK_RIGHT' in i['filename']:
                data_set['Radar_RR'] = i
            # data_list[sample_token]=data_set
  
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nus_extend', verbose=True)
    print(f"Now is: Scene-{scene_sec+1}_Sample-{sample_num}_LiDAR")
    nusc.render_pointcloud_in_image(sample_token, camera_channel='CAM_FRONT', pointsensor_channel='LIDAR_TOP', render_intensity=True)
    print(f"Now is: Scene-{scene_sec+1}_Sample-{sample_num}_Radar")
    nusc.render_pointcloud_in_image(sample_token, camera_channel='CAM_FRONT', pointsensor_channel='RADAR_FRONT')
    
    mm_data_vis(data_root, data_set)

def ref_3d_ana():
    ref_pts = torch.load('ref_pts.pt')
    ref_pts_cam = torch.load('ref_pts_cam.pt')
    
    down_sample = 3
    cmap = [np.array([1,0,0]),
              np.array([0,1,0]),
              np.array([0,0,1]),
              np.array([1,1,0]),
              np.array([1,0,1]),
              np.array([0,1,1])]
    
    ref_pts = ref_pts.squeeze().reshape(4,6,150,150,4)
    ref_pts_cam = ref_pts_cam.squeeze().reshape(4,6,150,150,4)
    
    ref_pts = ref_pts[0,0,0:-1:down_sample,0:-1:down_sample,:3].reshape(-1,3)
    c_orig = [[1,1,1] for i in range(ref_pts.size(0))]
    ref_pts_cam = ref_pts_cam[0,:,0:-1:down_sample,0:-1:down_sample,:3].reshape(-1,3)
    c_cam = [cmap for i in range(int(ref_pts_cam.size(0)/6))] 
    # c_cam = np.array(c_cam)
    
    # 构建 Open3D 中提供的 gemoetry 点云类 
    pts_orig = o3d.geometry.PointCloud() 
    pts_orig.points = o3d.utility.Vector3dVector(ref_pts) 
    pts_orig.colors = o3d.utility.Vector3dVector(c_orig) 
    pts_trans = o3d.geometry.PointCloud() 
    pts_trans.points = o3d.utility.Vector3dVector(ref_pts_cam)
    # pts_trans.colors = o3d.utility.Vector3dVector(c_cam)  
    
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 创建窗口,设置窗口名称
    vis.create_window(window_name="point_cloud")
    # 设置点云渲染参数
    opt = vis.get_render_option()
    # 设置背景色（这里为黑色）
    opt.background_color = np.array([50, 50, 50])/255
    # 设置渲染点的大小
    opt.point_size = 4.0
    # 添加点云
    vis.add_geometry(pts_orig)
    vis.add_geometry(pts_trans)
    
    vis.run()
  
  
def regst_test():
    # 实例化一个注册器用来管理模型
    MODELS = Registry('myModels')

    print("\n1")
    # 方式1: 在类的创建过程中, 使用函数装饰器进行注册(推荐)
    @MODELS.register_module()
    class ResNet(object):
        def __init__(self, depth):
            self.depth = depth
            print('Initialize ResNet{}'.format(depth))
    print("2")
    print(MODELS)
    """ 打印结果为:
    Registry(name=myModels, items={'ResNet': <class '__main__.ResNet'>, 'FPN': <class '__main__.FPN'>})
    """
    print("3")
    # 方式2: 完成类的创建后, 再显式调用register_module进行注册(不推荐)   
    class FPN(object):
        def __init__(self, in_channel):
            self.in_channel= in_channel
            print('Initialize FPN{}'.format(in_channel))
    MODELS.register_module(name='FPN', module=FPN)
    print("4")
    print(MODELS)
    
    
    # 配置参数, 一般cfg从配置文件中获取
    backbone_cfg = dict(type='ResNet', depth=101)
    neck_cfg = dict(type='FPN', in_channel=256)
    print("5")
    # 实例化模型(将配置参数传给模型的构造函数), 得到实例化对象
    
    cfg = Config.fromfile('tools/regst_cfg.py')
    
    my_neck = MODELS.build(cfg)
    print("6")
    my_backbone = MODELS.build(cfg)
    print("7")
    print(my_backbone, my_neck)
    print("8\n")
    """ 打印结果为:
    Initialize ResNet101
    Initialize FPN256
    <__main__.ResNet object at 0x000001E68E99E198> <__main__.FPN object at 0x000001E695044B38>
    """


def main():
    # pkl_read()
    # pth_read()
    # vis_gif_gen()
    # PR_anay()
    # cls_distr_ana()
    # samp_div()
    # origin_data_obt()
    # bin_pcd()
    # coord_vis()
    # test()
    # multi_modality_anay()
    # ref_3d_ana()
    # gpg.pkl_read()
    regst_test()


if __name__ == '__main__':
    main()
