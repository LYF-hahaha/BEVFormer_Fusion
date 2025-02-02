import pickle
import time
import numpy as np

# tk_pts_info文件生成
def pkl_read():
    # root_path = 'data/nus_extend'
    root_path = 'data/nuscenes'
    train_path = root_path + '/nuscenes_infos_temporal_train.pkl'
    val_path = root_path + '/nuscenes_infos_temporal_val.pkl'
    tk_pts_info_path = root_path + '/tk_pts_info.pkl'
    
    f1 = open(train_path, 'rb')
    a1 = pickle.load(f1) 
    f2 = open(val_path, 'rb')
    a2 = pickle.load(f2)
    
    tmp = {}
    for i in a1['infos']:
        tmp[i['token']] = i['lidar_path']
    for j in a2['infos']:
        tmp[j['token']] = j['lidar_path']
    
    with open(tk_pts_info_path, 'wb') as f:
        pickle.dump(tmp, f)
    
    print('done!') 


# 将输入的raw lidar pts投影到200*200的网格上
def density_grid_gen(pts_raw):
    cube_size = 100/200
    tmp_1 = np.zeros((200,200))
    tmp_2 = np.ones((200,200))*(-5)
    tmp_3 = np.ones((200,200))*(5)
    dnst_grid = np.stack([tmp_1,tmp_2,tmp_3],axis=2)  # 点的数量、z向最大高度、z向最小高度
    
    # 将±50m×±50m范围内的点云投影在200*200个网格中
    for i in pts_raw:
        if np.sqrt(i[0]**2+i[1]**2)>1 and abs(i[0])<50 and abs(i[1])<50:
            dimx=int(i[0]/cube_size)+100   # 实际米数/cube_size 得到该点能落入的网格位置（即落到第几个cube）
            dimy=int(i[1]/cube_size)+100
            dnst_grid[dimy][dimx][0] += 1
            if i[2] > dnst_grid[dimy][dimx][1]:
                dnst_grid[dimy][dimx][1] = i[2]
            if i[2] < dnst_grid[dimy][dimx][2]:
                dnst_grid[dimy][dimx][2] = i[2]
    return dnst_grid

# 把grid中pts数少于n或者z向高于2m的格子num和z都置零
# 参考:轿车在单个cube中的落点数为25，行人为6~9
def num_hdt_filt(dnst_grid, eps, h):
    tmp = dnst_grid    
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if tmp[i][j][1] > h or tmp[i][j][1]- tmp[i][j][2] < eps or tmp[i][j][0]>30:
                tmp[i][j][0] = 0
                tmp[i][j][1] = 2
                tmp[i][j][2] = -4
    return tmp


# 根据密度grid生成引导参考点
# （在每个网格的中心，在该网格中最高和最低点之间，均匀生成n个点，n是该网格中点的数量）

# 这里是处理current_frame的
# 在生成当前帧guide pts的时候，为减少在2d guide gen中再遍历一遍current_pts
# 就在这里遍历了，将guide_2d_current作为返回值之一返回
def guide_gen_curt(dnst_grid):
    guide_3d_pts = np.array([[0,0,0]])
    guide_2d_pts = np.array([[0,0]])
    for i in range(dnst_grid.shape[0]):
        for j in range(dnst_grid.shape[1]):
            if dnst_grid[i][j][0] != 0:
                pts_num = dnst_grid[i][j][0]
                z_max = dnst_grid[i][j][1]
                z_min = dnst_grid[i][j][2]
                z_list = np.array([(z_max-z_min)/pts_num*k+z_min for k in range(int(pts_num))])
                x_value = (j+0.5)/2-50  # 网格坐标转换成米制单位
                y_value = (i+0.5)/2-50
                x = np.array([x_value for k in range(int(pts_num))])
                y = np.array([y_value for k in range(int(pts_num))])
                pts = np.stack([x,y,z_list],axis=-1)
                guide_3d_pts = np.append(guide_3d_pts, values=pts, axis=0)
                guide_2d_pts = np.append(guide_2d_pts, values=[[(j+0.5)/200,1-(i+0.5)/200]], axis=0) # 用ratio记录位置
    return guide_3d_pts[1:,:], guide_2d_pts[1:,:]

# 这里处理2d引导点
# （可以只处理prev的引导点，也可以调用两次分别处理curt和prev的引导点）
def guide_gen_2d(dnst_grid):
    guide_2d_pts = np.array([[0,0]])
    for i in range(dnst_grid.shape[0]):
        for j in range(dnst_grid.shape[1]):
            if dnst_grid[i][j][0] != 0:
                guide_2d_pts = np.append(guide_2d_pts, values=[[(j+0.5)/200,1-(i+0.5)/200]], axis=0)
    return guide_2d_pts[1:,:]




# 生成了一个"拳击擂台"，用于验证6相机的视角
# 留，别删
def direct_pts_gen():
    pts = np.array([[0,0,0]])
    
    # (1)
    x1_1 = np.ones([5,])*(-20)
    y1_1 = np.ones([5,])*20
    z1_1 = np.linspace(-2,2,5)
    pts1_1 = np.stack([x1_1,y1_1,z1_1],axis=1)
    
    x1_2 = np.linspace(-10,10,21)
    y1_2 = np.ones([21,])*20
    z1_2 = np.zeros([21,])
    pts1_2 = np.stack([x1_2,y1_2,z1_2],axis=1)
    
    # (2)
    x2_1 = np.ones([10,])*(20)
    y2_1 = np.linspace(20,18,2)
    z2_1 = np.linspace(-2,2,5)
    Y,Z = np.meshgrid(y2_1, z2_1)
    pts2_1 = np.stack([x2_1,Y.reshape(-1),Z.reshape(-1)],axis=1)

    x2_2 = np.ones([42,])*(20)
    y2_2 = np.linspace(-10,10,21)
    z2_2 = np.linspace(-1,1,2)
    Y,Z = np.meshgrid(y2_2, z2_2)
    pts2_2 = np.stack([x2_2,Y.reshape(-1),Z.reshape(-1)],axis=1)

    # (3)
    x3_1 = np.linspace(20,16,3)
    y3_1 = np.ones([15,])*(-20)
    z3_1 = np.linspace(-2,2,5)
    X,Z = np.meshgrid(x3_1, z3_1)
    pts3_1 = np.stack([X.reshape(-1),y3_1,Z.reshape(-1)],axis=1)
    
    x3_2 = np.linspace(-10,10,21)
    y3_2 = np.ones([63,])*(-20) 
    z3_2 = np.linspace(-1,1,3)
    X,Z = np.meshgrid(x3_2, z3_2)
    pts3_2 = np.stack([X.reshape(-1),y3_2,Z.reshape(-1)],axis=1)

    # (4)
    x4_1 = np.ones([20,])*(-20)
    y4_1 = np.linspace(-20,-14,4)
    z4_1 = np.linspace(-2,2,5)
    Y,Z = np.meshgrid(y4_1, z4_1)
    pts4_1 = np.stack([x4_1,Y.reshape(-1),Z.reshape(-1)],axis=1)
    
    x4_2 = np.ones([84,])*(-20) 
    y4_2 = np.linspace(-10,10,21)
    z4_2 = np.linspace(-1.5,1.5,4)
    Y,Z = np.meshgrid(y4_2, z4_2)
    pts4_2 = np.stack([x4_2,Y.reshape(-1),Z.reshape(-1)],axis=1)
    
    pts = np.append(pts, pts1_1, axis=0)
    pts = np.append(pts, pts2_1, axis=0)
    pts = np.append(pts, pts3_1, axis=0)
    pts = np.append(pts, pts4_1, axis=0)
    
    pts = np.append(pts, pts1_2, axis=0)
    pts = np.append(pts, pts2_2, axis=0)
    pts = np.append(pts, pts3_2, axis=0)
    pts = np.append(pts, pts4_2, axis=0)
    
    pts = pts[1:,:].reshape([1,4,65,3])
    
    return pts   
