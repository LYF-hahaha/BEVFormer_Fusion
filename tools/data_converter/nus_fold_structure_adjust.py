import os
import shutil
from tqdm import tqdm


def file_move(data_path):

    train_path = os.path.join(data_path, 'train')
    prefix_name = 'v1.0-trainval'
    postfix_name = '_blobs'
    test_path = os.path.join(data_path, 'test')
    test_name = 'v1.0-test_blobs'
    
    data_model = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                  'LIDAR_TOP',
                  'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',]

    for i in range(1):  # train是10  test是1
        print(f"\nNow is sample folder {i+1}/10")
        for model in data_model:
            # samp_model_path = os.path.join(train_path, prefix_name+str(i+1).zfill(2)+postfix_name, 'samples', model)
            samp_model_path = os.path.join(test_path, test_name, 'samples', model)
            temp_samp_name = os.listdir(str(samp_model_path))
            for name in tqdm(temp_samp_name):
                tmp_source_path = os.path.join(str(samp_model_path), name)
                tmp_target_path = os.path.join(data_path, 'samples', model)
                assert os.path.exists(tmp_source_path)
                if not os.path.exists(tmp_target_path):
                    os.makedirs(tmp_target_path)
                shutil.move(str(tmp_source_path), str(tmp_target_path))

    for j in range(1):
        print(f"\nNow is sweep folder {j+1}/10")  
        for model in data_model:
            # swep_model_path = os.path.join(train_path, prefix_name+str(j+1).zfill(2)+postfix_name, 'sweeps', model)
            swep_model_path = os.path.join(test_path, test_name, 'sweeps', model)
            temp_swep_name = os.listdir(str(swep_model_path))
            for name in tqdm(temp_swep_name):
                tmp_source_path = os.path.join(str(swep_model_path), name)
                tmp_target_path = os.path.join(data_path, 'sweeps', model)
                assert os.path.exists(tmp_source_path)
                if not os.path.exists(tmp_target_path):
                    os.makedirs(tmp_target_path)
                shutil.move(str(tmp_source_path), str(tmp_target_path))


def main():
    path = '/root/workspace/BEVFormer_fusion/data/nus_extend'
    file_move(path)


if __name__ == '__main__':
    main()
