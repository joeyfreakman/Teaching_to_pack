import h5py
import redis
import numpy as np
import cv2
import json
import os
from tqdm import tqdm
from redis.exceptions import ConnectionError, TimeoutError
from redis.retry import Retry


def write_hdf5_to_redis(hdf5_path, redis_host='localhost', redis_port=6379, redis_db=0, password='P|ayw!th4am'):
    pool = redis.ConnectionPool(host=redis_host, port=redis_port, db=redis_db,password=password, 
                                socket_timeout=30, socket_connect_timeout=30,
                                max_connections=10)
    retry = Retry(retries=3, backoff=1)
    r = redis.Redis(connection_pool=pool, retry=retry,password=password)
    pipe = r.pipeline(transaction=False)
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            episode_id = hdf5_path.split('_')[-1].split('.')[0]
            
            # 写入action数据
            action_data = f['/action'][:]
            pipe.set(f'episode:{episode_id}:action', action_data.tobytes())
            
            # 写入图像数据
            for cam_name in f['/observations/images'].keys():
                images = f[f'/observations/images/{cam_name}'][:]
                compressed = f.attrs.get('compress', False)
                
                if compressed:
                    compress_len = f['/compress_len'][:]
                    for i, img in enumerate(images):
                        img_len = int(compress_len[list(f['/observations/images'].keys()).index(cam_name), i])
                        compressed_img = img[:img_len]
                        img = cv2.imdecode(compressed_img, 1)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pipe.rpush(f'episode:{episode_id}:images:{cam_name}', img.tobytes())
                else:
                    for img in images:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pipe.rpush(f'episode:{episode_id}:images:{cam_name}', img.tobytes())
                
                # 存储图像形状信息
                pipe.set(f'episode:{episode_id}:images:{cam_name}:shape', json.dumps(images.shape))
            
            # 执行所有命令
            pipe.execute()
        
        print(f"Successfully processed {hdf5_path}")
    except (ConnectionError, TimeoutError) as e:
        print(f"Redis connection error: {str(e)}")
        print("Please check if Redis server is running and accessible.")
    except Exception as e:
        print(f"Error processing {hdf5_path}: {str(e)}")
    finally:
        r.connection_pool.disconnect()

# 测试
hdf5_root = '/mnt/d/kit/ALR/dataset/ttp_compressed'
total_episodes = 50

for i in tqdm(range(total_episodes), desc="Processing episodes"):
    hdf5_path = os.path.join(hdf5_root, f'episode_{i}.hdf5')
    if os.path.exists(hdf5_path):
        write_hdf5_to_redis(hdf5_path)
    else:
        print(f"File not found: {hdf5_path}")

print("All episodes processed.")