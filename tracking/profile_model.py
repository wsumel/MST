import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib

import os
import numpy as np
# from lib.utils.misc import NestedTensor

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--script', type=str, default='mst', choices=['mst'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='vitb_256_mae_32x4_ep100_got', help='yaml configure file name')
    # parser.add_argument('--optimize', action='store_true', help='是否使用TorchScript优化')
    args = parser.parse_args()
    return args

def evaluate_vit(model, template, search,search_feat):
    model.eval()
    macs1, params1 = profile(model, inputs=(template, search), custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
    # Warmup
    T_w = 100
    T_t = 500
    with torch.no_grad():
        for _ in range(T_w):
            _ = model(template, search)
        times = []
        for _ in range(T_t):
            t1 = time.time()
            _ = model(template, search)
            t2 = time.time()
            times.append(t2-t1)
        
        avg_lat = np.mean(np.array(times))
        min_lat = np.min(np.array(times))
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("The min overall latency is %.2f ms" % (min_lat * 1000))

def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # device = "cpu"
    args = parse_args()
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "mst":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_mst
        model = model_constructor(cfg, training=False)
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        search_feat = torch.randn(bs, 256,192)
        template = template.to(device)
        search = search.to(device)
        search_feat = search_feat.to(device)
        model = model.to(device)
        
        evaluate_vit(model, template, search,search_feat)

    else:
        raise NotImplementedError
