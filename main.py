import torch
from models.PCA import PCA
from models.FF import FF
from models.IPCA import IPCA
from models.CA import CA0
from models.CA0Att import CA0Att
from models.CA0AttM import CA0AttM

import gc
import argparse
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
from utils import *
from analysis import *
import matplotlib.pyplot as plt
from itertools import product
import os
import hashlib

import warnings
warnings.filterwarnings('ignore')



def model_inference_and_predict(model):
    """
    Inference and Prediction of non NN models:
    Returns: model.name_inference.csv & model.name_inference.csv saved in path 'results'
    """
    mon_list = pd.read_pickle('data/mon_list.pkl')
    test_mons = mon_list.loc[mon_list >= model.test_period[0]]
    inference_result = []
    predict_result = []
    T_bar = tqdm(test_mons.groupby(test_mons.apply(lambda x: x//10000)), colour='red', desc=f'{model.name} Inferencing & Predicting')
    
    for g in T_bar: # rolling train
        T_bar.set_postfix({'Year': g[0]})
        model.train_model()
        
        for m in g[1].to_list():
            inference_result.append(model.inference(m)) # T * N * m 
            if not len(model.omit_char):
                predict_result.append(model.predict(m))
        # model refit (change train period and valid period)
        model.refit()
    
    if not len(model.omit_char):
        inference_result = pd.DataFrame(inference_result, index=test_mons, columns=CHARAS_LIST)
        inference_result.to_csv(f'results/inference/{model.name}_inference.csv')
        predict_result = pd.DataFrame(predict_result, index=test_mons, columns=CHARAS_LIST)
        predict_result.to_csv(f'results/predict/{model.name}_predict.csv')
    
    return inference_result
    
    
    
def model_inference_and_predict_CA(model):
    """
    Inference and Prediction of NN models:
    Returns: model.name_inference.csv & model.name_inference.csv saved in path 'results'
    """
    model = model.to('cuda')
    mon_list = pd.read_pickle('data/mon_list.pkl')
    test_mons = mon_list.loc[(mon_list >= model.test_period[0])]
    
    if not len(model.omit_char): # no omit characteristics
        inference_result = pd.DataFrame()
        predict_result = pd.DataFrame()
    else:
        inference_result = []
        
    T_bar = tqdm(test_mons.groupby(test_mons.apply(lambda x: x//10000)), colour='red', desc=f'{model.name} Inferencing & Predicting')
    
    stock_index = pd.Series(dtype=np.int64)
    for g in T_bar: # rolling train, refit once a year
        T_bar.set_postfix({'Year': g[0]})

        model.reset_weight()
        model.release_gpu()
        # release GPU memory
        for _ in range(6): # call function multiple times to clear the cuda cache
            torch.cuda.empty_cache()
            
        train_loss, val_loss = model.train_model()
        # plot loss
        plt.plot(train_loss, label='train_loss')
        plt.plot(val_loss, label='val_loss')
        plt.legend()
        plt.savefig(f'results/train_loss/{model.name}_loss_{g[0]}.png')
        plt.close()

        for m in g[1].to_list():
            m_stock_index, _, _, _ = model._get_item(m)
            stock_index = pd.concat([stock_index, pd.Series(m_stock_index)]).drop_duplicates().astype(int)

            if not len(model.omit_char): # no omit characteristics
                # move inference_R and predict_R to cpu
                inference_R = model.inference(m) # return (N, 1)
                inference_R = inference_R.cpu().detach().numpy()
                inference_R = pd.DataFrame(inference_R, index=m_stock_index, columns=[m])
                inference_result = pd.concat([inference_result.reset_index(drop=True), inference_R.reset_index(drop=True)], axis=1) # (N, T)
                
                predict_R = model.predict(m) # reutrn (N, 1)
                predict_R = predict_R.cpu().detach().numpy()
                predict_R = pd.DataFrame(predict_R, index=m_stock_index, columns=[m])
                predict_result = pd.concat([predict_result.reset_index(drop=True), predict_R.reset_index(drop=True)], axis=1) # (N, T)

            else:
                inference_R = model.inference(m) # return (N, m), m is the length of omit_char
                inference_result.append(inference_R) # (T, N, m)
            
        # refit: change train period and valid period
        model.refit()

    if not len(model.omit_char):
        inference_result = pd.DataFrame(inference_result.values.T, index=test_mons, columns=CHARAS_LIST)
        inference_result.to_csv(f'results/inference/{model.name}_inference.csv')
        
        predict_result = pd.DataFrame(predict_result.values.T, index=test_mons, columns=CHARAS_LIST)
        predict_result.to_csv(f'results/predict/{model.name}_predict.csv')

    # GC: release RAM memory(model)
    del model
    gc.collect()
    return inference_result



def git_push(msg):
    os.system('git add R_squares')
    os.system(f'git commit -m "{msg}"')
    os.system('git push')



def model_selection(model_type, model_K, omit_char=[]):
    allowed = ['FF', 'PCA', 'IPCA', 'CA0', 'CA0Att', 'CA0AttM']
    if model_type not in allowed:
        raise ValueError(f'No Such Model: {model_type}')
    
    if model_type == 'FF':
        return {
            'name': f'FF_{model_K}',
            'omit_char': [],
            'model': FF(K=model_K)
        } 
            
    elif model_type == 'PCA':
        return {
            'name': f'PCA_{model_K}',
            'omit_char': omit_char,
            'model': PCA(K=model_K, omit_char=omit_char)
        } 
        
    elif model_type == 'IPCA':
        return {
            'name': f'IPCA_{model_K}',
            'omit_char': omit_char,
            'model': IPCA(K=model_K, omit_char=omit_char)
        } 
        
    elif model_type == 'CA0':
        return {
            'name': f'CA0_{model_K}',
            'omit_char': omit_char,
            'model': CA0(hidden_size=model_K, lr=CA_LR, omit_char=omit_char)
        } 
            
    elif model_type == 'CA0Att':
        return {
            'name': f'CA0Att_{model_K}',
            'omit_char': omit_char,
            'model': CA0Att(hidden_size=model_K, lr=CA_LR, omit_char=omit_char)
        }
        
    elif model_type == 'CA0AttM':
        return {
            'name': f'CA0AttM_{model_K}',
            'omit_char': omit_char,
            'model': CA0AttM(hidden_size=model_K, lr=CA_LR, omit_char=omit_char)
        }
        
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults are set for omit-importance runs so "Run" in IDE works out of the box.
    parser.add_argument('--Model', type=str, default='FF PCA IPCA CA0 CA0Att CA0AttM')
    parser.add_argument('--K', type=str, default='3 4 5 6')
    parser.add_argument('--omit_char', type=str, default='ALL')

    args = parser.parse_args()
    
    if 'results' not in os.listdir('./'):
        os.mkdir('results')
    if 'train_loss' not in os.listdir('./results'):
        os.mkdir('results/train_loss')
    if 'inference' not in os.listdir('./results'):
        os.mkdir('results/inference')
    if 'predict' not in os.listdir('./results'):
        os.mkdir('results/predict')
    if 'omit_importance' not in os.listdir('./results'):
        os.mkdir('results/omit_importance')
    if 'imgs' not in os.listdir('./'):
        os.mkdir('imgs')
    if 'R_squares' not in os.listdir('./'):
        os.mkdir('R_squares')
        
        
    omit_chars = []
    if isinstance(args.omit_char, str) and len(args.omit_char.strip()) > 0:
        if args.omit_char.strip().upper() == "ALL":
            omit_chars = CHARAS_LIST
        else:
            omit_chars = args.omit_char.split()

    models_name = []
    R_square = []
    extra_total_R2 = {}

    # When running omit-char importance, some models don't implement omit logic (e.g., PCA uses modelBase.inference).
    omit_supported = {"PCA", "IPCA", "CA0", "CA0Att", "CA0AttM"}

    def _omit_cache_path(model_name, omit_list):
        omit_key = "ALL" if omit_list == CHARAS_LIST else " ".join(omit_list)
        digest = hashlib.md5(omit_key.encode("utf-8")).hexdigest()[:12]
        return os.path.join("results", "omit_importance", f"{model_name}_omit_{digest}.json")

    for g in product(args.Model.split(' '), args.K.split(' ')):
        # FF: only keep 6-factor variant (FF5 + UMD)
        if g[0] == "FF" and int(g[1]) != 6:
            continue

        omit_not_supported = len(omit_chars) and (g[0] not in omit_supported)
            
        model = model_selection(g[0], int(g[1]), omit_chars)
        inference_path = f"results/inference/{model['name']}_inference.csv"
        predict_path = f"results/predict/{model['name']}_predict.csv"
        has_saved_results = (not len(model['omit_char'])) and os.path.exists(inference_path)
        omit_cache_path = _omit_cache_path(model["name"], omit_chars) if len(omit_chars) else None
            
        print(f"{time.strftime('%a, %d %b %Y %H:%M:%S +0800', time.gmtime())} | Model: {model['name']} | {omit_chars}")
        print('name : ', model['name'])

        # If omit is not supported, still run total-R2 results generation, but do not contribute to omit-importance JSON.
        if omit_not_supported:
            print(f"{model['name']} does not support omit_char importance; run total R^2 only.")

        # Skip training/inference if cached omit-importance results exist for this model + omit list.
        if omit_cache_path and os.path.exists(omit_cache_path):
            try:
                with open(omit_cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_omit = cached.get("omit_char_run") or []
                cached_vals = cached.get("R2_importance") or []
                if cached_omit == omit_chars and len(cached_vals) == len(omit_chars):
                    print(f"Skip training for {model['name']} (found cached omit_importance).")
                    if not omit_not_supported:
                        models_name.append(model['name'])
                        R_square.extend(cached_vals)
                    del model
                    continue
            except Exception:
                # Cache corrupted/unreadable; fall through to re-run and overwrite.
                pass

        if not omit_not_supported:
            models_name.append(model['name'])

        if has_saved_results:
            print(f"Skip training for {model['name']} (found existing results).")
            inf_ret = pd.read_csv(inference_path)
        else:
            if model['name'].split('_')[0].startswith('CA'):
                print('model_inference_and_predict_CA')
                # if have omit char, inf_ret (T, N, m)
                inf_ret = model_inference_and_predict_CA(model['model'])  
            else:
                inf_ret = model_inference_and_predict(model['model'])
        
        gc.collect()    
        
        # Save total R^2   
        if not len(model['omit_char']):
            total_r2 = calculate_R2(model['model'], 'inference')
            if omit_not_supported:
                extra_total_R2[model["name"]] = total_r2
            else:
                R_square.append(total_r2)
            alpha_plot(model['model'], 'inference', save_dir='imgs')
            # alpha_plot(model['model'], 'predict', save_dir='alpha_imgs')
        else:
            inf_ret = np.array(inf_ret)
            omit_importance_vals = []
            for i in range(len(model['omit_char'])):
                inference_r = inf_ret[:, :, i] # T * N
                complete_r = inf_ret[:, :, -1]
                v = calculate_R2(None, None, inference_r, complete_r)
                omit_importance_vals.append(v)
                if not omit_not_supported:
                    R_square.append(v)

            # Persist per-model omit-importance cache to skip retraining on reruns.
            if omit_cache_path and (not omit_not_supported):
                with open(omit_cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model": model["name"],
                            "omit_char_run": omit_chars,
                            "R2_importance": omit_importance_vals,
                        },
                        f,
                        ensure_ascii=False,
                    )

        del model

    # save R_square to json
    p = time.localtime()
    time_str = "{:0>4d}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}".format(p.tm_year, p.tm_mon, p.tm_mday, p.tm_hour, p.tm_min, p.tm_sec)
    filename = f"R_squares/{time_str}.json"

    # Always record the full factor list for downstream plotting and debugging.
    # For total R^2 runs (no omit), store CHARAS_LIST in omit_char so the factor list is explicit,
    # and store the actual omit run list separately in omit_char_run.
    obj = {
        "models": models_name,
        "factor_list": CHARAS_LIST,
        "omit_char": omit_chars if len(omit_chars) else CHARAS_LIST,
        "omit_char_run": omit_chars,
        "result_type": "omit_importance" if len(omit_chars) else "total_R2",
        "R2_total": R_square,
        "extra_total_R2": extra_total_R2,
    }

    with open(filename, "w") as out_file:
        json.dump(obj, out_file, ensure_ascii=False)

    # git push
    # git_push(f"Run main.py")


    
