import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import tqdm

import os
import zipfile
from pathlib import Path
from joblib import delayed, Parallel
from itertools import product
from utils import CHARAS_LIST

import warnings
warnings.filterwarnings('ignore')


# --------------------------------------------------------------------------- #
# Same流程 as data_prepare.py，但保留 SIC2 列供 GNN 用
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = Path(os.environ.get('AUTOENCODER_DATA_DIR', r'E:\GNN in asset pricing\dataset\Autoencoder_data'))

required_paths = {
    'datashare_zip': RAW_DATA_DIR / 'datashare.zip',
    'month_ret': RAW_DATA_DIR / 'month_ret.pkl',
    'ff5': RAW_DATA_DIR / 'ff5.csv',
    'UMD': RAW_DATA_DIR / 'UMD.csv',
}
missing = [str(p) for p in required_paths.values() if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing required raw files: {missing}")

DATA_DIR.mkdir(exist_ok=True)

# month_ret.pkl
print('Reading month_ret.pkl', end=' ')
mon_ret = pd.read_pickle(required_paths['month_ret'])
mon_ret['ret-rf'] = pd.to_numeric(mon_ret['ret-rf'], errors='coerce')
_date_str = mon_ret['date'].astype(str).str.strip()
mask6 = _date_str.str.len() == 6  # YYYYMM
mask8 = _date_str.str.len() == 8  # YYYYMMDD
mon_ret['date_dt'] = pd.NaT
mon_ret.loc[mask6, 'date_dt'] = pd.to_datetime(_date_str[mask6] + '01', format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp('M')
mon_ret.loc[mask8, 'date_dt'] = pd.to_datetime(_date_str[mask8], format='%Y%m%d', errors='coerce').dt.to_period('M').dt.to_timestamp('M')
mon_ret.loc[~(mask6 | mask8), 'date_dt'] = pd.to_datetime(_date_str[~(mask6 | mask8)], errors='coerce').dt.to_period('M').dt.to_timestamp('M')
mon_ret = mon_ret.loc[mon_ret['date_dt'].notna()].copy()
mon_ret['date'] = mon_ret['date_dt'].dt.strftime('%Y%m%d').astype(int)
print('Done!')

# datashare.zip -> datashare DataFrame
print('Reading datashare.csv from datashare.zip', end=' ')
with zipfile.ZipFile(required_paths['datashare_zip'], 'r') as z:
    # detect header row (permno/date) for robustness, then read
    raw_lines = z.open('datashare.csv').read().decode('utf-8').splitlines()
    header_row = 0
    for i, line in enumerate(raw_lines[:10]):
        if 'permno' in line.lower() and 'date' in line.lower():
            header_row = i
            break
    with z.open('datashare.csv') as f:
        datashare = pd.read_csv(f, skiprows=header_row)
print('Done!')

if 'sic2' not in datashare.columns:
    raise ValueError('datashare.csv missing SIC2 column; required for GNN edges.')

# basic cleaning / type coercion
datashare['permno'] = datashare['permno'].astype(int)
datashare['DATE'] = datashare['DATE'].astype(int)
datashare['sic2'] = pd.to_numeric(datashare['sic2'], errors='coerce')
datashare['DATE_dt'] = pd.to_datetime(datashare['DATE'].astype(str), format='%Y%m%d', errors='coerce').dt.to_period('M').dt.to_timestamp('M')
missing_cols = set(CHARAS_LIST) - set(datashare.columns)
if missing_cols:
    raise ValueError(f'Missing characteristics in datashare: {missing_cols}')
datashare[CHARAS_LIST] = datashare[CHARAS_LIST].apply(pd.to_numeric, errors='coerce')

# keep only (permno, DATE) intersection to avoid missing returns
ds_idx = pd.MultiIndex.from_frame(datashare[['permno', 'DATE']])
mr_idx = pd.MultiIndex.from_frame(mon_ret[['permno', 'date']].rename(columns={'date': 'DATE'}))
common_idx = ds_idx.intersection(mr_idx)
common_df = common_idx.to_frame(index=False).rename(columns={'DATE': 'DATE'})
print(f"common (permno, DATE) rows: {len(common_df)}, permno count: {common_df['permno'].nunique()}")

datashare = datashare.merge(common_df, on=['permno', 'DATE'], how='inner')
mon_ret = mon_ret.merge(common_df.rename(columns={'DATE': 'date'}), on=['permno', 'date'], how='inner')

# overlapping months for downstream use
mon_list = datashare['DATE'].drop_duplicates().sort_values().reset_index(drop=True)
mon_list.to_pickle(DATA_DIR / 'mon_list.pkl')

# drop helper datetime columns and re-save aligned month_ret
datashare = datashare.drop(columns=['DATE_dt'])
mon_ret = mon_ret.drop(columns=['date_dt'])
mon_ret.to_pickle(DATA_DIR / 'month_ret.pkl')

# ff5 / UMD for FF model
pd.read_csv(required_paths['ff5']).to_csv(DATA_DIR / 'ff5.csv', index=False)
pd.read_csv(required_paths['UMD']).to_csv(DATA_DIR / 'UMD.csv', index=False)


def pre_process(date):
    cross_slice = datashare.loc[datashare.DATE == date].copy(deep=False)
    omitted_mask = 1.0 * np.isnan(cross_slice.loc[cross_slice['DATE'] == date])
    # fill nan values with each factors median
    cross_slice.loc[cross_slice.DATE == date] = cross_slice.fillna(0) + omitted_mask * cross_slice.median()
    # if all stocks' factor is nan, fill by zero
    cross_slice.loc[cross_slice.DATE == date] = cross_slice.fillna(0)

    re_df = []
    # rank normalization
    for col in CHARAS_LIST:
        series = cross_slice[col]
        de_duplicate_slice = pd.DataFrame(series.drop_duplicates().to_list(), columns=['chara'])
        series = pd.DataFrame(series.to_list(), columns=['chara'])
        # sort and assign rank, the same value should have the same rank
        de_duplicate_slice['sort_rank'] = de_duplicate_slice['chara'].argsort().argsort()
        rank = pd.merge(series, de_duplicate_slice, left_on='chara', right_on='chara', how='right')['sort_rank']
        # if all values are zero, the results will contain nan
        rank_normal = ((rank - rank.min())/(rank.max() - rank.min())*2 - 1)
        re_df.append(rank_normal)
    re_df = pd.DataFrame(re_df, index=CHARAS_LIST).T.fillna(0)
    re_df['permno'] = list(cross_slice['permno'].astype(int))
    re_df['DATE'] = list(cross_slice['DATE'].astype(int))
    re_df['sic2'] = list(cross_slice['sic2'].astype(int))
    
    return re_df[['permno', 'DATE', 'sic2'] + CHARAS_LIST]


def cal_portfolio_ret(it, df):
    d, f = it[0], it[1]
    # long portfolio, quantile 0.0~0.1; short portfolio, quantile 0.9~1.0
    long_portfolio = df.loc[df.DATE == d][['permno', f]].sort_values(by=f, ascending=False)[:df.loc[df.DATE == d].shape[0]//10]['permno'].to_list()
    short_portfolio = df.loc[df.DATE == d][['permno', f]].sort_values(by=f, ascending=False)[-df.loc[df.DATE == d].shape[0]//10:]['permno'].to_list()
    # long-short portfolio return
    long_ret = mon_ret.loc[mon_ret.date == d].drop_duplicates('permno').set_index('permno').reindex(long_portfolio)['ret-rf'].dropna().mean()
    short_ret = mon_ret.loc[mon_ret.date == d].drop_duplicates('permno').set_index('permno').reindex(short_portfolio)['ret-rf'].dropna().mean()
    chara_ret = 0.5*(long_ret - short_ret)
    
    return chara_ret


def cal_portfolio_charas(month, df):
    mon_portfolio_chara = []
    p_name = ['p_' + chr for chr in CHARAS_LIST]
    for chr in CHARAS_LIST:
        long_portfolio = df.loc[df.DATE == month].sort_values(by=chr, ascending=False).reset_index(drop=True)[:df.loc[df.DATE == month].shape[0]//10]['permno'].to_list()
        short_portfolio = df.loc[df.DATE == month].sort_values(by=chr, ascending=False).reset_index(drop=True)[-df.loc[df.DATE == month].shape[0]//10:]['permno'].to_list()
        
        long_charas = df.loc[df.DATE == month].set_index('permno').loc[long_portfolio][CHARAS_LIST]
        short_charas = df.loc[df.DATE == month].set_index('permno').loc[short_portfolio][CHARAS_LIST]
        
        mon_portfolio_chara.append([month] + (0.5*(long_charas.mean() - short_charas.mean())).to_list())

    return pd.DataFrame(mon_portfolio_chara, index=p_name, columns=['DATE']+CHARAS_LIST)



if __name__ == '__main__':
    # pre-process share data
    processed_df = Parallel(n_jobs=-1)(delayed(pre_process)(d) for d in tqdm(datashare.DATE.drop_duplicates().to_list(), colour='green', desc='Processing'))
    processed_df = pd.concat(processed_df)

    # calculate portfolio returns
    iter_list = list(product(datashare.DATE.drop_duplicates(), CHARAS_LIST))
    portfolio_rets = Parallel(n_jobs=-1)(delayed(cal_portfolio_ret)(it, df=processed_df) for it in tqdm(iter_list, colour='green', desc='Calculating'))
    portfolio_ret = pd.DataFrame(
        np.array(portfolio_rets).reshape(-1, 94),
        index=datashare.DATE.drop_duplicates(),
        columns=CHARAS_LIST
    ).reset_index().rename(columns={'index': 'DATE'})
    portfolio_ret[CHARAS_LIST] = portfolio_ret[CHARAS_LIST].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    print("\nPortfolio_ret NA counts (first 5 cols):")
    print(portfolio_ret[['DATE'] + CHARAS_LIST[:4]].isna().sum())
    
    
    # calculate portfolio characteristics
    mon_list = pd.read_pickle(DATA_DIR / 'mon_list.pkl')
    _portfolio_chara_set = Parallel(n_jobs=-1)(delayed(cal_portfolio_charas)(mon, df=processed_df) for mon in tqdm(mon_list, colour='yellow', desc='Calculating P characteristics'))
    p_charas = _portfolio_chara_set[0].copy(deep=False)
    for tdf in _portfolio_chara_set[1:]:
        p_charas = pd.concat([p_charas, tdf])
    
    # 保存带 SIC2 的版本，供 GNN 使用；也覆盖 datashare_re.pkl，兼容其它模型
    processed_df.to_pickle(DATA_DIR / 'datashare_re.pkl')
    processed_df.to_pickle(DATA_DIR / 'datashare_re_gnn.pkl')
    portfolio_ret.to_pickle(DATA_DIR / 'portfolio_ret.pkl')
    p_charas.to_pickle(DATA_DIR / 'p_charas.pkl')
