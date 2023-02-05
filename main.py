import sys
sys.path.append('C:/Users/STEPH/Documents/Scripts/finsec')
import finsec
import requests
import json
import pandas as pd
import pdb
print(finsec.__path__)
import open_figi_key
import time
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

OPEN_FIGI_API = "https://api.openfigi.com/v3/mapping"
HEADERS = {"X-OPENFIGI-APIKEY": "{}".format(open_figi_key.KEY)}

def dataframe_statistics(df, manager_name:str, cik:str):
    '''
    Build out the holdings dataframes with some basic statistics.
    '''
    df['Portfolio percentage'] = (df['Holding value'] / df['Holding value'].sum()) * 100
    df['Manager Name'] = manager_name
    df['CIK'] = cik
    return df

def write_cusips_to_txt(cusip_mappings, cusip_mapping_file = 'cusips.txt'):
    '''
    Write cusip mappings to txt. Minimizing the need for API calls. 
    '''
    with open(cusip_mapping_file, 'w') as f:
        f.write(str(cusip_mappings))
    return

def read_cusips_from_txt(cusip_mapping_file='cusips.txt'):
    '''
    Read cusip mappings from txt. Minimizing the need for API calls.
    '''
    try:
        with open(cusip_mapping_file) as f:
            cusip_mappings = eval(f.read())
    except:
        return []
    return cusip_mappings

def map_cusips_openfigi(cusips:list):
    '''
    Splits all openFIGI API calls down to lots of 100 so that they can be processed (maximum number of cusips in a POST is 100). 
    Returns cusip_mappings which is a list of tuples containing CUSIP and Ticker.
    '''
    cusips_lots = [cusips[x:x+100] for x in range(0, len(cusips), 100)]
    mappings = []
    for index, lot in enumerate(cusips_lots):
        try:
            query = [{"idType": "ID_CUSIP", "idValue": str(cusip)} for cusip in lot]
            open_figi_resp = json.loads(requests.post(OPEN_FIGI_API, json=query, headers=HEADERS).text)
            tickers = []
            for resp in open_figi_resp:
                if resp.get("warning"):
                    tickers.append('')
                else:
                    tickers.append(resp["data"][0].get('ticker'))
            mappings.append(list(zip(lot, tickers)))
        except:
            print("Exception caught on lot: {}".format(index))
            time.sleep(6)
        time.sleep(0.26)
    mappings = [item for sublist in mappings for item in sublist]   # Flatten list of lists to single list of tuples.
    no_cusip_mappings = [item for item in mappings if item[1] == '']
    cusip_mappings = [item for item in mappings if item[1] != '']
    return cusip_mappings, no_cusip_mappings

def map_cusips_cns(cusips, cns_file):
    '''
    Map cusips based on 'Fails to Deliver' Continuous Net Settlement (CNS) data. Fails to Deliver data contains the list of all shares that failed to be delivered as of a particular settlement date as recorded in the National Securities Clearing Corporation's ("NSCC") Continuous Net Settlement (CNS) system aggregated over all NSCC members. Typically this list contains thousands of potential mappings as it is common place for shares to fail to deliver. 

    Further information can be found here: https://www.sec.gov/data/foiadocsfailsdatahtm
    '''
    mapped_cusips_cns = []
    
    with open(cns_file) as f:
        cns = f.read().split('\n')
        for line in cns:
            components = line.split('|')
            if len(components)>3:
                mapped_cusips_cns.append((components[1],components[2]))
    mapped_cusips_cns = [t for t in (set(tuple(i) for i in mapped_cusips_cns))]   # Distinct mappings

    mapped_cusips_cns = [cns for cusip in cusips for cns in mapped_cusips_cns if cusip == cns[0]]     # Check if the list of unkowns is covered by cns
    mapped_cusips_cns_cusip_list = [item[0] for item in mapped_cusips_cns]  # Look at the list of CUSIPs where we have found mappings. 
    no_cusip_mappings = set(cusips) - set(mapped_cusips_cns_cusip_list)
    no_cusip_mappings = [(item,'') for item in no_cusip_mappings]

    return mapped_cusips_cns, no_cusip_mappings

def calc_pct_bought_sold(this_share_count, last_share_count):
    if this_share_count > 0 and last_share_count == 0:
        return 100
    else:
        return ((this_share_count/last_share_count)-1)*100
 
def calc_dollar_bought_sold(count_bought_sold, last_holding_val, last_share_count, this_holding_val, this_share_count):
    if this_share_count == 0:
        return count_bought_sold * (last_holding_val/last_share_count)
    else:
        return count_bought_sold * (this_holding_val/this_share_count)

def process_dataframe(this_qtr_df, last_qtr_df):
    this_qtr_df = this_qtr_df[['Manager Name', 'Ticker', 'Holding value','Share or principal amount count', 'Portfolio percentage']].groupby(['Manager Name', 'Ticker'],as_index=False).sum()
    last_qtr_df = last_qtr_df[['Manager Name', 'Ticker', 'Holding value','Share or principal amount count', 'Portfolio percentage']].groupby(['Manager Name', 'Ticker'],as_index=False).sum()

    processed_dataframe = pd.merge(last_qtr_df, this_qtr_df, how='outer', left_on=['Manager Name', 'Ticker'], right_on=['Manager Name', 'Ticker'], suffixes=['_last', '_this'])
    processed_dataframe = processed_dataframe.fillna({'Holding value_last':0,
       'Share or principal amount count_last':0, 
       'Portfolio percentage_last':0,
       'Holding value_this':0, 
       'Share or principal amount count_this':0,
       'Portfolio percentage_this':0})  # Fillna with zeroes to ensure calculations work. 
    
    processed_dataframe['pct bought/sold holdings'] = processed_dataframe.apply(lambda x: calc_pct_bought_sold(x['Share or principal amount count_this'],x['Share or principal amount count_last']), axis=1)
    processed_dataframe['pct bought/sold holdings weighted by portfolio pct invested'] = processed_dataframe.apply(lambda x: x["pct bought/sold holdings"] * (x["Portfolio percentage_this"]/100), axis=1)
    processed_dataframe['# bought/sold holdings'] = processed_dataframe['Share or principal amount count_this'] - processed_dataframe['Share or principal amount count_last']
    processed_dataframe['$ bought/sold holdings'] = processed_dataframe.apply(lambda x: calc_dollar_bought_sold(x['# bought/sold holdings'], x['Holding value_last'],x['Share or principal amount count_last'], x['Holding value_this'],x['Share or principal amount count_this']), axis=1)

    ticker_dataframe = processed_dataframe[['Ticker', 'Holding value_last','Share or principal amount count_last', 'Portfolio percentage_last','Holding value_this', 'Share or principal amount count_this','Portfolio percentage_this', 'pct bought/sold holdings','pct bought/sold holdings weighted by portfolio pct invested','# bought/sold holdings','$ bought/sold holdings']].groupby('Ticker',as_index=False).agg(value_held_last=('Holding value_last', 'sum'), count_held_last=('Share or principal amount count_last','sum'), mean_portfolio_pct_last=('Portfolio percentage_last','mean'),value_held_this=('Holding value_this', 'sum'), count_held_this=('Share or principal amount count_this','sum'), avg_portfolio_pct_this=('Portfolio percentage_this','mean'), avg_pct_bought_sold=('pct bought/sold holdings','mean'),avg_bought_sold_weight_by_portfolio_pct=('pct bought/sold holdings weighted by portfolio pct invested','mean'),total_bought_sold=('# bought/sold holdings','sum'), dollars_bought_sold=('$ bought/sold holdings','sum')).sort_values(by=['dollars_bought_sold'], ascending=False)
    
    return processed_dataframe, ticker_dataframe

def sns_heatmap(dataframe, key_column, key_column_format, chart_title, c_pos_neg):
    # Build out heatmap things
    dataframe['Position'] = range(1,len(dataframe) + 1)
    dataframe['y'] = [(x//10 + 1 if x%10 != 0 else (x//10)) for x in dataframe['Position']]
    dataframe['x'] = [(x%10 if x%10 != 0 else 10) for x in dataframe['Position']]
    # most_bought['normalized_dollars_bought_sold'] = (most_bought['dollars_bought_sold'] - most_bought['dollars_bought_sold'].min()) / (most_bought['dollars_bought_sold'].max()-most_bought['dollars_bought_sold'].min())
    # most_bought['natural_log'] = np.log(most_bought['normalized_dollars_bought_sold'])
    pivot_df = dataframe.pivot(index='y', columns='x', values=key_column)
    
    ticker_labels = np.asarray(dataframe['Ticker']).reshape((10,10))
    key_column_labels = np.asarray(dataframe[key_column]).reshape((10,10))
    if key_column_format == '$M':
        annot_labels = np.asarray(["{} \n ${:,.0f}M".format(ticker, dbs/1000000) for ticker, dbs in zip(ticker_labels.flatten(), key_column_labels.flatten())]).reshape((10,10))
    elif key_column_format == '$B':
        annot_labels = np.asarray(["{} \n ${:,.0f}B".format(ticker, dbs/1000000000) for ticker, dbs in zip(ticker_labels.flatten(), key_column_labels.flatten())]).reshape((10,10))
    elif key_column_format == '#T':
        annot_labels = np.asarray(["{} \n {:,.0f} Thousand".format(ticker, dbs/1000) for ticker, dbs in zip(ticker_labels.flatten(), key_column_labels.flatten())]).reshape((10,10))
    elif key_column_format == '%':
        annot_labels = np.asarray(["{} \n {:.2f} %".format(ticker, kcv) for ticker, kcv in zip(ticker_labels.flatten(), key_column_labels.flatten())]).reshape((10,10))
    else:
        annot_labels = np.asarray(["{}".format(ticker) for ticker in ticker_labels.flatten()]).reshape((10,10))
    
    fig, ax = plt.subplots(1, 1, figsize = (50, 50), dpi=300)

    cmap_val = 'BuGn' if c_pos_neg == 'pos' else 'OrRd_r'


    sns.heatmap(pivot_df, annot=annot_labels, annot_kws={"size":5}, fmt="", cmap=cmap_val, linewidths=0.25, square=False, yticklabels=False, xticklabels=False, ax=ax)
    # sns.set(font_scale=1.8)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(chart_title)
    plt.show()
    return

# --- Initialize some empty dataframes ---
all_this_qtr_holdings = pd.DataFrame()
all_last_qtr_holdings = pd.DataFrame()

# --- Iterate through the list of investment managers, collect this quarters holdings and last --- 
managers = json.load(open("fund_managers.json"))    # Get the list of fund managers under review. 
for manager in managers:
    cik, manager_name = manager['cik'], manager['name'] # Get the cik and manager name.
    fs = finsec.Filing(cik) # Initialize the finsec object. 
    this_qtr_holdings = fs.latest_13f_filing.sort_values(by='Holding value', ascending=False)   # Get the latest filing. 
    last_qtr_cover_page, detailed_last_qtr_holdings, last_qtr_holdings = fs.get_a_13f_filing("Q2-2022") # Get the last (available) quarters filing. 
    this_qtr_holdings = dataframe_statistics(this_qtr_holdings, manager_name, cik)
    last_qtr_holdings = dataframe_statistics(last_qtr_holdings, manager_name, cik)
    all_this_qtr_holdings = pd.concat([all_this_qtr_holdings, this_qtr_holdings], ignore_index=True, sort=False)
    all_last_qtr_holdings = pd.concat([all_last_qtr_holdings, last_qtr_holdings], ignore_index=True, sort=False)


# --- Get list of CUSIPs from all manager holdings ---
cusips = list(dict.fromkeys(all_this_qtr_holdings['CUSIP'].tolist() + all_last_qtr_holdings['CUSIP'].tolist()))

# --- Remove any CUSIPs that have been previously mapped (contained within the cusips text file) ---
existing_cusip_mappings = read_cusips_from_txt()
existing_cusip = [item[0] for item in existing_cusip_mappings]
cusips = list(set(cusips) - set(existing_cusip))

# --- Check openFIGI then SEC CNS records for CUSIP mappings ---
openfigi_cusip_mappings, no_cusip_mappings = map_cusips_openfigi(cusips)
cns_cusip_mappings, no_cusip_mappings = map_cusips_cns(no_cusip_mappings, 'cnsfails202301a')   # Pass in any cusips that had no openFIGI mappings, check if the cns function can find tickers for these. 
cusip_mappings = openfigi_cusip_mappings + cns_cusip_mappings + existing_cusip_mappings # Bring all of the cusip->ticker mappings together (openFIGI, CNS and any pre-existing mappings).
write_cusips_to_txt(cusip_mappings) # Write/overwrite the cusip mappings to a text file.

# --- Map tickers into the quarterly holding dataframes ---
all_this_qtr_holdings['Ticker'] = all_this_qtr_holdings['CUSIP'].map(dict(cusip_mappings))
all_last_qtr_holdings['Ticker'] = all_last_qtr_holdings['CUSIP'].map(dict(cusip_mappings))
all_this_qtr_holdings['Ticker'] = all_this_qtr_holdings['Ticker'].fillna('N/A')
all_last_qtr_holdings['Ticker'] = all_last_qtr_holdings['Ticker'].fillna('N/A')

all_last_qtr_holdings['Holding value'] = all_last_qtr_holdings['Holding value'].fillna(0)
all_last_qtr_holdings['Share or principal amount count'] = all_last_qtr_holdings['Share or principal amount count'].fillna(0)
all_last_qtr_holdings['Portfolio percentage'] = all_last_qtr_holdings['Portfolio percentage'].fillna(0)

all_this_qtr_holdings['Holding value'] = all_this_qtr_holdings['Holding value'].fillna(0)
all_this_qtr_holdings['Share or principal amount count'] = all_this_qtr_holdings['Share or principal amount count'].fillna(0)
all_this_qtr_holdings['Portfolio percentage'] = all_this_qtr_holdings['Portfolio percentage'].fillna(0)

# --- Create a processed dataframe ---
processed_dataframe, ticker_dataframe = process_dataframe(all_this_qtr_holdings, all_last_qtr_holdings)

# --- Build out heatmaps ---
most_bought = ticker_dataframe[:100].reindex()
sns_heatmap(most_bought, 'dollars_bought_sold', '$M', 'Most Bought Asset by Dollar Value', c_pos_neg='pos')

most_sold = ticker_dataframe[-100:].reindex()
sns_heatmap(most_sold, 'dollars_bought_sold', '$M', 'Most Sold Asset by Dollar Value', c_pos_neg='neg')

most_bought_sold_by_allocation = ticker_dataframe.sort_values(by=['avg_bought_sold_weight_by_portfolio_pct'], ascending=False)[:100]
sns_heatmap(most_bought_sold_by_allocation, 'avg_bought_sold_weight_by_portfolio_pct', '%', 'Most Bought Asset by Portfolio Allocation', c_pos_neg='pos')




# all_this_qtr_holdings = pd.read_excel('all_this_qtr_holdings.xlsx')
# all_last_qtr_holdings = pd.read_excel('all_last_qtr_holdings.xlsx')

# all_this_qtr_holdings.to_excel("all_this_qtr_holdings.xlsx")    # !!! Not req
# all_last_qtr_holdings.to_excel("all_last_qtr_holdings.xlsx")