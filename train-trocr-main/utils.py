import yaml
import pandas as pd

def load_annotations(config_path):
    with open(config_path) as stream: 
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    #read all train annotation files
    li = []
    dfs = []
    for i in config['train']:
        if i.endswith('txt'):
            with open(i, 'r', encoding='UTF-8') as f:
                annos = f.readlines()
            li += annos
        elif i.endswith('csv'):
            df = pd.read_csv(i)
            dfs.append(df)

    if len(li) != 0 and len(dfs)!=0:
        # replace all \n characters
        li = [sub.replace('\n', '') for sub in li]

        # create train_df
        txt_df = pd.DataFrame([sub.split("\t") for sub in li], columns=['file_name', 'text'])
        csv_df = pd.concat(dfs)
        train_df = pd.concat(txt_df, csv_df)
        train_df.reset_index(drop=True, inplace=True)
    elif len(li)!=0:
        # replace all \n characters
        li = [sub.replace('\n', '') for sub in li]

        # create train_df
        train_df = pd.DataFrame([sub.split("\t") for sub in li], columns=['file_name', 'text'])
    elif len(dfs)!=0:
        train_df = pd.concat(dfs)
        train_df.reset_index(drop=True, inplace=True)
    else:
        print('Did not create train DataFrame')

    #read all valid annotation files
    li = []
    dfs = []
    for i in config['valid']:
        if i.endswith('txt'):
            with open(i, 'r', encoding='UTF-8') as f:
                annos = f.readlines()
            li += annos
        elif i.endswith('csv'):
            df = pd.read_csv(i)
            dfs.append(df)

    if len(li) != 0 and len(dfs) != 0:
        # replace all \n characters
        li = [sub.replace('\n', '') for sub in li]

        # create train_df
        txt_df = pd.DataFrame([sub.split("\t") for sub in li], columns=['file_name', 'text'])
        csv_df = pd.concat(dfs)
        val_df = pd.concat(txt_df, csv_df)
        val_df.reset_index(drop=True, inplace=True)
    elif len(li) != 0:
        # replace all \n characters
        li = [sub.replace('\n', '') for sub in li]

        # create train_df
        val_df = pd.DataFrame([sub.split("\t") for sub in li], columns=['file_name', 'text'])   
    elif len(dfs) != 0:
        val_df = pd.concat(dfs)
        val_df.reset_index(drop=True, inplace=True)
    else:
        print('Did not create validation DataFrame')

    return train_df, val_df