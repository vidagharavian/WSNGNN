import pandas as pd

def load_data(file_path):
    df= pd.read_csv(file_path)
    return df

def get_edges(df:pd.DataFrame):
    edges = df.loc[:,[" id"," who CH"," Dist_To_CH"]]
    edges = edges.rename(columns={' id':'Src',' who CH':'Dst'," Dist_To_CH":"Weight"})
    edges['loop']=edges['Src']==edges['Dst']
    edges= edges[edges['loop']!=True]
    edges.drop(columns=['loop'],inplace=True)
    edges.to_csv("edges.csv",index=False)
    return edges

def get_features(df):
    """
     id	 Time	 Is_CH	 who CH	 Dist_To_CH	 ADV_S	 ADV_R	 JOIN_S	 JOIN_R	 SCH_S	 SCH_R	Rank	 DATA_S	 DATA_R	 Data_Sent_To_BS	 dist_CH_To_BS	 send_code 	Expaned Energy	Attack type
    :param df:
    :return:
    """
    feature = df[[" id"," ADV_S"," Time"," ADV_R"," JOIN_S"," JOIN_R"," SCH_S"," SCH_R","Rank"," DATA_S"," DATA_R"," Data_Sent_To_BS"," send_code "," dist_CH_To_BS" ,"Expaned Energy"]]
    feature.set_index(" id",inplace=True)
    feature.to_csv("features.csv")
    return feature

def get_label(df:pd.DataFrame):
    label = df[[" id",'Attack type']]
    label.set_index(" id", inplace=True)
    label.to_csv("label.csv")
    return label


df=load_data('WSN-DS.csv')
# edges = get_edges(df)
# feature = get_features(df)
label = get_label(df)


