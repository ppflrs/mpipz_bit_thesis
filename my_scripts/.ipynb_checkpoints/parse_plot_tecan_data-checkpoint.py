# Script Name   : parse_plot_tecan_data.py
# Author        : Pepe Flores & Ricardo Martinez
# Created       : 12th June 2019

# Description   : This script will parse tecan data and output scatterplots of the parsed data


###########
# imports #
###########
import numpy as np
import pandas as pd
import scipy

import argparse
import os
import xlrd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set(style='whitegrid')

    
##########################
# MAIN PARSING FUNCTIONS #
##########################
    
def parse_tecan_results_files(data_folder):
    """
    This script parses the tecan results files. Care should be taken when feeding the
    script 'Single' or 'Multiple' reads as feeding both simultaneously will result
    in a misformed dataframe
    """
    
    tecan_df_list = []
    multi_measurement_blocks = []
    
    for path, dirs, files in os.walk(data_folder):
        for file in [f for f in files if '.xlsx' in f]:
            
            tecan_results_file = os.path.join(path, file)
            work_sheets = xlrd.open_workbook(tecan_results_file).sheet_names()

            for sheet in work_sheets[::-1]:
                print(sheet)
                #tecan_worksheet = pd.read_excel(tecan_results_file, sheet_name=sheet, header=None)
                df = pd.read_excel(tecan_results_file, sheet_name=sheet, header=None)
                
                
                if df.empty:
                    # if the excel worksheet is empty continue onto the next one
                    continue
                
                else:
                    # if the worksheet is not empty, the script runs 
                    
                    # measurement start time is located on located on the 1st column of the row with string "Start Time:"
                    measurements_start_time = pd.to_datetime(df.loc[df[0]=='Start Time:'][1], dayfirst=True).reset_index(drop=True)
                    exp_start_time = min(measurements_start_time)
                    
                    # In Tecan results file with single measurements, the results block ends with "End Time:"
                    measurements_end_time = df.loc[df[0]=='End Time:'][1].index
                    
                    # Parsing is Multiple or Single measurements per well dependent
                    measurement_blocks = []
                    if "Multiple Reads per Well (Border)" in df.iloc[:,0].values:
                        
                        tecan_mode = 'multiple'
                    
                        #print("%r it's a multiple measurements tecan file" % sheet) 
                        # Get the indexes where the value is "Time [s]" which indicates
                        # the beginning of a new measurement block
                        measurement_block_start_indices = df[df[0] == "Time [s]"].index

                        # iterate trough the blocks
                        for block_idx in measurement_block_start_indices:
                            # well_ID
                            well_idx = block_idx - 1
                            
                            # measurement rep index
                            m1_idx = block_idx + 4
                            m2_idx = block_idx + 5
                            m3_idx = block_idx + 6
                            m4_idx = block_idx + 7


                            # Create a DF per measurements block where col_0 is Time [s], 
                            # and the other columns are the measurements
                            block_df = pd.concat([df.iloc[block_idx].dropna(), 
                                                  df.iloc[m1_idx].dropna(),                                                     
                                                  df.iloc[m2_idx].dropna(),
                                                  df.iloc[m3_idx].dropna(),
                                                  df.iloc[m4_idx].dropna()], axis=1)

                            # Rename the columns and drop the row that was used for naming the columns
                            block_df.columns = block_df.iloc[0].values
                            block_df =  block_df.reindex(block_df.index.drop(0))
                            block_df = block_df.rename(columns={'0;1':'rep1', '1;1':'rep2',
                                                                '1;0':'rep3', '0;0':'rep4'})
                            
                            # Add the Well column to the block DF
                            block_df["Well"] = df.iloc[well_idx,0]
                            block_df["Worksheet"] = sheet
                            
                            # Convert the "Time [s]" column to float and then to timedelta seconds
                            block_df["Time [s]"] = block_df["Time [s]"].astype('float').astype('timedelta64[s]')

                            # Add the seconds to the starting time
                            block_df["Measuring_time"] = measurements_start_time.loc[0] + block_df["Time [s]"]
                            
                            # Add the block df to the measurement_blocks list
                            multi_measurement_blocks.append(block_df)

                        # post-parsing processing of all the blocks into a dataframe
                        measurements_df = pd.concat(multi_measurement_blocks)
                        measurements_df["Filename"] = tecan_results_file
                        measurements_df["Measuring_type"] = tecan_mode
                        measurements_df["Row"] = measurements_df.Well.apply(lambda x: split_well(x)[0])
                        measurements_df["Column"] = measurements_df.Well.apply(lambda x: int(split_well(x)[1]))
                        
                        measurements_df = measurements_df.melt(id_vars=["Filename", "Worksheet", "Well", "Row","Column", 
                                                                        "Measuring_type", "Measuring_time"],
                                                               value_vars=['rep1', 'rep2', 'rep3', 'rep4'],
                                                               var_name="Replicate", value_name="Abs")
                        measurements_df['Abs'] = measurements_df['Abs'].astype(float)
                        
                    else:
                        
                        tecan_mode='single'
                                                
                        # In Tecan results file with single measurements, the results block starts with "Cycle Nr."
                        measurements_start = df[df[0]=='Cycle Nr.'].index
                        
                        for i in range(len(measurements_start_time)):
                            block_df = df.loc[measurements_start[i]:measurements_end_time[i]-4]
                            # deliminate where the measurements stop for that block
                            booldf = block_df.isnull()

                            if block_df.iloc[1].isna().sum() > 0:  
                                stop_column = booldf[booldf.any(axis=1)].idxmax(axis=1).iloc[1]
                                block_df = block_df.loc[:,:stop_column]
                                block_df = block_df[block_df.columns[:-1]]
                            block_df.columns = range(len(block_df.iloc[1]))
                            # adjust sampling_time
                            block_df.iloc[1,1:] = measurements_start_time.iloc[i]+pd.to_timedelta(block_df.iloc[1,1:],unit='s')

                            block_df = block_df.T
                            block_df.columns = block_df.iloc[0]
                            block_df = block_df.reindex(block_df.index.drop(0))

                            measurement_blocks.append(block_df)
                            
                        # post-processing
                        dfs = pd.concat(measurement_blocks)
                        dfs = dfs.drop(labels=["Cycle Nr."], axis=1)
                        dfs["Time [s]"] = pd.to_datetime(dfs["Time [s]"])
                        dfs = dfs.reset_index(drop=True)
                        
                        
                        not_wells = ['Time [s]', 'Temp. [°C]']
                        wells = [x for x in dfs.columns if x not in not_wells]
                        dfs = dfs.melt(value_vars=wells, id_vars=not_wells, value_name="Abs", var_name="Well")

                        dfs["Row"] = dfs.Well.apply(lambda x: split_well(x)[0])
                        dfs["Column"] = dfs.Well.apply(lambda x: int(split_well(x)[1]))
                        dfs["Filename"] = tecan_results_file
                        dfs["Worksheet"] = sheet
                        dfs["Measuring_type"] = tecan_mode
                        dfs['Abs'] = dfs['Abs'].astype(float)
                        tecan_df_list.append(dfs)

    

        if len(tecan_df_list)==0 and len(multi_measurement_blocks)==0:
            print("There is no data to parse!")
            return pd.DataFrame() # empty dataframe
        
        if len(tecan_df_list) > 0: 
            
            tecan_df = pd.concat(tecan_df_list)
            
            
            tecan_df["t[m]"] = (tecan_df["Time [s]"] - tecan_df["Time [s]"].min()).astype('timedelta64[m]')
            tecan_df["Replicate"] = 1
            
            tecan_df = tecan_df.rename(columns={"Time [s]":"Measuring_time"})
            tecan_df = tecan_df.drop(labels=["Temp. [°C]", "t[m]"], axis=1)

            if len(multi_measurement_blocks) > 0:
                tecan_df = pd.concat([tecan_df, measurements_df])
        
        else:
            tecan_df = measurements_df

            
        # Convert time to min
        experiment_start_time = min(tecan_df["Measuring_time"])
        tecan_df["t[m]"] = (tecan_df["Measuring_time"] - experiment_start_time).astype('timedelta64[m]')
        tecan_df = tecan_df.sort_values(["t[m]", "Column", "Well"])
        tecan_df = tecan_df.reset_index(drop=True)


        return tecan_df
    

def parse_metadata(metadata_folder):
    """
    Metadata refers to the 96-Well Plate (8-row, 12-column) arrangement in
    .csv format where one files contains name the sample housed in each 
    cell and the other the growth medium used. Ensure the provided metadata
    has columns (1-12) and rows(A-H) labeled as well.
    """
    
    
    files = os.listdir(metadata_folder)
    assert ['growth-media_arrangement.csv', 'samples_arrangement.csv'] == files
    
    flag=False
    for file in files:
        
        df = pd.read_csv(os.path.join(metadata_folder, file), index_col=0)
        
        df = df.dropna().reset_index()

        if 'samples' in file:
            df = df.melt(value_vars=df.columns[1:],
                         var_name="Column", id_vars="index",
                         value_name="Sample")
        
        else: 
            df = df.melt(value_vars=df.columns[1:],
                         var_name="Column", id_vars="index", 
                         value_name="Media")

        df = df.rename(columns={"index":"Row"})
        df["Well"] = df["Row"]+ df["Column"]
        df['Column'] = df['Column'].astype(int)

        if not flag:
            df0 = df.copy()
            flag=True
            
    return  df.merge(df0[["Well", "Media"]], on="Well")

def remove_blank_signal(df, mdf):
    """
    This function does returns two dataframes:
        1. df_mean --- this dataframe has the BLANK'S signal removed
        from the sample's signal on a per time, media, and replicate 
        basis. At every timepoint a blank's well signal is measured
        at 4 regions. Here the signals are subtracted by region and
        then averaged .
        
        2. df_regs --- this dataframe has the blank's signal removed
        from the sample's signal on a per time, media, and replicate
        basis. Here the signals are subtracted by region, but are NOT
        averaged. This is useful because it allows one to see the
        differences in the TECAN region measurements
        
    """
    
    # dataframe hosting the sample's name ID e.g. ICL_184BA or BLANK
    df = df.merge(mdf[["Well", "Media", "Sample"]], on="Well", how="left")

    # temp dataframe consisting only of the BLANK's values
    blank_df = df.loc[df["Sample"] == "BLANK"].copy()
    
    # merges the dataframes such that all the well replictes are aligned to the
    # blank's replicates, this makes subtraction on a per replicate basis possible
    df = df.merge(blank_df[["t[m]", "Media", 'Abs', 'Replicate']],
                  how='left',
                  on=["t[m]", "Media", 'Replicate'])
    
    
    
    # Subtraction (Sample_Abs -- Blank_Abs) then MEAN
    dfs = df.copy()
    dfs['Corr_Abs'] = dfs['Abs_x'] - dfs['Abs_y']
    df_regs = dfs.drop(labels=['Abs_x', 'Abs_y'], axis=1)    
    df_tmp = df_regs.groupby(['Well', 't[m]', 'Media']).agg('mean').reset_index()
    
    # only one row is needed from the df_reps as the aggregate-mean averages over 4 rows
    # we first drop the unnecessary rows and columns, reset the index and mer
    df_mean = df_regs[::4].drop(labels=['Replicate','Corr_Abs'], axis=1).reset_index(drop=True)
    df_mean = df_mean.merge(df_tmp[['Well', 't[m]', 'Media', 'Corr_Abs']],
                            how='left',
                            on=['Well', 't[m]', 'Media'])
    
    return df_mean, df_regs


def split_well(s):
    
    #split well into column row
    # https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number
    
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return head, tail


def create_strain_well_dictionary(df_mean):
    """
    This helper function extracts the wells (e.g. A1) associated
    with a given strain to facilitate plotting.
    """
    
    strains = df_mean['Sample'].unique()
    
    # strain at well dictionary
    wells_dict = {}
    for stn in strains:
        wells_dict[stn] = [well for well in df_mean[df_mean['Sample']==stn]['Well'].unique()]
        
    return wells_dict
    
    
    
######################
# PLOTTING FUNCTIONS #
######################
    
    
def plot_all_96W_format(df, y_axis_var="Corr_Abs"):
    """
    This function plots the growth measurements in corresponding 96-Well Format,
    i.e. (8rows x 12 cols)
    
    """
    
    cols = np.arange(1,13)
    rows = list(map(chr, range(65, 73)))
    
    g = sns.FacetGrid(df, col="Column", row="Row", aspect=2, col_order=cols,
                      row_order=rows, sharey=False)
    g = (g.map(sns.scatterplot, "t[m]", y_axis_var, marker=".").set_titles("{row_name}{col_name}"))
    
    return g


def plot_all_regions_96W_format(df, y_axis_var="Corr_Abs"):
    """
    This function plots the growth measurements in corresponding 96-Well Format, 
    for each mesurement region the Tecan output. 

    """
    cols = np.arange(1,13)
    rows = list(map(chr, range(65, 73)))
    
    g = sns.FacetGrid(df, col="Column", row="Row", aspect=2, col_order=cols,
                      row_order=rows, hue='Replicate', sharey=False)
    g = (g.map(sns.scatterplot, "t[m]", y_axis_var, marker=".").set_titles("{row_name}{col_name}"))
    
    return g
    

def plot_all_strains_media_format(df, y_axis_var='Corr_Abs'):
    """
    Plot the growth measurements of individual 'strain' or list of strains at
    their respective growth conditions--i.e. growth media format i.e. (N-rows x 3cols)
    """
    
    g=sns.FacetGrid(df, col='Media', row='Sample', aspect=2, col_order=['sup', 'tp', 'tsb'], sharey=False)
    g=(g.map(sns.scatterplot, 't[m]', y_axis_var).set_titles('{row_name}-{col_name}'))
    
    return g


def plot_select_strains_media_format(strains, df):
    """
    Plot the growth measurements of individual 'strain' or list of strains at
    their respective growth conditions--i.e. growth media format i.e. (N-rows x 3cols)
    """
    
    def plot(strains, df, y_axis_var='Corr_Abs'):
        """This does the actual plotting"""

        wells = []
        for s in strains:
            
            wells.append(wells_dict[s])
            
        wells = np.ravel(wells)
        df0 = pd.DataFrame(columns=df.columns)
        for well in wells:
            df0 = df0.append(df[df['Well']==well], ignore_index = True) 

        g = sns.FacetGrid(df0, col='Media', row='Sample', aspect=2, col_order=["sup", "tp", "tsb"], sharey=False)
        g = (g.map(sns.scatterplot, 't[m]', y_axis_var).set_titles('{row_name} - {col_name}'))
        
        plt.tight_layout()

        return g
    
    
    # below are the different types of input that can be used by
    # a user, this is parsed and pass to plot
    
    # dictionary of strains and the wells they're mapped to
    wells_dict = create_strain_well_dictionary(df)
    if type(strains)==str:
        strains = [strains]
        
        return plot(strains, df)
    
    else:
        assert type(strains)== list

        return plot(strains, df)


########
# MAIN #
########

def main():
    """
    Workhorse
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-in', '--data_folder',
                        required=True,
                        help="Path to tecan excel data")
    
    parser.add_argument('-mdf', '--metadata_folder',
                        required=True,
                        help='Path to metadata')
    
    parser.add_argument('-out','--output_folder',
                        required=True,
                        help='Path to save plots and parsed csv')

    parser.add_argument('-p', '--plot',
                        default=False,
                        help = 'returns the plots for all your data if True')
    
    parser.add_argument('-sl', '--strain_list',
                        required=False,
                        nargs='+',
                        help='returns plot for selected strains--do NOT separate with commas')
    
    args = parser.parse_args()
    
    dfp = parse_tecan_results_files(args.data_folder)
    mdf = parse_metadata(args.metadata_folder)
    
    
    df_mean, df_regs = remove_blank_signal(dfp, mdf)
    
    
    try:
        
        df_mean.to_csv(os.path.join(args.output_folder, 'parsed_kinetic.csv'))
        df_regs.to_csv(os.path.join(args.output_folder, 'parsed_kinetic_region.csv'))
        
               
    except:
        
        os.mkdir(args.output_folder)
        df_mean.to_csv(os.path.join(args.output_folder, 'parsed_kinetic.csv'))
        df_regs.to_csv(os.path.join(args.output_folder, 'parsed_kinetic_region.csv'))


    if args.plot:
        """
        return all growth scatterplots in 96W (8 x 12) and media (32 x 3) formats
        """
        
        g = plot_all_96W_format(df_mean)
        g.savefig(os.path.join(args.output_folder, 'all_strains_96W_format.png'))
        
        g = plot_all_strains_media_format(df_mean)
        g.savefig(os.path.join(args.output_folder, 'all_strains_media_format.png'))
        
        g = plot_all_regions_96W_format(df_regs)
        g.savefig(os.path.join(args.output_folder, 'all_strains_regions_96W_format.png'))

    if args.strain_list:
        """
        return growth scatterplots in media (n x 3) format
        """
        print(args.strain_list)
        g = plot_select_strains_media_format(args.strain_list, df_mean)
        g.savefig(os.path.join(args.output_folder, 'spc_strains_rename.png'))
        
if __name__ == '__main__':
    
    main()
    
    
    
    
