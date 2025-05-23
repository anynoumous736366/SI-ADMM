import numpy as np
import pandas as pd
import pickle


def build_slice_x(disease_name, dd_disease_data, non_binary):
    result = pd.DataFrame(np.zeros((num_drug, num_drug), dtype=int), columns=list(all_drugs), index=list(all_drugs))
    data = dd_disease_data[dd_disease_data['ICD_top_level_name'] == disease_name]
    for index, row in data.iterrows():
        drug1 = row['drugbank_id_A']
        drug2 = row['drugbank_id_B']
        if non_binary:
            result.at[drug1, drug2] = int(row['Combination_Count'])
            result.at[drug2, drug1] = int(row['Combination_Count'])
        else:
            result.at[drug1, drug2] = 1
            result.at[drug2, drug1] = 1

    return result


def build_slice_y(ddi_name, ddi_data):
    result = pd.DataFrame(np.zeros((num_drug, num_drug), dtype=int), columns=list(all_drugs), index=list(all_drugs))
    data = ddi_data[ddi_data['Y'] == ddi_name]
    for index, row in data.iterrows():
        drug1 = row['Drug1_ID']
        drug2 = row['Drug2_ID']
        result.at[drug1, drug2] = 1
        result.at[drug2, drug1] = 1

    return result


if __name__ == '__main__':
    non_binary = False

    x_file_name = 'cdcdb'
    y_file_name = 'tdc'

    print(f'x data: {x_file_name}, y data: {y_file_name}')
    # load dd disease and ddi data
    dd_disease_data = pd.read_csv(f'../data/CDCDB/processed/{x_file_name}.csv')
    ddi_data = pd.read_csv(f'../data/DDI/{y_file_name}.csv')

    # drop unnecessary data
    ddi_data.drop(columns=['Drug1', 'Drug2'], inplace=True)
    dd_disease_data.drop(columns=['drugA', 'drugB'], inplace=True)

    # get unique drugs
    unqiue_drugs_dd_disease = pd.unique(dd_disease_data[['drugbank_id_A', 'drugbank_id_B']].values.ravel())
    unique_drugs_ddi = pd.unique(ddi_data[['Drug1_ID', 'Drug2_ID']].values.ravel())

    # Finding the intersection
    common_drugs = set(unqiue_drugs_dd_disease) & set(unique_drugs_ddi)

    # filter out the entries that are not in the list of intersected drugs
    dd_disease_filter = (dd_disease_data['drugbank_id_A'].isin(common_drugs)) & (dd_disease_data['drugbank_id_B'].isin(common_drugs))
    dd_disease_filtered = dd_disease_data[dd_disease_filter]

    ddi_filter = (ddi_data['Drug1_ID'].isin(common_drugs)) & (ddi_data['Drug2_ID'].isin(common_drugs))
    ddi_filtered = ddi_data[ddi_filter]

    # get the final unique set of drugs
    result_ids = set(pd.unique(dd_disease_filtered[['drugbank_id_A', 'drugbank_id_B']].values.ravel())) & \
                 set(pd.unique(ddi_filtered[['Drug1_ID', 'Drug2_ID']].values.ravel()))

    # save the drug lists
    all_drugs = np.array(list(result_ids))
    num_drug = len(all_drugs)
    all_drugs_dict = {'drugs': all_drugs}
    pd.DataFrame(all_drugs_dict).to_csv(f'../data/{x_file_name}_{y_file_name}_all_drugs.csv', index=False)
    np.save(f'../data/{x_file_name}_{y_file_name}_all_drugs.npy', all_drugs)
    # filter again and save the data
    dd_disease_filter = (dd_disease_data['drugbank_id_A'].isin(all_drugs)) & (dd_disease_data['drugbank_id_B'].isin(all_drugs))
    dd_disease_filtered = dd_disease_data[dd_disease_filter]
    dd_disease_filtered.to_csv(f'../preprocessing_data/{x_file_name}_dddisease_filtered.csv', index=False)
    print(dd_disease_filtered.columns)
    print(f"x data has {dd_disease_filtered.shape[0]} rows")

    ddi_filter = (ddi_data['Drug1_ID'].isin(all_drugs)) & (ddi_data['Drug2_ID'].isin(all_drugs))
    ddi_filtered = ddi_data[ddi_filter]
    ddi_filtered.to_csv(f'../preprocessing_data/{y_file_name}_ddi_filtered.csv', index=False)
    print(f"y data has {ddi_filtered.shape[0]} rows")

    # get all the disease types
    disease_types = dd_disease_filtered['ICD_top_level_name'].unique()
    num_disease = len(disease_types)

    # build the x tensor
    x_tensor = {}
    for index, disease in enumerate(disease_types):
        x_tensor[str(index) + ' ' + disease] = build_slice_x(disease, dd_disease_filtered, non_binary)

    # save the x tensor
    if not non_binary:
        with open(f'../data/{x_file_name}_{y_file_name}_x.pickle', 'wb') as handle:
            pickle.dump(x_tensor, handle, protocol=4)
        with open(f'../baseline/ADMM/{x_file_name}_{y_file_name}_x.pickle', 'wb') as handle:
            pickle.dump(x_tensor, handle, protocol=4)
    else:
        with open(f'../data/{x_file_name}_{y_file_name}_x.pickle', 'wb') as handle:
            pickle.dump(x_tensor, handle, protocol=4)
        with open(f'../baseline/ADMM/{x_file_name}_{y_file_name}_z.pickle', 'wb') as handle:
            pickle.dump(x_tensor, handle, protocol=4)

    # print out the stats
    print('total number of drugs: {}'.format(num_drug))
    print('number of disease types: {}'.format(num_disease))
    print('total valid pair of drug disease: {}'.format(dd_disease_filtered.shape[0]))
    print('finish building tensor for drug drug disease!, sparsity: {}'
          .format(dd_disease_filtered.shape[0] / (num_drug * num_drug * num_disease)))

    # get all the DDI types
    ddi_types = ddi_filtered['Y'].unique()
    num_ddi_types = len(ddi_types)

    # build the y tensor
    y_tensor = {}
    for index, ddi_type in enumerate(ddi_types):
        y_tensor[str(index) + ' ' + str(ddi_type)] = build_slice_y(ddi_type, ddi_filtered)

    # save the y tensor
    with open(f'../data/{x_file_name}_{y_file_name}_y.pickle', 'wb') as handle:
        pickle.dump(y_tensor, handle, protocol=4)
    with open(f'../baseline/ADMM/{x_file_name}_{y_file_name}_y.pickle', 'wb') as handle:
        pickle.dump(y_tensor, handle, protocol=4)

    # print out the stats
    print('number of ddi types: {}'.format(num_ddi_types))
    print('total valid pair of ddi: {}'.format(ddi_filtered.shape[0]))
    print('finish building tensor for drug drug interaction!, sparsity: {}'
          .format(ddi_filtered.shape[0]/(num_drug*num_drug*num_ddi_types)))