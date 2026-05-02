import os
import argparse
import pwlf
import pandas as pd
import numpy as np

DYNA_ID = 'dyna_id'
ENTITY_ID = 'entity_id'
FLOW = 'traffic_flow'
SPEED = 'traffic_speed'
OCCUPANCY = 'traffic_occupancy'
FREE_SPEED = 'vf'
CONGESTION_DEGREE = 'p_cong'


def fd_param_identification(rhos, qs):
    model_pwlf = pwlf.PiecewiseLinFit(rhos, qs)
    breaks = model_pwlf.fit(2)
    rho_c = breaks[1]
    slopes = model_pwlf.slopes
    vf = slopes[0]
    w = slopes[1]
    b = model_pwlf.intercepts[1]
    return rho_c, vf, w, b

def entity_param_identification(raw_data ,id):
    id_data = raw_data[raw_data[ENTITY_ID] == id]
    param_idt_data = id_data.iloc[:len(id_data)//2]

    qs = param_idt_data[FLOW].to_numpy() # Flow
    vs = param_idt_data[SPEED].to_numpy() # Speed
    rhos = qs / vs # Density
    [rho_c, vf, w, b] = fd_param_identification(rhos, qs)
    result = {
        DYNA_ID: id,
        'type': 'phy',
        ENTITY_ID: id,
        'rho_c': rho_c,
        'vf': vf,
        'w': w,
    }
    return result

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



def main(data_name):
    base_path = './raw_data'
    data_path = f'{base_path}/{data_name}/{data_name}.dyna'

    raw_data = pd.read_csv(data_path)

    entity_list = raw_data[ENTITY_ID].unique()

    results = []
    for id in entity_list:
        flag = True
        while flag:
            try:
                result = entity_param_identification(id)
                results.append(result)
                flag = False
            except Exception as e:
                print(id)

    df = pd.DataFrame(results)
    ensure_dir(f'{base_path}/{data_name}')
    df.to_csv(f'{base_path}/{data_name}/{data_name}.nodefd', index=False)

    mapping = {}
    for item in results:
        mapping[item[ENTITY_ID]] = item[FREE_SPEED]

    cong_data = raw_data.copy()

    cong_data[FREE_SPEED] = cong_data[ENTITY_ID].map(mapping)

    cong_data[CONGESTION_DEGREE] = np.where(
        cong_data[SPEED] >= cong_data[FREE_SPEED],
        0.0,
        1 - (cong_data[SPEED] / cong_data[FREE_SPEED])
    )

    cong_data = cong_data.drop([FLOW, OCCUPANCY, SPEED, FREE_SPEED], axis=1)
    cong_data.to_csv(f'{base_path}/{data_name}/{data_name}.ext', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str, help='the name of dataset')
    args = parser.parse_args()
    data_name = args.dataset
    main(data_name)

