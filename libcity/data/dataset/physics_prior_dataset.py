import os
import datetime
import numpy as np
import pandas as pd
from libcity.data.dataset.pdformer_dataset import PDFormerDataset


class PhysicsPriorDataset(PDFormerDataset):

    def __init__(self, config):
        super().__init__(config)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'pdformer_point_phy_{}.npz'.format(self.parameters_str))
    
    def _load_ext(self):
        return self._load_ext_3d()
    
    def _load_ext_3d(self):
        self._logger.info("Loading file " + self.ext_file + '.ext')
        extfile = pd.read_csv(self.data_path + self.ext_file + '.ext')
        if self.ext_col != '':
            if isinstance(self.ext_col, list):
                ext_col = self.ext_col.copy()
            else:
                ext_col = [self.ext_col].copy()
            ext_col.insert(0, 'time')
            ext_col.insert(1, 'entity_id')
            extfile = extfile[ext_col]
        else:
            extfile = extfile[extfile.columns[2:]]
        self.timesolts = list(extfile['time'][:int(extfile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not extfile['time'].isna().any():
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        feature_dim = len(extfile.columns) - 2
        df = extfile[extfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i+len_time].values)
        data = np.array(data, dtype=np.float)
        data = data.swapaxes(0, 1)
        self._logger.info("Loaded file " + self.ext_file + '.ext' + ', shape=' + str(data.shape))
        return data

    def _add_external_information(self, df, ext_data=None):
        num_samples, num_nodes, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, dayofweek] = 1
            data_list.append(day_in_week)
        if ext_data is not None:
            data_list.append(ext_data)
        return np.concatenate(data_list, axis=-1)

    def _generate_data(self):
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()
        ext_data = None
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):
                ext_data = self._load_ext()
        x_list, y_list = [], []
        for filename in data_files:
            df = self._load_dyna(filename)
            self._logger.info("=" * 50)
            self._logger.info("Generating dataset from file " + filename + '.dyna')
            self._logger.info("Original data shape: " + str(df.shape))
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            x, y = self._generate_input_data(df)
            x_list.append(x)
            y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y
    
