from data_define import Storage
import pandas as pd
import numpy as np
import ast

class DataReader_pozo:
    """Read data stored in the format used by Pozo (2023)."""
    def __init__(self, data_path: str):
        self.data_path = 'input/' + data_path + '/'
        self.storages: list[Storage] = []

        # Read all input data
        self._read_data()

    def _read_data(self):
        # Read system-level data
        self._read_system_data()

        # Read ESS-unit data
        self._read_storage_data()

    def _read_system_data(self):
        # Read demand data
        demand_df = pd.read_csv(self.data_path + 'demand_profile.csv')
        self.load = demand_df['value'].to_numpy(dtype=float)
        self.T = len(self.load)

        # Read PV data
        pv_wind_df = pd.read_csv(self.data_path + 'PV_and_Wind_data_scenarios.csv')
        # pv_df = pv_wind_df[
        #     (pv_wind_df['Source'] == 'PV') & (pv_wind_df['Day'] == 2019)
        # ]  # Reserved interface for filtering by source type and date
        pv_df = pv_wind_df[
            (pv_wind_df['Source'] == 'PV')
        ]
        self.pv_power_factor = np.asarray(
            pv_df['Power'].map(ast.literal_eval).tolist(), dtype=float
        )
        self.pv_capacity = 50
        # self.pv_capacity = 35
        self.pv_power = self.pv_capacity * self.pv_power_factor
        self.signal = self.load - self.pv_power

        # Read electricity-price data
        scenarios_prices_df = pd.read_excel(self.data_path + 'scenarios_prices.xlsx')
        price = scenarios_prices_df.iloc[:, 1:].to_numpy(dtype=float).T
        self.price_dis = price
        self.price_ch = -price
        # Alternative random-price generation retained as a reserved interface
        # rng = np.random.default_rng(seed=42)
        # self.price_dis = -rng.uniform(10, 50, self.T)
        # self.price_ch= - self.price_dis
        # self.price_ch = rng.uniform(10, 50, self.T)

    def _read_storage_data(self):
        """Read ESS-unit data."""
        storage_df = pd.read_csv(self.data_path + 'ESS_data_SPTP.csv')
        # Read ESS-unit data
        self.ng = storage_df.shape[0]
        for g in range(self.ng):
            row = storage_df.iloc[g]
            es = Storage()
            es.id = g
            es.pch_max = float(row['PcMax'])
            es.pdis_max = float(row['PdMax'])
            es.eff_ch = float(row['eta_c'])
            es.eff_dis = float(row['eta_d'])
            es.emax = float(row['Emax'])
            es.emin = float(row['Emin'])
            es.init_e = float(row['E0'])
            self.storages.append(es)
