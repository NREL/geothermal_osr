# -*- coding: utf-8 -*-
"""Library-style code used for OSR modeling & analysis.

Written by: Dmitry Duplyakin (dmitry.duplyakin@nrel.gov)
in collaboration with the National Renewable Energy Laboratories.
"""

import sys
import logging
import os
import importlib.util
import pandas as pd
import numpy as np
from datetime import timedelta
from pyXSteam.XSteam import XSteam
from sklearn import preprocessing
from scipy.optimize import curve_fit
import math
import inspect

sys.path.append('../util')
import util
from polynomial import get_polynomial_func

loglevel = 'WARNING'
logging.basicConfig(level=os.environ.get("LOGLEVEL", loglevel))


class Reservoir:
    """This class allows capturing data for a single simulated scenario.

    The class includes data & methods for a single scenario simulated
    for a geothermal reservoir. It supports data in CMG and TETRAD-G formats.

    Args:
        config (str): Path to Python script with configuration options.

    Attributes:
        timeseries (dataframe): Raw timeseries for the loaded scenario.
        scaled_timeseries (dataframe): Scaled timeseries; available after
            calling scale_values().
    """

    def __init__(self, config, energy_calc=True, from_existing=None):
        if from_existing:
            """ This works similar to a copy constructor """ 
            self.config = from_existing.config
            self.name_mapper = from_existing.name_mapper
            self.timeseries = from_existing.timeseries.copy()
            self.scaled_timeseries = from_existing.scaled_timeseries.copy()
            
            self.gather_arrays()
            self.ranges = self.get_ranges()
            if energy_calc:
                self.estimate_energy_and_exergy()
                self.calculate_cumulative()
        else:
            self.config = config
            self.name_mapper = self.__get_mapper()
            self.timeseries = self.__get_timeseries()
            self.scaled_timeseries = pd.DataFrame()

            self.gather_arrays()
            self.ranges = self.get_ranges()
            if energy_calc:
                self.estimate_energy_and_exergy()
                self.calculate_cumulative()

    def __get_mapper(self):
        """Load name mapper from file specified in constructor."""
        # Importing a module by path; this implementation works for Python 3.5+
        mapper_spec = importlib.util.\
            spec_from_file_location("mapper",
                                    self.config.mapper_path)
        mapper = importlib.util.module_from_spec(mapper_spec)
        mapper_spec.loader.exec_module(mapper)
        return mapper.name_mapper

    def __get_timeseries(self):
        """Load historic or simulted reservoir data from a file."""
        if self.config.input_type == "CMG":
            # skiprows=2 was used for older CMG batches
            df = pd.read_excel(self.config.timeseries_file, skiprows=0)\
                .rename(columns=self.name_mapper)
            # Maybe comment out the following line if there are issues with Date
            #print(df.columns)
            #print(df.head(2).T)
            df["Date"] = pd.to_datetime(df["Date"], format='%y-%m-%d')
        elif self.config.input_type == "TETRADG":
            df = pd.read_excel(self.config.timeseries_file, skiprows=0)\
                .rename(columns=self.name_mapper)
            df["Date"] = df.t_day_raw.apply(lambda x: pd.to_datetime('2020-01-01') + timedelta(days=x))

        df.index = pd.to_datetime(df["Date"])

        if self.config.additional_timeseries_file:
            add_df = pd.read_excel(self.config.additional_timeseries_file,
                                   skiprows=2).rename(columns=self.name_mapper)
            add_df.index = pd.to_datetime(add_df["Date"])
            # These columns will become redundant after merging
            add_df.drop(columns=["Date", "t_day_raw"], inplace=True)
            df = pd.merge(df, add_df, left_index=True, right_index=True)

        df["t_epoch"] = (df.index.astype('int64')//1e9).astype('int64')

        return df

    def gather_arrays(self):
        """Form arrays with quantity names for producers & injectors."""
        self.gather_producer_arrays()
        self.gather_injector_arrays()

    def gather_producer_arrays(self):
        """Form convenience arrays with quantity names for all producers."""
        # String manipulation below needs to be coordinated with mapper values
        # (i.e., "pt" used for producer temperatures,
        # "pm" -- producer mass flow, "pp" -- producer pressure)
        temp_prefix = "pt"
        flow_prefix = "pm"
        pres_prefix = "pp"

        temp_ids = sorted([int(c.replace(temp_prefix, ""))
                           for c in self.timeseries.columns if
                           temp_prefix in c and c in
                           self.name_mapper.values()])
        flow_ids = sorted([int(c.replace(flow_prefix, ""))
                           for c in self.timeseries.columns if
                           flow_prefix in c and c in
                           self.name_mapper.values()])
        pres_ids = sorted([int(c.replace(pres_prefix, ""))
                           for c in self.timeseries.columns if
                           pres_prefix in c and c in
                           self.name_mapper.values()])
        logging.debug("temp_ids:" + str(temp_ids))
        logging.debug("flow_ids:" + str(flow_ids))
        logging.debug("pres_ids:" + str(pres_ids))

        if (temp_ids != flow_ids) or (flow_ids != pres_ids):
            raise ValueError("Mismatch between producers' temperature, "
                             "flow, and pressure inputs.")
        else:
            logging.debug("Match between producers' temperature, "
                          "flow, and pressure inputs is confirmed.")

        self.all_producer_temp = [temp_prefix + str(idx) for idx in temp_ids]
        self.all_producer_flow = [flow_prefix + str(idx) for idx in flow_ids]
        self.all_producer_pres = [pres_prefix + str(idx) for idx in pres_ids]

    def gather_injector_arrays(self):
        """Form convenience arrays with quantity names for all injectors."""
        # Like in the gather_producer_arrays(), this string manipulation
        #  needs to be coordinated with mapper values
        # (i.e., "it" used for injector temperatures,
        # "im" -- injector mass flow, "ip" -- injector pressure)
        temp_prefix = "it"
        flow_prefix = "im"
        pres_prefix = "ip"

        temp_ids = sorted([int(c.replace(temp_prefix, ""))
                           for c in self.timeseries.columns if
                           temp_prefix in c and c in
                           self.name_mapper.values()])
        flow_ids = sorted([int(c.replace(flow_prefix, ""))
                           for c in self.timeseries.columns if
                           flow_prefix in c and c in
                           self.name_mapper.values()])
        pres_ids = sorted([int(c.replace(pres_prefix, ""))
                           for c in self.timeseries.columns if
                           pres_prefix in c and c in
                           self.name_mapper.values()])
        logging.debug("temp_ids:" + str(temp_ids))
        logging.debug("flow_ids:" + str(flow_ids))
        logging.debug("pres_ids:" + str(pres_ids))

        # if (temp_ids != flow_ids) or (flow_ids != pres_ids):
        #     raise ValueError("Mismatch between injectors' temperature, "
        #                      "flow, and pressure inputs.")
        # else:
        #     logging.debug("Match between injectors' temperature, "
        #                   "flow, and pressure inputs is confirmed.")

        self.all_injector_temp = [temp_prefix + str(idx) for idx in temp_ids]
        self.all_injector_flow = [flow_prefix + str(idx) for idx in flow_ids]
        self.all_injector_pres = [pres_prefix + str(idx) for idx in pres_ids]

    def get_ranges(self, quantity_list=None):
        """Return df w/ (min, max) for values of all/specified quantities."""
        ranges = {}
        if not quantity_list:
            # If list isn't specified, get ranges for all quantitie
            quantity_list = self.all_injector_temp + self.all_injector_flow + \
                            self.all_injector_pres + \
                            self.all_producer_temp + self.all_producer_flow + \
                            self.all_producer_pres

        for q in quantity_list:
            ranges[q] = (self.timeseries[q].min(), self.timeseries[q].max())
        return pd.DataFrame(ranges, index=["min", "max"]).T

    def estimate_energy_and_exergy(self):
        """Calculate energy and exergy timeseries."""
        # Reference for using XSteam: https://pypi.org/project/pyXSteam/
        steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

        p0 = 1  # 1 bar
        T0 = 20  # 20 degrees C
        T0_K = T0 + 273.15

        # h_pt--Enthalpy as a function of pressure and temperature
        h0 = steamTable.h_pt(p0, T0)  # dead state enthalpy
        #print("h0:", h0)

        # s_pt -- Specific entropy as a function of pressure and temperature
        # (Returns saturated vapor enthalpy if mixture)
        s0 = steamTable.s_pt(p0, T0)  # dead state exergy

        # Assume 10 bar and 69 deg C power plant exit conditions
        # (some gets reinjected, some gets disposed off elsewhere)
        hinj = steamTable.h_pt(10, 69)
        #print("hinj:", hinj)

        # Empty arrays
        exergy = pd.Series(0, index=self.timeseries.index)
        energy = pd.Series(0, index=self.timeseries.index)

        # Goes by time
        for idx, row in self.timeseries.iterrows():

            # Old code with well selection based on specific flow thresholds:
            #
            # indicestotake = (row[self.all_producer_flow] > 2.0).values & \
            #                 (row[self.all_producer_temp] > 1.0).values
            #
            # flow_selected = [f for f, mask in
            #                  zip(self.all_producer_flow, indicestotake)
            #                  if mask]
            # temp_selected = [t for t, mask in
            #                  zip(self.all_producer_temp, indicestotake)
            #                  if mask]
            # pres_selected = [p for p, mask in
            #                  zip(self.all_producer_pres, indicestotake)
            #                  if mask]

            # Selecting all wells regargless of their flows
            flow_selected = self.all_producer_flow
            temp_selected = self.all_producer_temp
            pres_selected = self.all_producer_pres
            #print("flow_selected:", flow_selected)

            if len(flow_selected) > 0:
                # This should be constant over time
                TotalProdFlow = row[flow_selected].to_numpy().sum()

                # Average production temperature weighted by flow rate
                if TotalProdFlow > 0.000001:
                    AvgProdTemp = np.multiply(row[flow_selected].to_numpy(),
                                              row[temp_selected].to_numpy())\
                        .sum() / TotalProdFlow
                else:
                    AvgProdTemp = 0

                #print(idx, TotalProdFlow, AvgProdTemp)

                # Calculate production water enthalpy in kJ/kg
                # (needs pressure in bar and temperature in degrees C)
                # *** Confirmed that units are correct!
                h = steamTable.h_pt(self.config.geofluid_surface_pressure,
                                    AvgProdTemp)
                # Calculate production water exergy in kJ/kg/K
                s = steamTable.s_pt(self.config.geofluid_surface_pressure,
                                    AvgProdTemp)
                # Calculate production water density in kg/m3
                rho = steamTable.rho_pt(self.config.geofluid_surface_pressure,
                                        AvgProdTemp)
                #print(idx, h, s, rho)

                if self.config.flow_unit == "kKg/hour":
                    # Unit of exergy: MW
                    exergy.loc[idx] = TotalProdFlow/3600.0 * \
                        (h - h0 - T0_K*(s - s0))
                    #print(idx, "Exergy:", exergy.loc[idx], ";", TotalProdFlow, h, h0, T0_K, s, s0)
                elif self.config.flow_unit == "kKg/day":
                    exergy.loc[idx] = TotalProdFlow/24.0/3600.0 * \
                        (h - h0 - T0_K*(s - s0))
                elif self.config.flow_unit == "kg/day":
                    exergy.loc[idx] = TotalProdFlow/24.0/3600.0 * \
                        (h-h0-T0_K*(s-s0))/1000.0
                elif self.config.flow_unit == "m^3/day":
                    exergy.loc[idx] = TotalProdFlow/24.0/3600.0 * \
                        rho*(h-h0-T0_K*(s-s0))/1000.0
                else:
                    raise ValueError("Exergy Estimation: "
                                     "Unsupported unit "
                                     "is used for flow values."
                                     "Currently supported: "
                                     "kKg/day, kg/day, and m^3/day.")
            #print("Time:", idx)
            individual_energy_components = []
            for flow_col, temp_col, pres_col in \
                    zip(flow_selected, temp_selected, pres_selected):
                Flow = row[flow_col]
                Temp = row[temp_col]
                Pres = row[pres_col]

                # Calculate production water enthalpy in kJ/kg
                # (needs pressure in bar and temperature in degrees C)
                # *** 0.01: converting pressure from kPa to bar
                h = steamTable.h_pt(Pres*0.01, Temp)

                # Calculate production water exergy in kJ/kg/K
                s = steamTable.s_pt(Pres*0.01, Temp)

                # Calculate production water density in kg/m3
                rho = steamTable.rho_pt(Pres*0.01, Temp)

                if self.config.flow_unit == "kKg/hour":
                    # Unit of component: MW
                    individual_energy_components.append(
                        Flow/3600.0*(h-hinj))
                    #print("\t", individual_energy_components[-1])
                elif self.config.flow_unit == "kKg/day":
                    individual_energy_components.append(
                        Flow/24.0/3600.0*(h-hinj))
                elif self.config.flow_unit == "kg/day":
                    individual_energy_components.append(
                        Flow/24.0/3600.0*(h-hinj)/1000.0)
                elif self.config.flow_unit == "m^3/day":
                    individual_energy_components.append(
                        Flow/24.0/3600.0*rho*(h-hinj)/1000.0)
                else:
                    raise ValueError("Energy Estimation: "
                                     "Unsupported unit "
                                     "is used for flow values."
                                     "Currently supported: "
                                     "kKg/day, kg/day, and m^3/day.")

            energy.loc[idx] = np.sum(individual_energy_components)
            #print("\t\tSum:", np.sum(individual_energy_components))

        self.exergy = exergy
        self.energy = energy

    def _cumulative(self, s):
        res = pd.Series(0, index=s.index)
        for idx in range(1, len(s)):
            avg_value = (s[idx - 1] + s[idx]) / 2
            duration = (s.index[idx] - s.index[idx - 1]).total_seconds() / \
                3600.0
            res.loc[s.index[idx]] = avg_value * duration
        return res.cumsum()

    def calculate_cumulative(self):

        self.exergy_cumulative = self._cumulative(self.exergy)
        self.energy_cumulative = self._cumulative(self.energy)

    def add_starting_cumulative_energy(self, starting_value):
        self.energy_cumulative += starting_value

    def add_starting_cumulative_exergy(self, starting_value):
        self.exergy_cumulative += starting_value

    def scale_values(self, scaler=None, type="temp prod"):
        """Create scaled_timeseries dataframe with scaled quantity values."""
        if type == "temp prod":
            columns = self.all_producer_temp
            max_value = self.config.temp_scaled_max
        elif type == "temp inj":
            columns = self.all_injector_temp
            max_value = self.config.temp_scaled_max
        elif type == "pres prod":
            columns = self.all_producer_pres
            max_value = self.config.pres_scaled_max
        elif type == "pres inj":
            columns = self.all_injector_pres
            max_value = self.config.pres_scaled_max
        elif type == "flow prod":
            columns = self.all_producer_flow
            max_value = self.config.flow_scaled_max
        elif type == "flow inj":
            columns = self.all_injector_flow
            max_value = self.config.flow_scaled_max
        else:
            raise ValueError("Bad scaling type specified")

        if self.scaled_timeseries.empty:
            self.scaled_timeseries = self.timeseries.copy()

        if not scaler:
            # Use the range [0, max_value] where max_value comes from config

            curr_scaler = preprocessing.MinMaxScaler(
                feature_range=(0, max_value))

            # Use all wells for scaler fitting
            _ = curr_scaler.fit_transform(
                self.timeseries[columns].to_numpy()
                .reshape(-1, 1))

            # Scale one column at a time using the fitted scaler
            for c in columns:
                self.scaled_timeseries[c] = curr_scaler.transform(
                    self.timeseries[c].to_numpy().reshape(-1, 1))

            if "temp" in type:
                self.temp_scaler = curr_scaler
            elif "pres" in type:
                self.pres_scaler = curr_scaler
            elif "flow" in type:
                self.flow_scaler = curr_scaler
        else:
            # Use specified scaler (no fitting needed)
            curr_scaler = scaler
            for c in columns:
                self.scaled_timeseries[c] = curr_scaler.transform(
                    self.timeseries[c].to_numpy().reshape(-1, 1))

# For initial testing:
# bhs = Reservoir(config)
# print(bhs.name_mapper)
# print(bhs.timeseries.head())
# print(bhs.energy.head())


class ReservoirPredictionEnsemble:
    """This class allows manipulating data for multiple similated scenarios.

    The class is built upon a list of Reservoir objects and implements
    methods for data preparation and manipulation.

    Args:
        config (str): Path to Python script with configuration options.
        ensemble (list of Reservoir objects): all studied scenarios that
            correspond to a particular configuration of the used simulator.
    """

    def __init__(self, config, ensemble):
        self.config = config
        self.ensemble = ensemble
        self.count = len(self.ensemble)
        self.common_temp_scaler = None
        self.common_pres_scaler = None

        for idx, case in enumerate(self.ensemble[1:]):
            if case.name_mapper != self.ensemble[idx-1].name_mapper:
                raise ValueError("Mismatch in name mappers used in ensemble")
        self.common_name_mapper = self.ensemble[0].name_mapper

    def __getitem__(self, item):
        """Access individual scenario using format: <object>[0]."""
        return self.ensemble[item]

    def equalize_cumulative_values(self, historic_scenario_index=0):
        """Equalizes cumulative energy and exergy estimates across scenarios.

        It takes scenario at historic_scenario_index and adds its cumulative
        exergy and energy to other scenarios' cumulative timeseries.
        It does it to make the scenarios that cover, e.g., 2020-2040, match
        the scenario that also covers historic data, e.g., 1979-2040.
        """
        for idx, case in enumerate(self.ensemble):
            if idx != historic_scenario_index:

                # Take last cumulative values before/at the beginning
                # of timeseries for this case
                starting_energy = \
                    self.ensemble[historic_scenario_index].energy_cumulative[
                        self.ensemble[historic_scenario_index].energy.index <=
                        case.energy.index[0]].values[-1]
                starting_exergy = \
                    self.ensemble[historic_scenario_index].exergy_cumulative[
                        self.ensemble[historic_scenario_index].exergy.index <=
                        case.exergy.index[0]].values[-1]

                case.add_starting_cumulative_energy(starting_energy)
                case.add_starting_cumulative_exergy(starting_exergy)

    def scale(self, historic_scenario_index=0, scalers=None):
        """Implements consistent scaling for temperatures, pressures, and flows.

        Scalers are 'fitted' on the data for the scenario at
        historic_scenario_index and applied to other scenarios.
        If dict scalers is provided, no fitting will be done and specified
        scalers will be used.
        """
        if scalers:
            self.common_temp_scaler = scalers["temp"]
            for idx, case in enumerate(self.ensemble):
                case.scale_values(scaler=self.common_temp_scaler,
                                  type="temp prod")
                case.scale_values(scaler=self.common_temp_scaler,
                                  type="temp inj")

            self.common_pres_scaler = scalers["pres"]
            for idx, case in enumerate(self.ensemble):
                case.scale_values(scaler=self.common_pres_scaler,
                                  type="pres prod")
                case.scale_values(scaler=self.common_pres_scaler,
                                  type="pres inj")

            self.common_flow_scaler = scalers["flow"]
            for idx, case in enumerate(self.ensemble):
                case.scale_values(scaler=self.common_flow_scaler,
                                  type="flow prod")
                case.scale_values(scaler=self.common_flow_scaler,
                                  type="flow inj")
        else:
            self.ensemble[historic_scenario_index].scale_values(type="temp prod")
            self.common_temp_scaler = \
                self.ensemble[historic_scenario_index].temp_scaler
            for idx, case in enumerate(self.ensemble):
                if idx != historic_scenario_index:
                    case.scale_values(scaler=self.common_temp_scaler,
                                      type="temp prod")
            for idx, case in enumerate(self.ensemble):
                case.scale_values(scaler=self.common_temp_scaler, type="temp inj")

            self.ensemble[historic_scenario_index].scale_values(type="pres prod")
            self.common_pres_scaler = \
                self.ensemble[historic_scenario_index].pres_scaler
            for idx, case in enumerate(self.ensemble):
                if idx != historic_scenario_index:
                    case.scale_values(scaler=self.common_pres_scaler,
                                      type="pres prod")
            for idx, case in enumerate(self.ensemble):
                case.scale_values(scaler=self.common_pres_scaler, type="pres inj")

            self.ensemble[historic_scenario_index].scale_values(type="flow prod")
            self.common_flow_scaler = \
                self.ensemble[historic_scenario_index].flow_scaler
            for idx, case in enumerate(self.ensemble):
                if idx != historic_scenario_index:
                    case.scale_values(scaler=self.common_flow_scaler,
                                      type="flow prod")
            for idx, case in enumerate(self.ensemble):
                case.scale_values(scaler=self.common_flow_scaler, type="flow inj")

    def shared_scaled_time_index(self, start_at='2020-01-01'):
        """ This returns a mapper (dict) from timestamps to scaled/normalized values"""
        all_t = []
        start_at_timestamp = pd.to_datetime(start_at)
        for idx in range(self.count):
            subset = self.ensemble[idx].scaled_timeseries[
                self.ensemble[idx].scaled_timeseries.index >= start_at_timestamp]
            #print("Scenario:", idx)
            all_t.extend(list(subset.index))
        all_t = pd.Series(all_t).unique()
        #print(all_t)

        # Convert timestamps back to epoch
        t_epoch = pd.Series([(dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's') for dt in all_t])

        t_min = t_epoch.min()
        t_max = t_epoch.max()
        t_normalized = (t_epoch - t_min) / (t_max - t_min)

        #t_index_mapper = pd.DataFrame({"t": all_t, "scaled": t_normalized})
        t_index_mapper = {k:v for k,v in zip(all_t, t_normalized)}

        return t_index_mapper

    def split_cases(self, printing=True):
        """ Split scenarios into three sets: train, validate, test."""
        all_cases = range(self.count)

        train_case_count = int(self.count * self.config.train_ratio)
        val_case_count = int(self.count * self.config.val_ratio)
        test_case_count = self.count - train_case_count - val_case_count

        train_indices = np.random.choice(all_cases, train_case_count,
                                         replace=False)
        remaining_indices = [i for i in all_cases
                             if i not in train_indices]
        val_indices = np.random.choice(remaining_indices, val_case_count,
                                       replace=False)
        test_indices = [i for i in remaining_indices
                        if i not in val_indices]
        if printing:
            print({"Train": train_indices,
                   "Validate": val_indices,
                   "Test": test_indices})
        return train_indices, val_indices, test_indices

    def learning_data(self, quantity, start_at='2020-01-01',
                      test_indices=None, add_time_channel=True):
        """ Prepare a number of data subsets needed for learning."""

        start_at_timestamp = pd.to_datetime(start_at)

        # This code supports both original quantity names,
        # as well as names after renaming
        if quantity in self.common_name_mapper.keys():
            quantity = self.common_name_mapper[quantity]
        elif quantity not in self.common_name_mapper.values():
            raise ValueError("Unsupported quantity requested.")

        for idx, case in enumerate(self.ensemble):
            if quantity not in case.scaled_timeseries.columns:
                raise ValueError("Can't find quantity data in case: %d." % idx)

        t_mapper = self.shared_scaled_time_index(start_at=start_at)

        all_cases = range(self.count)

        if not test_indices:
            train_case_count = int(self.count * self.config.train_ratio)
            val_case_count = int(self.count * self.config.val_ratio)
            test_case_count = self.count - train_case_count - val_case_count

            train_indices = np.random.choice(all_cases, train_case_count,
                                             replace=False)
            remaining_indices = [i for i in all_cases
                                 if i not in train_indices]
            val_indices = np.random.choice(remaining_indices, val_case_count,
                                           replace=False)
            test_indices = [i for i in remaining_indices
                            if i not in val_indices]
        else:
            remaining_indices = [i for i in all_cases
                                 if i not in test_indices]
            train_case_count = int(self.count * self.config.train_ratio)
            train_indices = np.random.choice(remaining_indices, train_case_count,
                                             replace=False)
            val_indices = [i for i in remaining_indices
                           if i not in train_indices]

        split_info = {"Train": train_indices,
                      "Validate": val_indices,
                      "Test": test_indices}
        print(split_info)

        add_channels = self.ensemble[0].all_injector_temp + \
            self.ensemble[0].all_injector_pres + \
            self.ensemble[0].all_injector_flow
        if add_time_channel:
            add_channels += ["t_normalized"]

        # --- Prepare training set ---
        X_train = []
        y_train = []

        for idx in train_indices:

            # Main timeseries being learned and predicted

            subset = self.ensemble[idx].scaled_timeseries[
                self.ensemble[idx].scaled_timeseries.index >= start_at_timestamp].copy()
            subset["t_normalized"] = np.array(subset.index.map(t_mapper))

            t = subset[quantity].values

            X, y = util.split_sequence(t, self.config.n_steps)

            for c in add_channels:

                vals = subset[c].to_numpy()
                # Account for n_steps at the beginning -- skip those values
                # to maintain matching of timesteps between timeseries
                vals = vals[self.config.n_steps:]

                # Make it a column that is ready to be appended
                vals = vals.reshape(-1, 1)

                X = np.append(X, vals, 1)

            if len(X_train) == 0:
                X_train = X
            else:
                X_train = np.vstack([X_train, X])
            if len(y_train) == 0:
                y_train = y
            else:
                y_train = np.hstack([y_train, y])

        print("Training dataset shapes:",
              X_train.shape, y_train.shape)
        #return X_train, y_train

        # --- Prepare validaton set ---

        X_val = []
        y_val = []
        X_val_dict = {}
        y_val_dict = {}

        for idx in val_indices:

            # Main timeseries being learned and predicted
            subset = self.ensemble[idx].scaled_timeseries[
                self.ensemble[idx].scaled_timeseries.index >= start_at_timestamp].copy()
            subset["t_normalized"] = np.array(subset.index.map(t_mapper))

            t = subset[quantity].values

            X, y = util.split_sequence(t, self.config.n_steps)

            for c in add_channels:

                vals = subset[c].to_numpy()
                # Account for n_steps at the beginning -- skip those values
                # to maintain matching of timesteps between timeseries
                vals = vals[self.config.n_steps:]

                # Make it a column that is ready to be appended
                vals = vals.reshape(-1, 1)

                X = np.append(X, vals, 1)

            if len(X_val) == 0:
                X_val = X
            else:
                X_val = np.vstack([X_val, X])
            if len(y_val) == 0:
                y_val = y
            else:
                y_val = np.hstack([y_val, y])

            X_val_dict[idx] = X
            y_val_dict[idx] = y

        print("Validation dataset shapes:", X_val.shape, y_val.shape)

        # --- Prepare testing set ---
        X_test = []
        y_test = []
        X_test_dict = {}
        y_test_dict = {}

        for idx in test_indices:

            # Main timeseries being learned and predicted
            subset = self.ensemble[idx].scaled_timeseries[
                self.ensemble[idx].scaled_timeseries.index >= start_at_timestamp].copy()
            subset["t_normalized"] = np.array(subset.index.map(t_mapper))

            t = subset[quantity].values

            X, y = util.split_sequence(t, self.config.n_steps)

            for c in add_channels:

                vals = subset[c].to_numpy()
                # Account for n_steps at the beginning -- skip those values
                # to maintain matching of timesteps between timeseries
                vals = vals[self.config.n_steps:]

                # Make it a column that is ready to be appended
                vals = vals.reshape(-1, 1)

                X = np.append(X, vals, 1)

            if len(X_test) == 0:
                X_test = X
            else:
                X_test = np.vstack([X_test, X])
            if len(y_test) == 0:
                y_test = y
            else:
                y_test = np.hstack([y_test, y])

            X_test_dict[idx] = X
            y_test_dict[idx] = y

        print("Testing dataset shapes:", X_test.shape, y_test.shape)

        # return X_train_combined, y_train_combined, X_val_combined,
        #     y_val_combined, X_val_dict, y_val_dict, X_test_combined,
        #     y_test_combined, X_test_dict, y_test_dict, n_add_ch

        # return ReservoirMLModel(self.config,
        #                         X_train, y_train,
        #                         X_val, y_val,
        #                         X_test, y_test,
        #                         add_channels)
        return ReservoirMLModel(self.config,
                                X_train, y_train,
                                X_val, y_val,
                                X_val_dict, y_val_dict,
                                X_test, y_test,
                                X_test_dict, y_test_dict,
                                add_channels,
                                split_info)

    def get_curve_approximations(self, quantity, curve_func,
                                 start_at='2020-01-01',
                                 subtract_starting_points=False):
        """ Prepare a dataframe with coefficients of approximating curves.

        Args:
            quantity (str): one quantity to be approximated for all scenarios.
            curve_func (function): Function defining the type of approximation.
            start_at (str): date string representing the beginning of data
                being approximated.
        """

        t_mapper = self.shared_scaled_time_index(start_at=start_at)

        quantity_min = math.inf
        quantity_max = -math.inf

        ydata_df = pd.DataFrame(index=sorted(t_mapper.values()),
                                columns=range(self.count))
        yhat_df = pd.DataFrame(index=sorted(t_mapper.values()),
                               columns=range(self.count))

        columns = inspect.getargspec(curve_func).args
        if "x" in columns:
            columns.remove("x")

        coeff_df = pd.DataFrame(index=range(self.count), columns=columns)

        # Saving *scaled* starting points for quantity timeseries
        starting_points = pd.DataFrame(index=[quantity],
                                       columns=range(self.count))

        for i in range(self.count):

            one_traj = self[i].scaled_timeseries[quantity]

            one_traj = one_traj[one_traj.index >= pd.to_datetime(start_at)].copy()

            starting_points.at[quantity, i] = one_traj.tolist()[0]
            # Subtracting starting value from all values in timeseries
            if subtract_starting_points:
                one_traj = one_traj - starting_points.at[quantity, i]

            # Use timeseries (not scaled_timeseries) to find min/max
            min_candidate = self[i].timeseries[quantity].loc[one_traj.index].min()
            if min_candidate < quantity_min:
                quantity_min = min_candidate
            max_candidate = self[i].timeseries[quantity].loc[one_traj.index].max()
            if max_candidate > quantity_max:
                quantity_max = max_candidate

            xdata = np.array(one_traj.index.map(t_mapper))
            ydata = one_traj.values

            for x, y in zip(xdata, ydata):
                ydata_df.at[x, i] = y

            popt, pcov = curve_fit(curve_func, xdata, ydata, maxfev=10000)

            # Values that come out of obtained approximations
            # (can be used for error analysis)
            # y_hat = curve_func(xdata, *popt)

            coeff_df.loc[i] = popt

        quantity_min_max = pd.DataFrame([{"quantity": quantity,
                                         "quantity_min": quantity_min,
                                         "quantity_max": quantity_max}])
        return coeff_df, starting_points, quantity_min_max

