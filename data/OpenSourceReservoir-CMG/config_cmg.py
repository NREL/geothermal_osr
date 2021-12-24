# -*- coding: utf-8 -*-
"""Script that sets main configuration options.

Most of the other code scripts and notebooks import this file.

Written by: Dmitry Duplyakin (dmitry.duplyakin@nrel.gov)
in collaboration with the National Renewable Energy Laboratories.
"""

import logging
import os

loglevel = 'INFO'
logging.basicConfig(level=os.environ.get("LOGLEVEL", loglevel))

input_type="CMG"

# Path to a mapper file -- file that describes how to rename columns/inputs
# to shorter, convenient names throughout analysis code/notebooks
mapper_path = "../name_mappers/mapper_osr.py"

# Path to xlsx file with inut data
# This option will be reset in the notebook
timeseries_file = "..."

# File above will be merged with this one if specified
additional_timeseries_file = ""

# Unit used for measuring flow. Options: "kKg/day", "kg/day", and "m^3/day"
flow_unit = "kKg/day"

# Used to calculate enthalpy; Unit: bars
geofluid_surface_pressure = 25

# Maximum temperature after scaling
temp_scaled_max = 1.0

# Maximum pressure after scaling
pres_scaled_max = 1.0

# Maximum flow after scaling
flow_scaled_max = 1.0

# Number of steps (length of sequence) used for learning & prediction
n_steps = 8

# Ratios for train and vaidate subsets; the remaining fraction -- test subset
train_ratio = 0.80
val_ratio = 0.00
