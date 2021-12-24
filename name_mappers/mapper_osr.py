# -*- coding: utf-8 -*-
""" Name mapper used to rename columns from input xlsx files to
shorter, convenient names (e.g., which can be used as dataframe column names)
This particular mapper is tested with data/validation/case0 example.

Written by: Dmitry Duplyakin (dmitry.duplyakin@nrel.gov)
in collaboration with the National Renewable Energy Laboratories.
"""

name_mapper = {
    'I1_Mass_Flow_Rate (kg/day)': 'im1', 
    'I2_Mass_Flow_Rate (kg/day)': 'im2',
    'I3_Mass_Flow_Rate (kg/day)': 'im3', 
    'I4_Mass_Flow_Rate (kg/day)': 'im4',
    'P1_Mass_Flow_Rate (kg/day)': 'pm1', 
    'P2_Mass_Flow_Rate (kg/day)': 'pm2',
    'P3_Mass_Flow_Rate (kg/day)': 'pm3', 
    'P4_Mass_Flow_Rate (kg/day)': 'pm4',
    'P5_Mass_Flow_Rate (kg/day)': 'pm5', 
    'P6_Mass_Flow_Rate (kg/day)': 'pm6',
    'I1_Bottom-Hole_Pressure (kPa)': 'ip1', 
    'I2_Bottom-Hole_Pressure (kPa)': 'ip2',
    'I3_Bottom-Hole_Pressure (kPa)': 'ip3', 
    'I4_Bottom-Hole_Pressure (kPa)': 'ip4',
    'P1_Bottom-Hole_Pressure (kPa)': 'pp1', 
    'P2_Bottom-Hole_Pressure (kPa)': 'pp2',
    'P3_Bottom-Hole_Pressure (kPa)': 'pp3', 
    'P4_Bottom-Hole_Pressure (kPa)': 'pp4',
    'P5_Bottom-Hole_Pressure (kPa)': 'pp5', 
    'P6_Bottom-Hole_Pressure (kPa)': 'pp6',
    'I1_Bottom-Hole_Temperature (C)': 'it1', 
    'I2_Bottom-Hole_Temperature (C)': 'it2',
    'I3_Bottom-Hole_Temperature (C)': 'it3', 
    'I4_Bottom-Hole_Temperature (C)': 'it4',
    'P1_Bottom-Hole_Temperature (C)': 'pt1', 
    'P2_Bottom-Hole_Temperature (C)': 'pt2',
    'P3_Bottom-Hole_Temperature (C)': 'pt3', 
    'P4_Bottom-Hole_Temperature (C)': 'pt4',
    'P5_Bottom-Hole_Temperature (C)': 'pt5', 
    'P6_Bottom-Hole_Temperature (C)': 'pt6'
    }
