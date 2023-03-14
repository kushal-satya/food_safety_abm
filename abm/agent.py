# File to parse JSON output from Apple Health and convert into csv 
# We use exported data from Apple Health to create a csv file that can be used to create a graph of the data
# The data is exported from Apple Health as JSON and then parsed to create a csv file
# The csv file is then used to create a graph of the data
import json
import csv
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

# Read in the JSON file 
# Exported data from Apple Health is in JSON format
health = pd.read_json('health.json')

