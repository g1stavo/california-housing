#!/usr/bin/env python3

import os
import pandas as pd
import tarfile
from six.moves import urllib

housing_url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz'

def fetch_housing_data():
    urllib.request.urlretrieve(housing_url, 'housing.tgz')
    housing_tgz = tarfile.open('housing.tgz')
    housing_tgz.extractall()
    housing_tgz.close()

def load_housing_data():
    fetch_housing_data()
    return pd.read_csv('housing.csv')