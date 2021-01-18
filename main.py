#!/usr/bin/env python3

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import tarfile
import warnings
from pandas.plotting import scatter_matrix
from scipy.stats import randint
from six.moves import urllib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from utils import lnprintln, println

seed = 42
np.random.seed(seed)

mpl.rc('axes', labelsize=14) 
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# see https://github.com/scipy/scipy/issues/5998
warnings.filterwarnings(action='ignore', message='^internal gelsd')

download_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
housing_path = os.path.join('datasets', 'housing')
housing_url = download_root + 'datasets/housing/housing.tgz'

def fetch_housing_data(housing_url, housing_path):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data(housing_url, housing_path)

def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data(housing_path)
(housing.head())

sample = housing.sample(n = 10, random_state = seed)
println(sample)

housing.info()

lnprintln(housing['ocean_proximity'].value_counts())

println(housing.describe())

#housing.hist(bins=50, figsize=(20,15)) TODO
#plt.show() TODO

train_set, test_set = train_test_split(
    housing, test_size=0.2, random_state=seed)

println(housing['median_income'].describe())

mean = np.mean(housing['median_income'])
std = np.std(housing['median_income'])

housing['income_cat'] = pd.cut(
    housing['median_income'], 
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
    labels=[1, 2, 3, 4, 5]
)

#hist = housing['income_cat'].hist() TODO
#plt.show() TODO

println(housing['income_cat'].value_counts())

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

println(strat_test_set['income_cat'].value_counts() / len(strat_test_set))
println(housing['income_cat'].value_counts() / len(housing))

def income_cat_proportions(data): 
    return data['income_cat'].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=seed)

compare_props = pd.DataFrame({
    'Geral': income_cat_proportions(housing),
    'Estratificado': income_cat_proportions(strat_test_set),
    'Aleatorio': income_cat_proportions(test_set),
}).sort_index()

compare_props['Aleatório %erro'] = (
    100 * compare_props['Aleatorio'] / compare_props['Geral'] - 100)
compare_props['Estratificado %erro'] = (
    100 * compare_props['Estratificado'] / compare_props['Geral'] - 100)

println(compare_props)

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

housing = strat_train_set.copy()
# housing.plot(kind='scatter', x='longitude', y='latitude') TODO
# plt.show() TODO

#housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1) TODO
#plt.show() TODO

# housing.plot(
#     kind='scatter', x='longitude', 
#     y='latitude', alpha=0.4,
#     s=housing['population']/100, label='population', 
#     figsize=(10,7), c='median_house_value', 
#     cmap=plt.get_cmap('jet'), colorbar=True,
#     sharex=False # see https://github.com/pandas-dev/pandas/issues/10611
# ) TODO
# plt.legend() TODO
# plt.show() TODO

corr_matrix = housing.corr()
println(corr_matrix)

println(corr_matrix['median_house_value'].sort_values(ascending=False))

attributes = [
    'median_house_value', 'median_income', 
    'total_rooms', 'housing_median_age'
]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
# plt.axis([0, 16, 0, 550000])
# plt.show()

# Número de cômodos por familia (média)
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
# quartos/cômodos
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
# população/agregado familiar
housing['population_per_household']= housing['population']/housing['households']

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# housing.plot(kind='scatter', x='rooms_per_household', y='median_house_value', alpha=0.2)
# plt.axis([0, 5, 0, 520000])
# plt.show()

println(housing.describe())

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head() 
println(sample_incomplete_rows)

sample_incomplete_rows.dropna(subset=['total_bedrooms']) # option 1
sample_incomplete_rows.drop('total_bedrooms', axis=1) # option 2

median = housing['total_bedrooms'].median()
sample_incomplete_rows['total_bedrooms'].fillna(median, inplace=True) # option 3
println(sample_incomplete_rows)

imputer = SimpleImputer(strategy='median')

println(housing)

housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)

println(imputer.statistics_)
println(housing_num.median().values)

X = imputer.transform(housing_num) 
println(X)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index) 
println(housing_tr.head())

housing_cat = housing[['ocean_proximity']]
println(housing_cat.head(10))

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
println(type(housing_cat_encoded))

println(housing_cat_encoded[:10])
println(ordinal_encoder.categories_)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
println(housing_cat_1hot)

housing_cat_1hot.toarray()

cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

cat_encoder.categories_

println(housing.columns)

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, 3] / X[:, 6]
    population_per_household = X[:, 5] / X[:, 6]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, 4] / X[:, 3]
        return np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(
    add_extra_features, validate=False, kw_args={'add_bedrooms_per_room': False})

housing_extra_attribs = attr_adder.fit_transform(housing.values)

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns= list(housing.columns) + [
        'rooms_per_household', 'population_per_household'], 
    index=housing.index
)
println(housing_extra_attribs.head())

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
println(housing_num_tr)

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs), 
    ('cat', OneHotEncoder(), cat_attribs)
]) 

housing_prepared = full_pipeline.fit_transform(housing)
println(housing_prepared)
println(housing_prepared.shape)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels) 

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

println('Predictions:', lin_reg.predict(some_data_prepared))
println('Labels:', list(some_labels))
println(some_data_prepared)

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = MSE(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
println(lin_rmse)

lin_mae = MAE(housing_labels, housing_predictions)
println(lin_mae)

tree_reg = DecisionTreeRegressor(random_state= seed)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = MSE(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
println(tree_rmse)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10) 

tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())
    print()

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(
    lin_reg, housing_prepared, housing_labels, 
    scoring='neg_mean_squared_error', cv=10
)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels) 

housing_predictions = forest_reg.predict(housing_prepared) #Predizer
forest_mse = MSE(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
println(forest_rmse)

forest_scores = cross_val_score(
    forest_reg, housing_prepared, housing_labels, 
    scoring='neg_mean_squared_error', cv=10
)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=seed)

grid_search = GridSearchCV(
    forest_reg, param_grid, cv=5, 
    scoring='neg_mean_squared_error', return_train_score=True
)
grid_search.fit(housing_prepared, housing_labels)
println(grid_search.best_params_)
println(grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
print()

pd.DataFrame(grid_search.cv_results_)

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8)
}

forest_reg = RandomForestRegressor(random_state=seed)
rnd_search = RandomizedSearchCV(
    forest_reg, param_distributions=param_distribs, 
    n_iter=10, cv=5, 
    scoring='neg_mean_squared_error', random_state=seed
)
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
print()

feature_importances = grid_search.best_estimator_.feature_importances_
println(feature_importances)

extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted_attributes = sorted(zip(feature_importances, attributes), reverse=True)
println(sorted_attributes)
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = MSE(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)