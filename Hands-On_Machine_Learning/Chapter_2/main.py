import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Função para criar, baixar e extrair o dataset dentro de sua pasta :)
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Carregando o dataset
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing = load_housing_data()
print('\n\n')
print(housing.head())
print('\n\n')
print(housing.describe())

# Vendo via plot
import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20, 15))
plt.show()


# Test Set
import numpy as np
from zlib import crc32

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# Nesse dataset não existe um id, então vamos criar um
housing_with_id = housing.reset_index() # Adiciona coluna Index
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# Identifier unico para ser alocado no fim do dataset
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

# Cria atributo de categoria de renda
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

print('\n\n')
print('atributo de categoria de renda')
print(housing["income_cat"].hist())
housing["income_cat"].hist()
plt.show()

# Base de amostragem estratificada na categoria de renda
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print('\n\n')
print('Base de amostragem estratificada na categoria de renda')
print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

# Removendo o "income_cat" para fazer o data voltar ao estado original
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)
# Fim do Test Set :D


housing = strat_train_set.copy() # Criando copia do dataset

# Visualizando Data Geografica
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1,
                s=housing["population"] / 100, label="population", figsize=(10, 7),
                c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()
plt.show()

print('\n\n')
print('Correlação com o valor medio de casa')
corr_matrix = housing.corr() # Correlação
print('\n\n', corr_matrix["median_house_value"].sort_values(ascending=False))

# Checando correlações atraves do Pandas
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# Zoom no Gráfico de dispersão de correlações
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()


# Experimentando com combinações de atributos

# O numeros de quartos em um distrito não é muito util se você não sabe quantas casas tem la, o que você realmente quer
#é o número de quartos por casa... Então vamos criar esses novos atributs :D

housing['rooms_per_households'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
print('\n\n')
print('Correlação da Matriz')
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# Nada mal, o novo atributo bedrooms_per_room é muito mais correlacionado com a median_house_value do que o total_rooms ou total_bedrooms
# Aparentemente as casas com um numero menor de quartos e salas tendem a ser mais caras

# Preparando a Data para os algoritmos de Machine Learning
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Ok, a maioria dos algoritmos de ML não conseguem trabalhar com coisas faltando, então vamos criar algumas funções para tomar conta disso :D
# Você deve ter notado que o atributo total_bedrooms possui alguns valores faltando, então vamos arrumar isso. Você tem 3 opções
'''
    1 - Se livrar das cidades correspondentes
    2 - Se livrar do atributo inteiro
    3 - Setar os valores para algum valor (zero, significado, media, etc...)

 Você pode realizar isso utilizando o dopna(), drop() e fillna():
housing.dropna(subset=["total_bedrooms"])   # Opção 1

housing.drop("total_bedrooms", axis=1)      # Opção 2

median = housing["total_bedrooms"].median() # Opção 3
housing["total_bedrooms"].fillna(median, inplace=True)
'''

# O SKlearn provem um jeito de tomar conta desses valores inexistentes: SimpleImputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") # Substitui os valores inexistentes com sua média

housing_num = housing.drop("ocean_proximity", axis=1) # A média só pode ser feita com atributos numéricos, então criamos uma copia da data sem o atributo ocean_proximity

imputer.fit(housing_num) # Insere a instancia Imputer dentro do Data

print('\n\n')
print('Media dos atributos')
print(imputer.statistics_)

# Agora você pode usar esse Imputer "treinado" para transformar o Training Set substituindo valores inexistentes pela media aprendida

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# O ocean_proximity é texto então vamos converte-lo
housing_cat = housing["ocean_proximity"]

housing_cat_encoded, housing_categories = housing_cat.factorize()
print('\n\n')
print('ocean_proximity em numeros')
print(housing_cat_encoded[:10])
print('\n\n')

print(housing_categories)

print('\n\n')

# one-hot enconding para que o algoritmo não confunda valores distantes com valores próximos
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
print(housing_cat_1hot.toarray())

# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
        
# Transformation Pipelines
# O Scikit-Learn nos provem a classe PipeLine para ajudar com transformações,
# aqui temos um pequeno pipeline para atributos numericos
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):    
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")
            
        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self
        
    def transform(self, X):        
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])
            
            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
            else:
                X_mask[:, i] = valid_mask
                X[:, i][~valid_mask] = self.categories_[i][0]
                
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init_(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense"))
])

# Para juntar 2 pipelines podemos usar a classe FeatureUnion do sklearn
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])


housing_prepared = full_pipeline.fit_transform(housing)
print('Full Pipeline \n')
print(housing_prepared)


# Treinando e validando o no Training Set
# primeiramente vamos treinar um algoritmo de regressão linear








print('\n\n')
os.system('pause')
os.system('cls')
