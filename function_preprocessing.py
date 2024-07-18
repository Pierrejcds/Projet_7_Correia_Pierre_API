#Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
#Little feature selection is done and overfitting might be a problem since many features are related.
#The following key ideas were used:
#- Divide or subtract important features to get rates (like annuity and income)
#- In Bureau Data: create specific features for Active credits and Closed credits
#- In Previous Applications: create specific features for Approved and Refused applications
#- Modularity: one function for each table (except bureau_balance and application_test)
#- One-hot encoding for categorical features
#All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).

#Update 16/04/2024:
#- Added Payment Rate feature
#- Removed index from features

# Fonction de pré-processing empruntée et modifiée au kernel kaggle disponible à l'adresse : 
# https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager

# Visualisation graphique
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.model_selection import train_test_split

# Métriques
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve

# Modélisation
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

#import mlflow
#from mlflow.pyfunc import PythonModel

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    """
    Calcule le temps d'execution de la fonction associée.
    """
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, nan_as_category = True):
    """
    Fonction appliquant un encodage en one hot encoding pour les colonnes de type
    objet. On catégorise également les valeurs manquantes.
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    """
    - Concatène les sets d'entrainement et de test des données application ;
    - Applique un encodage de classification pour les données binaires et multi-classe ;
    - Suppression d'une valeur abérrante pour 'DAYS EMPLOYED' ;
    - Création de feature de pourcentage pour certaines valeurs.
    """
    # Read data and merge
    df = pd.read_csv('./data/application_train.csv', nrows=num_rows, encoding='utf-8')
    test_df = pd.read_csv('./data/application_test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = pd.concat([df, test_df], ignore_index=True)
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    """
    - Encodage one hot encoding des variables catégorielles ;
    - Merge de bureau et bureau_balance ;
    - Crée des variables numériques (mean, sum, min, max).
    """
    bureau = pd.read_csv('./data/bureau.csv', nrows = num_rows, encoding='utf-8')
    bb = pd.read_csv('./data/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    """
    - Suppression valeurs aberrantes ;
    - Crée des variables numériques (mean, sum, min, max) ;
    - Ajout des données de previous_application.
    """
    prev = pd.read_csv('./data/previous_application.csv', nrows = num_rows, encoding='utf-8')
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    """
    - Encodage des variables catégorielles ;
    - Encodage des variables numériques.
    """
    pos = pd.read_csv('./data/POS_CASH_balance.csv', nrows = num_rows, encoding='utf-8')
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    """
    - Encodage des variables catégorielles ;
    - Création variable versement ;
    - Nettoyage et création de variable sur les dates de versement ;
    - Encodage des variables numériques.
    """
    ins = pd.read_csv('./data/installments_payments.csv', nrows = num_rows, encoding='utf-8')
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    """
    - Encodage des variables catégorielles ;
    - Encodage des variables numériques.
    """
    cc = pd.read_csv('./data/credit_card_balance.csv', nrows = num_rows, encoding='utf-8')
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

def nettoyer_nom_colonne(nom_colonne):
    """
    Remplace les caractères spéciaux qui posaient problème dans le nommage des variables
    par des underscores pour remplacement des underscores en double.
    """
    # Remplacer les espaces et les caractères spéciaux par des underscores
    nom_colonne = nom_colonne.strip().replace(' ', '_')
    nom_colonne = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in nom_colonne)
    # Remplacer plusieurs underscores par un seul underscore
    nom_colonne = '_'.join(filter(None, nom_colonne.split('_')))
    return nom_colonne

def spliting_df(df, test_size):
    """
    Divise le dataframe en deux en excluant la variable cible et SK_ID_CURR qui ne nous sert plus.
    """
    X = df.drop(columns=['TARGET','SK_ID_CURR'])
    y = df['TARGET'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=48)

    return X_train, X_test, y_train, y_test

def suppression_nan(df, threshold_col=0.6, threshold_row=0.6):
    """
    - Suppression des valeurs manquantes caractérisées par les variables nan ou XDA ;
    - Suppression des lignes ne possédant pas de valeur cible ;
    - Suppression des lignes et des colonnes possédant un pourcentage définit par l'utilisateur
    de valeurs manquantes.
    """
    # Suppression des colonnes contenant les classes de valeurs manquantes qui ne présentent aucunes informations
    total_cols = len(df)
    columns_to_drop = [col for col in df.columns if '_nan_' in col or '_XDA_' in col]
    df = df.drop(columns=columns_to_drop, axis=1)
    remaining_cols = df.shape[0]
    print(f"Suppression de {len(columns_to_drop)} colonnes, {remaining_cols} colonnes restantes")

    # Suppression des lignes n'ayant pas de cible
    rows_to_drop = df['TARGET'].index[df['TARGET'].isnull()].tolist()
    df.drop(index=rows_to_drop, inplace=True)
    remaining_rows = df.shape[0]
    print(f"Suppression de {len(rows_to_drop)} lignes, {remaining_rows} lignes restantes")

    # Suppression des colonnes présentant trop de valeurs manquantes
    total_rows = len(df)
    columns_to_drop = [col for col in df if df[col].isnull().sum() / total_rows > threshold_col]
    df.drop(columns=columns_to_drop, inplace=True)
    remaining_columns = df.shape[1]
    print(f"Suppression de {len(columns_to_drop)} colonnes ayant plus de {threshold_col*100}% de valeurs manquantes, {remaining_columns} colonnes restantes")

    #Suppression des lignes présentant trop de valeurs manquantes (après traitement colonnes)
    rows_to_drop = df.index[df.isnull().sum(axis=1) / df.shape[1] > threshold_row].tolist()
    df.drop(index=rows_to_drop, inplace=True)
    remaining_rows = df.shape[0]
    print(f"Suppression de {len(rows_to_drop)} lignes, {remaining_rows} lignes restantes")

    return df

def evaluation(y_test, y_pred):
    """
    Teste la pertinence du modèle par : 
    - Plusieurs score de classification ; 
    - Matrice de confusion ;
    - Courbe ROC.
    """
    print(f"Score de classification :\n{classification_report(y_test, y_pred)}")
    print(f"Matrice de confusion :\n{confusion_matrix(y_test, y_pred)}")
    print("Courbe ROC :\n")
    fp, tp, seuils = roc_curve(y_test, y_pred)

    plt.plot(fp, tp, marker='.')
    plt.ylabel('Taux de vrais positifs (TP)')
    plt.xlabel('Taux de faux positifs (FP)')
    plt.title('Courbe ROC')
    plt.show()

class RFWrapper(BaseEstimator):
    """
    Classe personnalisée permettant d'appliquer un threshold via un predic_proba et
    ainsi choisir plusieurs valeurs de threshold dans un grid_search
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, threshold=0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.threshold = threshold
        self.rf_model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)

    def fit(self, X, y):
        self.rf_model.fit(X, y)
        return self

    def predict(self, X):
        return [proba[1] > self.threshold for proba in self.rf_model.predict_proba(X)]
    
    def predict_proba(self, X):
        return self.rf_model.predict_proba(X)


class LGBMWrapper(BaseEstimator):
    """
    Classe personnalisée permettant d'appliquer un threshold via un predict_proba et
    ainsi choisir plusieurs valeurs de threshold dans un grid_search
    """
    def __init__(self, n_estimators=100, max_depth=-1, num_leaves=31, min_child_samples=50, threshold=0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.threshold = threshold
        self.lgbm_model = LGBMClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, num_leaves=self.num_leaves, min_child_samples=self.min_child_samples)

    def fit(self, X, y):
        self.lgbm_model.fit(X, y)
        return self

    def predict(self, X):
        return [proba[1] > self.threshold for proba in self.lgbm_model.predict_proba(X)]
    
    def predict_proba(self, X):
        return self.lgbm_model.predict_proba(X)
    
"""class PredictProbaWrapper(PythonModel):

    def __init__(self):
        self.model = None

    def load_context(self, context):
        #Charge le modèle sauvegardé
        self.model = mlflow.sklearn.load_model(context.artifacts["model_path"])

    def predict(self, context, model_input, params=None):
        #Modifie la propriété du predict pour retourner un predict_proba à la place
        params = params or {"predict_method": "predict"}
        predict_method = params.get("predict_method")

        if predict_method == "predict":
            return self.model.predict(model_input)
        elif predict_method == "predict_proba":
            return self.model.predict_proba(model_input)
        elif predict_method == "predict_log_proba":
            return self.model.predict_log_proba(model_input)
        else:
            raise ValueError(f"The prediction method '{predict_method}' is not supported.")
"""
    

def bank_scoring(y_test, y_pred, cost_lost=-10, gain_win=1):
    """
    Fonction qui renvoie un score de rentabilité de la prédiction en comparant le gain réel et le gain potentiel total.
    Le score est compris entre -inf et 1 où 1 correspond à une correspondance parfaite et <0 correspond à une perte de capital.
    Par défaut le rapport entre perte et gain est de 10 contre 1, en cas de changement il est possible de modifier ce rapport 
    grâce aux paramètres cost_lost et gain_win.
    """
    # Extraire les valeurs issues de notre classification.
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # On calcul le gain de notre modèle de classification.
    potantial_gain = (tn*gain_win) + (fn*cost_lost)

    # On calcul le gain total possible à partir de nos données de test.
    total_positive = len(y_test[y_test == 1])
    max_gain_possible = total_positive * 1

    # On compare le gain de notre modèle avec le gain total possible.
    score = potantial_gain/max_gain_possible
    
    return score