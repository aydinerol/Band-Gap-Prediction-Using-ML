# Band Gap Prediction Using ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import time

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error, mean_absolute_percentage_error, explained_variance_score, max_error, mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance

# from mastml.feature_generators import ElementalFeatureGenerator


seed = 12345123
savepath = './data' 
randomize=False

def import_dataset(path=savepath, hasGeneratedFeatures = True):
    # Importing dataset
    if not hasGeneratedFeatures:
        data = pd.read_csv(path + "/citrination-export.csv", index_col=0)
    
    if hasGeneratedFeatures:
        data = pd.read_csv(path + '/AllData-Citrination-CDMM'+'.csv', index_col=0)
    
    data = clean_dataset(data)

    X = data.drop(columns=["Chemical formula","Band gap"])
    y = data["Band gap"]
    print('Import X:',X.shape[0], X.shape[1])
    return X, y

def clean_dataset(data):
    # Check if any string in the "Chemical formula" column does not match the regular expression pattern
    if not all(data["Chemical formula"].str.match(r"^[A-Za-z0-9]")):
        data['Chemical formula'] = data['Chemical formula'].str.replace('[$_{}]','') #Convert subscripts to normal
    
    if not all(data["Chemical formula"].str.match(r"^[0-9]")):
        data['Band gap'] = data['Band gap'].apply(pd.to_numeric, errors='coerce') ### Exclude rows that has ± sign

    # Exclude materials with band gap = 0
    data = data[data["Band gap"]>0]
    data = data.dropna()
    
    #Clean from the same elements by averaging the band gap of same/reoccuring materials
    data.groupby('Chemical formula').filter(lambda g: len(g) > 1).drop_duplicates(subset=['Chemical formula', 'Band gap'], keep='first').sort_values(by='Chemical formula')
    data.groupby("Chemical formula", as_index = False).mean()
    return data


def generate_features(X, y):
    from mastml.feature_generators import ElementalFeatureGenerator

    # Generating additional features using MAST-ML
    savepath = "./output"
    feature_types = ['composition_avg','max', 'min', 'difference']
    X, y = ElementalFeatureGenerator(composition_df=X,
                                feature_types=feature_types,
                                remove_constant_columns = False).evaluate(X=X, y=y, savepath=savepath, make_new_dir=False)

    return X, y


def split_dataset(X, y, evaluation_method='train_test_split', test_fraction=0.15, n_splits=5):
    """Split a dataset for evaluating a model.
    
    This function splits a dataset into train and test sets, or performs
    k-fold cross-validation, depending on the specified evaluation method.
    
    Args:
        X: The input data, as a NumPy array or Pandas DataFrame.
        y: The target data, as a NumPy array or Pandas Series.
        evaluation_method: The method for evaluating the model.
            Must be one of 'cross_validation' (default) or 'train_test_split'.
        test_size: The fraction of the data to reserve for testing,
            when using 'train_test_split' (default is 0.2).
        n_splits: The number of folds to use for cross-validation,
            when using 'cross_validation' (default is 5).
    
    Returns:
        If using 'train_test_split', returns a 4-tuple containing
        the train input data, test input data, train target data,
        and test target data.
        
        If using 'cross_validation', returns a generator object that
        yields train/test splits, as tuples containing the train
        input data, train target data, and test input data.
    """
    
    # Define the evaluation parameters
    if evaluation_method == 'cross_validation':
        # Use k-fold cross-validation
        cv = KFold(n_splits=n_splits)
        return cv.split(X, y)
    else:
        # Use train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=seed)
        return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type):
    # Training the model
    if model_type == "MLP":
        regressor = MLPRegressor(max_iter=1000)

    #elif model_type == "SVM":
        #SVC.fit(X_train, y_train)
    
    elif model_type == "Random Forest" or model_type == "RF":
        regressor = RandomForestRegressor(n_estimators=100, bootstrap=True, random_state=seed)

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return regressor.fit(X_train,y_train)
        

# Predict the response for test dataset
def predict(model, X_test):
    # Predicting using trained model
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(X, y_test, y_pred):
    # Evaluating model performance
    mae_score = mean_absolute_error(y_test, y_pred)
    rmse_score = mean_squared_error(y_test, y_pred, squared=False)
    r2_score_ = r2_score(y_test, y_pred)
    
    # n is sample size
    n = np.array(y_test).shape[0]
    # p is number of features
    p = len(X.columns)
    r2adj_score = 1 - (((1-r2_score_)*(n-1))/(n-p-1))

    return mae_score, rmse_score, r2_score_, r2adj_score
    
def metricsPrint(test_df, pred_df, X_test):
    
    mae_score = mean_absolute_error(test_df, pred_df)
    mse_score = mean_squared_error(test_df, pred_df)
    rmse_score = mean_squared_error(test_df, pred_df, squared=False)
    mape_score = mean_absolute_percentage_error(test_df, pred_df)
    evs_score = explained_variance_score(test_df, pred_df)
    me_score = max_error(test_df, pred_df)
    msle_score = mean_squared_log_error(test_df, pred_df)
    mae_score = median_absolute_error(test_df, pred_df)
    r2_score_ = r2_score(test_df, pred_df)
    r2adj_score = 1 - (1 - r2_score_) * (X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)
    # r2_score_adj_ = (1 - (1-r2_score)*(len(test_df)-1)/(len(test_df)-len(test_df.columns)-1))
    # r2_fpe_score = ((len(test_df)+len(test_df.columns))*r2_score_adj_-len(test_df.columns))/(len(test_df)+1)
    mpd_score = mean_poisson_deviance(test_df, pred_df)
    mgd_score = mean_gamma_deviance(test_df, pred_df)
    
    # Use `locals()` to automatically assign metric scores to variables, then print
    for metric_name, metric_score in locals().items():
        if metric_name.endswith('_score') and isinstance(metric_score, (int, float)):
            print(f'{metric_name}: {metric_score:.3f}')
    
def plot_results(y_test, y_pred):
    # Plotting actual vs predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Band Gap [eV]")
    plt.ylabel("Predicted Band Gap [eV]")
    plt.title("Actual vs Predicted Band Gap")
    plt.show()


# =============================================================================
# AUTOMATED FEATURE SELECTOR CLASS
# =============================================================================

def manageCorrelation(feat_data):
    # This block is used to find and to drop highly-correlated features in a given feature dataframe 
    corr_matrix = feat_data.corr(method='pearson').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    features_df_filt_lowcorr = feat_data.drop(to_drop, axis=1) # drop highly-correlated features
    return features_df_filt_lowcorr

def scale_dataset(X):
    columns = X.columns
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=columns)
    return X

def pull_features(X, y, n_best_features=3):
    # Pick n_best_features number of features quickly
    X = scale_dataset(X)
    random_forest_regressor = train_model(X, y, "RF")
    feat_importances = pd.Series(random_forest_regressor.feature_importances_, index=X.columns)
    initial_feature_list = list(feat_importances.nlargest(n_best_features).keys())

    print('Initial features are: '+ initial_feature_list[0]+' ,'+initial_feature_list[1]+' ,'+initial_feature_list[2])
    
    return initial_feature_list

def update_scores(mae_score, rmse_score, r2_score, r2adj_score, scores):
    # Update the scores stored in the scores dictionary
    scores['mae'].append(mae_score)
    scores['rmse'].append(rmse_score)
    scores['r2'].append(r2_score)
    scores['r2adj'].append(r2adj_score) 
    
    return scores

def shuffleList(featureList, randomize=True):
    # Manages the shuffling of a given list
    # When randomize is false, shuffles the list according to the seed
    if randomize==True:
        random.seed()
    else:
        random.seed(seed)
    return random.sample(list(featureList), len(featureList))

def typeChecker(scores):
    # Check if the object is an instance of the dict class
    dict_type = type(scores)

    # Make sure the type of the dictionary is dict
    if dict_type == dict:
      # The dictionary is a dict, so we can use it as a dictionary
      print("The dictionary is a dict")
    else:
      # The dictionary is not a dict, so we cannot use it as a dictionary
      print("The dictionary is not a dict")
      
def test_feature(feature, good_features_list, scores):
    
    mae_score = scores['mae'][-1]
    rmse_score = scores['rmse'][-1]
    r2_score_ = scores['r2'][-1]
    r2adj_score = scores['r2adj'][-1]
    
    
    # Check if the scores are the best results
    if scores['r2adj'][-1] < max(scores['r2adj']) or scores['r2'][-1] < max(scores['r2']) or scores['mae'][-1] > max(scores['mae']) or scores['rmse'][-1] > max(scores['rmse']):
        # Remove the last feature from the list of good features
        good_features_list = good_features_list[:-1]
    
    
    # Check if the feature is in the list of good features
    if feature == good_features_list[-1]:
        print('New score: '+"%.3f"%r2adj_score+'/'+"%.3f"%r2_score_+' :: - MAE: '+"%.3f"%mae_score+' - RMSE: '+"%.3f"%rmse_score+', Adding: '+feature)
      
    return good_features_list, scores

def find_features(X, y, good_features_list=[], n_best_features=3):
    scores = {
      'mae': [],
      'rmse': [],
      'r2': [],
      'r2adj': []
    }
    
    
    Xcopy = X.copy()
    # when no feature list is specified, 
    # change n_best_features to specify the number of features will be selected by the if-block below
    
    ## if initial features are not specified, find it automatically using rf.feature_importances
    if(len(good_features_list)<1):
        good_features_list = pull_features(X, y, n_best_features=3)

    X_ = pd.DataFrame(Xcopy, columns=good_features_list)
    
    X_train, X_test, y_train, y_test = split_dataset(X_, y)
    
    model = train_model(X_train, y_train, "RF")

    y_pred = predict(model, X_test)
    
    mae_score, rmse_score, r2_score_, r2adj_score = evaluate_model(X_train, y_test, y_pred)
    
    # Save scores into scores dictionary
    scores = update_scores(mae_score, rmse_score, r2_score_, r2adj_score, scores)
    print("%.3f"%r2adj_score+'/'+"%.3f"%r2_score_+' :: - MAE: '+"%.3f"%mae_score+' - RMSE: '+"%.3f"%rmse_score)
    
    
    ### Automation Part
    # Create a list of features to be tested in means of MAE, RMSE, R^2, ADJ-R^2
    features_2b_tried = Xcopy[Xcopy.columns.difference(good_features_list)] ## Features to be tested
    features_2b_tried_list = features_2b_tried.columns
    
    # Shuffle/randomize features
    shuffled_features = shuffleList(features_2b_tried_list, randomize=randomize) 
    i=0
    print('Finding features now.')
    for feature in shuffled_features:
        i=i+1
        print(f"{i+1}/{len(shuffled_features)}", end="\r", flush=True)
        good_features_list.append(feature) # Feature combination to be tested
        
        X_ = pd.DataFrame(Xcopy, columns=good_features_list)
        
        # Calculate and eliminate highly-correlated features
        X_ = manageCorrelation(X_)
        # Preprocess
        X_ = scale_dataset(X_)
        
        # Split
        X_train, X_test, y_train, y_test = split_dataset(X_, y)
        
        # Train model
        sub_model = train_model(X_train, y_train, "RF")
        
        # Get prediction
        y_pred = predict(sub_model, X_test)

        mae_score, rmse_score, r2_score_, r2adj_score = evaluate_model(X_, y_test, y_pred)
        scores = update_scores(mae_score, rmse_score, r2_score_, r2adj_score, scores)
        good_features_list, scores = test_feature(feature, good_features_list, scores)
        
        
    X = pd.DataFrame(Xcopy, columns=good_features_list)
    return X

# =============================================================================
# 
# =============================================================================

# Main program
def main():
    # Importing dataset
    X, y = import_dataset()

    # Generating additional features using MAST-ML
    # X,y = generate_features(X)

    # Find and select optimal features
    X = find_features(X, y)
    
    # Splitting dataset into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train the model using the training sets
    model = train_model(X_train, y_train, "RF")

    # Predicting using trained model
    y_pred = predict(model, X_test)

    # Evaluating model performance
    mae_score, rmse_score, r2_score_, r2adj_score = evaluate_model(X, y_test, y_pred)
    metricsPrint(y_test, y_pred, X_test)

    # Plotting actual vs predicted values
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()

