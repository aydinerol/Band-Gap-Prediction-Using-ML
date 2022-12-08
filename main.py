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


seed = 0

def import_dataset(path='', hasGeneratedFeatures = False):
    # Importing dataset
    if not hasGeneratedFeatures:
        data = pd.read_csv("data/citrination-export.csv")
    
    if hasGeneratedFeatures:
        data = pd.read_csv(path+'/AllData-Citrination-CDMM'+'.csv')
    
    data = clean_dataset(data)

    X = data.drop(columns=["Chemical formula","Band gap"])
    y = data["Band gap"]
    return X, y

def clean_dataset(data):
    # Check if any string in the "Chemical formula" column does not match the regular expression pattern
    if not all(data["Chemical formula"].str.match(r"^[A-Za-z0-9]")):
        data['Chemical formula'] = data['Chemical formula'].str.replace('[$_{}]','') #Convert subscripts to normal
    
    if not all(data["Band gap"].str.isalnum()):
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

def scale_dataset(X):
    columns = X.columns
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=columns)
    return X

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)
        return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type):
    # Training the model
    if model_type == "MLP":
        regressor = MLPRegressor(max_iter=1000)

    #elif model_type == "SVM":
        #SVC.fit(X_train, y_train)
    
    elif model_type == "Random Forest" or model_type == "RF":
        regressor = RandomForestRegressor(n_estimators=100)

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
    r2adj_score = 1 - (1 - r2_score_) * (X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1)
    return mae_score, rmse_score, r2_score_, r2adj_score
    
def metricsPrint(test_df, pred_df):
    
    mae_score = mean_absolute_error(test_df, pred_df)
    mse_score = mean_squared_error(test_df, pred_df)
    rmse_score = mean_squared_error(test_df, pred_df, squared=False)
    mape_score = mean_absolute_percentage_error(test_df, pred_df)
    evs_score = explained_variance_score(test_df, pred_df)
    me_score = max_error(test_df, pred_df)
    msle_score = mean_squared_log_error(test_df, pred_df)
    mae_score = median_absolute_error(test_df, pred_df)
    r2_score_ = r2_score(test_df, pred_df)
    r2_score_adj_ = (1 - (1-r2_score)*(len(test_df)-1)/(len(test_df)-len(test_df.columns)-1))
    r2_fpe_score = ((len(test_df)+len(test_df.columns))*r2_score_adj_-len(test_df.columns))/(len(test_df)+1)
    mpd_score = mean_poisson_deviance(test_df, pred_df)
    mgd_score = mean_gamma_deviance(test_df, pred_df)
    
    # Use `locals()` to automatically assign metric scores to variables, then print
    for metric_name, metric_score in locals().items():
        if metric_name.endswith('_score') and isinstance(metric_score, (int, float)):
            print(f'{metric_name}: {metric_score:.3f}')
    
def plot_results(y_test, y_pred):
    # Plotting actual vs predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Band Gap")
    plt.ylabel("Predicted Band Gap")
    plt.title("Actual vs Predicted Band Gap")
    plt.show()

# Main program
def main():
    # Importing dataset
    X, y = import_dataset()

    # Generating additional features using MAST-ML
    # X,y = generate_features(X)

    # Find and select optimal features
    # X = FeatureSelectionPipeline().find_features(X, y)
    X = find_features(X, y)
    
    # Splitting dataset into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train the model using the training sets
    model = train_model(X_train, y_train, "RF")

    # Predicting using trained model
    y_pred = predict(model, X_test)

    # Evaluating model performance
    #evaluate_model(y_test, y_pred)
    metricsPrint(y_test, y_pred)

    # Plotting actual vs predicted values
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()


# class FeatureSelectionPipeline:


# random_forest_reg = RandomForestRegressor(random_state=seed, criterion='mse', n_estimators=500, bootstrap=True)
# test_fraction = 0.15
#preprocessor = SklearnPreprocessor(preprocessor='StandardScaler', as_frame=True)

def manageCorrelation(feat_data):
    # This block is used to find and to drop highly-correlated features in a given feature dataframe 
    corr_matrix = feat_data.corr(method='pearson').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
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
   
    return initial_feature_list

def shuffleList(featureList, randomize=True):
    # Manages the shuffling of a given list
    # When randomize is false, shuffles the list according to the seed
    if randomize==True:
        random.seed()
    else:
        random.seed(seed)
    return random.sample(list(featureList), len(featureList))

def test_feature(feature, scores, good_features_list):
    
    mae_score = scores['mae']
    rmse_score = scores['rmse']
    r2_score = scores['r2']
    r2adj_score = scores['r2adj']
    
    # Check if the scores are the best results
    if scores['r2adj'] < max(scores['r2adj']) or scores['r2'] < max(scores['r2']) or scores['mae'] > max(scores['mae']) or scores['rmse'] > max(scores['rmse']):
        # Remove the last feature from the list of good features
        good_features_list = good_features_list[:-1]
    
    
    # Check if the feature is in the list of good features
    if feature in good_features_list:
      print('New score: '+"%.3f"%scores['r2adj']+'/'+"%.3f"%scores['r2']+' :: - MAE: '+"%.3f"%scores['mae']+' - RMSE: '+"%.3f"%scores['rmse']+', Adding: '+good_features_list[-1])
        
    # Update the scores in the dictionary
    scores = update_scores(scores, mae_score, rmse_score, r2_score, r2adj_score)
    
    return scores, good_features_list
    

# def get_scores(r2adj_list, r2_list, mae_list, rmse_list ):

def update_scores(scores, mae_score, rmse_score, r2_score, r2adj_score):
    # Update the scores in the scores dictionary
    scores['mae'].append(mae_score)
    scores['rmse'].append(rmse_score)
    scores['r2'].append(r2_score)
    scores['r2adj'].append(r2adj_score)    
    return scores


# feature = "AtomicBla"
randomize=True

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

    X = pd.DataFrame(X, columns=good_features_list)
    
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    model = train_model(X, y, "RF")

    y_pred = predict(model, X_test)
    
    mae_score, rmse_score, r2_score, r2adj_score = evaluate_model(X_train, y_test, y_pred)
    print(mae_score, rmse_score, r2_score, r2adj_score)
    
    # Save scores in scores dictionary
    scores = evaluate_model(X_train, y_test, y_pred)
    
    print(scores)
    
    ### Automation Part
    # Create a list of features to be tested in means of MAE, RMSE, R^2, ADJ-R^2
    features_2b_tried = Xcopy[Xcopy.columns.difference(good_features_list)] ## Features to be tested
    features_2b_tried_list = features_2b_tried.columns
    
    # Shuffle/randomize features
    shuffled_features = shuffleList(features_2b_tried_list, randomize=randomize) 

    print('Finding features now.')
    for feature in shuffled_features:
        print('Testing: '+feature)
        good_features_list.append(feature) # Feature to be tested
        X = pd.DataFrame(Xcopy, columns=good_features_list)
        
        # Calculate and eliminate highly-correlated features
        X = manageCorrelation(X)
        # Preprocess
        X = scale_dataset(X)
        
        # Split
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        
        # Train model
        model = train_model(X_train, y_train, "RF")
        
        # Get prediction
        y_pred = predict(model, X_test)

        scores = evaluate_model(X, y_test, y_pred)
        scores, good_features_list = test_feature(feature, scores, good_features_list)

    X = pd.DataFrame(Xcopy, columns=good_features_list)
    return X
