import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")
if __name__ == '__main__':

    X_train_scaled = pd.read_csv('train_X0.csv',delimiter=',')
    X_test = pd.read_csv('X_test0.csv',delimiter=',')
    ytrain = pd.read_csv('train_y0.csv',delimiter=',')
    X_train_scaled=X_train_scaled.iloc[:,1:-1]
    X_test=X_test.iloc[:,1:-1]

    print(X_train_scaled.shape)
    print(X_test.shape)

    y_train=ytrain.iloc[:,-1].values.T
    X_train_scaled = X_train_scaled.drop(['q05'], axis=1)
    X_train_scaled = X_train_scaled.drop(['q99'], axis=1)
    X_train_scaled = X_train_scaled.drop(['q01'], axis=1)
    X_train_scaled = X_train_scaled.drop(['q95'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_q01_roll_mean_100'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_q95_roll_mean_100'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_max_roll_mean_50'], axis=1)
    X_train_scaled = X_train_scaled.drop(['ave_roll_mean_100'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_max_roll_std_100'], axis=1)
    X_train_scaled = X_train_scaled.drop(['abs_median'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_q99_roll_mean_50'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_q99_roll_std_50'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_q01_roll_mean_50'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_Rmax'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_abs_max'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_max_first_50000'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_max_first_10000'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_ave_roll_std_100'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_ave_roll_std_50'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_std_roll_std_50'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_min_last_10000'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_q95_roll_std_100'], axis=1)
    X_train_scaled = X_train_scaled.drop(['fft_min_first_50000'], axis=1)


    scaler = StandardScaler()
    scaler.fit(X_train_scaled)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train_scaled))
    X_train_scaled.to_csv("scaled_train_X_new0.csv")

    param_grid = {
        'n_estimators': [(100)],
        'max_depth': [(5)]
    }
    from sklearn.model_selection import GridSearchCV
    rf = RandomForestRegressor()
    grid_obj = GridSearchCV(rf,param_grid,cv=10)
    grid_obj.fit(X_train_scaled,y_train)
    y_pred = grid_obj.predict(X_train_scaled)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_train.flatten(), y_pred)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('actual', fontsize=12)
    plt.ylabel('predicted', fontsize=12)
    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
    plt.show()

    score = mean_absolute_error(y_train.flatten(), y_pred)
    print(score)

    submission = pd.read_csv('sample_submission.csv', index_col='seg_id')
    X_test = X_test.drop(['q05'], axis=1)
    X_test = X_test.drop(['q99'], axis=1)
    X_test = X_test.drop(['q01'], axis=1)
    X_test = X_test.drop(['q95'], axis=1)
    X_test = X_test.drop(['fft_q01_roll_mean_100'], axis=1)
    X_test = X_test.drop(['fft_q95_roll_mean_100'], axis=1)
    X_test = X_test.drop(['fft_max_roll_mean_50'], axis=1)
    X_test = X_test.drop(['ave_roll_mean_100'], axis=1)
    X_test = X_test.drop(['fft_max_roll_std_100'], axis=1)
    X_test = X_test.drop(['abs_median'], axis=1)
    X_test = X_test.drop(['fft_q99_roll_mean_50'], axis=1)
    X_test = X_test.drop(['fft_q99_roll_std_50'], axis=1)
    X_test = X_test.drop(['fft_q01_roll_mean_50'], axis=1)
    X_test = X_test.drop(['fft_Rmax'], axis=1)
    X_test = X_test.drop(['fft_abs_max'], axis=1)
    X_test = X_test.drop(['fft_max_first_50000'], axis=1)
    X_test = X_test.drop(['fft_max_first_10000'], axis=1)
    X_test = X_test.drop(['fft_ave_roll_std_100'], axis=1)
    X_test = X_test.drop(['fft_ave_roll_std_50'], axis=1)
    X_test = X_test.drop(['fft_std_roll_std_50'], axis=1)
    X_test = X_test.drop(['fft_min_last_10000'], axis=1)
    X_test = X_test.drop(['fft_q95_roll_std_100'], axis=1)
    X_test = X_test.drop(['fft_min_first_50000'], axis=1)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled)
    X_test.to_csv('X_test_scaled_new0.csv')

    submission['time_to_failure'] = grid_obj.predict(X_test_scaled)
    submission.to_csv('submission.csv')
