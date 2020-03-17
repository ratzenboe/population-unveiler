import os
import sys

from astropy.table import Table
import numpy as np
import pandas as pd
import math

import pickle
import time
from random import randint

from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
from sklearn.utils import shuffle

import argparse


def md(fit_data):
    """
    Mean distance between members and centroid of point cloud
    :param fit_data: Data matrix, point cloud
    :return: mean distance between centroid and points, centroid position
    """
    com_tot = np.sum(fit_data)/fit_data.shape[0]
    dist_to_com = (fit_data-com_tot)
    md = np.sum(np.sqrt(np.square(dist_to_com).sum(axis=1)))/fit_data.shape[0]
    return md, com_tot


# Important: in the Meingast et al. (2019) paper a pre-cut is applied, thus, to validate the velocity distribution we make the same cut when comparing the vel-distrs
def validate_velocity(df_train_vr, df_predict_vr):
    """
    Estimate the number of outliers (in df_predict_vr) outside the 3-sigma range in the three marginal velocity distributions
    :param df_train_vr: Training data (pandas DataFrame) containing the 3 velocity features
    :param df_predict_vr: Infered data points (pandas DataFrame) containing the 3 velocity features
    :return: Average contamination fraction
    """
    v_cyl_cols = ['vr_cylinder', 'vphi_cylinder', 'vz_cylinder']
    tot_nb = 0
    for col in v_cyl_cols:
        # how much data is within three times the standard deviation
        mean_train, std_train = df_train_vr[col].mean(), 3*df_train_vr[col].std()
        nb_outside = np.sum(df_predict_vr[col]>mean_train+std_train) + np.sum(df_predict_vr[col]<mean_train-std_train)
        tot_nb += nb_outside/df_predict_vr.shape[0]
    return tot_nb/len(v_cyl_cols)


def hyperparam_search(X_train, y_train, X_full, data, target, hyper_params, test_params, cut_meingast):
    """
    Hyperparameter search in OCSVM
    :param X_train: training data set (pandas DataFrame) with which the OCSVM model is trained; shape: (n_samples, n_features)
    :param y_train: target data set (numpy array with ones; shape: (n_samples,))
    :param X_full: Data to infer membership of (pandas DataFrame)
    :param data: data set combining training and prediction set (1st input parameter for 'validate_velocity' function)
    :param target: full training data with additional featues compared to X_train (2nd input parameter for 'validate_velocity' function)
    :param hyper_params: Hyper-parameters for OCSVM and search region
    :param test_params: Validation parameters
    :param cut_meingast: Source qualitiy filter described in Meingast et al. (2019) applied to the full data
    :return: Accepted hyper-parameters and classifiers in the ensemble + stability(=y_pred)
    """
    print_progress = False

    y_pred_all_summed = np.zeros(shape=(X_full.shape[0],), dtype=int)
    nu_accepted, gamma_accepted = [], []
    accepted_clf_list = []
    y_pred_all_list = []
    c_pos_list = []
    for ith_search in range(hyper_params['n_searches']):
        kernel = np.random.choice(hyper_params['kernel_list'])
        # we want to draw as many sample in buckets from 10^-8 - 10^-7 than in the 10^-4 - 10^-3
        # so we first draw the exponent and then draw the multiplier
        gamma = np.random.uniform(1, 10)*10**(randint(*hyper_params['gamma_exponent_range']))
        nu = np.random.uniform(*hyper_params['nu_range'])
        c_pos = np.random.uniform(*hyper_params['c_pos_range'])
        # scale training and prediction set accordingly
        X_train_final = X_train.copy()
        X_full_final = X_full.copy()
        X_train_final[pos_cols] *= c_pos
        X_full_final[pos_cols] *= c_pos
        # define classifier
        clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

        X_train_shuffled, y_train_shuffled = shuffle(X_train_final.values, y_train)
        scores = cross_val_score(estimator=clf, X=X_train_shuffled, y=y_train_shuffled, scoring='accuracy', cv=hyper_params['k_folds'], n_jobs=1)
        if print_progress:
            print('{}\nGamma: {}   nu: {}\nScores mean: {}  std: {}'.format(50*'-',gamma, nu, scores.mean(), scores.std()))
        # We only want models which at least fit to a certain protion of the training data
        if scores[scores!=0.].shape[0]==hyper_params['k_folds']:
            if scores.mean()>test_params['min_mean_acc'] and scores.std()<test_params['max_std']:
                # Make predictions on the large DS
                clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
                # Store potentially good classifiers
                clf_list = []
                # Train model with bootstrap samples and predict ~5 times
                # Each data point in the full dataset which is predicted at least 3 times (>50%) is then added to the full data set
                y_pred_bootstrap = np.zeros(shape=(X_full_final.shape[0],), dtype=int)
                rs = ShuffleSplit(n_splits=hyper_params['k_folds'], train_size=hyper_params['train_size'])
                for train_index, test_index in rs.split(X_train_final.values):
                    clf_i = clone(clf)
                    clf_i.fit(X_train_final.values[train_index])
                    clf_list.append(clf_i)
                    y_pred_full = clf_i.predict(X_full_final.values)
                    y_pred_full[y_pred_full==-1] = 0
                    y_pred_bootstrap += y_pred_full
                # Only robust points (>1/2 of times predicted) are used in next steps
                min_folds_detected = hyper_params['k_folds']//2
                y_pred_bootstrap[y_pred_bootstrap<min_folds_detected] = 0
                y_pred_bootstrap[y_pred_bootstrap>=min_folds_detected] = 1

                # Test if the shape is the same: via MD comparison
                pts_in_hull = y_pred_bootstrap[y_pred_bootstrap==1].shape[0]
                if print_progress:
                    print(f'{pts_in_hull} points found')
                if pts_in_hull>test_params['min_pts_full'] and pts_in_hull<test_params['max_pts_full']:
                    # Compute MD of found cluster and compare it to the original md
                    # training data (recalculate because we scale positional axes)
                    md_orig_vel, com_orig_vel = md(X_train_final[vel_cols])
                    md_orig_pos, com_orig_pos = md(X_train_final[pos_cols])
                    # prediction data
                    md_full_vel, com_full_vel = md(X_full_final.loc[y_pred_bootstrap==1, vel_cols])
                    md_full_pos, com_full_pos = md(X_full_final.loc[y_pred_bootstrap==1, pos_cols])
                    # check if in bounds
                    is_in_md_vel_range = (md_full_vel/md_orig_vel<(1-test_params['md_margin_pm'])) and (md_full_vel/md_orig_vel>(test_params['md_min_perc']))
                    is_in_md_pos_range = (md_full_pos/md_orig_pos<(1-test_params['md_margin_xyz'])) and (md_full_pos/md_orig_pos>(test_params['md_min_perc']))
                    if print_progress:
                        print(f'md_full_vel/md_orig_vel: {md_full_vel/md_orig_vel}')
                        print(f'md_full_pos/md_orig_pos: {md_full_pos/md_orig_pos}')
                    if is_in_md_vel_range and is_in_md_pos_range:
                        # Check if COMs are approximately aligned
                        com_dist_vel = np.sqrt(np.square(com_orig_vel - com_full_vel).sum())
                        com_dist_pos = np.sqrt(np.square(com_orig_pos - com_full_pos).sum())
                        is_in_com_range_vel = (com_dist_vel/md_orig_vel<test_params['com_margin'])
                        is_in_com_range_pos = (com_dist_pos/md_orig_pos<test_params['com_margin'])
                        if print_progress:
                            print(f'com_dist_vel/md_orig_vel: {com_dist_vel/md_orig_vel}')
                            print(f'com_dist_pos/md_orig_pos: {com_dist_pos/md_orig_pos}')
                        if is_in_com_range_vel and is_in_com_range_pos:
                            v_cyl_cols = ['vr_cylinder', 'vphi_cylinder', 'vz_cylinder']
                            vr_diff = validate_velocity(df_predict_vr=data.loc[cut_meingast & (y_pred_bootstrap==1), v_cyl_cols], df_train_vr=target)
                            if print_progress:
                                print(f'vr_diff: {vr_diff}')
                            if vr_diff<test_params['max_vr_overflow']:
                                y_pred_all_summed += y_pred_bootstrap
                                if print_progress:
                                    print('------- Found suitable classifier! --------')
                                nu_accepted.append(nu)
                                gamma_accepted.append(gamma)
                                accepted_clf_list.append(clf_list)
                                c_pos_list.append(c_pos)
    return {'y_pred': y_pred_all_summed, 'nus': nu_accepted, 'gammas': gamma_accepted, 'c_pos': c_pos_list, 'clf_list': accepted_clf_list}


def file_reader(fpath, contains_colums=None):
    """
    Read in data set as pandas data frame
    :param fpath: Path to the data file
    :param contains_colums: List of columns that have to be present in the data set
    """
    if fpath.endswith('.pkl'):
        data = pd.read_pickle(fpath)

    elif fpath.endswith('.csv'):
        data = pd.read_csv(fpath)

    elif fpath.endswith('.fits'):
        data = Table.read(fpath).to_pandas()

    else:
        msg = """Currently supported input data formats include '.pkl', '.csv', and '.fits'.
              If the data is stored in another format add the proper file importer function in the 'file_reader' function.
              """
        raise TypeError(msg)

    # Check if the right columns are contained in the dataframe
    if isinstance(contains_colums, list):
        # If the input columns are not featured in the data we trow a KeyError
        if not set(contains_colums) <= set(data.columns.tolist()):
            msg = "The data set is missing at least one of the following columns {}".format(contains_colums)
            raise KeyError(msg)

    return data



def main():
    # Necessary columns
    column_list = ['phot_bp_mean_mag', 'phot_bp_mean_flux_over_error',
                   'phot_rp_mean_mag', 'phot_rp_mean_flux_over_error',
                   'parallax', 'parallax_error', 'pmra', 'pmra_error',
                   'pmdec', 'pmdec_error',
                   'radial_velocity', 'radial_velocity_error',
                   'astrometric_sigma5d_max']

    # Load data
    train_data = file_reader(fpath_train, contains_colums=column_list+pos_cols+vel_cols)
    train_data['target'] = 0
    # Data where we want to infer the group membership
    predict_data = file_reader(fpath_predict, contains_colums=column_list+pos_cols+vel_cols)
    predict_data['target'] = 1
    # Full data (needed for radial velocity validation)
    data = pd.concat([train_data, predict_data])

    # Simple error propagation on the bp & rp errors (used in the next step)
    data['phot_bp_mean_mag_error'] = 2.5/math.log(10)* (1/data['phot_bp_mean_flux_over_error'])
    data['phot_rp_mean_mag_error'] = 2.5/math.log(10)* (1/data['phot_rp_mean_flux_over_error'])
    # Source qualitiy filter defined in by Meingast et al. (2019)
    cut_px = np.abs(data['parallax_error']/data['parallax'])<0.5
    cut_pmra = data['pmra_error']/data['pmra']<0.5
    cut_pmdec = data['pmdec_error']/data['pmdec']<0.5
    cut_vr = np.abs(data['radial_velocity_error']/data['radial_velocity'])<0.5
    cut_bp = np.abs(data['phot_bp_mean_mag_error']/data['phot_bp_mean_mag'])<0.5
    cut_rp = np.abs(data['phot_rp_mean_mag_error']/data['phot_rp_mean_mag'])<0.5
    cut_s5d = np.abs(data['astrometric_sigma5d_max'])<0.5
    cut_meingast = cut_px & cut_pmra & cut_s5d & cut_rp & cut_bp & cut_vr &cut_pmdec

    # Define hyperparameter search parameters and ranges
    hyper_params = {'n_searches': n_searches_per_process, 'k_folds': 10, 'kernel_list': ['rbf'],
                    'gamma_exponent_range': (-2, 1),  'nu_range': (0.01, 0.7),
                    'train_size': 0.8, 'c_pos_range': (cp_lo,cp_hi)}
    test_params = {'min_mean_acc': 0.5, 'max_std': 0.15, 'min_pts_full': 500, 'max_pts_full': max_pts_full,
                   'md_margin_xyz': md_margin_pos, 'md_margin_pm': md_margin_vel, 'md_min_perc': 0.5,
                   'com_margin': com_margin, 'max_vr_overflow': 0.25}

    # Preprocessing
    train_cols = pos_cols + vel_cols
    # training data
    X_train_stddev = train_data[train_cols].std()
    X_train, y_train = train_data[train_cols]/X_train_stddev, np.ones(shape=(train_data.shape[0],))
    # Full data set
    X_full = data[train_cols]/X_train_stddev
    y_pred_all_summed = np.zeros(shape=(X_full.shape[0],), dtype=int)

    # Start hyperparameter search on all available cores
    import multiprocessing
    nb_processes = min(n_cores, multiprocessing.cpu_count())
    print('Starting hyperparameter search with {} processes...'.format(nb_processes))
    st = time.time()
    pool = multiprocessing.Pool(processes=nb_processes)
    hyper_param_list = [pool.apply_async(hyperparam_search, (X_train, y_train, X_full, data, train_data, hyper_params, test_params, cut_meingast)) for i in range(nb_processes)]
    pool.close()
    pool.join()
    hp_list = [hp_dic.get() for hp_dic in hyper_param_list]
    et = time.time()
    print('Done [took {:.1f}sec]'.format(et-st))

    # Extract 
    y_pred_all_summed = np.zeros(shape=(X_full.shape[0],), dtype=int)
    nu_accepted, gamma_accepted, c_pos_accepted = [], [], []
    clf_accepted = []
    for hp_dic in hp_list:
        y_pred_all_summed += hp_dic['y_pred'] # This is the stability criterion
        nu_accepted.extend(hp_dic['nus'])
        gamma_accepted.extend(hp_dic['gammas'])
        c_pos_accepted.extend(hp_dic['c_pos'])
        clf_accepted.extend(hp_dic['clf_list'])

    # --------------- Saving files -------------------
    rand_id_str = str(np.random.randint(999999)).zfill(6)
    np.save(os.path.join(data_path,'y_pred_finished_300pc_id-{}.npy'.format(rand_id_str)), y_pred_all_summed)
    np.save(os.path.join(data_path,'nu_accepted_300pc_id-{}.npy'.format(rand_id_str)), nu_accepted)
    np.save(os.path.join(data_path,'gamma_accepted_300pc_id-{}.npy'.format(rand_id_str)), gamma_accepted)
    np.save(os.path.join(data_path,'c_pos_accepted_300pc_id-{}.npy'.format(rand_id_str)), c_pos_accepted)
    with open(os.path.join(data_path,'models_list_accepted_300pc_id-{}.pkl'.format(rand_id_str)), 'wb') as ofile:
        pickle.dump(clf_accepted, ofile)


if __name__=='__main__':
    # Global defined (appear also in function)
    pos_cols = ['X', 'Y', 'Z']
    vel_cols = ['pmra', 'pmdec']

    # commend line parser
    user_argv = None
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--max_pts_full', help='int: maximum number of stream members (default: 5000)',
                        action='store', type=int, default=5000)
    parser.add_argument('--md_margin_vel', help='float: Minimum deviation difference (in percent) of the original to predicted stream members (default: 0.01)',
                        action='store', type=float, default=0.01)
    parser.add_argument('--md_margin_pos', help='float: Minimum deviation difference (in percent) of the original to predicted stream members (default: 0.01)',
                        action='store', type=float, default=0.01)
    parser.add_argument('--com_margin', help='float: Deviation parameter of the original to predicted centroid (default: 0.2)',
                        action='store', type=float, default=0.2)
    parser.add_argument('--cp_lo', help='float: scaling factor multiplying the positional features against the proper motion variables (lowerst value)',
                        action='store', type=float, default=0.1)
    parser.add_argument('--cp_hi', help='float: scaling factor multiplying the positional features against the proper motion variables (highest value)',
                        action='store', type=float, default=10)
    # Number of search points parameters
    parser.add_argument('--n_searches', help='int: Number of hyper-parameters searched by single CPU (default: 1000 ~3h runtime)',
                        action='store', type=int, default=1000)
    parser.add_argument('--n_cores', help='int: Number of cores to run the program on',
                        action='store', type=int, default=20)
    # Data and data storage
    parser.add_argument('--fpath_train', help='str: Training data path',
                        action='store', type=str)
    parser.add_argument('--fpath_predict', help='str: Prediction data path',
                        action='store', type=str)
    parser.add_argument('--save_dir', help='str: Directory path to save the output files in',
                        action='store', type=str, default='./')


    command_line_args = parser.parse_args(user_argv)
    max_pts_full = command_line_args.max_pts_full
    md_margin_vel = command_line_args.md_margin_vel
    md_margin_pos = command_line_args.md_margin_pos
    com_margin = command_line_args.com_margin
    cp_lo = command_line_args.cp_lo
    cp_hi = command_line_args.cp_hi

    n_searches_per_process = command_line_args.n_searches
    n_cores = command_line_args.n_cores

    fpath_train = command_line_args.fpath_train
    fpath_predict = command_line_args.fpath_predict
    data_path = command_line_args.save_dir

    main()
