from multiprocessing import Pool, TimeoutError
import time
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
import nltk
from nltk.corpus import stopwords
from datetime import timedelta
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
import math
import string 
from six.moves.html_parser import HTMLParser
import urllib2
import json
import time
from functools import wraps
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import EnglishStemmer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

def vectorize(series, threshold_low = 0, threshold_high = None):
	if threshold_high is not None:
		vectorizer = CountVectorizer(stop_words = 'english', min_df = threshold_low + 1, max_df = threshold_high)
	else:
		vectorizer = CountVectorizer(stop_words = 'english', min_df = threshold_low + 1)
		
	vectorized = vectorizer.fit_transform(series).toarray()
	feat_names = vectorizer.get_feature_names()
	return vectorized, feat_names

def make_pca(bag_to_use, pca_components):
	pca = PCA(n_components = pca_components)
	pca.fit(bag_to_use)
	return pca.transform(bag_to_use)

""" pred_series = panda series """
def simulate_helper(filtered_df, pred_series):
    
    profits = []

    for index, pred in enumerate(pred_series):
        curr_share_value = filtered_df.iloc[index]['price_before']
        new_share_value = filtered_df.iloc[index]['price_after']
        actual = filtered_df.iloc[index]['y']
        if (pred == 1): # buy
            profits += [new_share_value - curr_share_value]

        elif (pred == 0): # sell
             profits += [curr_share_value - new_share_value]

#     print 'Average profits:', np.mean(profits)
#     print 'Total profits:', np.sum(profits)
#     print 
    return np.mean(profits)

def simulate(bag_to_use, model_to_use, filtered_df):
    filtered_df['random'] = ''
    filtered_df['model'] = ''
    print "before"
    for i in range (0, filtered_df.shape[0]):
        filtered_df['random'].values[i] = round(np.random.rand())
    print "after"
    filtered_df['model'] = model_to_use.predict(bag_to_use)
    # calculated from simulate_helper(filtered_df, np.zeros((filtered_df.shape[0], 1)) + 1)
    always_buy_average_score = 0.0743334265734
#     print 'Random (50/50):'
#     simulate_helper(filtered_df, filtered_df['random'])
#    print 'Model prediction:'
    model_average_score = simulate_helper(filtered_df, filtered_df['model'])
#     print 'Always buy:'
#     simulate_helper(filtered_df, np.zeros((filtered_df.shape[0], 1)) + 1)
#     print 'Always sell:'
#     simulate_helper(filtered_df, np.zeros((filtered_df.shape[0], 1)))
#     print 'Fortune teller:'
#     simulate_helper(filtered_df, filtered_df['y'])
    return model_average_score - always_buy_average_score

def f(id, filtered_df, pos_stem, name):
	result = []
	for n in range(1,4):
		for pca_n in [0,2]:
			print "Worker "+ str(id) +" : " + str(n) + "/3 and " + str(pca_n+1) + "/2"
			vectorized_1gram, vectorized_1gram_names = vectorize(pos_stem, n)

			if pca_n != 0:
				# project to the data onto the two axes
				bag_to_use = make_pca(vectorized_1gram, pca_n)
			else:
				bag_to_use = vectorized_1gram

			y = filtered_df['y'].values

			x_train, x_test, y_train, y_test = train_test_split(bag_to_use, 
												y, 
												test_size = 0.4, 
												random_state = 42)

			# to keep track of the best model
			best_avg = 0
			best_trees_avg = None
			best_depth_avg = None
			best_nodes_avg = None

			### RF CV
			# parameters for tuning
			n_trees = np.arange(10, 200, 20) 
			depths = np.arange(2, 10)
			leaf_nodes = np.arange(2, 10)
			num_folds = 4

			# iterate through trees and depths
			for nodes in leaf_nodes:
				for trees in n_trees:
					for depth in depths:
						# cross validation for every experiment
						k_folds = KFold(x_train.shape[0], n_folds = num_folds, shuffle = True)
						scores = []

						# for each fold
						for train_indices, validation_indices in k_folds:
							# generate training data
							x_train_cv = x_train[train_indices]
							y_train_cv = y_train[train_indices]
							# generate validation data
							x_validate = x_train[validation_indices]
							y_validate = y_train[validation_indices]

							# fit random forest on training data
							rf = RandomForest(n_estimators = trees, max_depth = depth, max_leaf_nodes = nodes, class_weight = 'balanced')
							rf.fit(x_train_cv, y_train_cv)
							print "HITTING"
							# score on validation data
							scores += [simulate(x_validate, rf, filtered_df)]

						# record and report accuracy
						average_score = np.mean(scores, axis = 0)

						# update our record of the best parameters see so far
						if np.mean(average_score) >= best_avg:
							best_avg = np.mean(average_score)
							best_trees_avg = trees
							best_depth_avg = depth
							best_nodes_avg = nodes

			result += [(name,n,pca_n, best_trees_avg, best_depth_avg, best_nodes_avg, best_avg)]
	return result

def filter_misc(series, pos = None, stem = False):
	new_series = pd.Series(index = series.index)

	if stem == True:
		# does not stem stopwords
		stemmer = EnglishStemmer(ignore_stopwords = True)


	if pos is not None:
		for index, text in enumerate(series):
			new_series.iloc[index] = ' '.join([y for y,tag in nltk.pos_tag(nltk.word_tokenize(text)) if tag in pos])
		pos_flag = True
	else:
		pos_flag = False
		
	if stem is True:
		# if both pos and stem
		if pos_flag == True:
			use_series = new_series
		# just stem
		else:
			use_series = series
		for index, text in enumerate(use_series):
			text_list = text.split()
			stemmed_text = []
			for word in text_list:
				stemmed_text += [stemmer.stem(word)]
			new_series.iloc[index] = ' '.join(stemmed_text)
			
	return new_series

""" Function returns the nearest date closest to the given date
		stock_df: a dataframe
		date: a date in date format 
		time_type: "before" or "after"
			"before": finds the nearest business day before the given date
			"after": finds the nearest business day after the given date """
def find_nearest_biz_day(stock_df, date, time_type, counter = 0):
	if counter > 10:
		return None
	price = stock_df[stock_df['date'] == date]['close'].values
	if time_type == 'after':
		# if date exists (i.e., not weekend or holiday)
		if price.size != 0:
			return date
		# if weekend or holiday
		else:
			new_date = date + timedelta(days = 1)
			counter += 1
			return find_nearest_biz_day(stock_df, date + timedelta(days = 1), 'after', counter)
	elif time_type == 'before' or time_type == 'before and done':
		day_before_price = stock_df[stock_df['date'] == date - timedelta(days = 1)]['close'].values
		# if date exists (i.e., not weekend or holiday)
		if price.size != 0 and day_before_price != 0 and time_type == 'before':
			return date - timedelta(days = 1)
		elif price.size != 0 and time_type == 'before and done':
			return date
		# if weekend or holiday
		else:
			counter+= 1
			return find_nearest_biz_day(stock_df, date - timedelta(days = 1), 'before and done', counter)

if __name__ == '__main__':

	filtered_df = pd.read_excel('10_year_filtered_data.xlsx')
	filtered_df_pos = deepcopy(filtered_df)
	filtered_df_stem = deepcopy(filtered_df)
	filtered_df_pos_stem = deepcopy(filtered_df)

	filtered_df_pos['headline'] = filter_misc(filtered_df_pos['headline'], pos = ['NN', 'JJ'], stem = False)
	filtered_df_stem['headline'] = filter_misc(filtered_df_stem['headline'], pos = None, stem = True)
	filtered_df_pos_stem['headline'] = filter_misc(filtered_df_pos_stem['headline'], pos = ['NN', 'JJ'], stem = True)

	# load stock price
	stock_price = pd.read_csv('12-4-06-to-12-3-16-Quotes.csv', parse_dates = [0], keep_date_col = True, encoding = 'cp1252')
	stock_price.head(n=5)

	results = []
	before_p = []
	after_p = []
	price_list = []

	# iterate through news article's weekday along with corresponding published date
	for date in filtered_df['pub_date']:
		compare_date_after = find_nearest_biz_day(stock_price, date, 'after')
		compare_date_before = find_nearest_biz_day(stock_price, date, 'before')
		if compare_date_after is None:
			print 'No after:', date
		if compare_date_before is None:
			print 'No before:', date
		price_after = stock_price[stock_price['date'] == compare_date_after]['close'].values
		price_before = stock_price[stock_price['date'] == compare_date_before]['close'].values
		
		# stock price the day after is higher than stock price day before, encode binary 1
		price_diff = price_after[0] - price_before[0]
		threshold = 0.005 * -1*price_before[0]
		if price_diff <= threshold:
			results += [0]
		# no increase or decrease in stock price, encode binary 0
		else:
			results += [1]
			
		# true before and after values
		before_p += [price_before[0]]
		after_p += [price_after[0]]

	filtered_df['price_before'] = before_p
	filtered_df['price_after'] = after_p
	filtered_df['price_diff'] = filtered_df['price_after'] - filtered_df['price_before']
	filtered_df['y'] = results

	print "STARTING POOL WORKERS"

	a = [1,2,3,4]

	pool = Pool(processes=4)              # start 4 worker processes
	multiple_results = [pool.apply_async(f, (a.pop(), filtered_df, pos_stem,name,)) for pos_stem, name in zip([filtered_df['headline'], filtered_df_pos['headline'], filtered_df_stem['headline'], filtered_df_pos_stem['headline']], ['no pos no stem', 'yes pos no stem', 'no pos yes stem', 'yes pos yes stem'])]
	
	for res in multiple_results:
		print res.get()