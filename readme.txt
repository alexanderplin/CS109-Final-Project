File Descriptions:

[1-5].*                           : Files that should be run in sequential order
10_year_data.xlsx                 : All articles
10_year_filtered_data.xlsx        : Articles that contain exclusively mention Apple
12-4-06-to-12-3-16-Quotes.csv     : Stock Quotes for AAPL
best_rf_thresh_gt.xlsx            : hypertuned random forests models threshold > 0.5% = class 1
best_rf_thresh_lt.xlsx            : hypertuned random forests models threshold < 0.5% = class 0
gt_scores.xlsx                    : scores of best_rf_thresh_gt models
lt_scores.xlsx                    : scores of best_rf_thresh_lt models
hypertune_lt_thresh.py            : python script that hypertunes random forest threshold < 0.5% = class 0
hypertune_gt_thresh.py            : python script that hypertunes random forest threshold > 0.5% = class 1 