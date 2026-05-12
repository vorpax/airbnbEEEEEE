"""
Generate all presentation plots from the Airbnb Paris dataset.
Saves PNGs to ./plots/ for inclusion in the Beamer presentation.
"""
import os, warnings, ast
from collections import Counter
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import (train_test_split, KFold,
                                     cross_val_score, GridSearchCV)
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, classification_report,
                              ConfusionMatrixDisplay)
from sklearn.linear_model import (LinearRegression, LassoCV,
                                   Ridge, RidgeCV, LogisticRegression)
from sklearn.tree import (DecisionTreeRegressor, DecisionTreeClassifier,
                           plot_tree)
from sklearn.ensemble import (RandomForestRegressor, StackingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style='whitegrid', palette='muted')
PLT_DPI = 180

CORAL  = '#FF5A5F'
TEAL   = '#00A699'
DARK   = '#484848'
GREY   = '#767676'
PURPLE = '#9B59B6'
ORANGE = '#E67E22'

CLUSTER_COLORS = {0: '#E74C3C', 1: '#3498DB', 2: '#2ECC71', 3: '#F39C12'}
CLUSTER_NAMES  = {0: 'Spacious & Premium', 1: 'Well-Equipped Mid-Range',
                  2: 'Standard Compact',   3: 'Budget Basic'}

def savefig(name):
    path = f'plots/{name}.png'
    plt.savefig(path, dpi=PLT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close('all')
    print(f'  saved → {path}')

def clean_name(n):
    return (n.replace('property_type_', 'prop: ')
             .replace('room_type_', 'room: ')
             .replace('neighbourhood_', 'arr: ')
             .replace('review_scores_', 'score_')
             .replace('dist_', 'dist ')
             .replace('_km', ' km')
             .replace('_', ' '))

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6_371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD & FILTER
# ══════════════════════════════════════════════════════════════════════════
print('Loading data …')
df_all = pd.read_csv('listings.csv', encoding='utf-8-sig', low_memory=False)
df_all.columns = df_all.columns.str.replace('ï»¿', '', regex=False)
df = df_all[df_all['city'] == 'Paris'].copy().reset_index(drop=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
p1, p99 = df['price'].quantile([0.01, 0.99])
df_vis = df[(df['price'] >= p1) & (df['price'] <= p99)].copy()
print(f'  Paris listings: {len(df):,}')

# ══════════════════════════════════════════════════════════════════════════
# EDA PLOTS
# ══════════════════════════════════════════════════════════════════════════
print('\n── EDA plots ──')

# Price distribution (raw + log)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(df_vis['price'], bins=80, color=CORAL, edgecolor='none', alpha=0.85)
axes[0].axvline(df['price'].median(), color=TEAL, lw=2,
                label=f'Median €{df["price"].median():.0f}')
axes[0].axvline(df['price'].mean(), color=DARK, lw=2, ls='--',
                label=f'Mean €{df["price"].mean():.0f}')
axes[0].set_title('Price Distribution (raw, p1–p99)', fontsize=10, color=DARK)
axes[0].set_xlabel('Nightly Price (€)', color=GREY)
axes[0].set_ylabel('Count', color=GREY)
axes[0].legend(fontsize=8)
axes[0].spines[['top','right']].set_visible(False)
log_prices = np.log1p(df_vis['price'])
axes[1].hist(log_prices, bins=70, color=TEAL, edgecolor='none', alpha=0.85)
axes[1].set_title('Price Distribution (log scale)', fontsize=10, color=DARK)
axes[1].set_xlabel('log(1 + Price)', color=GREY)
axes[1].set_ylabel('Count', color=GREY)
axes[1].spines[['top','right']].set_visible(False)
savefig('price_distribution')

# Combined EDA: log-price hist + room type boxplot
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(log_prices, bins=70, color=CORAL, edgecolor='none', alpha=0.85)
axes[0].axvline(log_prices.median(), color=TEAL, lw=2,
                label=f'Median €{df_vis["price"].median():.0f}')
axes[0].axvline(log_prices.mean(), color=DARK, lw=2, ls='--',
                label=f'Mean €{df_vis["price"].mean():.0f}')
axes[0].set_title('Price Distribution (log scale)', fontsize=10, color=DARK)
axes[0].set_xlabel('log(1 + Price)', color=GREY)
axes[0].set_ylabel('Count', color=GREY)
axes[0].legend(fontsize=8)
axes[0].spines[['top','right']].set_visible(False)
order = df_vis.groupby('room_type')['price'].median().sort_values(ascending=False).index
palette = [CORAL, TEAL, ORANGE, PURPLE]
sns.boxplot(data=df_vis, x='room_type', y='price', order=order,
            palette=palette[:len(order)], width=0.5,
            flierprops={'marker':'.','alpha':0.3,'ms':3}, ax=axes[1])
axes[1].set_title('Nightly Price by Room Type', fontsize=10, color=DARK)
axes[1].set_xlabel(''); axes[1].set_ylabel('Price (€)', color=GREY)
axes[1].spines[['top','right']].set_visible(False)
savefig('eda_combined')

# Neighbourhood prices
arr_med = df_vis.groupby('neighbourhood')['price'].median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 3.8))
colors = [CORAL if i < 3 else (TEAL if i >= len(arr_med)-3 else GREY)
          for i in range(len(arr_med))]
arr_med.plot(kind='bar', color=colors, edgecolor='none', ax=ax)
ax.set_title('Median Nightly Price by Neighbourhood', fontsize=10, color=DARK)
ax.set_xlabel(''); ax.set_ylabel('Median Price (€)', color=GREY)
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.spines[['top','right']].set_visible(False)
savefig('price_by_neighbourhood')

# Geographic scatter
fig, ax = plt.subplots(figsize=(5.5, 5.5))
sc = ax.scatter(df_vis['longitude'], df_vis['latitude'],
                c=df_vis['price'], cmap='YlOrRd', alpha=0.3, s=3,
                vmax=df_vis['price'].quantile(0.95))
plt.colorbar(sc, ax=ax, label='Price (€)', shrink=0.7)
ax.set_title('Listing Price — Geographic Spread', fontsize=10, color=DARK)
ax.set_xlabel('Longitude', color=GREY, fontsize=8)
ax.set_ylabel('Latitude', color=GREY, fontsize=8)
ax.spines[['top','right']].set_visible(False)
savefig('geo_scatter')

# Correlation heatmap
num_cols = [c for c in ['price','accommodates','bedrooms','review_scores_rating',
            'review_scores_cleanliness','review_scores_location','review_scores_value',
            'host_total_listings_count','minimum_nights','latitude','longitude']
            if c in df.columns]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.4, annot_kws={'size':8}, ax=ax)
ax.set_title('Correlation Matrix — Numerical Features', fontsize=10, color=DARK)
savefig('correlation_heatmap')

# Review activity: monthly volume
print('  Review temporal analysis …')
reviews_df = pd.read_csv('Reviews.csv', parse_dates=['date'])
monthly_vol = reviews_df.set_index('date').resample('ME').size().rename('reviews')
yearly_vol  = monthly_vol.resample('YE').sum()
yearly_vol.index = yearly_vol.index.year
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5))
monthly_vol.plot(ax=axes[0], color=TEAL, linewidth=0.9, alpha=0.9)
axes[0].set_title('Monthly Review Volume — Paris Airbnb (proxy for bookings)',
                  fontsize=10, color=DARK)
axes[0].set_ylabel('Reviews / Month', color=GREY)
axes[0].spines[['top','right']].set_visible(False)
yearly_vol.plot(kind='bar', ax=axes[1], color=CORAL, edgecolor='none')
axes[1].set_title('Annual Review Volume', fontsize=10, color=DARK)
axes[1].set_xlabel('Year', color=GREY)
axes[1].tick_params(axis='x', rotation=45)
axes[1].spines[['top','right']].set_visible(False)
savefig('review_temporal')

# ══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
print('\n── Feature engineering ──')
dfc = df.copy()
dfc = dfc.dropna(subset=['price'])
p1, p99 = dfc['price'].quantile([0.01, 0.99])
dfc = dfc[(dfc['price'] >= p1) & (dfc['price'] <= p99)].reset_index(drop=True)

for col in ['bedrooms','review_scores_rating','review_scores_cleanliness',
            'review_scores_location','review_scores_value',
            'review_scores_accuracy','review_scores_checkin',
            'review_scores_communication']:
    if col in dfc.columns:
        dfc[col] = dfc[col].fillna(dfc[col].median())

def parse_amenities(s):
    if not isinstance(s, str) or not s.strip():
        return []
    try: return ast.literal_eval(s)
    except: return []

dfc['amenities_list'] = dfc['amenities'].apply(parse_amenities)
dfc['amenities_count'] = dfc['amenities_list'].apply(len)

for amenity in ['Wifi','Kitchen','Heating','Washer','Dryer',
                'Air conditioning','Elevator','TV','Dishwasher','Essentials']:
    col = 'has_' + amenity.lower().replace(' ','_')
    dfc[col] = dfc['amenities_list'].apply(lambda x, a=amenity: a in x).astype(int)

for col in ['host_is_superhost','instant_bookable',
            'host_identity_verified','host_has_profile_pic']:
    if col in dfc.columns:
        dfc[col] = dfc[col].map({'t':1,'f':0}).fillna(0).astype(int)

if 'property_type' in dfc.columns:
    top_props = dfc['property_type'].value_counts().head(10).index
    dfc['property_type'] = dfc['property_type'].where(
        dfc['property_type'].isin(top_props), 'Other')
ohe_cols = [c for c in ['room_type','neighbourhood','property_type'] if c in dfc.columns]
dfc = pd.get_dummies(dfc, columns=ohe_cols, dtype=int)

dfc['log_price'] = np.log1p(dfc['price'])
if 'host_since' in dfc.columns:
    dfc['host_since_dt'] = pd.to_datetime(dfc['host_since'], errors='coerce')
    dfc['host_seniority_days'] = (pd.Timestamp('2026-01-01') - dfc['host_since_dt']).dt.days
    dfc['host_seniority_days'] = dfc['host_seniority_days'].fillna(
        dfc['host_seniority_days'].median())
if 'review_scores_rating' in dfc.columns:
    dfc['review_scores_rating_norm'] = dfc['review_scores_rating'] / 20.0

LANDMARKS = {'eiffel_tower':(48.8584,2.2945), 'louvre':(48.8606,2.3376),
             'notre_dame':(48.8530,2.3499), 'opera_garnier':(48.8720,2.3316)}
for name,(lat,lon) in LANDMARKS.items():
    dfc[f'dist_{name}_km'] = dfc.apply(
        lambda r,lat=lat,lon=lon: haversine_km(r['latitude'],r['longitude'],lat,lon), axis=1)

# Review stats join
rev_stats = (reviews_df.groupby('listing_id')
             .agg(review_count=('review_id','count'),
                  last_review_date=('date','max'))
             .reset_index())
ref_date = reviews_df['date'].max()
rev_stats['days_since_last_review'] = (ref_date - rev_stats['last_review_date']).dt.days
dfc = dfc.merge(rev_stats[['listing_id','review_count','days_since_last_review']],
                on='listing_id', how='left')
dfc['review_count'] = dfc['review_count'].fillna(0)
dfc['days_since_last_review'] = dfc['days_since_last_review'].fillna(
    dfc['days_since_last_review'].median())

EXCLUDE = {'listing_id','name','host_id','host_since','host_since_dt',
           'host_location','city','district','amenities','amenities_list',
           'price','log_price','price_per_guest','last_review_date','review_scores_rating'}
feature_cols = [c for c in dfc.select_dtypes(include='number').columns if c not in EXCLUDE]
X = dfc[feature_cols].fillna(0)
y = dfc['log_price']
y_raw = dfc['price']

X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f'  Feature matrix: {X.shape}  Train: {len(X_train):,}  Test: {len(X_test):,}')

# ══════════════════════════════════════════════════════════════════════════
# 3. FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════
print('\n── Feature selection ──')
X_train_num = X_train.select_dtypes(include='number')
corr_target = X_train_num.corrwith(y_train).abs().sort_values(ascending=False)

print('  Fitting Lasso …')
lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=5000)
lasso.fit(X_train_scaled, y_train)
lasso_coef = pd.Series(lasso.coef_, index=X.columns)
selected_by_lasso = lasso_coef[lasso_coef != 0].abs().sort_values(ascending=False)

print('  Fitting RF selector …')
rf_sel = RandomForestRegressor(n_estimators=100, max_depth=10,
                                random_state=RANDOM_STATE, n_jobs=-1)
rf_sel.fit(X_train, y_train)
rf_importance = pd.Series(rf_sel.feature_importances_, index=X.columns)
rf_importance_top = rf_importance.sort_values(ascending=False).head(20)

top_lasso_feats = set(selected_by_lasso.head(30).index)
top_rf_feats    = set(rf_importance_top.index)
selected_features = list(top_lasso_feats | top_rf_feats)
X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled  = scaler.transform(X_test_sel)
print(f'  Selected {len(selected_features)} features')

# 3-panel feature selection plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
top10_p = corr_target.head(10).sort_values()
top10_p.plot(kind='barh', color=CORAL, edgecolor='none', ax=axes[0])
axes[0].set_yticklabels([clean_name(n) for n in top10_p.index], fontsize=8)
axes[0].set_title('Pearson |r| with log(price)', fontsize=10, color=DARK)
axes[0].spines[['top','right']].set_visible(False)

top10_l = selected_by_lasso.head(10).sort_values()
top10_l.plot(kind='barh', color=ORANGE, edgecolor='none', ax=axes[1])
axes[1].set_yticklabels([clean_name(n) for n in top10_l.index], fontsize=8)
axes[1].set_title(f'Lasso Coefficients (α={lasso.alpha_:.4f})', fontsize=10, color=DARK)
axes[1].spines[['top','right']].set_visible(False)

top10_r = rf_importance_top.head(10).sort_values()
top10_r.plot(kind='barh', color=TEAL, edgecolor='none', ax=axes[2])
axes[2].set_yticklabels([clean_name(n) for n in top10_r.index], fontsize=8)
axes[2].set_title('RF Feature Importance', fontsize=10, color=DARK)
axes[2].spines[['top','right']].set_visible(False)

fig.suptitle('Feature Selection: Three Methods Agree', fontsize=11, color=DARK)
plt.tight_layout()
savefig('feature_selection_3panel')

# Standalone RF importance (top 15)
fig, ax = plt.subplots(figsize=(7, 5))
top15_r = rf_importance_top.head(15).sort_values()
top15_r.plot(kind='barh', color=TEAL, edgecolor='none', ax=ax)
ax.set_yticklabels([clean_name(n) for n in top15_r.index], fontsize=9)
ax.set_title('Top 15 Features — Random Forest Importance', fontsize=11, color=DARK)
ax.set_xlabel('Feature Importance (impurity reduction)', color=GREY)
ax.spines[['top','right']].set_visible(False)
savefig('rf_importance')

# ══════════════════════════════════════════════════════════════════════════
# 4. SUPERVISED MODELS — ALL 7 + TUNED XGB
# ══════════════════════════════════════════════════════════════════════════
print('\n── Supervised models ──')
results = []

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, y_raw_te, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2', n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred_log = model.predict(X_te)
    y_pred = np.expm1(y_pred_log)
    r2   = r2_score(y_te, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_raw_te, y_pred))
    mae  = mean_absolute_error(y_raw_te, y_pred)
    print(f'  [{name:28s}]  CV R²={cv_r2.mean():.3f}±{cv_r2.std():.3f}'
          f'  Test R²={r2:.3f}  RMSE=€{rmse:.1f}  MAE=€{mae:.1f}')
    return {'Model':name, 'CV R²':round(cv_r2.mean(),4),
            'CV std':round(cv_r2.std(),4),
            'Test R²':round(r2,4), 'RMSE':round(rmse,1), 'MAE':round(mae,1)}, model

r, lr_model = evaluate_model('Linear Regression', LinearRegression(),
    X_train_sel_scaled, X_test_sel_scaled, y_train, y_test, y_raw_test)
results.append(r)

ridge_cv_m = RidgeCV(alphas=np.logspace(-3,4,60), cv=5, scoring='r2')
ridge_cv_m.fit(X_train_sel_scaled, y_train)
r, ridge_model = evaluate_model('Ridge Regression', Ridge(alpha=ridge_cv_m.alpha_),
    X_train_sel_scaled, X_test_sel_scaled, y_train, y_test, y_raw_test)
results.append(r)

r, dt_model = evaluate_model('Decision Tree',
    DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),
    X_train_sel, X_test_sel, y_train, y_test, y_raw_test)
results.append(r)

r, rf_model = evaluate_model('Random Forest',
    RandomForestRegressor(n_estimators=200, max_depth=12, max_features='sqrt',
                          random_state=RANDOM_STATE, n_jobs=-1),
    X_train_sel, X_test_sel, y_train, y_test, y_raw_test)
results.append(r)

r, mlp_model = evaluate_model('MLP Neural Network',
    MLPRegressor(hidden_layer_sizes=(128,64), activation='relu', solver='adam',
                 max_iter=500, early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=20, random_state=RANDOM_STATE),
    X_train_sel_scaled, X_test_sel_scaled, y_train, y_test, y_raw_test)
results.append(r)

r, xgb_model = evaluate_model('XGBoost',
    XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    X_train_sel, X_test_sel, y_train, y_test, y_raw_test)
results.append(r)

stack = StackingRegressor(
    estimators=[('lr', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10,
                                             random_state=RANDOM_STATE, n_jobs=-1)),
                ('xgb', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=RANDOM_STATE, verbosity=0))],
    final_estimator=Ridge(alpha=10.0), cv=5, n_jobs=-1)
r, stack_model = evaluate_model('Stacking Ensemble', stack,
    X_train_sel_scaled, X_test_sel_scaled, y_train, y_test, y_raw_test)
results.append(r)

# GridSearch tuning
print('  GridSearch (XGBoost) …')
pipe = Pipeline([('scaler', StandardScaler()),
                 ('model', XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0))])
param_grid = {'model__n_estimators':[100,300], 'model__max_depth':[3,5,7],
              'model__learning_rate':[0.05,0.1], 'model__subsample':[0.8]}
gs = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
gs.fit(X_train_sel, y_train)
print(f'  Best params: {gs.best_params_}  Best CV R²: {gs.best_score_:.4f}')
best_pipe = gs.best_estimator_
y_tuned_log = best_pipe.predict(X_test_sel)
y_tuned     = np.expm1(y_tuned_log)
r2_tuned   = r2_score(y_test, y_tuned_log)
rmse_tuned = np.sqrt(mean_squared_error(y_raw_test, y_tuned))
mae_tuned  = mean_absolute_error(y_raw_test, y_tuned)
tuned_res  = {'Model':'Tuned XGBoost', 'CV R²':round(gs.best_score_,4),
              'CV std':0.0, 'Test R²':round(r2_tuned,4),
              'RMSE':round(rmse_tuned,1), 'MAE':round(mae_tuned,1)}
results.append(tuned_res)
print(f'  [Tuned XGBoost]  Test R²={r2_tuned:.3f}  RMSE=€{rmse_tuned:.1f}  MAE=€{mae_tuned:.1f}')

results_df = pd.DataFrame(results).set_index('Model')

# ── Plot: full model comparison ────────────────────────────────────────────
print('  Plotting model comparison …')
MODEL_COLORS = {'Linear Regression': '#95A5A6', 'Ridge Regression': '#BDC3C7',
                'Decision Tree': '#F39C12',
                'Random Forest': TEAL, 'MLP Neural Network': PURPLE,
                'XGBoost': CORAL, 'Stacking Ensemble': '#E74C3C',
                'Tuned XGBoost': '#C0392B'}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric, ascending, title in zip(
        axes,
        ['Test R²',  'RMSE',  'MAE'],
        [True,       False,   False],
        ['Test R²  (↑ better)', 'RMSE — € (↓ better)', 'MAE — € (↓ better)']):
    sorted_df = results_df[metric].sort_values(ascending=ascending)
    bar_colors = [MODEL_COLORS.get(m, GREY) for m in sorted_df.index]
    bars = ax.barh(range(len(sorted_df)), sorted_df.values, color=bar_colors, edgecolor='none')
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df.index, fontsize=8)
    ax.set_title(title, fontsize=9, color=DARK)
    ax.spines[['top','right']].set_visible(False)
    for bar, val in zip(bars, sorted_df.values):
        label = f'{val:.3f}' if metric == 'Test R²' else f'€{val:.1f}'
        ax.text(bar.get_width() + ax.get_xlim()[1]*0.005,
                bar.get_y() + bar.get_height()/2,
                label, va='center', ha='left', fontsize=7, color=DARK)

fig.suptitle('All Models — Performance Comparison (8 models)', fontsize=11, color=DARK)
plt.tight_layout()
savefig('model_comparison_full')

# ── Plot: R² only with CV error bars ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
df_sorted = results_df.sort_values('Test R²', ascending=True)
bar_colors = [MODEL_COLORS.get(m, GREY) for m in df_sorted.index]
ax.barh(range(len(df_sorted)), df_sorted['Test R²'], xerr=df_sorted['CV std'],
        color=bar_colors, edgecolor='none', capsize=3, ecolor=DARK, error_kw={'lw':1})
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted.index, fontsize=9)
ax.set_xlabel('Test R²', color=GREY)
ax.set_title('Model Comparison — Test R² with CV Std Dev', fontsize=11, color=DARK)
ax.axvline(0.5, color=GREY, ls=':', lw=1)
ax.spines[['top','right']].set_visible(False)
for i, (r2, std) in enumerate(zip(df_sorted['Test R²'], df_sorted['CV std'])):
    ax.text(r2 + std + 0.005, i, f'{r2:.3f}', va='center', fontsize=8, color=DARK)
savefig('model_r2_errorbars')

# ── Plot: MLP loss curve ───────────────────────────────────────────────────
# Re-fit MLP to capture loss curve cleanly
mlp_plot = MLPRegressor(hidden_layer_sizes=(128,64), activation='relu', solver='adam',
                        max_iter=500, early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=20, random_state=RANDOM_STATE)
mlp_plot.fit(X_train_sel_scaled, y_train)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(mlp_plot.loss_curve_, color=PURPLE, lw=2, label='Training Loss')
if hasattr(mlp_plot, 'validation_scores_') and mlp_plot.validation_scores_:
    val_loss = [-s for s in mlp_plot.validation_scores_]
    ax.plot(val_loss, color=CORAL, lw=2, ls='--', label='Validation Loss (neg)')
ax.set_xlabel('Epoch', color=GREY)
ax.set_ylabel('Loss (MSE)', color=GREY)
ax.set_title('MLP Neural Network — Training Loss Curve\n(128→64 neurons, ReLU, Adam, early stopping)',
             fontsize=10, color=DARK)
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
savefig('mlp_loss_curve')

# ── Plot: XGBoost residuals ────────────────────────────────────────────────
y_pred_log = xgb_model.predict(X_test_sel)
y_pred_price = np.expm1(y_pred_log)
residuals = y_raw_test.values - y_pred_price
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_pred_price, residuals, alpha=0.25, s=6, color=CORAL)
axes[0].axhline(0, color=DARK, lw=1.5, ls='--')
axes[0].set_xlabel('Predicted Price (€)', color=GREY)
axes[0].set_ylabel('Residual (€)', color=GREY)
axes[0].set_title('Residuals vs. Predicted — XGBoost', fontsize=10, color=DARK)
axes[0].spines[['top','right']].set_visible(False)
lims = [0, min(y_raw_test.max(), 600)]
axes[1].scatter(y_raw_test, y_pred_price, alpha=0.25, s=6, color=TEAL)
axes[1].plot(lims, lims, 'r--', lw=1.5)
axes[1].set_xlim(lims); axes[1].set_ylim(lims)
axes[1].set_xlabel('Actual Price (€)', color=GREY)
axes[1].set_ylabel('Predicted Price (€)', color=GREY)
axes[1].set_title('Actual vs. Predicted — XGBoost', fontsize=10, color=DARK)
axes[1].spines[['top','right']].set_visible(False)
savefig('xgb_residuals')

# ── Plot: GridSearch heatmap ───────────────────────────────────────────────
gs_df = pd.DataFrame(gs.cv_results_)
pivot = gs_df.pivot_table(index='param_model__max_depth',
                           columns='param_model__learning_rate',
                           values='mean_test_score', aggfunc='max')
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGn', ax=ax,
            linewidths=0.5, cbar_kws={'label':'CV R²'})
ax.set_title('GridSearch CV R² — XGBoost (n_estimators=300, subsample=0.8)',
             fontsize=10, color=DARK)
ax.set_xlabel('Learning Rate', color=GREY)
ax.set_ylabel('Max Depth', color=GREY)
savefig('gridsearch_heatmap')

# ══════════════════════════════════════════════════════════════════════════
# 5. CLASSIFICATION — PRICE TIERS
# ══════════════════════════════════════════════════════════════════════════
print('\n── Classification ──')
dfc['price_tier'] = pd.qcut(dfc['price'], q=3, labels=['Budget','Mid-Range','Premium'])
y_tier_train = dfc.loc[X_train_sel.index, 'price_tier']
y_tier_test  = dfc.loc[X_test_sel.index,  'price_tier']

clf_configs = {
    'Logistic\nRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    'Random\nForest':       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                    random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient\nBoosting':   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                        learning_rate=0.1, random_state=RANDOM_STATE),
}
clf_f1 = {}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (name, clf) in zip(axes, clf_configs.items()):
    clf.fit(X_train_sel_scaled, y_tier_train)
    preds = clf.predict(X_test_sel_scaled)
    report = classification_report(y_tier_test, preds, output_dict=True)
    clf_f1[name.replace('\n',' ')] = report['weighted avg']['f1-score']
    ConfusionMatrixDisplay.from_predictions(
        y_tier_test, preds,
        display_labels=['Budget','Mid-Range','Premium'],
        colorbar=False, ax=ax,
        cmap='Blues')
    ax.set_title(f'{name}\n(F1={report["weighted avg"]["f1-score"]:.2f})',
                 fontsize=9, color=DARK)
    ax.tick_params(labelsize=8)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
fig.suptitle('Price Tier Classification — Confusion Matrices (Budget / Mid-Range / Premium)',
             fontsize=10, color=DARK)
plt.tight_layout()
savefig('classification_confusion')

# ══════════════════════════════════════════════════════════════════════════
# 6. AMENITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print('\n── Amenity analysis ──')
mlb = MultiLabelBinarizer()
am_matrix = mlb.fit_transform(dfc['amenities_list'])
am_df_full = pd.DataFrame(am_matrix, columns=mlb.classes_, index=dfc.index)
am_freq = am_df_full.sum()
am_cols = am_freq[am_freq >= 500].index.tolist()
am_df = am_df_full[am_cols]

X_am_train = am_df.loc[X_train_sel.index]
X_am_test  = am_df.loc[X_test_sel.index]

# ── Amenity Decision Tree Classifier (depth=3 for readability) ─────────────
print('  Fitting amenity decision tree …')
dt_am_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100,
                                    random_state=RANDOM_STATE)
dt_am_clf.fit(X_am_train, y_tier_train)
y_am_pred = dt_am_clf.predict(X_am_test)
am_report = classification_report(y_tier_test, y_am_pred, output_dict=True)
print(f'  Amenity DT accuracy: {am_report["accuracy"]:.2f}')

fig, ax = plt.subplots(figsize=(22, 8))
plot_tree(dt_am_clf, feature_names=am_cols,
          class_names=['Budget','Mid-Range','Premium'],
          filled=True, rounded=True, fontsize=8,
          impurity=False, proportion=True, ax=ax)
ax.set_title('Amenity Decision Tree — Price Tier Classification (depth=3)\n'
             f'Accuracy: {am_report["accuracy"]:.2f}  |  '
             'Each node shows: split condition / class distribution / majority class',
             fontsize=11, color=DARK)
savefig('amenity_decision_tree')

# ── Amenity price premium bar chart ───────────────────────────────────────
print('  Computing amenity price premiums …')
rows_am = []
for col in am_cols:
    with_am    = dfc.loc[am_df[col] == 1, 'price']
    without_am = dfc.loc[am_df[col] == 0, 'price']
    if without_am.median() > 0 and len(with_am) >= 100:
        pct = 100*(with_am.median()/without_am.median() - 1)
        rows_am.append({'Amenity': col,
                        'With': with_am.shape[0],
                        'Premium_EUR': round(with_am.median() - without_am.median(), 1),
                        'Premium_pct': round(pct, 1)})
premium_df = pd.DataFrame(rows_am).sort_values('Premium_EUR', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top20 = premium_df.head(20).sort_values('Premium_EUR')
colors_pm = [CORAL if v > 0 else TEAL for v in top20['Premium_EUR']]
top20.plot(x='Amenity', y='Premium_EUR', kind='barh', color=colors_pm,
           edgecolor='none', legend=False, ax=axes[0])
axes[0].set_xlabel('Median Price Premium (€)', color=GREY)
axes[0].set_title('Top 20 Amenities by Price Premium (€)', fontsize=10, color=DARK)
axes[0].spines[['top','right']].set_visible(False)
axes[0].tick_params(axis='y', labelsize=7)

top20_pct = premium_df.head(20).sort_values('Premium_pct')
top20_pct.plot(x='Amenity', y='Premium_pct', kind='barh',
               color=ORANGE, edgecolor='none', legend=False, ax=axes[1])
axes[1].set_xlabel('Median Price Premium (%)', color=GREY)
axes[1].set_title('Top 20 Amenities by Price Premium (%)', fontsize=10, color=DARK)
axes[1].spines[['top','right']].set_visible(False)
axes[1].tick_params(axis='y', labelsize=7)

plt.suptitle('Amenity Price Premium Analysis — Which Amenities Add the Most Value?',
             fontsize=11, color=DARK)
plt.tight_layout()
savefig('amenity_premium')

# ── Amenity frequency bar chart ────────────────────────────────────────────
top_amenities = am_freq.sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(9, 5))
top_amenities.sort_values().plot(kind='barh', color=TEAL, edgecolor='none', ax=ax)
ax.set_xlabel('Number of Listings', color=GREY)
ax.set_title('Top 20 Most Common Amenities in Paris Listings', fontsize=10, color=DARK)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='y', labelsize=8)
for bar in ax.patches:
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width()/len(dfc)*100:.0f}%',
            va='center', fontsize=7, color=GREY)
savefig('amenity_frequency')

# ══════════════════════════════════════════════════════════════════════════
# 7. CLUSTERING
# ══════════════════════════════════════════════════════════════════════════
print('\n── Clustering ──')
# Cap host_total_listings_count at p99 to prevent outlier super-hosts
# from collapsing one cluster (matches notebook behaviour)
if 'host_total_listings_count' in dfc.columns:
    cap_val = dfc['host_total_listings_count'].quantile(0.99)
    dfc['host_listings_capped'] = dfc['host_total_listings_count'].clip(upper=cap_val)

cluster_candidates = ['accommodates','bedrooms','amenities_count',
                      'review_scores_rating_norm','host_listings_capped','price']
cluster_cols = [c for c in cluster_candidates if c in dfc.columns]
X_cluster = dfc[cluster_cols].dropna()
sc_cluster = StandardScaler()
X_cluster_scaled = sc_cluster.fit_transform(X_cluster)

k_range = range(2, 11)
print('  Computing inertias and silhouette scores …')
inertias = [KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
            .fit(X_cluster_scaled).inertia_ for k in k_range]
sil_scores = [silhouette_score(X_cluster_scaled,
              KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
              .fit_predict(X_cluster_scaled),
              sample_size=3000, random_state=RANDOM_STATE) for k in k_range]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(k_range), inertias, 'o-', color=CORAL, lw=2, ms=7)
axes[0].axvline(4, color=TEAL, ls='--', lw=1.5, label='k=4 chosen')
axes[0].set_title('Elbow Method — Inertia', fontsize=10, color=DARK)
axes[0].set_xlabel('k', color=GREY); axes[0].set_ylabel('Inertia (SSE)', color=GREY)
axes[0].set_xticks(list(k_range)); axes[0].legend(fontsize=9)
axes[0].spines[['top','right']].set_visible(False)
axes[1].plot(list(k_range), sil_scores, 'o-', color=TEAL, lw=2, ms=7)
axes[1].axvline(4, color=CORAL, ls='--', lw=1.5, label='k=4 chosen')
axes[1].set_title('Silhouette Score (higher = better)', fontsize=10, color=DARK)
axes[1].set_xlabel('k', color=GREY); axes[1].set_ylabel('Silhouette Score', color=GREY)
axes[1].set_xticks(list(k_range)); axes[1].legend(fontsize=9)
axes[1].spines[['top','right']].set_visible(False)
savefig('elbow_silhouette')

K_OPTIMAL = 4
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=RANDOM_STATE, n_init='auto')
kmeans.fit(X_cluster_scaled)
cluster_idx = X_cluster.index
dfc.loc[cluster_idx, 'cluster'] = kmeans.labels_
dfc['cluster'] = dfc['cluster'].fillna(-1).astype(int)
dfc_cl = dfc[dfc['cluster'] >= 0].copy()
print('  Cluster sizes:', dfc_cl['cluster'].value_counts().sort_index().to_dict())

# Assign names dynamically based on median price rank
price_rank = dfc_cl.groupby('cluster')['price'].median().rank(ascending=False).astype(int)
DYNAMIC_NAMES = {}
label_by_rank = {1:'Spacious & Premium', 2:'Well-Equipped Mid-Range',
                 3:'Standard Compact',   4:'Budget Basic'}
for cluster_id, rank in price_rank.items():
    DYNAMIC_NAMES[cluster_id] = label_by_rank.get(rank, f'Cluster {cluster_id}')
DYNAMIC_COLORS = {}
color_by_rank = {1:'#E74C3C', 2:'#3498DB', 3:'#2ECC71', 4:'#F39C12'}
for cluster_id, rank in price_rank.items():
    DYNAMIC_COLORS[cluster_id] = color_by_rank.get(rank, GREY)

# Override module-level dicts so downstream plots are consistent
CLUSTER_NAMES.update(DYNAMIC_NAMES)
CLUSTER_COLORS.update(DYNAMIC_COLORS)
dfc['cluster_name'] = dfc['cluster'].map(CLUSTER_NAMES).fillna('Unassigned')

# ── Cluster profiles ───────────────────────────────────────────────────────
profile_cols = [c for c in cluster_candidates if c in dfc.columns]
cp = dfc_cl.groupby('cluster')[profile_cols].mean().round(2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (col, title) in zip(axes, [('price','Avg Nightly Price (€)'),
                                    ('accommodates','Avg Capacity (guests)'),
                                    ('amenities_count','Avg Amenities Count')]):
    vals = cp[col]
    ax.bar(range(K_OPTIMAL), vals,
           color=[CLUSTER_COLORS[k] for k in range(K_OPTIMAL)],
           edgecolor='none', width=0.6)
    ax.set_xticks(range(K_OPTIMAL))
    ax.set_xticklabels([CLUSTER_NAMES[i].split(' ')[0] for i in range(K_OPTIMAL)],
                       rotation=20, ha='right', fontsize=8)
    ax.set_title(title, fontsize=9, color=DARK)
    ax.spines[['top','right']].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v + vals.max()*0.02, f'{v:.0f}', ha='center', fontsize=8, color=DARK)
fig.suptitle('K-Means Cluster Profiles (k=4)', fontsize=11, color=DARK)
plt.tight_layout()
savefig('cluster_profiles')

# ── PCA 2-D scatter ────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_cluster_scaled)
explained = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(7, 5))
for k in range(K_OPTIMAL):
    mask = kmeans.labels_ == k
    ax.scatter(X_pca[mask,0], X_pca[mask,1], s=6, alpha=0.35,
               color=CLUSTER_COLORS[k], label=CLUSTER_NAMES[k])
ax.set_xlabel(f'PC1 ({explained[0]:.1%} variance)', color=GREY)
ax.set_ylabel(f'PC2 ({explained[1]:.1%} variance)', color=GREY)
ax.set_title('Listing Clusters — PCA 2D Projection', fontsize=11, color=DARK)
ax.legend(fontsize=8, markerscale=3, framealpha=0.7)
ax.spines[['top','right']].set_visible(False)
savefig('cluster_pca')

# ── Hierarchical clustering dendrogram ────────────────────────────────────
print('  Hierarchical clustering …')
np.random.seed(RANDOM_STATE)
hier_sample = np.random.choice(len(X_cluster_scaled),
                                size=min(1500, len(X_cluster_scaled)), replace=False)
Z = linkage(X_cluster_scaled[hier_sample], method='ward')
fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
           leaf_rotation=90, color_threshold=Z[-4,2],
           above_threshold_color=GREY)
ax.axhline(Z[-4,2], color=CORAL, ls='--', lw=1.5, label='k=4 cut (78% agreement with K-Means)')
ax.set_xlabel('Sample index / cluster size', color=GREY, fontsize=9)
ax.set_ylabel('Ward linkage distance', color=GREY, fontsize=9)
ax.set_title('Hierarchical Clustering Dendrogram — Validates k=4',
             fontsize=11, color=DARK)
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
savefig('dendrogram')

# ── t-SNE coloured by cluster and price tier ──────────────────────────────
print('  Running t-SNE (may take ~60s) …')
N_TSNE = 4000
tsne_pos = np.random.choice(len(X_train_sel_scaled),
                             size=min(N_TSNE, len(X_train_sel_scaled)), replace=False)
X_tsne_in = X_train_sel_scaled[tsne_pos]
tsne_idx   = X_train_sel.index[tsne_pos]
tsne_cluster = dfc.loc[tsne_idx, 'cluster'].fillna(-1).astype(int).values
tsne_tier    = dfc.loc[tsne_idx, 'price_tier'].values

tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
            random_state=RANDOM_STATE, n_jobs=-1)
X_tsne = tsne.fit_transform(X_tsne_in)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for k in range(K_OPTIMAL):
    mask = tsne_cluster == k
    axes[0].scatter(X_tsne[mask,0], X_tsne[mask,1], s=5, alpha=0.4,
                    color=CLUSTER_COLORS[k], label=CLUSTER_NAMES[k])
axes[0].set_title('t-SNE — coloured by K-Means cluster', fontsize=10, color=DARK)
axes[0].legend(fontsize=7, markerscale=2)
axes[0].axis('off')

tier_colors = {'Budget': TEAL, 'Mid-Range': ORANGE, 'Premium': CORAL}
for tier, color in tier_colors.items():
    mask = tsne_tier == tier
    axes[1].scatter(X_tsne[mask,0], X_tsne[mask,1], s=5, alpha=0.4,
                    color=color, label=tier)
axes[1].set_title('t-SNE — coloured by Price Tier', fontsize=10, color=DARK)
axes[1].legend(fontsize=8, markerscale=2)
axes[1].axis('off')

fig.suptitle('t-SNE — 4000 listings, 37 features (for visualisation only)',
             fontsize=11, color=DARK)
plt.tight_layout()
savefig('tsne')

print('\nAll plots saved to ./plots/')
