"""
Streamlit demo — Airbnb Paris Pricing Analysis
Run: streamlit run app.py
"""
import ast
import os
import warnings
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
PLOTS = ROOT / "plots"
RANDOM_STATE = 42

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb Paris — Pricing Analytics",
    page_icon="🏙️",
    layout="wide",
)

# ── helpers ───────────────────────────────────────────────────────────────────
LANDMARKS = {
    "eiffel_tower": (48.8584, 2.2945),
    "louvre": (48.8606, 2.3376),
    "notre_dame": (48.8530, 2.3499),
    "opera_garnier": (48.8720, 2.3316),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6_371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def parse_amenities(s):
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def clean_name(n):
    return (
        n.replace("property_type_", "prop: ")
        .replace("room_type_", "room: ")
        .replace("neighbourhood_", "arr: ")
        .replace("review_scores_", "score_")
        .replace("dist_", "dist ")
        .replace("_km", " km")
        .replace("_", " ")
    )


# ── data loading & feature engineering (cached) ───────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data…")
def load_data():
    df_all = pd.read_csv(ROOT / "listings.csv", encoding="utf-8-sig", low_memory=False)
    df_all.columns = df_all.columns.str.replace("ï»¿", "", regex=False)
    df = df_all[df_all["city"] == "Paris"].copy().reset_index(drop=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    reviews_df = pd.read_csv(ROOT / "Reviews.csv", parse_dates=["date"])

    dfc = df.dropna(subset=["price"]).copy()
    p1, p99 = dfc["price"].quantile([0.01, 0.99])
    dfc = dfc[(dfc["price"] >= p1) & (dfc["price"] <= p99)].reset_index(drop=True)

    for col in [
        "bedrooms", "review_scores_rating", "review_scores_cleanliness",
        "review_scores_location", "review_scores_value",
        "review_scores_accuracy", "review_scores_checkin", "review_scores_communication",
    ]:
        if col in dfc.columns:
            dfc[col] = dfc[col].fillna(dfc[col].median())

    dfc["amenities_list"] = dfc["amenities"].apply(parse_amenities)
    dfc["amenities_count"] = dfc["amenities_list"].apply(len)

    KEY_AMENITIES = [
        "Wifi", "Kitchen", "Heating", "Washer", "Dryer",
        "Air conditioning", "Elevator", "TV", "Dishwasher", "Essentials",
    ]
    for amenity in KEY_AMENITIES:
        col = "has_" + amenity.lower().replace(" ", "_")
        dfc[col] = dfc["amenities_list"].apply(lambda x, a=amenity: a in x).astype(int)

    for col in ["host_is_superhost", "instant_bookable", "host_identity_verified", "host_has_profile_pic"]:
        if col in dfc.columns:
            dfc[col] = dfc[col].map({"t": 1, "f": 0}).fillna(0).astype(int)

    if "property_type" in dfc.columns:
        top_props = dfc["property_type"].value_counts().head(10).index
        dfc["property_type"] = dfc["property_type"].where(dfc["property_type"].isin(top_props), "Other")

    ohe_cols = [c for c in ["room_type", "neighbourhood", "property_type"] if c in dfc.columns]
    dfc = pd.get_dummies(dfc, columns=ohe_cols, dtype=int)

    dfc["log_price"] = np.log1p(dfc["price"])

    if "host_since" in dfc.columns:
        dfc["host_since_dt"] = pd.to_datetime(dfc["host_since"], errors="coerce")
        dfc["host_seniority_days"] = (pd.Timestamp("2026-01-01") - dfc["host_since_dt"]).dt.days
        dfc["host_seniority_days"] = dfc["host_seniority_days"].fillna(dfc["host_seniority_days"].median())

    if "review_scores_rating" in dfc.columns:
        dfc["review_scores_rating_norm"] = dfc["review_scores_rating"] / 20.0

    for name, (lat, lon) in LANDMARKS.items():
        dfc[f"dist_{name}_km"] = dfc.apply(
            lambda r, lat=lat, lon=lon: haversine_km(r["latitude"], r["longitude"], lat, lon), axis=1
        )

    if "host_total_listings_count" in dfc.columns:
        cap_val = dfc["host_total_listings_count"].quantile(0.99)
        dfc["host_listings_capped"] = dfc["host_total_listings_count"].clip(upper=cap_val)

    rev_stats = (
        reviews_df.groupby("listing_id")
        .agg(review_count=("review_id", "count"), last_review_date=("date", "max"))
        .reset_index()
    )
    ref_date = reviews_df["date"].max()
    rev_stats["days_since_last_review"] = (ref_date - rev_stats["last_review_date"]).dt.days
    dfc = dfc.merge(
        rev_stats[["listing_id", "review_count", "days_since_last_review"]],
        on="listing_id", how="left",
    )
    dfc["review_count"] = dfc["review_count"].fillna(0)
    dfc["days_since_last_review"] = dfc["days_since_last_review"].fillna(
        dfc["days_since_last_review"].median()
    )

    EXCLUDE = {
        "listing_id", "name", "host_id", "host_since", "host_since_dt",
        "host_location", "city", "district", "amenities", "amenities_list",
        "price", "log_price", "price_per_guest", "last_review_date", "review_scores_rating",
    }
    feature_cols = [c for c in dfc.select_dtypes(include="number").columns if c not in EXCLUDE]
    return dfc, feature_cols


@st.cache_resource(show_spinner="Training model (first run ~60s)…")
def train_model():
    dfc, feature_cols = load_data()

    X = dfc[feature_cols].fillna(0)
    y = dfc["log_price"]
    y_raw = dfc["price"]

    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection: Lasso ∪ RF top-20
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=5000)
    lasso.fit(X_train_scaled, y_train)
    lasso_coef = pd.Series(lasso.coef_, index=X.columns)
    top_lasso = set(lasso_coef[lasso_coef != 0].abs().sort_values(ascending=False).head(30).index)

    rf_sel = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf_sel.fit(X_train, y_train)
    rf_imp = pd.Series(rf_sel.feature_importances_, index=X.columns)
    top_rf = set(rf_imp.sort_values(ascending=False).head(20).index)

    selected = list(top_lasso | top_rf)

    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]

    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train_sel, y_train)

    y_pred_log = model.predict(X_test_sel)
    y_pred = np.expm1(y_pred_log)
    metrics = {
        "R²": round(r2_score(y_test, y_pred_log), 3),
        "RMSE (€)": round(np.sqrt(mean_squared_error(y_raw_test, y_pred)), 1),
        "MAE (€)": round(mean_absolute_error(y_raw_test, y_pred), 1),
    }

    return model, selected, feature_cols, metrics


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=120)
st.sidebar.title("Airbnb Paris\nPricing Analytics")
st.sidebar.markdown("**HEC Paris — Business Analytics**  \nMay 2026")
st.sidebar.divider()

# Trigger model training in background so it's ready
with st.sidebar:
    if st.button("🔄 Load model", help="Pre-load the model for faster predictions"):
        train_model()
        st.success("Model ready!")

# ── main tabs ─────────────────────────────────────────────────────────────────
tab_eda, tab_feat, tab_models, tab_clusters, tab_pred = st.tabs([
    "🏠 EDA",
    "📊 Feature Analysis",
    "🤖 Models",
    "🗺️ Clusters",
    "💰 Price Predictor",
])


# ── TAB 1: EDA ────────────────────────────────────────────────────────────────
with tab_eda:
    st.header("Exploratory Data Analysis — Paris Airbnb")

    dfc_eda, _ = load_data()
    p1, p99 = dfc_eda["price"].quantile([0.01, 0.99])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total listings", f"{len(dfc_eda):,}")
    col2.metric("Median price", f"€{dfc_eda['price'].median():.0f}")
    col3.metric("Mean price", f"€{dfc_eda['price'].mean():.0f}")
    col4.metric("Price range (p1–p99)", f"€{p1:.0f} – €{p99:.0f}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Price distribution")
        st.image(str(PLOTS / "eda_combined.png"), use_container_width=True)
    with c2:
        st.subheader("Price by neighbourhood")
        st.image(str(PLOTS / "price_by_neighbourhood.png"), use_container_width=True)

    st.subheader("Geographic spread")
    c3, c4 = st.columns([1, 1])
    with c3:
        st.image(str(PLOTS / "geo_scatter.png"), use_container_width=True)
    with c4:
        st.subheader("Review activity over time")
        st.image(str(PLOTS / "review_temporal.png"), use_container_width=True)

    st.subheader("Correlation matrix — numerical features")
    st.image(str(PLOTS / "correlation_heatmap.png"), use_container_width=True)


# ── TAB 2: FEATURE ANALYSIS ───────────────────────────────────────────────────
with tab_feat:
    st.header("Feature Selection & Importance")
    st.markdown(
        "Three independent methods (Pearson, Lasso L1, Random Forest) agree on the top drivers of price."
    )

    st.subheader("Three-method comparison")
    st.image(str(PLOTS / "feature_selection_3panel.png"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Random Forest importance (top 15)")
        st.image(str(PLOTS / "rf_importance.png"), use_container_width=True)
    with c2:
        st.subheader("Amenity price premium")
        st.image(str(PLOTS / "amenity_premium.png"), use_container_width=True)

    st.subheader("Amenity frequency")
    st.image(str(PLOTS / "amenity_frequency.png"), use_container_width=True)


# ── TAB 3: MODELS ─────────────────────────────────────────────────────────────
with tab_models:
    st.header("Supervised Model Comparison")
    st.markdown(
        "8 models trained on log-price, evaluated with 5-fold CV. "
        "Target: log(1 + price). Metrics back-transformed to euros."
    )

    st.subheader("All models — R², RMSE, MAE")
    st.image(str(PLOTS / "model_comparison_full.png"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("R² with cross-validation std dev")
        st.image(str(PLOTS / "model_r2_errorbars.png"), use_container_width=True)
    with c2:
        st.subheader("GridSearch heatmap — XGBoost tuning")
        st.image(str(PLOTS / "gridsearch_heatmap.png"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("XGBoost — residuals & actual vs predicted")
        st.image(str(PLOTS / "xgb_residuals.png"), use_container_width=True)
    with c4:
        st.subheader("MLP Neural Network — training loss curve")
        st.image(str(PLOTS / "mlp_loss_curve.png"), use_container_width=True)

    st.subheader("Price tier classification — confusion matrices")
    st.image(str(PLOTS / "classification_confusion.png"), use_container_width=True)


# ── TAB 4: CLUSTERS ───────────────────────────────────────────────────────────
with tab_clusters:
    st.header("Unsupervised Clustering — Listing Segments")
    st.markdown(
        "K-Means with k=4, validated by elbow method, silhouette score, and hierarchical clustering."
    )

    st.subheader("Choosing k — elbow & silhouette")
    st.image(str(PLOTS / "elbow_silhouette.png"), use_container_width=True)

    st.subheader("Cluster profiles")
    st.image(str(PLOTS / "cluster_profiles.png"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("PCA 2D projection")
        st.image(str(PLOTS / "cluster_pca.png"), use_container_width=True)
    with c2:
        st.subheader("t-SNE visualisation")
        st.image(str(PLOTS / "tsne.png"), use_container_width=True)

    st.subheader("Hierarchical clustering dendrogram")
    st.image(str(PLOTS / "dendrogram.png"), use_container_width=True)

    st.subheader("Amenity decision tree (depth=3)")
    st.image(str(PLOTS / "amenity_decision_tree.png"), use_container_width=True)


# ── TAB 5: PRICE PREDICTOR ────────────────────────────────────────────────────
with tab_pred:
    st.header("💰 Interactive Price Predictor")
    st.markdown(
        "Enter your listing's characteristics to get an estimated nightly price "
        "based on our best XGBoost model (R² ≈ 0.63)."
    )

    # Load data to get reference values for the form
    dfc_ref, feature_cols = load_data()

    # Build neighbourhood list
    neigh_cols = [c for c in dfc_ref.columns if c.startswith("neighbourhood_")]
    neighbourhoods = sorted([c.replace("neighbourhood_", "") for c in neigh_cols])

    # Build room type list
    room_cols = [c for c in dfc_ref.columns if c.startswith("room_type_")]
    room_types = sorted([c.replace("room_type_", "") for c in room_cols])

    # Build property type list
    prop_cols = [c for c in dfc_ref.columns if c.startswith("property_type_")]
    prop_types = sorted([c.replace("property_type_", "") for c in prop_cols])

    # Median lat/lon per neighbourhood (for distance features)
    orig_df = pd.read_csv(ROOT / "listings.csv", encoding="utf-8-sig", low_memory=False)
    orig_df.columns = orig_df.columns.str.replace("ï»¿", "", regex=False)
    orig_df = orig_df[orig_df["city"] == "Paris"]
    neigh_coords = (
        orig_df.groupby("neighbourhood")[["latitude", "longitude"]].median().to_dict("index")
    )

    st.divider()
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("Property details")
        neighbourhood = st.selectbox("Neighbourhood (arrondissement)", neighbourhoods,
                                      index=neighbourhoods.index("Buttes-Montmartre") if "Buttes-Montmartre" in neighbourhoods else 0)
        room_type = st.selectbox("Room type", room_types,
                                  index=room_types.index("Entire home/apt") if "Entire home/apt" in room_types else 0)
        prop_type = st.selectbox("Property type", prop_types,
                                  index=prop_types.index("Entire rental unit") if "Entire rental unit" in prop_types else 0)
        accommodates = st.slider("Accommodates (guests)", 1, 16, 2)
        bedrooms = st.slider("Bedrooms", 0, 10, 1)

        st.subheader("Amenities")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            has_wifi = st.checkbox("Wi-Fi", value=True)
            has_kitchen = st.checkbox("Kitchen", value=True)
            has_heating = st.checkbox("Heating", value=True)
            has_washer = st.checkbox("Washer")
            has_dryer = st.checkbox("Dryer")
        with col_a2:
            has_air = st.checkbox("Air conditioning")
            has_elevator = st.checkbox("Elevator")
            has_tv = st.checkbox("TV")
            has_dishwasher = st.checkbox("Dishwasher")
            has_essentials = st.checkbox("Essentials", value=True)

    with c_right:
        st.subheader("Host & ratings")
        review_score = st.slider("Review score (out of 5)", 1.0, 5.0, 4.5, 0.1)
        host_is_superhost = st.checkbox("Superhost")
        instant_bookable = st.checkbox("Instant bookable", value=True)
        host_seniority = st.slider("Host seniority (years)", 0, 15, 3)
        amenities_count = st.slider("Total amenities count", 0, 80, 25)

        st.subheader("Booking policy")
        minimum_nights = st.slider("Minimum nights", 1, 30, 2)
        availability_365 = st.slider("Availability (days/year)", 0, 365, 180)

    st.divider()
    predict_btn = st.button("🔮 Predict price", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Computing prediction…"):
            model, selected_features, feature_cols, train_metrics = train_model()

            # Build a baseline row (median of training data)
            median_row = dfc_ref[feature_cols].fillna(0).median()
            row = median_row.copy()

            # Override with user inputs
            if "accommodates" in row.index:
                row["accommodates"] = accommodates
            if "bedrooms" in row.index:
                row["bedrooms"] = bedrooms
            if "amenities_count" in row.index:
                row["amenities_count"] = amenities_count
            if "host_is_superhost" in row.index:
                row["host_is_superhost"] = int(host_is_superhost)
            if "instant_bookable" in row.index:
                row["instant_bookable"] = int(instant_bookable)
            if "minimum_nights" in row.index:
                row["minimum_nights"] = minimum_nights
            if "availability_365" in row.index:
                row["availability_365"] = availability_365
            if "host_seniority_days" in row.index:
                row["host_seniority_days"] = host_seniority * 365
            if "review_scores_rating_norm" in row.index:
                row["review_scores_rating_norm"] = review_score / 5.0

            # Amenity flags
            amenity_map = {
                "has_wifi": has_wifi, "has_kitchen": has_kitchen, "has_heating": has_heating,
                "has_washer": has_washer, "has_dryer": has_dryer,
                "has_air_conditioning": has_air, "has_elevator": has_elevator,
                "has_tv": has_tv, "has_dishwasher": has_dishwasher, "has_essentials": has_essentials,
            }
            for col, val in amenity_map.items():
                if col in row.index:
                    row[col] = int(val)

            # OHE: zero out all neighbourhood/room/property columns, then set the chosen one
            for c in [x for x in row.index if x.startswith("neighbourhood_")]:
                row[c] = 0
            neigh_col = f"neighbourhood_{neighbourhood}"
            if neigh_col in row.index:
                row[neigh_col] = 1

            for c in [x for x in row.index if x.startswith("room_type_")]:
                row[c] = 0
            rt_col = f"room_type_{room_type}"
            if rt_col in row.index:
                row[rt_col] = 1

            for c in [x for x in row.index if x.startswith("property_type_")]:
                row[c] = 0
            pt_col = f"property_type_{prop_type}"
            if pt_col in row.index:
                row[pt_col] = 1

            # Distance features from neighbourhood median coordinates
            coords = neigh_coords.get(neighbourhood, {"latitude": 48.8566, "longitude": 2.3522})
            lat, lon = coords["latitude"], coords["longitude"]
            for name, (lm_lat, lm_lon) in LANDMARKS.items():
                dist_col = f"dist_{name}_km"
                if dist_col in row.index:
                    row[dist_col] = haversine_km(lat, lon, lm_lat, lm_lon)

            # Predict
            X_pred = row[selected_features].values.reshape(1, -1)
            log_pred = model.predict(X_pred)[0]
            price_pred = np.expm1(log_pred)

            # Confidence interval (±1 MAE)
            mae = train_metrics["MAE (€)"]
            low = max(0, price_pred - mae)
            high = price_pred + mae

        st.success("Prediction complete!")
        res1, res2, res3 = st.columns(3)
        res1.metric("Estimated nightly price", f"€{price_pred:.0f}")
        res2.metric("Lower bound (−1 MAE)", f"€{low:.0f}")
        res3.metric("Upper bound (+1 MAE)", f"€{high:.0f}")

        with st.expander("Model info"):
            st.markdown(f"**Model**: XGBoost (300 estimators, depth=5, lr=0.05)")
            st.markdown(f"**Test R²**: {train_metrics['R²']}  |  **RMSE**: €{train_metrics['RMSE (€)']}  |  **MAE**: €{train_metrics['MAE (€)']}")
            st.markdown(f"**Selected features**: {len(selected_features)}")
            st.caption("Price predicted on log-scale, then back-transformed via expm1.")
