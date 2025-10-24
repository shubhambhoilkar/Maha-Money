# financial_analysis_ml.py
# Description: 
# Analyze user financial data, 
# determine risk state,
# predict future behavior,
# recommend investments.

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error

# ----------------------------
# 1. Load and Parse JSON Data
# ----------------------------
def sum_field(d, key):
    """Safely extract numeric values from asset/liability entries."""
    if key not in d:
        return 0.0
    vals = d[key]
    if not isinstance(vals, list):
        vals = [vals]
    total = 0.0
    for v in vals:
        if isinstance(v, (int, float)):
            total += v
        elif isinstance(v, str):
            try:
                total += float(v.replace(",", ""))
            except:
                pass
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    total += vv
                elif isinstance(vv, str):
                    try:
                        total += float(vv.replace(",", ""))
                    except:
                        pass
        elif isinstance(v, list):
            for item in v:
                try:
                    total += float(item)
                except:
                    pass
    print("total: ", total)
    return total


def compute_financial_features(profile):
    """Compute financial features from JSON profile."""
    assets = profile["properties"]["financialData"]["assets"]
    liabilities = profile["properties"]["financialData"]["liabilities"]

    asset_total_keys = [
        "savingAccountTotal", "currentAccountTotal", "liquidMutualFundsTotal",
        "fixedDepositsTotal", "recurringDepositTotal", "ppfSelfTotal", "ppfSpouseTotal",
        "preciousMetalsTotal", "immovableAssetsTotal", "vehiclesTotal", "businessPartnership",
        "sharesTotal", "epfTotal", "bonds", "debtMutualFundsTotal", "postOfficeDeposits", "NSC"
    ]

    liability_total_keys = [
        "personalLoanTotal", "homeLoanTotal", "carLoanTotal",
        "otherLoanTotal", "otherBillsOutstandingTotal", "creditDueTotal", "taxesDueTotal"
    ]

    total_assets = sum(sum_field(assets, k) for k in asset_total_keys)
    total_liabilities = sum(sum_field(liabilities, k) for k in liability_total_keys)

    net_worth = total_assets - total_liabilities
    liquidity = sum_field(assets, "savingAccountTotal") + \
                sum_field(assets, "currentAccountTotal") + \
                sum_field(assets, "liquidMutualFundsTotal") + \
                sum_field(assets, "fixedDepositsTotal") + \
                sum_field(assets, "recurringDepositTotal") + \
                sum_field(assets, "postOfficeDeposits")

    equity_exposure = (sum_field(assets, "sharesTotal") / sum_field(assets, "equityMutualFunds"))*100
    debt_exposure = sum_field(assets, "bonds") + sum_field(assets, "debtMutualFundsTotal") + sum_field(assets, "corporateDeposits")
    debt_ratio = total_liabilities / total_assets if total_assets > 0 else 0
  
    features = pd.DataFrame([{
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "net_worth": net_worth,
        "liquidity": liquidity,
        "equity_exposure": equity_exposure,
        "debt_exposure": debt_exposure,
        "debt_ratio": debt_ratio
    }])
    return features


# -------------------------------------
# 2. Unsupervised Risk-State Clustering
# -------------------------------------
def cluster_risk(features):
    """Cluster financial features into Low/Moderate/High risk states."""
    # Create synthetic population for clustering baseline
    np.random.seed(42)
    population = []
    for _ in range(400):
        ta = 10000 + np.random.lognormal(8, 1)
        tl = ta * np.random.beta(1, 5)
        liq = np.random.uniform(0, ta * 0.5)
        eq = np.random.uniform(0, ta * 0.6)
        de = np.random.uniform(0, ta * 0.2)
        dr = tl / (ta + 1e-6)
        population.append([ta, tl, ta - tl, liq, eq, de, dr])

    pop_df = pd.DataFrame(population, columns=features.columns)
    pop_df = pd.concat([pop_df, features], ignore_index=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(pop_df)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=15)
    labels = kmeans.fit_predict(X)

    # Identify user’s cluster
    user_label = labels[-1]

    # Map clusters -> risk states by net worth ordering
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=features.columns)
    centers_df["cluster"] = range(len(centers_df))
    centers_sorted = centers_df.sort_values("net_worth", ascending=False).reset_index(drop=True)

    states = ["Low financial risk (stable)", "Moderate financial risk", "High financial risk (vulnerable)"]
    mapping = {int(row["cluster"]): states[idx] for idx, row in centers_sorted.iterrows()}

    user_state = mapping[int(user_label)]
    return user_state, centers_df


# ---------------------------------------------------
# 3. Behavioral Prediction (Investment Propensity)
# ---------------------------------------------------
def make_synthetic_data(n=2000):
    """Generate synthetic financial-behavior dataset."""
    rows = []
    for _ in range(n):
        ta = 5e4 + np.random.lognormal(8, 1)
        tl = ta * np.random.beta(1, 4)
        liq = np.random.uniform(0, ta * 0.6)
        eq = np.random.uniform(0, ta * 0.6)
        dr = tl / (ta + 1e-6)
        p = 1 / (1 + np.exp(3 * dr - (liq / ta) * 5 + np.random.normal(0, 0.5)))
        label = np.random.rand() < p
        rows.append([ta, tl, liq, eq, dr, int(label)])
    return pd.DataFrame(rows, columns=["total_assets", "total_liabilities", "liquidity", "equity_exposure", "debt_ratio", "increase_invest"])


def train_behavior_models():
    """Train logistic & regression models on synthetic data."""
    synth = make_synthetic_data(2500)
    X = synth[["total_assets", "total_liabilities", "liquidity", "equity_exposure", "debt_ratio"]]
    y = synth["increase_invest"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=500)
    clf.fit(Xs, y)

    # Regression for expected monthly change
    synth["monthly_change"] = np.tanh((synth["liquidity"]/synth["total_assets"] - synth["debt_ratio"]) * 3 + np.random.normal(0, 0.2, len(synth)))
    reg = LinearRegression()
    reg.fit(Xs, synth["monthly_change"])

    auc = roc_auc_score(y, clf.predict_proba(Xs)[:, 1])
    mse = mean_squared_error(synth["monthly_change"], reg.predict(Xs))
    print(f"[INFO] Synthetic model metrics — Logistic AUC: {auc:.3f}, Regression MSE: {mse:.4f}")
    return clf, reg, scaler


# ----------------------------
# 4. Generate Recommendations
# ----------------------------
def make_recommendations(user_state):
    if "Low financial risk" in user_state:
        return [
            "Preserve capital with high-quality debt instruments (PPF, short-term FDs).",
            "Continue SIPs into balanced or debt-heavy mutual funds for steady growth.",
            "Use liquidity to top-up retirement and tax-saving instruments."
        ]
    elif "Moderate" in user_state:
        return [
            "Maintain a balanced portfolio: diversify between equity and debt funds.",
            "Gradually increase SIPs into equity mutual funds; keep emergency fund ≥6 months expenses.",
            "Rebalance portfolio every 6–12 months."
        ]
    else:
        return [
            "Reduce high-cost debt (credit card, personal loan) before new investments.",
            "Increase liquidity; focus on debt instruments until net worth stabilizes.",
            "Avoid large illiquid purchases until debt ratio improves."
        ]


# ----------------------------
# 5. Main Execution Pipeline
# ----------------------------
if __name__ == "__main__":
    # ---- Load Data ----
    with open("C:\\Users\\Developer\\Shubham_files\\Maha_Money\\machine_learning\\user_profile.json", "r") as f:
        profile = json.load(f)

    features = compute_financial_features(profile)
    print("\n[Computed Financial Features]\n", features)

    # ---- Cluster Risk ----
    risk_state, centers = cluster_risk(features)
    print(f"\n[User Financial Risk State]: {risk_state}")

    # ---- Predict Behavior ----
    clf, reg, scaler = train_behavior_models()
    X_user = scaler.transform(features[["total_assets","total_liabilities","liquidity","equity_exposure","debt_ratio"]])
    prob_increase = clf.predict_proba(X_user)[0, 1]
    monthly_change = reg.predict(X_user)[0]

    # ---- Build Report ----
    recs = make_recommendations(risk_state)
    report = {
        "financial_features": features.to_dict(orient="records")[0],
        "risk_state": risk_state,
        "prob_increase_invest_next_6m": float(prob_increase),
        "expected_monthly_investment_change_frac": float(monthly_change),
        "recommendations": recs
    }

    with open("financial_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n[Report Saved] → financial_analysis_report.json")

    # ---- Display Summary ----
    print("\n[Summary]")
    print(json.dumps(report, indent=2))

    # ---- Plot Overview ----
    plt.bar(["Assets", "Liabilities", "Net Worth"],
            [features["total_assets"].iloc[0],
             features["total_liabilities"].iloc[0],
             features["net_worth"].iloc[0]])
    plt.title("Assets vs Liabilities Overview")
    plt.xlabel("Category")
    plt.ylabel("Amount (INR)")
    plt.tight_layout()
    plt.show()
