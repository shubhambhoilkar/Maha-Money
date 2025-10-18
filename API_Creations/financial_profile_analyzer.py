# financial_profile_analyzer.py
"""
Financial profile analyzer module.
Provides analyze_customer_profile(profile_json) -> analysis dict.

This module is intentionally dependency-light (only uses Python stdlib + matplotlib optional).
"""

import math
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from decimal import Decimal , InvalidOperation

# ---------- Helpers ----------
def to_float(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "").replace("₹", "").replace("INR", "")
        return float(s) if s else 0.0
    except Exception:
        return 0.0

def to_decimal(x) ->Decimal:
    try: 
        if x is None:
            return Decimal('0')
        if isinstance(x,(int, float, Decimal)):
            return Decimal(str(x))
        s = str(x).strip().replace(",","").replace("₹", "").replace("INR","")
        return Decimal(s) if s else Decimal('0')
    except (InvalidOperation, TypeError):
        return Decimal('0')

def safe_list(obj) -> list:
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]

def sum_field(lst: list, key: Optional[str] = None) -> float:
    total = 0.0
    for item in safe_list(lst):
        if isinstance(item, dict):
            if key:
                total += to_float(item.get(key, 0))
            else:
                # if no key specified, try typical numeric keys
                # but keep conservative: don't sum entire dict blindly
                for k in ("balance","amount","currentValue","currentMarketValue","currentValuation","maturityAmount","depositAmount","investmentAmount","initialAmount"):
                    if k in item:
                        total += to_float(item.get(k))
        else:
            total += to_float(item)
    return total

# ---------- Core analysis ----------
def analyze_customer_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single customer's profile JSON (structure as provided earlier).
    Returns dict with summary, classification, recommendations, and computed features.
    """

    props = profile.get("properties", {}) or {}
    fin = props.get("financialData", {}) or {}
    assets = fin.get("assets", {}) or {}
    liab = fin.get("liabilities", {}) or {}
    personal = props.get("personalData", {}) or {}

    # --- Liquid asset components ---
    bank_saving = sum(to_float(a.get("balance", 0)) for a in safe_list(assets.get("savingAccounts", [])))
    bank_current = sum(to_float(a.get("balance", 0)) for a in safe_list(assets.get("currentAccount", [])))
    liquid_mf_redemp = sum(to_float(x.get("redemptionAmount", 0)) for x in safe_list(assets.get("liquidMutualFunds", [])))
    liquid_mf_total = sum_field(assets.get("liquidMutualFundsTotal", []))
    debt_mf_total = sum_field(assets.get("debtMutualFundsTotal", []))
    fd_total = sum_field(assets.get("fixedDepositsTotal", [])) or sum(to_float(x.get("depositAmount",0)) for x in safe_list(assets.get("fixedDeposits",[])))
    rd_total = sum_field(assets.get("recurringDepositTotal", []))
    po_total = sum_field(assets.get("postOfficeDeposits", []))
    epf_total = sum_field(assets.get("epfTotal", []))
    shares_total = sum_field(assets.get("sharesTotal", []))

    liquid_assets = bank_saving + bank_current + liquid_mf_redemp + liquid_mf_total + debt_mf_total + fd_total + rd_total + po_total + epf_total + shares_total

    # --- Illiquid / big assets ---
    precious = sum_field(assets.get("preciousMetalsTotal", [])) or sum(to_float(x.get("currentValue",0)) for x in safe_list(assets.get("preciousMetals",[])))
    immovable = sum_field(assets.get("immovableAssetsTotal", [])) or sum(to_float(x.get("currentMarketValue",0)) for x in safe_list(assets.get("immovableAssets",[])))
    vehicles = sum_field(assets.get("vehiclesTotal", [])) or sum(to_float(x.get("currentValuation",0)) for x in safe_list(assets.get("vehicles",[])))
    art_antiques = sum_field(assets.get("artAntiques", []))
    business_part = sum_field(assets.get("businessPartnership", []))

    illiquid_assets = precious + immovable + vehicles + art_antiques + business_part

    # --- Totals reported (fallback) ---
    total_assets_reported = sum_field(personal.get("assets", [])) or (liquid_assets + illiquid_assets)
    total_liabilities_reported = sum_field(personal.get("liabilities", [])) or sum_field(liab.get("homeLoanTotal",[]))
    net_worth_reported = sum_field(personal.get("netWorth", [])) or (total_assets_reported - total_liabilities_reported)

    # --- Liabilities & EMIs ---
    monthly_emi = 0.0
    loan_outstanding = {}
    for loan_key in ["homeLoan", "carLoan", "personalLoan", "otherLoans"]:
        loan_sum = 0.0
        for l in safe_list(liab.get(loan_key, [])):
            loan_sum += to_float(l.get("loanAmount", l.get("loan_amount", 0)))
            monthly_emi += to_float(l.get("monthlyEMI", l.get("monthly_emi", 0)))
        loan_outstanding[loan_key] = loan_sum

    # --- Credit card utilization ---
    cc_list = safe_list(liab.get("creditCardDue", []))
    credit_balance = sum(to_float(c.get("currentBalance", c.get("balance",0))) for c in cc_list)
    credit_avail = sum(to_float(c.get("avialCredit", c.get("availableCredit",0))) for c in cc_list)
    credit_limit = (credit_balance + credit_avail) if (credit_balance + credit_avail) > 0 else None
    credit_utilization = (credit_balance / credit_limit) if credit_limit else None

    # --- LTV if possible (use first loan entry) ---
    home_ltv = None
    home_loans = safe_list(liab.get("homeLoan", []))
    if home_loans:
        h = home_loans[0]
        loan_amt = to_float(h.get("loanAmount", h.get("loan_amount", 0)))
        prop_val = to_float(h.get("propertyValue", h.get("property_value", 0)))
        if prop_val > 0:
            home_ltv = loan_amt / prop_val

    car_ltv = None
    car_loans = safe_list(liab.get("carLoan", []))
    if car_loans:
        c = car_loans[0]
        loan_amt = to_float(c.get("loanAmount", c.get("loan_amount", 0)))
        purch_price = to_float(c.get("purchPrice", c.get("purchasePrice", 0)))
        if purch_price > 0:
            car_ltv = loan_amt / purch_price

    # --- concentration shares ---
    base_assets = total_assets_reported if total_assets_reported else (liquid_assets + illiquid_assets)
    precious_share = (precious / base_assets) if base_assets else 0.0
    real_estate_share = (immovable / base_assets) if base_assets else 0.0
    liquid_share = (liquid_assets / base_assets) if base_assets else 0.0
    illiquid_share = (illiquid_assets / base_assets) if base_assets else 0.0

    # --- liquidity category (absolute thresholds; you can adjust or make dynamic) ---
    # Suggested tiers: < 50k: Low | 50k-200k: Moderate | >200k: High
    if liquid_assets < 50_000:
        liquidity_category = "Low"
    elif liquid_assets < 200_000:
        liquidity_category = "Moderate"
    else:
        liquidity_category = "High"

    # --- risk suggestions for investment suitability (low/moderate/high) ---
    # We derive risk capacity by combining liquidity, illiquid concentration and credit stress.
    # Simple rule-based:
    risk_capacity_score = 0.0
    # liquidity strong reduces score
    if liquidity_category == "Low":
        risk_capacity_score += 2.0
    elif liquidity_category == "Moderate":
        risk_capacity_score += 1.0
    # credit utilization impact
    if credit_utilization is not None:
        if credit_utilization > 0.7:
            risk_capacity_score += 2.0
        elif credit_utilization > 0.3:
            risk_capacity_score += 1.0
    else:
        risk_capacity_score += 0.5  # unknown -> moderate penalty
    # concentration: large illiquid share reduces capacity
    if illiquid_share > 0.6:
        risk_capacity_score += 2.0
    elif illiquid_share > 0.3:
        risk_capacity_score += 1.0

    # Normalize to label: 0-1 -> low risk capacity; higher score worse
    # Score range roughly 0..6 -> map to final capacity
    if risk_capacity_score <= 1.5:
        investment_risk_capacity = "High"      # can take higher-risk investments
    elif risk_capacity_score <= 3.5:
        investment_risk_capacity = "Moderate"
    else:
        investment_risk_capacity = "Low"       # should avoid high-risk investments

    # --- financial behavior classification based on holdings/ liabilities ---
    behavior = "Balanced"
    # compute percent of assets that are growth (approx using shares + equity MFs)
    growth_assets = sum_field(assets.get("equityMutualFunds", [])) + sum_field(assets.get("equityMutualFundsTotal", [])) + shares_total
    growth_share = (growth_assets / base_assets) if base_assets else 0.0

    # debt indicator
    debt_ratio = (total_liabilities_reported / total_assets_reported) if total_assets_reported else 0.0

    if debt_ratio > 0.5 or (monthly_emi > 0 and monthly_emi > (0.5 * 0)):  # monthlySalary unknown, keep debt_ratio primary
        behavior = "Under Debt"
    elif growth_share > 0.3:
        behavior = "Investor"
    elif liquid_share > 0.6 and illiquid_share < 0.2:
        behavior = "Safe-Side"
    else:
        behavior = "Balanced"

    # --- suggestions / recommendations ---
    recommendations: List[str] = []
    if credit_utilization is not None:
        if credit_utilization > 0.7:
            recommendations.append("Pay down credit card balances — aim <30% utilization.")
        elif credit_utilization > 0.3:
            recommendations.append("Reduce credit utilization to improve credit profile.")
    if liquidity_category != "High":
        recommendations.append("Build emergency fund: target 6 months of monthly obligations.")
    if illiquid_share > 0.5:
        recommendations.append("Consider gradual diversification from illiquid assets into liquid investments (debt/equity funds).")
    if investment_risk_capacity == "Low":
        recommendations.append("Prefer low-risk instruments: debt funds, FDs, PPF until liquidity improves.")
    elif investment_risk_capacity == "Moderate":
        recommendations.append("Mix of debt and balanced/hybrid funds; small SIPs into equity for long-term growth.")
    else:
        recommendations.append("Can increase exposure to equities and growth funds with SIP discipline.")

    # --- assemble result ---
    analysis = {
        "profileId": profile.get("itemId"),
        "summary": {
            "totalAssetsReported": total_assets_reported,
            "totalLiabilitiesReported": total_liabilities_reported,
            "netWorthReported": net_worth_reported,
            "liquidAssets": round(liquid_assets, 2),
            "illiquidAssets": round(illiquid_assets, 2),
            "liquidityCategory": liquidity_category,
            "liquidShare": round(liquid_share, 3),
            "illiquidShare": round(illiquid_share, 3),
            "preciousShare": round(precious_share, 3),
            "realEstateShare": round(real_estate_share, 3),
            "monthlyEMI": round(monthly_emi, 2),
            "creditUtilization": round(credit_utilization, 4) if credit_utilization is not None else None,
        },
        "classification": {
            "investmentRiskCapacity": investment_risk_capacity,
            "financialBehavior": behavior,
            "debtRatio": round(debt_ratio, 3),
            "growthShare": round(growth_share, 3)
        },
        "recommendations": recommendations,
        # internal details for debugging / dashboard signals
        "_internal": {
            "precious": precious,
            "immovable": immovable,
            "vehicles": vehicles,
            "epf": epf_total,
            "shares": shares_total,
            "fd_total": fd_total,
            "rd_total": rd_total,
            "debt_mf_total": debt_mf_total,
            "bank_saving": bank_saving,
            "bank_current": bank_current,
            "monthly_emi": monthly_emi,
            "home_ltv": round(home_ltv, 3) if home_ltv is not None else None,
            "car_ltv": round(car_ltv, 3) if car_ltv is not None else None,
            "riskCapacityScore": round(risk_capacity_score, 2)
        }
    }

    return analysis

# optional quick plotting functions (useful for local debugging)
def save_asset_pie(analysis: Dict[str, Any], filename: str = "asset_allocation.png"):
    s = analysis["summary"]
    liquid = s["liquidAssets"]
    imm = s["illiquidAssets"]
    other = max(0, (s["totalAssetsReported"] - (liquid + imm))) if s.get("totalAssetsReported") else 0
    labels = ["Liquid", "Illiquid", "Other"]
    values = [liquid, imm, other]
    plt.figure(figsize=(6,6))
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title("Asset Allocation")
    plt.savefig(filename)
    plt.close()
