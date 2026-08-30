from __future__ import annotations

import numpy as np
import pandas as pd


def build_benchmark_datasets() -> dict[str, pd.DataFrame]:
    """Build small deterministic company-style datasets with known structure."""
    products = pd.DataFrame({
        "product": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
        "category": ["Core", "Core", "Growth", "Growth", "Value", "Value"],
        "supplier": ["S1", "S1", "S2", "S2", "S3", "S3"],
        "cost": [20.0, 30.0, 15.0, 25.0, 10.0, 12.0],
        "stock": [90, 80, 110, 70, 140, 120],
    })
    customers = pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(60)],
        "segment": ["Consumer", "Business", "Enterprise"] * 20,
        "region": ["North", "South", "East", "West"] * 15,
        "signup_date": pd.date_range("2021-01-01", periods=60, freq="15D"),
        "age": [22 + (i * 7) % 43 for i in range(60)],
        "churn": [1 if i % 7 in {0, 1} else 0 for i in range(60)],
    })

    rows = []
    product_names = products["product"].tolist()
    base_price = dict(zip(product_names, [60, 80, 45, 70, 35, 40]))
    year_adjustment = {"Alpha": -18, "Beta": -10, "Gamma": 8, "Delta": 2, "Epsilon": 5, "Zeta": -3}
    order = 1
    for month_index, date in enumerate(pd.date_range("2024-01-01", periods=24, freq="MS")):
        for offset in range(5):
            product = product_names[(month_index + offset) % len(product_names)]
            customer_index = (month_index * 5 + offset) % 60
            quantity = 1 + (month_index + offset) % 5
            price = base_price[product] + month_index
            if date.year == 2025:
                price += year_adjustment[product]
            revenue = float(quantity * price)
            rows.append({
                "date": date,
                "order_id": f"O{order:04d}",
                "customer_id": f"C{customer_index:03d}",
                "product": product,
                "category": products.set_index("product").loc[product, "category"],
                "region": customers.set_index("customer_id").loc[f"C{customer_index:03d}", "region"],
                "quantity": quantity,
                "unit_price": float(price),
                "revenue": revenue,
                "returned": int((quantity == 5 and product in {"Beta", "Delta"}) or order % 17 == 0),
            })
            order += 1
    sales = pd.DataFrame(rows)
    sales.loc[3, "unit_price"] = np.nan
    sales.loc[10, "revenue"] = sales["revenue"].max() * 8

    events = sales[["customer_id", "date"]].iloc[:80].copy()
    events["event"] = ["visit", "support"] * 40
    return {"sales": sales, "customers": customers, "products": products, "events": events}
