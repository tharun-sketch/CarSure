from flask import Flask, jsonify, request, render_template
import sqlite3
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import uuid
import re

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "carsure.db"
MODEL_PATH = BASE_DIR / "ml" / "carsure_price_model.pkl"

CURRENT_YEAR = 2026

# ---------------------------------------------------------
# Load ML model
# ---------------------------------------------------------
model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Database connection
# ---------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------
# Step 16: create valuation history table
# ---------------------------------------------------------
def ensure_valuation_table():
    """
    Create/migrate valuation_requests without deleting existing records.

    This migration supports older Step 15/Step 16 schemas and recovers
    fair-range values from common legacy column names or a text range.
    """
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS valuation_requests (
            valuation_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            variant TEXT NOT NULL,
            fuel_type TEXT NOT NULL,
            transmission TEXT NOT NULL,
            manufacture_year INTEGER NOT NULL,
            current_km INTEGER NOT NULL,
            owner_count INTEGER NOT NULL,
            registration_location TEXT NOT NULL,
            ai_predicted_value REAL NOT NULL,
            fair_buying_min REAL,
            fair_buying_max REAL,
            fair_buying_range TEXT,
            data_status TEXT
        )
    """)

    def get_columns():
        return {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(valuation_requests)"
            ).fetchall()
        }

    columns = get_columns()

    required_columns = {
        "fair_buying_min": "REAL",
        "fair_buying_max": "REAL",
        "fair_buying_range": "TEXT",
        "data_status": "TEXT",
    }

    for column_name, column_type in required_columns.items():
        if column_name not in columns:
            conn.execute(
                f"ALTER TABLE valuation_requests ADD COLUMN "
                f"{column_name} {column_type}"
            )

    columns = get_columns()

    # Recover fair-range values from common older column names.
    min_aliases = [
        "fair_min",
        "fair_range_min",
        "fair_buying_range_min",
        "fair_buying_minimum",
        "fair_buying_low",
    ]
    max_aliases = [
        "fair_max",
        "fair_range_max",
        "fair_buying_range_max",
        "fair_buying_maximum",
        "fair_buying_high",
    ]

    min_source = next((c for c in min_aliases if c in columns), None)
    max_source = next((c for c in max_aliases if c in columns), None)

    if min_source:
        conn.execute(f"""
            UPDATE valuation_requests
            SET fair_buying_min = CAST("{min_source}" AS REAL)
            WHERE fair_buying_min IS NULL
              AND "{min_source}" IS NOT NULL
        """)

    if max_source:
        conn.execute(f"""
            UPDATE valuation_requests
            SET fair_buying_max = CAST("{max_source}" AS REAL)
            WHERE fair_buying_max IS NULL
              AND "{max_source}" IS NOT NULL
        """)

    # Recover from a single text field such as:
    # "₹5,90,630 - ₹6,52,802"
    rows = conn.execute("""
        SELECT valuation_id, fair_buying_range
        FROM valuation_requests
        WHERE fair_buying_range IS NOT NULL
          AND (fair_buying_min IS NULL OR fair_buying_max IS NULL)
    """).fetchall()

    for row in rows:
        raw = str(row["fair_buying_range"])

        # Remove commas and currency symbols before extracting numbers.
        cleaned = raw.replace(",", "")
        numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", cleaned)

        if len(numbers) >= 2:
            minimum = float(numbers[0])
            maximum = float(numbers[1])

            conn.execute("""
                UPDATE valuation_requests
                SET fair_buying_min = ?,
                    fair_buying_max = ?
                WHERE valuation_id = ?
            """, (
                minimum,
                maximum,
                row["valuation_id"]
            ))

    conn.commit()
    conn.close()


ensure_valuation_table()


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def money(value):
    if value is None:
        return None
    return round(float(value))


# ---------------------------------------------------------
# Data confidence
# ---------------------------------------------------------
def build_data_confidence(
    vehicle,
    accidents,
    services,
    insurance,
    condition,
    km_verification
):
    available = 0
    total = 6

    if vehicle:
        available += 1
    if accidents:
        available += 1
    if services:
        available += 1
    if insurance:
        available += 1
    if condition:
        available += 1
    if km_verification and km_verification.get("service_readings"):
        available += 1

    percentage = round((available / total) * 100)

    if percentage >= 80:
        status = "High"
    elif percentage >= 50:
        status = "Medium"
    else:
        status = "Low"

    return {
        "status": status,
        "percentage": percentage,
        "available_sections": available,
        "total_sections": total,
        "note": (
            "Confidence reflects the availability of documented development "
            "data in CarSure. It is not a guarantee of vehicle condition."
        )
    }


# ---------------------------------------------------------
# Vehicle history score
# ---------------------------------------------------------
def calculate_history_score(data):
    accidents = data.get("accidents", [])
    services = data.get("service_history", [])
    insurance = data.get("insurance")
    condition = data.get("condition")
    km = data.get("km_verification", {})

    accident_score = 20

    if accidents:
        for accident in accidents:
            severity = str(accident.get("severity", "")).lower()

            if severity == "minor":
                accident_score -= 5
            elif severity == "moderate":
                accident_score -= 8
            elif severity == "major":
                accident_score -= 12
            else:
                accident_score -= 5

    accident_score = max(0, min(20, accident_score))

    service_score = 20
    if not services:
        service_score = 10
    elif len(services) == 1:
        service_score = 14
    elif len(services) >= 2:
        service_score = 16

    insurance_score = 10

    if insurance:
        documented_claims = sum(
            1 for accident in accidents
            if str(accident.get("insurance_claim", "")).lower() == "yes"
        )

        stored_claims = int(insurance.get("claim_count") or 0)

        claim_count = max(documented_claims, stored_claims)

        if claim_count >= 2:
            insurance_score = 5
        elif claim_count == 1:
            insurance_score = 7

    ownership_score = 10
    owner_count = int(data["vehicle"].get("owner_count") or 1)

    if owner_count >= 4:
        ownership_score = 5
    elif owner_count == 3:
        ownership_score = 7
    elif owner_count == 2:
        ownership_score = 9

    km_score = 15
    if km.get("status") == "Inconsistent":
        km_score = 5
    elif km.get("status") == "No service records":
        km_score = 10

    condition_score = 21
    if condition:
        negative_values = {
            "poor",
            "damaged",
            "major damage",
            "bad",
            "flood",
            "fire"
        }

        condition_values = [
            str(value).lower()
            for key, value in condition.items()
            if key not in ("vehicle_id", "condition_id", "inspection_source", "inspection_date")
            and value is not None
        ]

        negatives = sum(value in negative_values for value in condition_values)

        if negatives >= 3:
            condition_score = 12
        elif negatives >= 1:
            condition_score = 17

    total = (
        accident_score
        + service_score
        + insurance_score
        + ownership_score
        + km_score
        + condition_score
    )

    return {
        "score": total,
        "max_score": 100,
        "breakdown": {
            "accident_history": {
                "score": accident_score,
                "max": 20
            },
            "service_history": {
                "score": service_score,
                "max": 20
            },
            "insurance_history": {
                "score": insurance_score,
                "max": 10
            },
            "ownership": {
                "score": ownership_score,
                "max": 10
            },
            "km_verification": {
                "score": km_score,
                "max": 15
            },
            "condition": {
                "score": condition_score,
                "max": 25
            }
        },
        "note": (
            "This is a transparent development score based only on "
            "documented CarSure records."
        )
    }


# ---------------------------------------------------------
# AI valuation
# ---------------------------------------------------------
def predict_vehicle_value(data):
    required = [
        "brand",
        "model",
        "variant",
        "fuel_type",
        "transmission",
        "manufacture_year",
        "current_km",
        "owner_count",
        "registration_location"
    ]

    missing = [
        field for field in required
        if data.get(field) in (None, "")
    ]

    if missing:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing)
        )

    manufacture_year = int(data["manufacture_year"])
    current_km = int(data["current_km"])
    owner_count = int(data["owner_count"])

    if manufacture_year < 1990 or manufacture_year > CURRENT_YEAR:
        raise ValueError(
            f"Manufacture year must be between 1990 and {CURRENT_YEAR}."
        )

    if current_km < 0:
        raise ValueError("Current KM cannot be negative.")

    if owner_count < 1 or owner_count > 10:
        raise ValueError("Owner count must be between 1 and 10.")

    vehicle_age = CURRENT_YEAR - manufacture_year

    model_input = pd.DataFrame([{
        "brand": str(data["brand"]),
        "model": str(data["model"]),
        "variant": str(data["variant"]),
        "fuel_type": str(data["fuel_type"]),
        "transmission": str(data["transmission"]),
        "manufacture_year": manufacture_year,
        "vehicle_age": vehicle_age,
        "current_km": current_km,
        "owner_count": owner_count,
        "registration_location": str(data["registration_location"])
    }])

    prediction = float(model.predict(model_input)[0])
    prediction = max(0, prediction)

    fair_min = prediction * 0.95
    fair_max = prediction * 1.05

    return {
        "ai_predicted_value": money(prediction),
        "fair_buying_range": {
            "min": money(fair_min),
            "max": money(fair_max)
        },
        "vehicle_age": vehicle_age,
        "input": {
            "brand": str(data["brand"]),
            "model": str(data["model"]),
            "variant": str(data["variant"]),
            "fuel_type": str(data["fuel_type"]),
            "transmission": str(data["transmission"]),
            "manufacture_year": manufacture_year,
            "current_km": current_km,
            "owner_count": owner_count,
            "registration_location": str(data["registration_location"])
        },
        "depreciation_note": (
            "The estimate is generated from the CarSure development/sample "
            "training dataset. It is not a guaranteed market or transaction price."
        ),
        "data_status": "Development/sample model"
    }


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------
# Step 15 / 16: Estimate vehicle value
# ---------------------------------------------------------
@app.route("/api/estimate", methods=["POST"])
def estimate_vehicle():
    try:
        data = request.get_json(silent=True) or {}

        result = predict_vehicle_value(data)

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ---------------------------------------------------------
# Step 15: Save valuation
# ---------------------------------------------------------
@app.route("/api/save-valuation", methods=["POST"])
def save_valuation():
    try:
        data = request.get_json(silent=True) or {}

        result = predict_vehicle_value(data)

        valuation_id = (
            "VAL-"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6].upper()
        )

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db_connection()

        conn.execute("""
            INSERT INTO valuation_requests (
                valuation_id,
                created_at,
                brand,
                model,
                variant,
                fuel_type,
                transmission,
                manufacture_year,
                current_km,
                owner_count,
                registration_location,
                ai_predicted_value,
                fair_buying_min,
                fair_buying_max,
                fair_buying_range,
                data_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            valuation_id,
            created_at,
            result["input"]["brand"],
            result["input"]["model"],
            result["input"]["variant"],
            result["input"]["fuel_type"],
            result["input"]["transmission"],
            result["input"]["manufacture_year"],
            result["input"]["current_km"],
            result["input"]["owner_count"],
            result["input"]["registration_location"],
            result["ai_predicted_value"],
            result["fair_buying_range"]["min"],
            result["fair_buying_range"]["max"],
            f"₹{result['fair_buying_range']['min']:,.0f} - ₹{result['fair_buying_range']['max']:,.0f}",
            result["data_status"]
        ))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Valuation saved successfully.",
            "valuation_id": valuation_id,
            "created_at": created_at,
            **result
        })

    except sqlite3.IntegrityError:
        return jsonify({
            "success": False,
            "error": "Unable to create a unique valuation record. Please try again."
        }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ---------------------------------------------------------
# Step 16: Valuation history
# ---------------------------------------------------------
@app.route("/api/valuation-history", methods=["GET"])
def valuation_history():
    try:
        conn = get_db_connection()

        rows = conn.execute("""
            SELECT
                valuation_id,
                created_at,
                brand,
                model,
                variant,
                fuel_type,
                transmission,
                manufacture_year,
                current_km,
                owner_count,
                registration_location,
                ai_predicted_value,
                fair_buying_min,
                fair_buying_max,
                data_status
            FROM valuation_requests
            ORDER BY created_at DESC
        """).fetchall()

        conn.close()

        valuations = []

        for row in rows:
            item = dict(row)

            item["ai_predicted_value"] = money(
                item["ai_predicted_value"]
            )

            minimum = item.get("fair_buying_min")
            maximum = item.get("fair_buying_max")

            # Fallback for an older Step 15 record.
            if (minimum is None or maximum is None) and item.get("fair_buying_range"):
                numbers = re.findall(
                    r"[0-9]+(?:\\.[0-9]+)?",
                    str(item["fair_buying_range"])
                )
                if len(numbers) >= 2:
                    minimum = float(numbers[0])
                    maximum = float(numbers[1])

            item["fair_buying_range"] = {
                "min": money(minimum),
                "max": money(maximum)
            }

            del item["fair_buying_min"]
            del item["fair_buying_max"]

            valuations.append(item)

        return jsonify({
            "success": True,
            "count": len(valuations),
            "valuations": valuations
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# Step 16: Single valuation details
# ---------------------------------------------------------
@app.route("/api/valuation/<valuation_id>", methods=["GET"])
def valuation_details(valuation_id):
    try:
        conn = get_db_connection()

        row = conn.execute("""
            SELECT *
            FROM valuation_requests
            WHERE valuation_id = ?
        """, (valuation_id,)).fetchone()

        conn.close()

        if row is None:
            return jsonify({
                "success": False,
                "error": "Valuation not found."
            }), 404

        item = dict(row)

        item["ai_predicted_value"] = money(
            item["ai_predicted_value"]
        )

        minimum = item.get("fair_buying_min")
        maximum = item.get("fair_buying_max")

        if (minimum is None or maximum is None) and item.get("fair_buying_range"):
            numbers = re.findall(
                r"[0-9]+(?:\\.[0-9]+)?",
                str(item["fair_buying_range"])
            )
            if len(numbers) >= 2:
                minimum = float(numbers[0])
                maximum = float(numbers[1])

        item["fair_buying_range"] = {
            "min": money(minimum),
            "max": money(maximum)
        }

        del item["fair_buying_min"]
        del item["fair_buying_max"]

        return jsonify({
            "success": True,
            "valuation": item
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# Vehicle list/search
# ---------------------------------------------------------
@app.route("/api/vehicles", methods=["GET"])
def get_vehicles():
    conn = get_db_connection()

    rows = conn.execute("""
        SELECT
            vehicle_id,
            brand,
            model,
            variant,
            colour,
            fuel_type,
            transmission,
            manufacture_year,
            registration_date,
            registration_year,
            registration_location,
            current_km,
            owner_count,
            original_ex_showroom_price,
            original_onroad_price,
            current_market_value,
            dealer_asking_price
        FROM vehicles
        ORDER BY vehicle_id
    """).fetchall()

    conn.close()

    return jsonify({
        "success": True,
        "vehicles": [dict(row) for row in rows]
    })


# ---------------------------------------------------------
# Complete vehicle report
# ---------------------------------------------------------
@app.route("/api/report/<vehicle_id>", methods=["GET"])
def get_report(vehicle_id):
    conn = get_db_connection()

    vehicle_row = conn.execute(
        "SELECT * FROM vehicles WHERE vehicle_id = ?",
        (vehicle_id,)
    ).fetchone()

    if vehicle_row is None:
        conn.close()
        return jsonify({
            "success": False,
            "error": "Vehicle not found."
        }), 404

    vehicle = dict(vehicle_row)

    accident_rows = conn.execute(
        """
        SELECT *
        FROM accidents
        WHERE vehicle_id = ?
        ORDER BY accident_date ASC
        """,
        (vehicle_id,)
    ).fetchall()

    service_rows = conn.execute(
        """
        SELECT *
        FROM service_history
        WHERE vehicle_id = ?
        ORDER BY service_date ASC
        """,
        (vehicle_id,)
    ).fetchall()

    insurance_row = conn.execute(
        """
        SELECT *
        FROM insurance
        WHERE vehicle_id = ?
        ORDER BY insurance_to DESC
        LIMIT 1
        """,
        (vehicle_id,)
    ).fetchone()

    condition_row = conn.execute(
        """
        SELECT *
        FROM vehicle_condition
        WHERE vehicle_id = ?
        ORDER BY inspection_date DESC
        LIMIT 1
        """,
        (vehicle_id,)
    ).fetchone()

    market_rows = conn.execute(
        """
        SELECT *
        FROM market_prices
        WHERE vehicle_id = ?
        """,
        (vehicle_id,)
    ).fetchall()

    conn.close()

    accidents = [dict(row) for row in accident_rows]
    services = [dict(row) for row in service_rows]
    insurance = dict(insurance_row) if insurance_row else None
    condition = dict(condition_row) if condition_row else None
    market_prices = [dict(row) for row in market_rows]

    service_readings = []

    for service in services:
        if service.get("odometer_km") is not None:
            service_readings.append({
                "date": service.get("service_date"),
                "km": service.get("odometer_km"),
                "source": service.get("service_center")
            })

    current_km = int(vehicle.get("current_km") or 0)

    documented_kms = [
        int(item["km"])
        for item in service_readings
        if item.get("km") is not None
    ]

    km_issues = []

    if documented_kms:
        if any(
            documented_kms[index] > documented_kms[index + 1]
            for index in range(len(documented_kms) - 1)
        ):
            km_issues.append(
                "Service odometer readings are not consistently increasing."
            )

        if max(documented_kms) > current_km:
            km_issues.append(
                "A documented service odometer reading is higher than the current KM."
            )

        km_status = "Inconsistent" if km_issues else "Consistent"

    else:
        km_status = "No service records"

    latest_documented_service_km = (
        documented_kms[-1] if documented_kms else None
    )

    km_verification = {
        "current_km": current_km,
        "latest_documented_service_km": latest_documented_service_km,
        "service_readings": service_readings,
        "status": km_status,
        "issues": km_issues,
        "evidence_note": (
            "KM verification is based only on documented service readings. "
            "CarSure does not claim that the current odometer reading is original "
            "without supporting evidence."
        )
    }

    documented_claims = sum(
        1 for accident in accidents
        if str(accident.get("insurance_claim", "")).lower() == "yes"
    )

    if insurance:
        stored_claims = int(insurance.get("claim_count") or 0)

        if documented_claims > stored_claims:
            insurance["claim_count"] = documented_claims
            insurance["claim_count_source"] = (
                "Reconciled from documented accident claim records"
            )
        else:
            insurance["claim_count_source"] = "Insurance record"

    try:
        model_result = predict_vehicle_value({
            "brand": vehicle["brand"],
            "model": vehicle["model"],
            "variant": vehicle["variant"],
            "fuel_type": vehicle["fuel_type"],
            "transmission": vehicle["transmission"],
            "manufacture_year": vehicle["manufacture_year"],
            "current_km": vehicle["current_km"],
            "owner_count": vehicle["owner_count"],
            "registration_location": vehicle["registration_location"]
        })

        ai_predicted_value = model_result["ai_predicted_value"]
        fair_buying_range = model_result["fair_buying_range"]

    except Exception:
        ai_predicted_value = None
        fair_buying_range = None

    market_listing_average = None

    if market_prices:
        market_listing_average = money(
            sum(
                float(item.get("market_price", 0))
                for item in market_prices
            ) / len(market_prices)
        )

    original_onroad = vehicle.get("original_onroad_price")
    current_market_value = vehicle.get("current_market_value")
    dealer_asking_price = vehicle.get("dealer_asking_price")

    vehicle_age = CURRENT_YEAR - int(vehicle["manufacture_year"])

    if original_onroad:
        retention = (
            float(current_market_value) / float(original_onroad)
        ) * 100
    else:
        retention = None

    if original_onroad:
        depreciation = (
            1
            - (
                float(current_market_value)
                / float(original_onroad)
            )
        ) * 100
    else:
        depreciation = None

    valuation_analysis = {
        "age_years": vehicle_age,
        "current_km": current_km,
        "owner_count": vehicle["owner_count"],
        "market_value_retention_percent": (
            round(retention, 2) if retention is not None else None
        ),
        "depreciation_from_original_onroad_percent": (
            round(depreciation, 2) if depreciation is not None else None
        ),
        "ai_vs_market_difference": (
            money(ai_predicted_value - current_market_value)
            if ai_predicted_value is not None
            and current_market_value is not None
            else None
        ),
        "ai_vs_listing_difference": (
            money(ai_predicted_value - market_listing_average)
            if ai_predicted_value is not None
            and market_listing_average is not None
            else None
        )
    }

    data = {
        "vehicle": vehicle,
        "accidents": accidents,
        "service_history": services,
        "insurance": insurance,
        "condition": condition,
        "market_prices": market_prices,
        "km_verification": km_verification
    }

    history_score = calculate_history_score(data)

    data_confidence = build_data_confidence(
        vehicle,
        accidents,
        services,
        insurance,
        condition,
        km_verification
    )

    return jsonify({
        "success": True,
        "vehicle": vehicle,
        "accidents": accidents,
        "service_history": services,
        "insurance": insurance,
        "condition": condition,
        "market_prices": market_prices,
        "km_verification": km_verification,
        "vehicle_history_score": history_score,
        "data_confidence": data_confidence,
        "ai_predicted_value": ai_predicted_value,
        "fair_buying_range": fair_buying_range,
        "dealer_asking_price": dealer_asking_price,
        "market_listing_average": market_listing_average,
        "dealer_price_difference": (
            money(dealer_asking_price - market_listing_average)
            if dealer_asking_price is not None
            and market_listing_average is not None
            else None
        ),
        "valuation_analysis": valuation_analysis,
        "data_status": "Development/sample data and model"
    })


# ---------------------------------------------------------
# Vehicle detail endpoint
# ---------------------------------------------------------
@app.route("/api/vehicle/<vehicle_id>", methods=["GET"])
def get_vehicle(vehicle_id):
    return get_report(vehicle_id)


# ---------------------------------------------------------
# Run application
# ---------------------------------------------------------
if __name__ == "__main__":
    print("CarSure server starting...")
    print("Database:", DB_PATH)
    print("Model:", MODEL_PATH)
    print("Open: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )