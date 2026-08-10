import csv
import io
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from flask import Flask, Response, redirect, render_template, request, url_for
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn_test_report.csv"
DB_PATH = BASE_DIR / "crm.db"

APP_NAME = "CRM Thông Minh"
APP_FULL_NAME = (
    "Hệ thống CRM thông minh hỗ trợ dự đoán khách hàng rời bỏ "
    "và chăm sóc khách hàng"
)

SEGMENT_HIGH = "Nguy cơ cao"
SEGMENT_MEDIUM = "Cần quan tâm"
SEGMENT_SAFE = "An toàn"
STATUS_PENDING = "Chưa chăm sóc"
STATUS_DONE = "Đã chăm sóc"
STATUS_FAILED = "Gửi lỗi"
RESEND_API_URL = "https://api.resend.com/emails"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS care_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_code TEXT NOT NULL,
                segment TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_care_customer ON care_history(customer_code)"
        )
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(care_history)").fetchall()
        }
        migrations = {
            "recipient_email": "TEXT",
            "email_subject": "TEXT",
            "email_content": "TEXT",
            "provider_message_id": "TEXT",
            "error_message": "TEXT",
        }
        for column, column_type in migrations.items():
            if column not in existing_columns:
                conn.execute(
                    f"ALTER TABLE care_history ADD COLUMN {column} {column_type}"
                )


def get_db_rows(query, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def normalize_probability(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1)


def normalize_prediction(series):
    values = pd.to_numeric(series, errors="coerce")
    return (values.fillna(0) >= 0.5).astype(int)


def get_segment(probability):
    if probability >= 0.60:
        return SEGMENT_HIGH
    if probability >= 0.30:
        return SEGMENT_MEDIUM
    return SEGMENT_SAFE


def get_care_action(segment):
    if segment == SEGMENT_HIGH:
        return "Voucher 20% + ưu tiên liên hệ giữ chân"
    if segment == SEGMENT_MEDIUM:
        return "Gửi ưu đãi và gợi ý sản phẩm phù hợp"
    return "Tích điểm và chăm sóc định kỳ"


def badge_class(segment):
    if segment == SEGMENT_HIGH:
        return "danger"
    if segment == SEGMENT_MEDIUM:
        return "warning"
    return "success"


def load_care_status_map():
    rows = get_db_rows(
        """
        SELECT customer_code
        FROM care_history
        WHERE status = ?
        GROUP BY customer_code
        """,
        (STATUS_DONE,),
    )
    return {row["customer_code"]: STATUS_DONE for row in rows}


def load_data():
    required_cols = [
        "ThucTe_RoiBo",
        "XacSuat_LSTM",
        "DuDoan_LSTM",
        "XacSuat_XGBoost",
        "DuDoan_XGBoost",
    ]

    if not DATA_PATH.exists():
        return pd.DataFrame(columns=required_cols)

    df = pd.read_csv(DATA_PATH)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    df["ThucTe_RoiBo"] = pd.to_numeric(df["ThucTe_RoiBo"], errors="coerce")
    df = df[df["ThucTe_RoiBo"].isin([0, 1])].copy()
    df["ThucTe_RoiBo"] = df["ThucTe_RoiBo"].astype(int)

    df["XacSuat_LSTM"] = normalize_probability(df["XacSuat_LSTM"])
    df["XacSuat_XGBoost"] = normalize_probability(df["XacSuat_XGBoost"])
    df["DuDoan_LSTM"] = normalize_prediction(df["DuDoan_LSTM"])
    df["DuDoan_XGBoost"] = normalize_prediction(df["DuDoan_XGBoost"])

    df["XacSuat_TrungBinh"] = (df["XacSuat_LSTM"] + df["XacSuat_XGBoost"]) / 2
    df["XacSuat_Ensemble"] = df["XacSuat_TrungBinh"]
    df["DuDoan_Ensemble"] = (df["XacSuat_Ensemble"] >= 0.5).astype(int)

    df = df.reset_index(drop=True)
    df["MaHienThi"] = [f"KH{i + 1:03d}" for i in range(len(df))]
    df["PhanKhuc"] = df["XacSuat_Ensemble"].apply(get_segment)
    df["ChamSoc"] = df["PhanKhuc"].apply(get_care_action)
    df["SegmentClass"] = df["PhanKhuc"].apply(badge_class)

    care_status = load_care_status_map()
    df["TrangThaiChamSoc"] = (
        df["MaHienThi"].map(care_status).fillna(STATUS_PENDING)
    )

    return df


def find_customer(customer_code):
    code = (customer_code or "").strip().upper()
    if not code:
        return None
    df = load_data()
    matched = df[df["MaHienThi"].str.upper() == code]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def compute_metrics(df):
    empty_metric = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "cm": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
    }
    if df.empty:
        return {
            "LSTM": empty_metric.copy(),
            "XGBoost": empty_metric.copy(),
            "Ensemble": empty_metric.copy(),
        }

    y_true = df["ThucTe_RoiBo"].astype(int)
    models = {
        "LSTM": df["DuDoan_LSTM"].astype(int),
        "XGBoost": df["DuDoan_XGBoost"].astype(int),
        "Ensemble": df["DuDoan_Ensemble"].astype(int),
    }
    results = {}
    for name, y_pred in models.items():
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        results[name] = {
            "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
            "precision": round(
                precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
                2,
            ),
            "recall": round(
                recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
                2,
            ),
            "f1": round(
                f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
                2,
            ),
            "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    return results


def recent_history(limit=20, customer_code=None):
    if customer_code:
        return get_db_rows(
            """
            SELECT customer_code, segment, action, status, created_at, updated_at,
                   recipient_email, email_subject, email_content,
                   provider_message_id, error_message
            FROM care_history
            WHERE customer_code = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (customer_code, limit),
        )
    return get_db_rows(
        """
        SELECT customer_code, segment, action, status, created_at, updated_at,
               recipient_email, email_subject, email_content,
               provider_message_id, error_message
        FROM care_history
        ORDER BY updated_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )


def save_care_action(
    customer_code,
    recipient_email=None,
    email_subject=None,
    email_content=None,
    status=STATUS_DONE,
    provider_message_id=None,
    error_message=None,
):
    customer = find_customer(customer_code)
    if not customer:
        return False

    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    existing = get_db_rows(
        """
        SELECT id FROM care_history
        WHERE customer_code = ? AND status = ?
        ORDER BY id DESC LIMIT 1
        """,
        (customer["MaHienThi"], status),
    )
    with sqlite3.connect(DB_PATH) as conn:
        if existing:
            conn.execute(
                """
                UPDATE care_history
                SET segment = ?, action = ?, status = ?, updated_at = ?,
                    recipient_email = ?, email_subject = ?, email_content = ?,
                    provider_message_id = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    customer["PhanKhuc"],
                    customer["ChamSoc"],
                    status,
                    now,
                    recipient_email,
                    email_subject,
                    email_content,
                    provider_message_id,
                    error_message,
                    existing[0]["id"],
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO care_history
                    (customer_code, segment, action, status, created_at, updated_at,
                     recipient_email, email_subject, email_content,
                     provider_message_id, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    customer["MaHienThi"],
                    customer["PhanKhuc"],
                    customer["ChamSoc"],
                    status,
                    now,
                    now,
                    recipient_email,
                    email_subject,
                    email_content,
                    provider_message_id,
                    error_message,
                ),
            )
    return True


def default_email_subject(customer):
    return f"CRM Thông Minh - Đề xuất chăm sóc khách hàng {customer['MaHienThi']}"


def default_email_content(customer):
    return (
        f"Xin chào khách hàng {customer['MaHienThi']},\n\n"
        "CRM Thông Minh ghi nhận quý khách thuộc nhóm cần được chăm sóc phù hợp. "
        f"Đề xuất hiện tại: {customer['ChamSoc']}.\n\n"
        "Nhân viên phụ trách sẽ liên hệ và hỗ trợ quý khách trong thời gian sớm nhất.\n\n"
        "Trân trọng,\nCRM Thông Minh"
    )


def send_resend_email(recipient_email, subject, content):
    api_key = os.environ.get("RESEND_API_KEY")
    from_email = os.environ.get("RESEND_FROM_EMAIL")
    if not api_key or not from_email:
        return False, None, "Thiếu RESEND_API_KEY hoặc RESEND_FROM_EMAIL."

    payload = {
        "from": from_email,
        "to": [recipient_email],
        "subject": subject,
        "text": content,
    }
    request_payload = json.dumps(payload).encode("utf-8")
    req = Request(
        RESEND_API_URL,
        data=request_payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as response:
            response_body = response.read().decode("utf-8")
            data = json.loads(response_body) if response_body else {}
            return True, data.get("id"), None
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, None, f"Resend HTTP {exc.code}: {body}"
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, None, str(exc)


def filter_customers(df, segment=None, status=None):
    filtered = df.copy()
    if segment and segment != "all":
        filtered = filtered[filtered["PhanKhuc"] == segment]
    if status and status != "all":
        filtered = filtered[filtered["TrangThaiChamSoc"] == status]
    return filtered


def get_retention_context(df):
    total_high = int((df["PhanKhuc"] == SEGMENT_HIGH).sum()) if not df.empty else 0
    cared = int((df["TrangThaiChamSoc"] == STATUS_DONE).sum()) if not df.empty else 0
    pending = int((df["TrangThaiChamSoc"] == STATUS_PENDING).sum()) if not df.empty else 0
    total = len(df)
    processed_rate = round(cared / total * 100, 2) if total else 0
    return {
        "total_high": total_high,
        "cared": cared,
        "pending": pending,
        "processed_rate": processed_rate,
    }


def customers_csv_response(customers, filename):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Ma KH",
            "LSTM",
            "XGBoost",
            "Ensemble",
            "Phan khuc",
            "De xuat CRM",
            "Trang thai cham soc",
        ]
    )
    for customer in customers:
        writer.writerow(
            [
                customer.get("MaHienThi", ""),
                round(float(customer.get("XacSuat_LSTM", 0)) * 100, 2),
                round(float(customer.get("XacSuat_XGBoost", 0)) * 100, 2),
                round(float(customer.get("XacSuat_Ensemble", 0)) * 100, 2),
                customer.get("PhanKhuc", ""),
                customer.get("ChamSoc", ""),
                customer.get("TrangThaiChamSoc", ""),
            ]
        )
    csv_data = "\ufeff" + output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


init_db()


def build_dashboard_context():
    df = load_data()
    metrics = compute_metrics(df)

    total_customer = len(df)
    actual_churn = int(df["ThucTe_RoiBo"].sum()) if not df.empty else 0
    actual_rate = round(actual_churn / total_customer * 100, 2) if total_customer else 0

    high_risk = int((df["PhanKhuc"] == SEGMENT_HIGH).sum()) if not df.empty else 0
    medium_risk = int((df["PhanKhuc"] == SEGMENT_MEDIUM).sum()) if not df.empty else 0
    safe_customer = int((df["PhanKhuc"] == SEGMENT_SAFE).sum()) if not df.empty else 0
    cared = int((df["TrangThaiChamSoc"] == STATUS_DONE).sum()) if not df.empty else 0

    top_risk = (
        df.sort_values("XacSuat_Ensemble", ascending=False).head(10).to_dict("records")
        if not df.empty
        else []
    )

    return {
        "total_customer": total_customer,
        "actual_churn": actual_churn,
        "actual_rate": actual_rate,
        "predicted_lstm": int(df["DuDoan_LSTM"].sum()) if not df.empty else 0,
        "predicted_xgb": int(df["DuDoan_XGBoost"].sum()) if not df.empty else 0,
        "predicted_ensemble": int(df["DuDoan_Ensemble"].sum()) if not df.empty else 0,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "safe_customer": safe_customer,
        "cared": cared,
        "not_cared": max(total_customer - cared, 0),
        "top_risk": top_risk,
        "metrics": metrics,
    }


@app.context_processor
def inject_globals():
    return {
        "APP_NAME": APP_NAME,
        "APP_FULL_NAME": APP_FULL_NAME,
        "SEGMENT_HIGH": SEGMENT_HIGH,
        "SEGMENT_MEDIUM": SEGMENT_MEDIUM,
        "SEGMENT_SAFE": SEGMENT_SAFE,
        "STATUS_PENDING": STATUS_PENDING,
        "STATUS_DONE": STATUS_DONE,
        "STATUS_FAILED": STATUS_FAILED,
    }


@app.route("/")
def tong_quan():
    return render_template("tong_quan.html", **build_dashboard_context())


@app.route("/du-doan-roi-bo")
def du_doan_roi_bo():
    return render_template(
        "du_doan_roi_bo.html",
        customers=load_data().to_dict("records"),
    )


@app.route("/phan-khuc")
def phan_khuc():
    df = load_data()
    return render_template(
        "phan_khuc_khach_hang.html",
        customers=df.to_dict("records"),
        high=int((df["PhanKhuc"] == SEGMENT_HIGH).sum()) if not df.empty else 0,
        medium=int((df["PhanKhuc"] == SEGMENT_MEDIUM).sum()) if not df.empty else 0,
        safe=int((df["PhanKhuc"] == SEGMENT_SAFE).sum()) if not df.empty else 0,
    )


@app.route("/cham-soc")
def cham_soc():
    df = load_data()
    if not df.empty:
        df["priority"] = df["PhanKhuc"].map({SEGMENT_HIGH: 0, SEGMENT_MEDIUM: 1}).fillna(2)
        df = df.sort_values(["priority", "XacSuat_Ensemble"], ascending=[True, False])

    not_cared = int((df["TrangThaiChamSoc"] == STATUS_PENDING).sum()) if not df.empty else 0
    cared = int((df["TrangThaiChamSoc"] == STATUS_DONE).sum()) if not df.empty else 0
    high_pending = (
        int(((df["PhanKhuc"] == SEGMENT_HIGH) & (df["TrangThaiChamSoc"] == STATUS_PENDING)).sum())
        if not df.empty
        else 0
    )

    return render_template(
        "cham_soc_khach_hang.html",
        customers=df.to_dict("records"),
        history=recent_history(20),
        not_cared=not_cared,
        cared=cared,
        high_pending=high_pending,
    )


@app.route("/cham-soc/gui/<customer_code>", methods=["POST"])
def gui_cham_soc(customer_code):
    customer = find_customer(customer_code)
    next_url = request.form.get("next") or url_for("cham_soc")
    if not customer:
        return redirect(next_url)

    recipient_email = (request.form.get("recipient_email") or "").strip()
    email_subject = (request.form.get("email_subject") or "").strip()
    email_content = (request.form.get("email_content") or "").strip()
    if not recipient_email or not email_subject or not email_content:
        save_care_action(
            customer_code,
            recipient_email=recipient_email,
            email_subject=email_subject,
            email_content=email_content,
            status=STATUS_FAILED,
            error_message="Thiếu email người nhận, tiêu đề hoặc nội dung.",
        )
        return redirect(f"{next_url}?email_status=error")

    success, provider_message_id, error_message = send_resend_email(
        recipient_email, email_subject, email_content
    )
    if success:
        save_care_action(
            customer_code,
            recipient_email=recipient_email,
            email_subject=email_subject,
            email_content=email_content,
            status=STATUS_DONE,
            provider_message_id=provider_message_id,
        )
        return redirect(f"{next_url}?email_status=success")

    save_care_action(
        customer_code,
        recipient_email=recipient_email,
        email_subject=email_subject,
        email_content=email_content,
        status=STATUS_FAILED,
        error_message=error_message,
    )
    return redirect(f"{next_url}?email_status=error")


@app.route("/chien-dich-giu-chan")
def chien_dich_giu_chan():
    df = load_data()
    segment = request.args.get("segment", "all")
    status = request.args.get("status", "all")
    if not df.empty:
        df["priority"] = df["PhanKhuc"].map({SEGMENT_HIGH: 0, SEGMENT_MEDIUM: 1}).fillna(2)
        df = df.sort_values(["priority", "XacSuat_Ensemble"], ascending=[True, False])
    filtered_df = filter_customers(df, segment, status)
    return render_template(
        "chien_dich_giu_chan.html",
        customers=filtered_df.to_dict("records"),
        campaign=get_retention_context(df),
        segment_filter=segment,
        status_filter=status,
    )


@app.route("/chien-dich-giu-chan/export")
def export_chien_dich_giu_chan():
    df = load_data()
    segment = request.args.get("segment", "all")
    status = request.args.get("status", "all")
    filtered_df = filter_customers(df, segment, status)
    return customers_csv_response(
        filtered_df.to_dict("records"),
        "chien-dich-giu-chan.csv",
    )


@app.route("/xuat-bao-cao/khach-nguy-co-cao.csv")
def xuat_bao_cao_nguy_co_cao():
    df = load_data()
    high_risk = df[df["PhanKhuc"] == SEGMENT_HIGH] if not df.empty else df
    return customers_csv_response(
        high_risk.to_dict("records"),
        "khach-nguy-co-cao.csv",
    )


@app.route("/phan-tich", methods=["GET", "POST"])
def phan_tich():
    result = None
    error = None
    manual_result = None
    customers = load_data()[["MaHienThi", "PhanKhuc"]].to_dict("records")

    if request.method == "POST":
        mode = request.form.get("mode", "lookup")
        if mode == "lookup":
            code = request.form.get("ma_khach_hang")
            result = find_customer(code)
            if not result:
                error = f"Không tìm thấy khách hàng {code or ''} trong CSV."
        else:
            try:
                lstm = float(request.form.get("lstm", ""))
                xgboost = float(request.form.get("xgboost", ""))
            except ValueError:
                error = "Vui lòng nhập xác suất hợp lệ."
            else:
                if lstm > 1:
                    lstm = lstm / 100
                if xgboost > 1:
                    xgboost = xgboost / 100
                lstm = max(0, min(lstm, 1))
                xgboost = max(0, min(xgboost, 1))
                ensemble = (lstm + xgboost) / 2
                segment = get_segment(ensemble)
                manual_result = {
                    "MaHienThi": request.form.get("manual_code") or "Khách hàng mới",
                    "XacSuat_LSTM": lstm,
                    "XacSuat_XGBoost": xgboost,
                    "XacSuat_Ensemble": ensemble,
                    "DuDoan_Ensemble": int(ensemble >= 0.5),
                    "ThucTe_RoiBo": None,
                    "PhanKhuc": segment,
                    "ChamSoc": get_care_action(segment),
                    "SegmentClass": badge_class(segment),
                    "TrangThaiChamSoc": STATUS_PENDING,
                }

    return render_template(
        "phan_tich_khach_hang.html",
        result=result,
        manual_result=manual_result,
        error=error,
        customers=customers,
    )


@app.route("/danh-gia-mo-hinh")
def danh_gia_mo_hinh():
    metrics = compute_metrics(load_data())
    best = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        best[metric] = max(metrics, key=lambda model: metrics[model][metric])
    return render_template("danh_gia_mo_hinh.html", metrics=metrics, best=best)


@app.route("/khach-hang/<customer_code>")
def chi_tiet_khach_hang(customer_code):
    customer = find_customer(customer_code)
    if not customer:
        return render_template(
            "chi_tiet_khach_hang.html",
            customer=None,
            history=[],
            customer_code=customer_code,
        )
    return render_template(
        "chi_tiet_khach_hang.html",
        customer=customer,
        history=recent_history(20, customer["MaHienThi"]),
        customer_code=customer_code,
        default_subject=default_email_subject(customer),
        default_content=default_email_content(customer),
        email_status=request.args.get("email_status"),
    )


# Legacy URLs are kept as redirects so old bookmarks do not break after the rename.
@app.route("/churn")
def churn():
    return redirect(url_for("du_doan_roi_bo"))


@app.route("/segmentation")
def segmentation():
    return redirect(url_for("phan_khuc"))


@app.route("/care")
def care():
    return redirect(url_for("cham_soc"))


@app.route("/care/send/<customer_code>", methods=["POST"])
def send_care(customer_code):
    return redirect(url_for("chi_tiet_khach_hang", customer_code=customer_code))


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    return redirect(url_for("phan_tich"))


@app.route("/evaluation")
def evaluation():
    return redirect(url_for("danh_gia_mo_hinh"))


@app.route("/customer/<customer_code>")
def customer_detail(customer_code):
    return redirect(url_for("chi_tiet_khach_hang", customer_code=customer_code))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
