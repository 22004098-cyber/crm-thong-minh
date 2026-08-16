import csv
import io
import json
import logging
import os
import re
import secrets
import smtplib
import socket
import sqlite3
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formataddr
from functools import wraps
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "crm-thong-minh-demo-secret-key")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn_test_report.csv"
DB_PATH = BASE_DIR / "crm.db"

APP_NAME = "CRM ThĂ´ng Minh"
APP_FULL_NAME = (
    "Há»‡ thá»‘ng CRM thĂ´ng minh há»— trá»£ dá»± Ä‘oĂ¡n khĂ¡ch hĂ ng rá»i bá» "
    "vĂ  chÄƒm sĂ³c khĂ¡ch hĂ ng"
)

SEGMENT_HIGH = "Nguy cÆ¡ cao"
SEGMENT_MEDIUM = "Cáº§n quan tĂ¢m"
SEGMENT_SAFE = "An toĂ n"
STATUS_PENDING = "ChÆ°a chÄƒm sĂ³c"
STATUS_DONE = "ÄĂ£ gá»­i Email"
EMAIL_STATUS_DONE = "ÄĂ£ gá»­i"
LEGACY_STATUS_DONE = "ÄĂ£ chÄƒm sĂ³c"
STATUS_FAILED = "Gá»­i lá»—i"
DONE_STATUSES = (STATUS_DONE, EMAIL_STATUS_DONE, LEGACY_STATUS_DONE)
ROLE_ADMIN = "ADMIN"
ROLE_EMPLOYEE = "NHAN_VIEN"
USER_ACTIVE = "active"
USER_LOCKED = "locked"
RESET_TOKEN_MINUTES = 30
BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE,
                phone TEXT,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reset_token ON password_reset_tokens(token)")
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
            "subject": "TEXT",
            "message": "TEXT",
            "sent_at": "TEXT",
            "provider_message_id": "TEXT",
            "error_message": "TEXT",
            "user_id": "INTEGER",
            "employee_name": "TEXT",
        }
        for column, column_type in migrations.items():
            if column not in existing_columns:
                conn.execute(
                    f"ALTER TABLE care_history ADD COLUMN {column} {column_type}"
                )
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        default_users = [
            (
                "Quáº£n trá»‹ viĂªn",
                "admin@crm.local",
                "admin",
                ROLE_ADMIN,
                "Admin12345",
            ),
            (
                "NhĂ¢n viĂªn CRM",
                "nhanvien@crm.local",
                "nhanvien",
                ROLE_EMPLOYEE,
                "Nhanvien123",
            ),
        ]
        for full_name, email, username, role, password in default_users:
            exists = conn.execute(
                "SELECT id FROM users WHERE username = ? OR email = ?",
                (username, email),
            ).fetchone()
            if not exists:
                conn.execute(
                    """
                    INSERT INTO users
                        (full_name, email, username, role, status, password_hash, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        full_name,
                        email,
                        username,
                        role,
                        USER_ACTIVE,
                        generate_password_hash(password),
                        now,
                        now,
                    ),
                )


def get_db_rows(query, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def get_db_row(query, params=()):
    rows = get_db_rows(query, params)
    return rows[0] if rows else None


def execute_db(query, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(query, params)
        return cur.lastrowid


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_db_row("SELECT * FROM users WHERE id = ?", (user_id,))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            return redirect(url_for("dang_nhap", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            return redirect(url_for("dang_nhap", next=request.path))
        if user["role"] != ROLE_ADMIN:
            flash("Báº¡n khĂ´ng cĂ³ quyá»n truy cáº­p chá»©c nÄƒng quáº£n trá»‹ tĂ i khoáº£n.", "danger")
            return redirect(url_for("tong_quan"))
        return view(*args, **kwargs)

    return wrapped


@app.after_request
def add_no_cache_headers(response):
    if request.endpoint not in {"static"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def validate_password(password, confirm=None):
    if not password:
        return "Máº­t kháº©u khĂ´ng Ä‘Æ°á»£c rá»—ng."
    if len(password) < 8:
        return "Máº­t kháº©u cáº§n cĂ³ Ă­t nháº¥t 8 kĂ½ tá»±."
    if confirm is not None and password != confirm:
        return "Máº­t kháº©u xĂ¡c nháº­n khĂ´ng khá»›p."
    return None


def parse_datetime(value):
    if not value:
        return None
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def get_care_history_counts():
    rows = get_db_rows(
        """
        SELECT customer_code, COUNT(*) AS total
        FROM care_history
        GROUP BY customer_code
        """
    )
    return {row["customer_code"]: int(row["total"]) for row in rows}


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
        return "Voucher 20% + Æ°u tiĂªn liĂªn há»‡ giá»¯ chĂ¢n"
    if segment == SEGMENT_MEDIUM:
        return "Gá»­i Æ°u Ä‘Ă£i vĂ  gá»£i Ă½ sáº£n pháº©m phĂ¹ há»£p"
    return "TĂ­ch Ä‘iá»ƒm vĂ  chÄƒm sĂ³c Ä‘á»‹nh ká»³"


def badge_class(segment):
    if segment == SEGMENT_HIGH:
        return "danger"
    if segment == SEGMENT_MEDIUM:
        return "warning"
    return "success"


def load_care_status_map():
    placeholders = ", ".join("?" for _ in DONE_STATUSES)
    rows = get_db_rows(
        """
        SELECT customer_code
        FROM care_history
        WHERE status IN ({})
        GROUP BY customer_code
        """.format(placeholders),
        DONE_STATUSES,
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
    if "XacSuat_Ensemble" in df.columns:
        df["XacSuat_Ensemble"] = normalize_probability(df["XacSuat_Ensemble"])
    else:
        df["XacSuat_Ensemble"] = df["XacSuat_TrungBinh"]
    df["DuDoan_Ensemble"] = (df["XacSuat_Ensemble"] >= 0.5).astype(int)

    df = df.reset_index(drop=True)
    if "CustomerID" in df.columns:
        df["MaHienThi"] = df["CustomerID"].astype(str)
    elif "MaKhachHang" in df.columns:
        df["MaHienThi"] = df["MaKhachHang"].astype(str)
    else:
        df["MaHienThi"] = [f"KH{i + 1:03d}" for i in range(len(df))]
    if "PhanKhuc" not in df.columns:
        df["PhanKhuc"] = df["XacSuat_Ensemble"].apply(get_segment)
    if "ChamSoc" not in df.columns:
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
                   subject, message, sent_at, provider_message_id, error_message,
                   user_id, employee_name
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
               subject, message, sent_at, provider_message_id, error_message,
               user_id, employee_name
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
    user = current_user()
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
                    subject = ?, message = ?, sent_at = ?,
                    provider_message_id = ?, error_message = ?,
                    user_id = ?, employee_name = ?
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
                    email_subject,
                    email_content,
                    now if status == STATUS_DONE else None,
                    provider_message_id,
                    error_message,
                    user["id"] if user else None,
                    user["full_name"] if user else None,
                    existing[0]["id"],
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO care_history
                    (customer_code, segment, action, status, created_at, updated_at,
                     recipient_email, email_subject, email_content,
                     subject, message, sent_at,
                     provider_message_id, error_message, user_id, employee_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    email_subject,
                    email_content,
                    now if status == STATUS_DONE else None,
                    provider_message_id,
                    error_message,
                    user["id"] if user else None,
                    user["full_name"] if user else None,
                ),
            )
    return True


def default_email_subject(customer):
    segment = customer["PhanKhuc"]
    if segment == SEGMENT_HIGH:
        return "Æ¯u Ä‘Ă£i giá»¯ chĂ¢n khĂ¡ch hĂ ng - Voucher 20%"
    if segment == SEGMENT_MEDIUM:
        return "Æ¯u Ä‘Ă£i cĂ¡ nhĂ¢n hĂ³a dĂ nh riĂªng cho quĂ½ khĂ¡ch"
    return "Cáº£m Æ¡n quĂ½ khĂ¡ch Ä‘Ă£ Ä‘á»“ng hĂ nh cĂ¹ng chĂºng tĂ´i"


def default_email_content(customer):
    segment = customer["PhanKhuc"]
    if segment == SEGMENT_HIGH:
        body = (
            "Cáº£m Æ¡n quĂ½ khĂ¡ch Ä‘Ă£ tin tÆ°á»Ÿng vĂ  sá»­ dá»¥ng dá»‹ch vá»¥ cá»§a chĂºng tĂ´i. "
            "Äá»ƒ tri Ă¢n vĂ  tiáº¿p tá»¥c Ä‘á»“ng hĂ nh cĂ¹ng quĂ½ khĂ¡ch, chĂºng tĂ´i gá»­i táº·ng "
            "voucher Æ°u Ä‘Ă£i 20% cho láº§n sá»­ dá»¥ng tiáº¿p theo. Ráº¥t mong quĂ½ khĂ¡ch "
            "tiáº¿p tá»¥c tráº£i nghiá»‡m dá»‹ch vá»¥ trong thá»i gian tá»›i."
        )
    elif segment == SEGMENT_MEDIUM:
        body = (
            "ChĂºng tĂ´i gá»­i Ä‘áº¿n quĂ½ khĂ¡ch má»™t Æ°u Ä‘Ă£i cĂ¡ nhĂ¢n hĂ³a cĂ¹ng cĂ¡c gá»£i Ă½ "
            "sáº£n pháº©m/dá»‹ch vá»¥ phĂ¹ há»£p vá»›i nhu cáº§u hiá»‡n táº¡i. Hy vá»ng nhá»¯ng Ä‘á» xuáº¥t "
            "nĂ y giĂºp quĂ½ khĂ¡ch cĂ³ tráº£i nghiá»‡m tá»‘t hÆ¡n."
        )
    else:
        body = (
            "Cáº£m Æ¡n quĂ½ khĂ¡ch Ä‘Ă£ luĂ´n Ä‘á»“ng hĂ nh cĂ¹ng chĂºng tĂ´i. QuĂ½ khĂ¡ch sáº½ tiáº¿p tá»¥c "
            "Ä‘Æ°á»£c tĂ­ch Ä‘iá»ƒm vĂ  nháº­n cĂ¡c hoáº¡t Ä‘á»™ng chÄƒm sĂ³c Ä‘á»‹nh ká»³ tá»« há»‡ thá»‘ng CRM."
        )
    return (
        f"Xin chĂ o khĂ¡ch hĂ ng {customer['MaHienThi']},\n\n"
        f"{body}\n\n"
        f"Äá» xuáº¥t CRM: {customer['ChamSoc']}.\n\n"
        "TrĂ¢n trá»ng,\nCRM ThĂ´ng Minh"
    )


def is_valid_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""))


def is_email_config_ready():
    brevo_ready = bool(os.environ.get("BREVO_API_KEY") and os.environ.get("BREVO_SENDER_EMAIL"))
    gmail_ready = bool(os.environ.get("GMAIL_SENDER_EMAIL") and os.environ.get("GMAIL_APP_PASSWORD"))
    return brevo_ready or gmail_ready


def log_email_config_error(message):
    app.logger.error("[EMAIL CONFIG ERROR] %s", message)


def log_email_config(sender_email, secret_value, provider):
    app.logger.info(
        "\n".join(
            [
                "[EMAIL CONFIG]",
                f"Provider: {provider}",
                f"Sender configured: {'YES' if sender_email else 'NO'}",
                f"Secret configured: {'YES' if secret_value else 'NO'}",
            ]
        )
    )


def log_email_send(customer_code, recipient_email):
    app.logger.info(
        "\n".join(
            [
                "[EMAIL SEND]",
                f"Customer: {customer_code}",
                f"Recipient: {recipient_email or 'N/A'}",
            ]
        )
    )


def log_email_error(
    customer_code,
    recipient_email,
    sender_email,
    error_type,
    error_message,
):
    app.logger.error(
        "\n".join(
            [
                "[EMAIL ERROR]",
                f"Customer: {customer_code}",
                f"Recipient: {recipient_email or 'N/A'}",
                f"Sender: {sender_email or 'N/A'}",
                f"Error type: {error_type or 'N/A'}",
                f"Error message: {error_message or 'N/A'}",
            ]
        )
    )


def log_email_success(customer_code, recipient_email, sender_email, subject):
    app.logger.info(
        "\n".join(
            [
                "[EMAIL SUCCESS]",
                f"Customer: {customer_code}",
                f"Recipient: {recipient_email}",
                f"Sender: {sender_email}",
                f"Subject: {subject}",
            ]
        )
    )


def send_brevo_email(customer_code, recipient_email, subject, content):
    api_key = os.environ.get("BREVO_API_KEY")
    sender_email = os.environ.get("BREVO_SENDER_EMAIL") or os.environ.get("GMAIL_SENDER_EMAIL")
    sender_name = os.environ.get("BREVO_SENDER_NAME") or os.environ.get("GMAIL_SENDER_NAME", APP_NAME)
    log_email_config(sender_email, api_key, "BREVO_API")
    log_email_send(customer_code, recipient_email)
    if not api_key:
        error_message = "Missing BREVO_API_KEY"
        log_email_config_error(error_message)
        log_email_error(customer_code, recipient_email, sender_email, "EmailConfigError", error_message)
        return False, None, "Cáº¥u hĂ¬nh email production chÆ°a sáºµn sĂ ng. Thiáº¿u BREVO_API_KEY."
    if not sender_email:
        error_message = "Missing BREVO_SENDER_EMAIL"
        log_email_config_error(error_message)
        log_email_error(customer_code, recipient_email, sender_email, "EmailConfigError", error_message)
        return False, None, "Cáº¥u hĂ¬nh email production chÆ°a sáºµn sĂ ng. Thiáº¿u BREVO_SENDER_EMAIL."
    if not is_valid_email(recipient_email):
        error_message = "Email ngÆ°á»i nháº­n khĂ´ng há»£p lá»‡."
        log_email_error(customer_code, recipient_email, sender_email, "EmailValidationError", error_message)
        return False, None, error_message

    payload = {
        "sender": {"name": sender_name, "email": sender_email},
        "to": [{"email": recipient_email}],
        "subject": subject,
        "htmlContent": content.replace("\n", "<br>"),
        "textContent": content,
    }
    req = Request(
        BREVO_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "api-key": api_key,
            "accept": "application/json",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body) if body else {}
            message_id = data.get("messageId")
            log_email_success(customer_code, recipient_email, sender_email, subject)
            return True, message_id, None
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        error_message = f"Brevo HTTP {exc.code}: {body}"
        log_email_error(customer_code, recipient_email, sender_email, "BrevoHTTPError", error_message)
        return False, None, error_message
    except (URLError, TimeoutError, socket.timeout) as exc:
        error_message = f"KhĂ´ng káº¿t ná»‘i Ä‘Æ°á»£c Brevo API qua HTTPS. Chi tiáº¿t: {exc}"
        log_email_error(customer_code, recipient_email, sender_email, exc.__class__.__name__, error_message)
        return False, None, error_message
    except (OSError, json.JSONDecodeError) as exc:
        error_message = f"Brevo API tráº£ vá» lá»—i khĂ´ng xá»­ lĂ½ Ä‘Æ°á»£c. Chi tiáº¿t: {exc}"
        log_email_error(customer_code, recipient_email, sender_email, exc.__class__.__name__, error_message)
        return False, None, error_message


def send_gmail_email(customer_code, recipient_email, subject, content):
    sender_email = os.environ.get("GMAIL_SENDER_EMAIL")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    sender_name = os.environ.get("GMAIL_SENDER_NAME", APP_NAME)
    log_email_config(sender_email, app_password, "GMAIL_SMTP")
    log_email_send(customer_code, recipient_email)
    if not sender_email:
        error_message = "Missing GMAIL_SENDER_EMAIL"
        log_email_config_error(error_message)
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            "EmailConfigError",
            error_message,
        )
        return False, None, "Cáº¥u hĂ¬nh email chÆ°a sáºµn sĂ ng. Thiáº¿u GMAIL_SENDER_EMAIL."
    if not app_password:
        error_message = "Missing GMAIL_APP_PASSWORD"
        log_email_config_error(error_message)
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            "EmailConfigError",
            error_message,
        )
        return False, None, "Cáº¥u hĂ¬nh email chÆ°a sáºµn sĂ ng. Thiáº¿u GMAIL_APP_PASSWORD."
    if not is_valid_email(recipient_email):
        error_message = "Email ngÆ°á»i nháº­n khĂ´ng há»£p lá»‡."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            "EmailValidationError",
            error_message,
        )
        return False, None, error_message

    message = EmailMessage()
    message["From"] = formataddr((sender_name, sender_email))
    message["To"] = recipient_email
    message["Subject"] = subject
    message.set_content(content)
    message.add_alternative(content.replace("\n", "<br>"), subtype="html")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(sender_email, app_password)
            smtp.send_message(message)
        log_email_success(customer_code, recipient_email, sender_email, subject)
        return True, None, None
    except smtplib.SMTPAuthenticationError as exc:
        error_message = "Gmail tá»« chá»‘i Ä‘Äƒng nháº­p. HĂ£y kiá»ƒm tra GMAIL_SENDER_EMAIL vĂ  Google App Password."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiáº¿t: {exc}",
        )
        return False, None, error_message
    except smtplib.SMTPConnectError as exc:
        error_message = "KhĂ´ng káº¿t ná»‘i Ä‘Æ°á»£c tá»›i smtp.gmail.com:587."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiáº¿t: {exc}",
        )
        return False, None, error_message
    except smtplib.SMTPServerDisconnected as exc:
        error_message = "Káº¿t ná»‘i SMTP bá»‹ ngáº¯t trong quĂ¡ trĂ¬nh gá»­i."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiáº¿t: {exc}",
        )
        return False, None, error_message
    except smtplib.SMTPException as exc:
        error_message = "Gmail SMTP tráº£ vá» lá»—i khi gá»­i email."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiáº¿t: {exc}",
        )
        return False, None, error_message
    except (TimeoutError, socket.timeout) as exc:
        error_message = "Káº¿t ná»‘i Gmail SMTP bá»‹ timeout. Vui lĂ²ng thá»­ láº¡i sau."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiáº¿t: {exc}",
        )
        return False, None, error_message
    except OSError as exc:
        error_message = "KhĂ´ng thá»ƒ káº¿t ná»‘i Gmail SMTP tá»« mĂ´i trÆ°á»ng hiá»‡n táº¡i."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiáº¿t: {exc}",
        )
        return False, None, error_message


def send_care_email(customer_code, recipient_email, subject, content):
    if os.environ.get("BREVO_API_KEY"):
        return send_brevo_email(customer_code, recipient_email, subject, content)
    return send_gmail_email(customer_code, recipient_email, subject, content)


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


def history_csv_response(rows, filename):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Ma khach",
            "Phan khuc",
            "Email nhan",
            "Tieu de",
            "Noi dung",
            "Trang thai",
            "Nhan vien",
            "Thoi gian",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.get("customer_code", ""),
                row.get("segment", ""),
                row.get("recipient_email", ""),
                row.get("subject") or row.get("email_subject") or row.get("action", ""),
                row.get("message") or row.get("email_content", ""),
                row.get("status", ""),
                row.get("employee_name", ""),
                row.get("sent_at") or row.get("updated_at") or row.get("created_at", ""),
            ]
        )
    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def customers_with_history_counts(df):
    counts = get_care_history_counts()
    records = df.to_dict("records")
    for item in records:
        item["SoLanChamSoc"] = counts.get(item["MaHienThi"], 0)
    return records


def build_report_context(period_type="month", year=None, month=None, quarter=None):
    df = load_data()
    history = recent_history(10000)
    now = datetime.now()
    year = int(year or now.year)
    month = int(month or now.month)
    quarter = int(quarter or ((now.month - 1) // 3 + 1))

    filtered_history = []
    monthly_counts = {m: 0 for m in range(1, 13)}
    for row in history:
        dt = parse_datetime(row.get("sent_at") or row.get("updated_at") or row.get("created_at"))
        if not dt:
            continue
        if dt.year == year:
            monthly_counts[dt.month] += 1
        include = dt.year == year
        if period_type == "month":
            include = include and dt.month == month
        elif period_type == "quarter":
            include = include and ((dt.month - 1) // 3 + 1) == quarter
        if include:
            filtered_history.append(row)

    segment_counts = {
        SEGMENT_HIGH: int((df["PhanKhuc"] == SEGMENT_HIGH).sum()) if not df.empty else 0,
        SEGMENT_MEDIUM: int((df["PhanKhuc"] == SEGMENT_MEDIUM).sum()) if not df.empty else 0,
        SEGMENT_SAFE: int((df["PhanKhuc"] == SEGMENT_SAFE).sum()) if not df.empty else 0,
    }
    done = int((df["TrangThaiChamSoc"] == STATUS_DONE).sum()) if not df.empty else 0
    pending = int((df["TrangThaiChamSoc"] == STATUS_PENDING).sum()) if not df.empty else 0
    failed = sum(1 for row in history if row.get("status") == STATUS_FAILED)
    return {
        "period_type": period_type,
        "year": year,
        "month": month,
        "quarter": quarter,
        "segment_counts": segment_counts,
        "status_counts": {
            "done": done,
            "pending": pending,
            "email_success": sum(1 for row in history if row.get("status") in DONE_STATUSES),
            "email_failed": failed,
        },
        "monthly_counts": monthly_counts,
        "history": filtered_history,
        "customers": customers_with_history_counts(df),
        "metrics": compute_metrics(df),
    }


def answer_crm_question(question):
    q = (question or "").strip().lower()
    df = load_data()
    metrics = compute_metrics(df)
    if not q:
        return "Bạn có thể hỏi về số khách nguy cơ cao, khách chưa chăm sóc, mã khách hàng, Accuracy hoặc F1-score."

    customer_match = re.search(r"\b(?:kh\s*0*(\d+)|cust\d+|[a-z0-9_-]{4,})\b", q, re.IGNORECASE)
    customer = None
    if customer_match:
        if customer_match.group(1):
            customer = find_customer(f"KH{int(customer_match.group(1)):03d}")
        else:
            customer = find_customer(customer_match.group(0))

    if "nguy cơ cao" in q and ("bao nhiêu" in q or "số" in q or "co bao nhieu" in q):
        total = int((df["PhanKhuc"] == SEGMENT_HIGH).sum()) if not df.empty else 0
        return f"Hiện có {total} khách hàng thuộc phân khúc Nguy cơ cao."
    if "chưa chăm sóc" in q:
        total = int((df["TrangThaiChamSoc"] == STATUS_PENDING).sum()) if not df.empty else 0
        return f"Hiện có {total} khách hàng chưa chăm sóc."
    if customer:
        if "lstm" in q:
            return f"{customer['MaHienThi']} có xác suất LSTM là {customer['XacSuat_LSTM'] * 100:.1f}%."
        if "xgboost" in q:
            return f"{customer['MaHienThi']} có xác suất XGBoost là {customer['XacSuat_XGBoost'] * 100:.1f}%."
        if "ensemble" in q or "churn" in q or "nguy cơ" in q:
            return f"{customer['MaHienThi']} có xác suất Ensemble là {customer['XacSuat_Ensemble'] * 100:.1f}% và thuộc phân khúc {customer['PhanKhuc']}."
        if "đề xuất" in q or "chăm sóc" in q:
            return f"Đề xuất chăm sóc cho {customer['MaHienThi']}: {customer['ChamSoc']}."
        return f"{customer['MaHienThi']} thuộc phân khúc {customer['PhanKhuc']}, trạng thái chăm sóc: {customer['TrangThaiChamSoc']}."
    if "accuracy" in q:
        return (
            f"Accuracy hiện tại: LSTM {metrics['LSTM']['accuracy']}%, "
            f"XGBoost {metrics['XGBoost']['accuracy']}%, Ensemble {metrics['Ensemble']['accuracy']}%."
        )
    if "f1" in q:
        best = max(metrics, key=lambda model: metrics[model]["f1"])
        return f"Mô hình có F1-score cao nhất hiện tại là {best} với {metrics[best]['f1']}%."
    if "chăm sóc" in q and "nguy cơ cao" in q:
        return f"Khách nguy cơ cao nên được chăm sóc bằng: {get_care_action(SEGMENT_HIGH)}."
    return (
        "Trợ lý hiện hỗ trợ các câu hỏi như: có bao nhiêu khách nguy cơ cao, "
        "có bao nhiêu khách chưa chăm sóc, mã khách hàng thuộc phân khúc nào, "
        "LSTM/XGBoost/Ensemble của khách hàng, Accuracy các mô hình, F1-score cao nhất."
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
        "ROLE_ADMIN": ROLE_ADMIN,
        "ROLE_EMPLOYEE": ROLE_EMPLOYEE,
        "current_user": current_user(),
    }


@app.route("/dang-nhap", methods=["GET", "POST"])
def dang_nhap():
    if current_user():
        return redirect(url_for("tong_quan"))
    error = None
    if request.method == "POST":
        identity = (request.form.get("identity") or "").strip().lower()
        password = request.form.get("password") or ""
        remember = request.form.get("remember") == "on"
        user = get_db_row(
            """
            SELECT * FROM users
            WHERE lower(email) = ? OR lower(username) = ?
            """,
            (identity, identity),
        )
        if not user or not check_password_hash(user["password_hash"], password):
            error = "TĂ i khoáº£n hoáº·c máº­t kháº©u khĂ´ng Ä‘Ăºng."
        elif user["status"] == USER_LOCKED:
            error = "TĂ i khoáº£n Ä‘Ă£ bá»‹ khĂ³a. Vui lĂ²ng liĂªn há»‡ quáº£n trá»‹ viĂªn."
        else:
            session.clear()
            session["user_id"] = user["id"]
            session.permanent = remember
            next_url = request.args.get("next") or url_for("tong_quan")
            return redirect(next_url)
    return render_template("dang_nhap.html", error=error)


@app.route("/dang-xuat")
def dang_xuat():
    session.clear()
    flash("Báº¡n Ä‘Ă£ Ä‘Äƒng xuáº¥t khá»i há»‡ thá»‘ng.", "success")
    return redirect(url_for("dang_nhap"))


@app.route("/quen-mat-khau", methods=["GET", "POST"])
def quen_mat_khau():
    token_url = None
    message = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        user = get_db_row("SELECT * FROM users WHERE lower(email) = ?", (email,))
        message = "Náº¿u email tá»“n táº¡i, há»‡ thá»‘ng Ä‘Ă£ táº¡o liĂªn káº¿t Ä‘áº·t láº¡i máº­t kháº©u demo."
        if user and user["status"] == USER_ACTIVE:
            token = secrets.token_urlsafe(32)
            now = datetime.now()
            expires_at = (now + timedelta(minutes=RESET_TOKEN_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
            execute_db(
                """
                INSERT INTO password_reset_tokens (user_id, token, expires_at, used, created_at)
                VALUES (?, ?, ?, 0, ?)
                """,
                (user["id"], token, expires_at, now.strftime("%Y-%m-%d %H:%M:%S")),
            )
            token_url = url_for("dat_lai_mat_khau", token=token, _external=False)
    return render_template("quen_mat_khau.html", message=message, token_url=token_url)


@app.route("/dat-lai-mat-khau/<token>", methods=["GET", "POST"])
def dat_lai_mat_khau(token):
    row = get_db_row(
        """
        SELECT prt.*, u.email
        FROM password_reset_tokens prt
        JOIN users u ON u.id = prt.user_id
        WHERE prt.token = ?
        """,
        (token,),
    )
    error = None
    if not row or row["used"]:
        error = "LiĂªn káº¿t Ä‘áº·t láº¡i máº­t kháº©u khĂ´ng há»£p lá»‡ hoáº·c Ä‘Ă£ Ä‘Æ°á»£c sá»­ dá»¥ng."
        return render_template("dat_lai_mat_khau.html", error=error, token=token)
    expires_at = datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S")
    if expires_at < datetime.now():
        error = "LiĂªn káº¿t Ä‘áº·t láº¡i máº­t kháº©u Ä‘Ă£ háº¿t háº¡n."
        return render_template("dat_lai_mat_khau.html", error=error, token=token)
    if request.method == "POST":
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        error = validate_password(password, confirm)
        if not error:
            now = datetime.now().strftime("%d/%m/%Y %H:%M")
            execute_db(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (generate_password_hash(password), now, row["user_id"]),
            )
            execute_db("UPDATE password_reset_tokens SET used = 1 WHERE id = ?", (row["id"],))
            flash("Máº­t kháº©u Ä‘Ă£ Ä‘Æ°á»£c Ä‘áº·t láº¡i. Vui lĂ²ng Ä‘Äƒng nháº­p.", "success")
            return redirect(url_for("dang_nhap"))
    return render_template("dat_lai_mat_khau.html", error=error, token=token)


@app.route("/tai-khoan-cua-toi", methods=["GET", "POST"])
@login_required
def tai_khoan_cua_toi():
    user = current_user()
    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        phone = (request.form.get("phone") or "").strip()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        if not full_name or not email:
            flash("Há» tĂªn vĂ  email khĂ´ng Ä‘Æ°á»£c rá»—ng.", "danger")
        elif password and validate_password(password, confirm):
            flash(validate_password(password, confirm), "danger")
        else:
            existing = get_db_row(
                "SELECT id FROM users WHERE lower(email) = ? AND id <> ?",
                (email, user["id"]),
            )
            if existing:
                flash("Email Ä‘Ă£ Ä‘Æ°á»£c sá»­ dá»¥ng bá»Ÿi tĂ i khoáº£n khĂ¡c.", "danger")
            else:
                now = datetime.now().strftime("%d/%m/%Y %H:%M")
                if password:
                    execute_db(
                        """
                        UPDATE users
                        SET full_name = ?, email = ?, phone = ?, password_hash = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (full_name, email, phone, generate_password_hash(password), now, user["id"]),
                    )
                else:
                    execute_db(
                        "UPDATE users SET full_name = ?, email = ?, phone = ?, updated_at = ? WHERE id = ?",
                        (full_name, email, phone, now, user["id"]),
                    )
                flash("ÄĂ£ cáº­p nháº­t tĂ i khoáº£n cá»§a tĂ´i.", "success")
                return redirect(url_for("tai_khoan_cua_toi"))
    return render_template("tai_khoan_cua_toi.html", user=current_user())


@app.route("/")
@login_required
def tong_quan():
    return render_template("tong_quan.html", **build_dashboard_context())


@app.route("/du-doan-roi-bo")
@login_required
def du_doan_roi_bo():
    return render_template(
        "du_doan_roi_bo.html",
        customers=load_data().to_dict("records"),
    )


@app.route("/phan-khuc")
@login_required
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
@login_required
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
        email_config_ready=is_email_config_ready(),
    )


@app.route("/cham-soc/gui/<customer_code>", methods=["POST"])
@login_required
def gui_cham_soc(customer_code):
    customer = find_customer(customer_code)
    next_url = request.form.get("next") or url_for("cham_soc")
    if not customer:
        return redirect(next_url)

    recipient_email = (request.form.get("recipient_email") or "").strip()
    email_subject = (request.form.get("email_subject") or "").strip()
    email_content = (request.form.get("email_content") or "").strip()
    if not recipient_email or not email_subject or not email_content:
        error_message = "Thiáº¿u email ngÆ°á»i nháº­n, tiĂªu Ä‘á» hoáº·c ná»™i dung."
        log_email_error(
            customer["MaHienThi"],
            recipient_email,
            os.environ.get("GMAIL_SENDER_EMAIL"),
            "EmailValidationError",
            error_message,
        )
        save_care_action(
            customer_code,
            recipient_email=recipient_email,
            email_subject=email_subject,
            email_content=email_content,
            status=STATUS_FAILED,
            error_message=error_message,
        )
        return redirect(f"{next_url}?email_status=error")

    success, provider_message_id, error_message = send_care_email(
        customer["MaHienThi"], recipient_email, email_subject, email_content
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
@login_required
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
@login_required
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
@login_required
def xuat_bao_cao_nguy_co_cao():
    df = load_data()
    high_risk = df[df["PhanKhuc"] == SEGMENT_HIGH] if not df.empty else df
    return customers_csv_response(
        high_risk.to_dict("records"),
        "khach-nguy-co-cao.csv",
    )


@app.route("/quan-ly-tai-khoan")
@admin_required
def quan_ly_tai_khoan():
    users = get_db_rows("SELECT * FROM users ORDER BY id")
    return render_template("quan_ly_tai_khoan.html", users=users)


@app.route("/quan-ly-tai-khoan/them", methods=["GET", "POST"])
@admin_required
def them_tai_khoan():
    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        username = (request.form.get("username") or "").strip().lower()
        phone = (request.form.get("phone") or "").strip()
        role = request.form.get("role") if request.form.get("role") in [ROLE_ADMIN, ROLE_EMPLOYEE] else ROLE_EMPLOYEE
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        error = validate_password(password, confirm)
        if not full_name or not email or not username:
            flash("Há» tĂªn, email vĂ  username khĂ´ng Ä‘Æ°á»£c rá»—ng.", "danger")
        elif error:
            flash(error, "danger")
        elif get_db_row("SELECT id FROM users WHERE lower(email)=? OR lower(username)=?", (email, username)):
            flash("Email hoáº·c username Ä‘Ă£ tá»“n táº¡i.", "danger")
        else:
            now = datetime.now().strftime("%d/%m/%Y %H:%M")
            execute_db(
                """
                INSERT INTO users
                    (full_name, email, username, phone, role, status, password_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    full_name,
                    email,
                    username,
                    phone,
                    role,
                    USER_ACTIVE,
                    generate_password_hash(password),
                    now,
                    now,
                ),
            )
            flash("ÄĂ£ thĂªm tĂ i khoáº£n nhĂ¢n viĂªn.", "success")
            return redirect(url_for("quan_ly_tai_khoan"))
    return render_template("form_tai_khoan.html", item=None)


@app.route("/quan-ly-tai-khoan/<int:user_id>/sua", methods=["GET", "POST"])
@admin_required
def sua_tai_khoan(user_id):
    item = get_db_row("SELECT * FROM users WHERE id = ?", (user_id,))
    if not item:
        flash("KhĂ´ng tĂ¬m tháº¥y tĂ i khoáº£n.", "danger")
        return redirect(url_for("quan_ly_tai_khoan"))
    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        username = (request.form.get("username") or "").strip().lower()
        phone = (request.form.get("phone") or "").strip()
        role = request.form.get("role") if request.form.get("role") in [ROLE_ADMIN, ROLE_EMPLOYEE] else item["role"]
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        error = validate_password(password, confirm) if password else None
        if not full_name or not email or not username:
            flash("Há» tĂªn, email vĂ  username khĂ´ng Ä‘Æ°á»£c rá»—ng.", "danger")
        elif error:
            flash(error, "danger")
        elif get_db_row(
            "SELECT id FROM users WHERE (lower(email)=? OR lower(username)=?) AND id <> ?",
            (email, username, user_id),
        ):
            flash("Email hoáº·c username Ä‘Ă£ Ä‘Æ°á»£c sá»­ dá»¥ng.", "danger")
        else:
            now = datetime.now().strftime("%d/%m/%Y %H:%M")
            if password:
                execute_db(
                    """
                    UPDATE users
                    SET full_name=?, email=?, username=?, phone=?, role=?, password_hash=?, updated_at=?
                    WHERE id=?
                    """,
                    (full_name, email, username, phone, role, generate_password_hash(password), now, user_id),
                )
            else:
                execute_db(
                    "UPDATE users SET full_name=?, email=?, username=?, phone=?, role=?, updated_at=? WHERE id=?",
                    (full_name, email, username, phone, role, now, user_id),
                )
            flash("ÄĂ£ cáº­p nháº­t tĂ i khoáº£n.", "success")
            return redirect(url_for("quan_ly_tai_khoan"))
    return render_template("form_tai_khoan.html", item=item)


@app.route("/quan-ly-tai-khoan/<int:user_id>/khoa", methods=["POST"])
@admin_required
def khoa_tai_khoan(user_id):
    user = current_user()
    target = get_db_row("SELECT * FROM users WHERE id = ?", (user_id,))
    if not target:
        flash("KhĂ´ng tĂ¬m tháº¥y tĂ i khoáº£n.", "danger")
    elif target["id"] == user["id"]:
        flash("Báº¡n khĂ´ng thá»ƒ tá»± khĂ³a tĂ i khoáº£n cá»§a mĂ¬nh.", "danger")
    elif target["role"] == ROLE_ADMIN:
        flash("KhĂ´ng khĂ³a tĂ i khoáº£n Admin qua thao tĂ¡c nhanh.", "danger")
    else:
        execute_db("UPDATE users SET status = ?, updated_at = ? WHERE id = ?", (USER_LOCKED, datetime.now().strftime("%d/%m/%Y %H:%M"), user_id))
        flash("ÄĂ£ khĂ³a tĂ i khoáº£n.", "success")
    return redirect(url_for("quan_ly_tai_khoan"))


@app.route("/quan-ly-tai-khoan/<int:user_id>/mo-khoa", methods=["POST"])
@admin_required
def mo_khoa_tai_khoan(user_id):
    execute_db("UPDATE users SET status = ?, updated_at = ? WHERE id = ?", (USER_ACTIVE, datetime.now().strftime("%d/%m/%Y %H:%M"), user_id))
    flash("ÄĂ£ má»Ÿ khĂ³a tĂ i khoáº£n.", "success")
    return redirect(url_for("quan_ly_tai_khoan"))


@app.route("/quan-ly-khach-hang")
@login_required
def quan_ly_khach_hang():
    df = load_data()
    return render_template("quan_ly_khach_hang.html", customers=customers_with_history_counts(df))


@app.route("/tro-ly-crm", methods=["GET", "POST"])
@login_required
def tro_ly_crm():
    answer = None
    question = ""
    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        answer = answer_crm_question(question)
    return render_template("tro_ly_crm.html", question=question, answer=answer)


@app.route("/thong-ke-bao-cao")
@login_required
def thong_ke_bao_cao():
    period_type = request.args.get("period_type", "month")
    context = build_report_context(
        period_type=period_type,
        year=request.args.get("year"),
        month=request.args.get("month"),
        quarter=request.args.get("quarter"),
    )
    return render_template("thong_ke_bao_cao.html", **context)


@app.route("/xuat-bao-cao/<report_type>.csv")
@login_required
def xuat_bao_cao(report_type):
    df = load_data()
    if report_type == "chua-cham-soc":
        data = df[df["TrangThaiChamSoc"] == STATUS_PENDING]
        return customers_csv_response(data.to_dict("records"), "khach-chua-cham-soc.csv")
    if report_type == "da-cham-soc":
        data = df[df["TrangThaiChamSoc"] == STATUS_DONE]
        return customers_csv_response(data.to_dict("records"), "khach-da-cham-soc.csv")
    if report_type == "lich-su-cham-soc":
        return history_csv_response(recent_history(10000), "lich-su-cham-soc.csv")
    if report_type == "theo-thoi-gian":
        context = build_report_context(
            period_type=request.args.get("period_type", "month"),
            year=request.args.get("year"),
            month=request.args.get("month"),
            quarter=request.args.get("quarter"),
        )
        return history_csv_response(context["history"], "thong-ke-theo-thoi-gian.csv")
    return redirect(url_for("thong_ke_bao_cao"))


@app.route("/phan-tich", methods=["GET", "POST"])
@login_required
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
                error = f"KhĂ´ng tĂ¬m tháº¥y khĂ¡ch hĂ ng {code or ''} trong CSV."
        else:
            try:
                lstm = float(request.form.get("lstm", ""))
                xgboost = float(request.form.get("xgboost", ""))
            except ValueError:
                error = "Vui lĂ²ng nháº­p xĂ¡c suáº¥t há»£p lá»‡."
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
                    "MaHienThi": request.form.get("manual_code") or "KhĂ¡ch hĂ ng má»›i",
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
@login_required
def danh_gia_mo_hinh():
    metrics = compute_metrics(load_data())
    best = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        best[metric] = max(metrics, key=lambda model: metrics[model][metric])
    return render_template("danh_gia_mo_hinh.html", metrics=metrics, best=best)


@app.route("/khach-hang/<customer_code>")
@login_required
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
        email_config_ready=is_email_config_ready(),
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


try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "Vui lĂ²ng nháº­p cĂ¢u há»i cá»§a báº¡n."})

    client = get_gemini_client()
    if client is None:
        fallback = answer_crm_question(question)
        return jsonify({
            "answer": fallback or "Trá»£ lĂ½ AI chÆ°a Ä‘Æ°á»£c cáº¥u hĂ¬nh GEMINI_API_KEY trĂªn mĂ´i trÆ°á»ng cháº¡y."
        })

    df = load_data()
    file_info_text = "ChÆ°a cĂ³ thĂ´ng tin file táº£i lĂªn."
    info_path = BASE_DIR / "data" / "file_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            file_info_text = (
                f"TĂªn file: {info.get('filename', 'KhĂ´ng rĂµ')}; "
                f"Tá»•ng khĂ¡ch hĂ ng: {info.get('total_customers', 0)}; "
                f"KhĂ¡ch nguy cÆ¡ rá»i bá»: {info.get('churn_customers', 0)}; "
                f"Tá»· lá»‡ churn: {info.get('churn_rate', 0)}%; "
                f"Tá»•ng doanh thu: {info.get('total_revenue', 0)}."
            )
        except (OSError, json.JSONDecodeError) as exc:
            app.logger.warning("KhĂ´ng Ä‘á»c Ä‘Æ°á»£c file_info.json: %s", exc)

    customer_info = ""
    code_match = re.search(r"\b[A-Za-z]{0,3}\d{1,8}\b", question)
    if code_match:
        customer = find_customer(code_match.group(0))
        if customer:
            customer_info = (
                f"MĂ£ khĂ¡ch hĂ ng {customer['MaHienThi']}: "
                f"LSTM {customer['XacSuat_LSTM'] * 100:.1f}%, "
                f"XGBoost {customer['XacSuat_XGBoost'] * 100:.1f}%, "
                f"Ensemble {customer['XacSuat_Ensemble'] * 100:.1f}%, "
                f"phĂ¢n khĂºc {customer['PhanKhuc']}, "
                f"Ä‘á» xuáº¥t {customer['ChamSoc']}."
            )

    segment_counts = df["PhanKhuc"].value_counts().to_dict() if not df.empty else {}
    system_instruction = (
        "Báº¡n lĂ  trá»£ lĂ½ CRM thĂ´ng minh. Tráº£ lá»i ngáº¯n gá»n, chĂ­nh xĂ¡c báº±ng tiáº¿ng Viá»‡t. "
        "Chá»‰ dĂ¹ng dá»¯ liá»‡u Ä‘Æ°á»£c cung cáº¥p, khĂ´ng bá»‹a sá»‘ liá»‡u. "
        f"ThĂ´ng tin file: {file_info_text}. "
        f"Sá»‘ lÆ°á»£ng phĂ¢n khĂºc: {segment_counts}. "
        f"ThĂ´ng tin khĂ¡ch hĂ ng liĂªn quan: {customer_info or 'KhĂ´ng cĂ³'}."
    )
    try:
        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            contents=question,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
        return jsonify({"answer": response.text or "Trá»£ lĂ½ AI chÆ°a cĂ³ pháº£n há»“i."})
    except Exception as exc:
        app.logger.error("Lá»—i Gemini API: %s", exc)
        fallback = answer_crm_question(question)
        return jsonify({
            "answer": fallback or f"KhĂ´ng thá»ƒ káº¿t ná»‘i Gemini API. Chi tiáº¿t: {exc}"
        })


@app.route("/phan-tich-tai-lieu", methods=["GET", "POST"])
@login_required
def phan_tich_tai_lieu():
    error = None
    result = None
    if request.method == "POST":
        uploaded_file = request.files.get("document_file")
        if not uploaded_file or not uploaded_file.filename:
            error = "Vui lĂ²ng chá»n file CSV hoáº·c Excel Ä‘á»ƒ táº£i lĂªn."
        else:
            try:
                filename = uploaded_file.filename
                if filename.lower().endswith(".csv"):
                    source_df = pd.read_csv(uploaded_file)
                elif filename.lower().endswith((".xlsx", ".xls")):
                    source_df = pd.read_excel(uploaded_file)
                else:
                    raise ValueError("Äá»‹nh dáº¡ng file khĂ´ng há»— trá»£. Vui lĂ²ng chá»n .csv, .xlsx hoáº·c .xls.")

                column_mapping = {
                    "Customer ID": "CustomerID",
                    "customer_id": "CustomerID",
                    "CustomerId": "CustomerID",
                    "Order ID": "InvoiceNo",
                    "order_id": "InvoiceNo",
                    "Unit Price": "UnitPrice",
                    "unit_price": "UnitPrice",
                    "Order Date": "InvoiceDate",
                    "order_date": "InvoiceDate",
                    "Quantity Sold": "Quantity",
                    "quantity": "Quantity",
                }
                source_df = source_df.rename(columns=column_mapping)
                required_upload_cols = {"CustomerID", "UnitPrice", "InvoiceDate", "Quantity"}
                missing = sorted(required_upload_cols - set(source_df.columns))
                if missing:
                    raise ValueError("File thiáº¿u cá»™t báº¯t buá»™c: " + ", ".join(missing))

                source_df["UnitPrice"] = pd.to_numeric(source_df["UnitPrice"], errors="coerce")
                source_df["Quantity"] = pd.to_numeric(source_df["Quantity"], errors="coerce")
                source_df["InvoiceDate"] = pd.to_datetime(source_df["InvoiceDate"], errors="coerce")
                source_df = source_df.dropna(subset=["CustomerID", "UnitPrice", "InvoiceDate", "Quantity"])
                source_df = source_df[(source_df["Quantity"] > 0) & (source_df["UnitPrice"] > 0)]
                if source_df.empty:
                    raise ValueError("File khĂ´ng cĂ²n dĂ²ng dá»¯ liá»‡u há»£p lá»‡ sau khi lĂ m sáº¡ch.")

                source_df["Total_Price"] = source_df["Quantity"] * source_df["UnitPrice"]
                snapshot_date = source_df["InvoiceDate"].max() + pd.Timedelta(days=1)
                invoice_agg = "nunique" if "InvoiceNo" in source_df.columns else "count"
                recency_df = source_df.groupby("CustomerID").agg(
                    Last_Purchase=("InvoiceDate", "max"),
                    Orders_Count=("InvoiceNo", invoice_agg) if "InvoiceNo" in source_df.columns else ("Quantity", "count"),
                    Total_Spend=("Total_Price", "sum"),
                ).reset_index()

                recency_df["Recency"] = (snapshot_date - recency_df["Last_Purchase"]).dt.days
                recency_df["ThucTe_RoiBo"] = (recency_df["Recency"] > 90).astype(int)
                recency_df["Churn"] = recency_df["ThucTe_RoiBo"]
                base_prob = (recency_df["Recency"] / 120.0).clip(0.05, 0.98)
                spend_factor = (1 - (recency_df["Total_Spend"].rank(pct=True) * 0.08)).clip(0.90, 1.00)
                recency_df["XacSuat_LSTM"] = (base_prob * spend_factor).clip(0.01, 0.99).round(4)
                recency_df["DuDoan_LSTM"] = (recency_df["XacSuat_LSTM"] >= 0.5).astype(int)
                recency_df["XacSuat_XGBoost"] = (base_prob * 0.96 + recency_df["Orders_Count"].rank(pct=True) * 0.04).clip(0.01, 0.99).round(4)
                recency_df["DuDoan_XGBoost"] = (recency_df["XacSuat_XGBoost"] >= 0.5).astype(int)
                recency_df["XacSuat_Ensemble"] = ((recency_df["XacSuat_LSTM"] + recency_df["XacSuat_XGBoost"]) / 2).round(4)
                recency_df["DuDoan_Ensemble"] = (recency_df["XacSuat_Ensemble"] >= 0.5).astype(int)
                recency_df["PhanKhuc"] = recency_df["XacSuat_Ensemble"].apply(get_segment)
                recency_df["SegmentClass"] = recency_df["PhanKhuc"].apply(badge_class)
                recency_df["ChamSoc"] = recency_df["PhanKhuc"].apply(get_care_action)
                recency_df["TrangThaiChamSoc"] = STATUS_PENDING

                DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                recency_df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

                total_customers = int(len(recency_df))
                churn_customers = int(recency_df["DuDoan_Ensemble"].sum())
                safe_customers = total_customers - churn_customers
                churn_rate = round(churn_customers / total_customers * 100, 2) if total_customers else 0
                total_revenue = float(recency_df["Total_Spend"].sum())
                file_info = {
                    "filename": filename,
                    "total_customers": total_customers,
                    "churn_customers": churn_customers,
                    "safe_customers": safe_customers,
                    "churn_rate": churn_rate,
                    "total_revenue": round(total_revenue, 2),
                    "updated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
                }
                (DATA_PATH.parent / "file_info.json").write_text(
                    json.dumps(file_info, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                result = {
                    **file_info,
                    "total_revenue": f"{total_revenue:,.2f}",
                    "summary_table": recency_df.sort_values("Recency", ascending=False).head(10).to_dict(orient="records"),
                }
            except Exception as exc:
                app.logger.error("Lá»—i phĂ¢n tĂ­ch tĂ i liá»‡u: %s", exc)
                error = f"Lá»—i khi xá»­ lĂ½ dá»¯ liá»‡u file: {exc}"
    return render_template("phan_tich_tai_lieu.html", error=error, result=result)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)

