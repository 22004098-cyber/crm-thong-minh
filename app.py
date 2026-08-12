import csv
import io
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

import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
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

APP_NAME = "CRM Thông Minh"
APP_FULL_NAME = (
    "Hệ thống CRM thông minh hỗ trợ dự đoán khách hàng rời bỏ "
    "và chăm sóc khách hàng"
)

SEGMENT_HIGH = "Nguy cơ cao"
SEGMENT_MEDIUM = "Cần quan tâm"
SEGMENT_SAFE = "An toàn"
STATUS_PENDING = "Chưa chăm sóc"
STATUS_DONE = "Đã gửi Email"
EMAIL_STATUS_DONE = "Đã gửi"
LEGACY_STATUS_DONE = "Đã chăm sóc"
STATUS_FAILED = "Gửi lỗi"
DONE_STATUSES = (STATUS_DONE, EMAIL_STATUS_DONE, LEGACY_STATUS_DONE)
ROLE_ADMIN = "ADMIN"
ROLE_EMPLOYEE = "NHAN_VIEN"
USER_ACTIVE = "active"
USER_LOCKED = "locked"
RESET_TOKEN_MINUTES = 30


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
                "Quản trị viên",
                "admin@crm.local",
                "admin",
                ROLE_ADMIN,
                "Admin12345",
            ),
            (
                "Nhân viên CRM",
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
            flash("Bạn không có quyền truy cập chức năng quản trị tài khoản.", "danger")
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
        return "Mật khẩu không được rỗng."
    if len(password) < 8:
        return "Mật khẩu cần có ít nhất 8 ký tự."
    if confirm is not None and password != confirm:
        return "Mật khẩu xác nhận không khớp."
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
        return "Ưu đãi giữ chân khách hàng - Voucher 20%"
    if segment == SEGMENT_MEDIUM:
        return "Ưu đãi cá nhân hóa dành riêng cho quý khách"
    return "Cảm ơn quý khách đã đồng hành cùng chúng tôi"


def default_email_content(customer):
    segment = customer["PhanKhuc"]
    if segment == SEGMENT_HIGH:
        body = (
            "Cảm ơn quý khách đã tin tưởng và sử dụng dịch vụ của chúng tôi. "
            "Để tri ân và tiếp tục đồng hành cùng quý khách, chúng tôi gửi tặng "
            "voucher ưu đãi 20% cho lần sử dụng tiếp theo. Rất mong quý khách "
            "tiếp tục trải nghiệm dịch vụ trong thời gian tới."
        )
    elif segment == SEGMENT_MEDIUM:
        body = (
            "Chúng tôi gửi đến quý khách một ưu đãi cá nhân hóa cùng các gợi ý "
            "sản phẩm/dịch vụ phù hợp với nhu cầu hiện tại. Hy vọng những đề xuất "
            "này giúp quý khách có trải nghiệm tốt hơn."
        )
    else:
        body = (
            "Cảm ơn quý khách đã luôn đồng hành cùng chúng tôi. Quý khách sẽ tiếp tục "
            "được tích điểm và nhận các hoạt động chăm sóc định kỳ từ hệ thống CRM."
        )
    return (
        f"Xin chào khách hàng {customer['MaHienThi']},\n\n"
        f"{body}\n\n"
        f"Đề xuất CRM: {customer['ChamSoc']}.\n\n"
        "Trân trọng,\nCRM Thông Minh"
    )


def is_valid_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""))


def is_email_config_ready():
    return bool(os.environ.get("GMAIL_SENDER_EMAIL") and os.environ.get("GMAIL_APP_PASSWORD"))


def log_email_config_error(message):
    app.logger.error("[EMAIL CONFIG ERROR] %s", message)


def log_email_config(sender_email, app_password):
    app.logger.info(
        "\n".join(
            [
                "[EMAIL CONFIG]",
                f"Sender configured: {'YES' if sender_email else 'NO'}",
                f"Password configured: {'YES' if app_password else 'NO'}",
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


def send_gmail_email(customer_code, recipient_email, subject, content):
    sender_email = os.environ.get("GMAIL_SENDER_EMAIL")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    sender_name = os.environ.get("GMAIL_SENDER_NAME", APP_NAME)
    log_email_config(sender_email, app_password)
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
        return False, None, "Cấu hình email chưa sẵn sàng. Thiếu GMAIL_SENDER_EMAIL."
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
        return False, None, "Cấu hình email chưa sẵn sàng. Thiếu GMAIL_APP_PASSWORD."
    if not is_valid_email(recipient_email):
        error_message = "Email người nhận không hợp lệ."
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
        error_message = "Gmail từ chối đăng nhập. Hãy kiểm tra GMAIL_SENDER_EMAIL và Google App Password."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiết: {exc}",
        )
        return False, None, error_message
    except smtplib.SMTPConnectError as exc:
        error_message = "Không kết nối được tới smtp.gmail.com:587."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiết: {exc}",
        )
        return False, None, error_message
    except smtplib.SMTPServerDisconnected as exc:
        error_message = "Kết nối SMTP bị ngắt trong quá trình gửi."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiết: {exc}",
        )
        return False, None, error_message
    except smtplib.SMTPException as exc:
        error_message = "Gmail SMTP trả về lỗi khi gửi email."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiết: {exc}",
        )
        return False, None, error_message
    except (TimeoutError, socket.timeout) as exc:
        error_message = "Kết nối Gmail SMTP bị timeout. Vui lòng thử lại sau."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiết: {exc}",
        )
        return False, None, error_message
    except OSError as exc:
        error_message = "Không thể kết nối Gmail SMTP từ môi trường hiện tại."
        log_email_error(
            customer_code,
            recipient_email,
            sender_email,
            exc.__class__.__name__,
            f"{error_message} Chi tiết: {exc}",
        )
        return False, None, error_message


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
        return "Bạn có thể hỏi về số khách nguy cơ cao, khách chưa chăm sóc, KH001, Accuracy hoặc F1-score."

    customer_match = re.search(r"kh\s*0*(\d+)", q, re.IGNORECASE)
    customer = None
    if customer_match:
        customer = find_customer(f"KH{int(customer_match.group(1)):03d}")

    if "nguy cơ cao" in q and ("bao nhiêu" in q or "số" in q):
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
        "có bao nhiêu khách chưa chăm sóc, KH001 thuộc phân khúc nào, "
        "LSTM/XGBoost/Ensemble của KH001, Accuracy các mô hình, F1-score cao nhất."
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
            error = "Tài khoản hoặc mật khẩu không đúng."
        elif user["status"] == USER_LOCKED:
            error = "Tài khoản đã bị khóa. Vui lòng liên hệ quản trị viên."
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
    flash("Bạn đã đăng xuất khỏi hệ thống.", "success")
    return redirect(url_for("dang_nhap"))


@app.route("/quen-mat-khau", methods=["GET", "POST"])
def quen_mat_khau():
    token_url = None
    message = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        user = get_db_row("SELECT * FROM users WHERE lower(email) = ?", (email,))
        message = "Nếu email tồn tại, hệ thống đã tạo liên kết đặt lại mật khẩu demo."
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
        error = "Liên kết đặt lại mật khẩu không hợp lệ hoặc đã được sử dụng."
        return render_template("dat_lai_mat_khau.html", error=error, token=token)
    expires_at = datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S")
    if expires_at < datetime.now():
        error = "Liên kết đặt lại mật khẩu đã hết hạn."
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
            flash("Mật khẩu đã được đặt lại. Vui lòng đăng nhập.", "success")
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
            flash("Họ tên và email không được rỗng.", "danger")
        elif password and validate_password(password, confirm):
            flash(validate_password(password, confirm), "danger")
        else:
            existing = get_db_row(
                "SELECT id FROM users WHERE lower(email) = ? AND id <> ?",
                (email, user["id"]),
            )
            if existing:
                flash("Email đã được sử dụng bởi tài khoản khác.", "danger")
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
                flash("Đã cập nhật tài khoản của tôi.", "success")
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
        error_message = "Thiếu email người nhận, tiêu đề hoặc nội dung."
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

    success, provider_message_id, error_message = send_gmail_email(
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
            flash("Họ tên, email và username không được rỗng.", "danger")
        elif error:
            flash(error, "danger")
        elif get_db_row("SELECT id FROM users WHERE lower(email)=? OR lower(username)=?", (email, username)):
            flash("Email hoặc username đã tồn tại.", "danger")
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
            flash("Đã thêm tài khoản nhân viên.", "success")
            return redirect(url_for("quan_ly_tai_khoan"))
    return render_template("form_tai_khoan.html", item=None)


@app.route("/quan-ly-tai-khoan/<int:user_id>/sua", methods=["GET", "POST"])
@admin_required
def sua_tai_khoan(user_id):
    item = get_db_row("SELECT * FROM users WHERE id = ?", (user_id,))
    if not item:
        flash("Không tìm thấy tài khoản.", "danger")
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
            flash("Họ tên, email và username không được rỗng.", "danger")
        elif error:
            flash(error, "danger")
        elif get_db_row(
            "SELECT id FROM users WHERE (lower(email)=? OR lower(username)=?) AND id <> ?",
            (email, username, user_id),
        ):
            flash("Email hoặc username đã được sử dụng.", "danger")
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
            flash("Đã cập nhật tài khoản.", "success")
            return redirect(url_for("quan_ly_tai_khoan"))
    return render_template("form_tai_khoan.html", item=item)


@app.route("/quan-ly-tai-khoan/<int:user_id>/khoa", methods=["POST"])
@admin_required
def khoa_tai_khoan(user_id):
    user = current_user()
    target = get_db_row("SELECT * FROM users WHERE id = ?", (user_id,))
    if not target:
        flash("Không tìm thấy tài khoản.", "danger")
    elif target["id"] == user["id"]:
        flash("Bạn không thể tự khóa tài khoản của mình.", "danger")
    elif target["role"] == ROLE_ADMIN:
        flash("Không khóa tài khoản Admin qua thao tác nhanh.", "danger")
    else:
        execute_db("UPDATE users SET status = ?, updated_at = ? WHERE id = ?", (USER_LOCKED, datetime.now().strftime("%d/%m/%Y %H:%M"), user_id))
        flash("Đã khóa tài khoản.", "success")
    return redirect(url_for("quan_ly_tai_khoan"))


@app.route("/quan-ly-tai-khoan/<int:user_id>/mo-khoa", methods=["POST"])
@admin_required
def mo_khoa_tai_khoan(user_id):
    execute_db("UPDATE users SET status = ?, updated_at = ? WHERE id = ?", (USER_ACTIVE, datetime.now().strftime("%d/%m/%Y %H:%M"), user_id))
    flash("Đã mở khóa tài khoản.", "success")
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


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
