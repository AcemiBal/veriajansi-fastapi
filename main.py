from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
import io
import os
import uuid
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt

app = FastAPI()

os.makedirs("static/charts", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

last_analysis_context = None

def init_db():
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS upload_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            company TEXT,
            email TEXT,
            phone TEXT,
            data_type TEXT,
            file_name TEXT,
            row_count INTEGER,
            col_count INTEGER,
            upload_time TEXT
        )"""
    )
    # eski tabloda phone kolonu yoksa eklemeye çalış
    try:
        c.execute("ALTER TABLE upload_logs ADD COLUMN phone TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()

def log_upload(name, company, email, phone, data_type, file_name, row_count, col_count):
    init_db()
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO upload_logs (name, company, email, phone, data_type, file_name, row_count, col_count, upload_time) VALUES (?,?,?,?,?,?,?,?,?)",
        (name, company, email, phone, data_type, file_name, row_count, col_count, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def resolve_theme_class(data_type: str) -> str:
    dt = (data_type or "").lower()
    if "enerji" in dt:
        return "theme-enerji"
    if "uretim" in dt or "üretim" in dt:
        return "theme-uretim"
    if "satis" in dt or "satış" in dt:
        return "theme-satis"
    if "depo" in dt or "stok" in dt:
        return "theme-depo"
    if "kalite" in dt:
        return "theme-kalite"
    return "theme-default"

def detect_column(df: pd.DataFrame, keywords):
    lowered = {col: col.lower() for col in df.columns}
    for col, low in lowered.items():
        for kw in keywords:
            if kw in low:
                return col
    return None

def create_charts(df: pd.DataFrame, numeric_cols, datetime_cols):
    chart_paths = []

    # 1) Line chart
    if numeric_cols:
        first_num = numeric_cols[0]
        chart_id = str(uuid.uuid4())
        path = f"static/charts/{chart_id}.png"
        plt.figure()
        try:
            df[first_num].plot(kind="line")
            plt.title(first_num)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            chart_paths.append("/static/charts/" + chart_id + ".png")
        except Exception:
            plt.close()

    # 2) Histogram
    if numeric_cols:
        first_num = numeric_cols[0]
        chart_id2 = str(uuid.uuid4())
        path2 = f"static/charts/{chart_id2}.png"
        plt.figure()
        try:
            df[first_num].hist(bins=20)
            plt.title(first_num + " - Histogram")
            plt.tight_layout()
            plt.savefig(path2)
            plt.close()
            chart_paths.append("/static/charts/" + chart_id2 + ".png")
        except Exception:
            plt.close()

    # 3) İlk 5 sayısal kolon ortalamaları
    if len(numeric_cols) >= 2:
        means = df[numeric_cols].mean(numeric_only=True).sort_values(ascending=False)[:5]
        chart_id3 = str(uuid.uuid4())
        path3 = f"static/charts/{chart_id3}.png"
        plt.figure(figsize=(6, 3))
        try:
            means.plot(kind="bar")
            plt.title("İlk 5 sayısal kolon ortalamaları")
            plt.tight_layout()
            plt.savefig(path3)
            plt.close()
            chart_paths.append("/static/charts/" + chart_id3 + ".png")
        except Exception:
            plt.close()

    # 4) Günlük toplam (zaman serisi)
    if datetime_cols and numeric_cols:
        dt_col = datetime_cols[0]
        num_col = numeric_cols[0]
        chart_id4 = str(uuid.uuid4())
        path4 = f"static/charts/{chart_id4}.png"
        try:
            ts_df = df.copy()
            ts_df[dt_col] = pd.to_datetime(ts_df[dt_col], errors="coerce")
            ts_df = ts_df.dropna(subset=[dt_col])
            ts_df = ts_df.set_index(dt_col).sort_index()
            daily = ts_df[num_col].resample("D").sum()
            if not daily.empty:
                plt.figure(figsize=(6, 3))
                daily.plot(kind="line")
                plt.title(f"Günlük toplam - {num_col}")
                plt.tight_layout()
                plt.savefig(path4)
                plt.close()
                chart_paths.append("/static/charts/" + chart_id4 + ".png")
        except Exception:
            try:
                plt.close()
            except Exception:
                pass

    # 5) Korelasyon ısı haritası
    try:
        num_df = df.select_dtypes(include=["int64", "float64"])
        if not num_df.empty and num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True)
            chart_id5 = str(uuid.uuid4())
            path5 = f"static/charts/{chart_id5}.png"
            plt.figure(figsize=(4, 3))
            plt.imshow(corr, aspect="auto")
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=6)
            plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
            plt.title("Korelasyon Isı Haritası", fontsize=8)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(path5)
            plt.close()
            chart_paths.append("/static/charts/" + chart_id5 + ".png")
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

    # 6) Eksik veri ısı haritası
    try:
        if df.shape[0] > 0 and df.shape[1] > 0:
            chart_id6 = str(uuid.uuid4())
            path6 = f"static/charts/{chart_id6}.png"
            plt.figure(figsize=(4, 3))
            plt.imshow(df.isna(), aspect="auto")
            plt.title("Eksik Veri Haritası", fontsize=8)
            plt.xlabel("Kolonlar", fontsize=6)
            plt.ylabel("Satırlar", fontsize=6)
            plt.tight_layout()
            plt.savefig(path6)
            plt.close()
            chart_paths.append("/static/charts/" + chart_id6 + ".png")
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

    return chart_paths

def analyze_dataframe(df: pd.DataFrame, data_type: str):
    row_count = len(df)
    col_count = len(df.columns)
    columns = df.columns.tolist()

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    numeric_cols = numeric_df.columns.tolist()

    summary_stats = {}
    for col in numeric_cols[:5]:
        summary_stats[col] = {
            "min": float(numeric_df[col].min()),
            "max": float(numeric_df[col].max()),
            "mean": float(numeric_df[col].mean()),
            "sum": float(numeric_df[col].sum()),
        }

    na_counts = df.isna().sum().to_dict()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    has_time_series = len(datetime_cols) > 0

    total_missing = int(df.isna().sum().sum())
    total_cells = int(row_count * col_count) if row_count and col_count else 0
    quality_score = 0.0
    if total_cells > 0:
        quality_score = float(100 * (1 - (total_missing / total_cells)))

    top_missing_col = max(na_counts, key=lambda k: na_counts[k]) if na_counts else None
    var_cols = {}
    if not numeric_df.empty:
        for col in numeric_cols:
            series = numeric_df[col].dropna()
            var_cols[col] = float(series.var()) if len(series) > 1 else 0.0
    top_var_col = max(var_cols, key=lambda k: var_cols[k]) if var_cols else None

    # Kategorik analiz
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    top_categories = {}
    for col in cat_cols[:3]:
        vc = df[col].value_counts().head(5)
        top_categories[col] = {str(idx): int(val) for idx, val in vc.items()}

    chart_paths = create_charts(df, numeric_cols, datetime_cols)
    sample_html = df.head(20).to_html(classes="table table-striped", border=0)

    # İnteraktif grafik için ilk sayısal kolondan örnek seri hazırla
    chart_data = None
    if numeric_cols:
        first_num = numeric_cols[0]
        series = df[first_num].dropna().head(100)
        labels = list(range(1, len(series) + 1))
        chart_data = {
            "label": first_num,
            "labels": labels,
            "values": [float(v) for v in series.values],
        }

    domain_insights = []
    data_type_lower = (data_type or "").lower()
    if "enerji" in data_type_lower:
        col_kwh = detect_column(df, ["kwh", "energy", "enerji", "kw "])
        if col_kwh and col_kwh in numeric_cols:
            total_kwh = float(df[col_kwh].sum())
            avg_kwh = float(df[col_kwh].mean())
            domain_insights.append(f"Toplam enerji tüketimi (yaklaşık): {total_kwh:,.2f} birim.")
            domain_insights.append(f"Ortalama tüketim: {avg_kwh:,.2f} birim.")
    elif "uretim" in data_type_lower or "üretim" in data_type_lower:
        col_qty = detect_column(df, ["adet", "miktar", "qty", "quantity", "output"])
        if col_qty and col_qty in numeric_cols:
            total_qty = float(df[col_qty].sum())
            domain_insights.append(f"Toplam üretim miktarı (yaklaşık): {total_qty:,.0f} birim.")
    elif "satis" in data_type_lower or "satış" in data_type_lower:
        col_amount = detect_column(df, ["tutar", "amount", "sale", "ciro", "revenue", "price"])
        if col_amount and col_amount in numeric_cols:
            total_sales = float(df[col_amount].sum())
            domain_insights.append(f"Toplam satış cirosu (yaklaşık): {total_sales:,.2f} birim.")

    if not domain_insights:
        domain_insights.append("Veriniz üzerinde genel istatistikler çıkarıldı. Detaylı analiz için özel modeller eklenebilir.")

    return {
        "row_count": row_count,
        "col_count": col_count,
        "columns": columns,
        "numeric_cols": numeric_cols,
        "summary_stats": summary_stats,
        "na_counts": na_counts,
        "datetime_cols": datetime_cols,
        "has_time_series": has_time_series,
        "sample_html": sample_html,
        "chart_paths": chart_paths,
        "chart_data": chart_data,
        "domain_insights": domain_insights,
        "total_missing": total_missing,
        "total_cells": total_cells,
        "quality_score": quality_score,
        "top_missing_col": top_missing_col,
        "top_var_col": top_var_col,
        "top_categories": top_categories,
    }
@app.get("/", response_class=HTMLResponse)
async def home():
    return """<h3>Veri Analiz Asistanı v10 (FastAPI)</h3>
    <p><a href='/upload'>Veri yükleme formu</a></p>
    <p><a href='/admin/uploads?key=veriadmin'>Admin log ekranı</a></p>"""

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/api/upload", response_class=HTMLResponse)
async def upload_file(
    request: Request,
    name: str = Form(...),
    company: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    data_type: str = Form(...),
    file: UploadFile = File(...),
):
    global last_analysis_context

    contents = await file.read()
    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        return HTMLResponse(f"<h3>Dosya okunamadı:</h3><pre>{e}</pre>", status_code=400)

    analysis = analyze_dataframe(df, data_type)
    log_upload(name, company, email, phone, data_type, file.filename, analysis["row_count"], analysis["col_count"])

    theme = resolve_theme_class(data_type)
    context = {
        "request": request,
        "name": name,
        "company": company,
        "email": email,
        "phone": phone,
        "data_type": data_type,
        "file_name": file.filename,
        "theme": theme,
        **analysis,
    }
    last_analysis_context = {k: v for k, v in context.items() if k != "request"}

    return templates.TemplateResponse("report.html", context)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    global last_analysis_context
    if not last_analysis_context:
        return HTMLResponse(
            "<h3>Henüz dashboard için bir veri yüklenmedi.</h3><p>Önce /upload üzerinden bir dosya gönderin.</p>",
            status_code=200,
        )
    theme = last_analysis_context.get("theme", resolve_theme_class(last_analysis_context.get("data_type", "")))
    context = {"request": request, "theme": theme, **last_analysis_context}
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/admin/uploads", response_class=HTMLResponse)
async def admin_uploads(request: Request):
    key = request.query_params.get("key")
    if key != "veriadmin":
        return HTMLResponse("<h3>Yetkisiz erişim</h3>", status_code=401)

    init_db()
    # filtre parametreleri
    name_f = request.query_params.get("name") or ""
    email_f = request.query_params.get("email") or ""
    company_f = request.query_params.get("company") or ""
    dtype_f = request.query_params.get("data_type") or ""
    date_from = request.query_params.get("date_from") or ""
    date_to = request.query_params.get("date_to") or ""

    conn = sqlite3.connect("data.db")
    base_q = "SELECT * FROM upload_logs WHERE 1=1"
    params = []
    if name_f:
        base_q += " AND name LIKE ?"
        params.append(f"%{name_f}%")
    if email_f:
        base_q += " AND email LIKE ?"
        params.append(f"%{email_f}%")
    if company_f:
        base_q += " AND company LIKE ?"
        params.append(f"%{company_f}%")
    if dtype_f:
        base_q += " AND data_type = ?"
        params.append(dtype_f)
    if date_from:
        base_q += " AND upload_time >= ?"
        params.append(date_from)
    if date_to:
        base_q += " AND upload_time <= ?"
        params.append(date_to)

    base_q += " ORDER BY id DESC"
    df_logs = pd.read_sql_query(base_q, conn, params=params)
    conn.close()

    logs = df_logs.to_dict(orient="records")
    filters = {
        "name": name_f,
        "email": email_f,
        "company": company_f,
        "data_type": dtype_f,
        "date_from": date_from,
        "date_to": date_to,
        "key": key,
    }
    return templates.TemplateResponse("admin_uploads.html", {"request": request, "logs": logs, "filters": filters})

@app.get("/admin/uploads.csv", response_class=PlainTextResponse)
async def admin_uploads_csv(request: Request):
    key = request.query_params.get("key")
    if key != "veriadmin":
        return PlainTextResponse("unauthorized", status_code=401)

    init_db()
    conn = sqlite3.connect("data.db")
    df_logs = pd.read_sql_query("SELECT * FROM upload_logs ORDER BY id DESC", conn)
    conn.close()
    csv_data = df_logs.to_csv(index=False)
    return PlainTextResponse(csv_data, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=upload_logs.csv"
    })
