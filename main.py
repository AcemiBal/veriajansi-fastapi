from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

app = FastAPI()

# Templates & Static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Yüklemeleri hafızada tutmak için basit liste
UPLOAD_HISTORY = []


# -----------------------------
# ANA SAYFA
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {"request": request}
    )


# -----------------------------
# VERİ YÜKLEME FORMU (GET)
# -----------------------------
@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse(
        "form.html",
        {"request": request}
    )


# -----------------------------
# VERİ YÜKLEME (POST)
# -----------------------------
@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    domain: str = Form(None),
    description: str = Form(None)
):

    # Dosya okuma
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
            data_type = "CSV"
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
            data_type = "Excel"
        else:
            return templates.TemplateResponse(
                "form.html",
                {"request": request, "error": "Sadece CSV veya XLSX yükleyebilirsiniz."}
            )
    except Exception as e:
        return templates.TemplateResponse(
            "form.html",
            {"request": request, "error": f"Dosya okunamadı: {str(e)}"}
        )

    # Temel analiz
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    total_cells = rows * cols
    total_missing = int(df.isna().sum().sum())

    # Basit kalite skoru
    if total_cells > 0:
        quality_score = round(100 - (total_missing / total_cells * 100), 1)
    else:
        quality_score = 0

    # Özet istatistik
    summary_stats = {}
    for col in numeric_cols:
        summary_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "sum": float(df[col].sum())
        }

    # Grafik kaydetme
    os.makedirs("static/charts", exist_ok=True)
    chart_paths = []

    if len(numeric_cols) > 0:
        for col in numeric_cols[:4]:  # en fazla 4 grafik
            plt.figure()
            df[col].plot(kind="line")
            chart_file = f"static/charts/{col}.png"
            plt.savefig(chart_file)
            plt.close()
            chart_paths.append("/" + chart_file)

    # Yükleme geçmişine ekle
    record = {
        "id": len(UPLOAD_HISTORY) + 1,
        "file_name": file.filename,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_type": data_type,
        "row_count": rows,
        "col_count": cols,
        "quality_score": quality_score,
        "status": "success",
        "detail_url": f"/dashboard?upload_id={len(UPLOAD_HISTORY) + 1}",
    }
    UPLOAD_HISTORY.append(record)

    # Domain içgörüleri (demo)
    domain_insights = []
    if domain == "energy":
        domain_insights.append("Enerji verisi tespit edildi. Tüketim trend analizi önerilir.")
    elif domain == "production":
        domain_insights.append("Üretim verisi. Proses döngü süreleri incelenebilir.")
    elif domain == "quality":
        domain_insights.append("Kalite verisi. Hata Pareto analizi uygulanabilir.")
    else:
        domain_insights.append("Genel veri analizi tamamlandı.")

    # Dashboard’a gönder
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "file_name": file.filename,
            "data_type": data_type,
            "row_count": rows,
            "col_count": cols,
            "numeric_cols": numeric_cols,
            "total_cells": total_cells,
            "total_missing": total_missing,
            "quality_score": quality_score,
            "summary_stats": summary_stats,
            "chart_paths": chart_paths,
            "domain_insights": domain_insights,
        },
    )


# -----------------------------
# YÜKLEME GEÇMİŞİ (ADMIN)
# -----------------------------
@app.get("/admin/uploads", response_class=HTMLResponse)
async def admin_uploads(request: Request):
    return templates.TemplateResponse(
        "admin_uploads.html",
        {"request": request, "uploads": UPLOAD_HISTORY}
    )


# -----------------------------
# DASHBOARD
# -----------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, upload_id: int):
    # Kayıt bul
    record = next((u for u in UPLOAD_HISTORY if u["id"] == upload_id), None)
    if not record:
        return {"error": "Kayıt bulunamadı."}

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "file_name": record["file_name"],
            "data_type": record["data_type"],
            "row_count": record["row_count"],
            "col_count": record["col_count"],
            "numeric_cols": [],
            "total_cells": record["row_count"] * record["col_count"],
            "total_missing": 0,
            "quality_score": record["quality_score"],
            "summary_stats": {},
            "chart_paths": [],
            "domain_insights": ["Geçmiş yükleme dashboard görüntüsü"],
        }
    )
