#!/usr/bin/env python3
"""
Dashboard Eficiencia de Facturadores - TRT FOOD
Fuente: CSV exportado desde SAP HANA (OINV + INV1 + OUSR + OCRD)
Ejecutar: python dashboard_facturadores.py
Dependencias: pip install pandas numpy plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import itertools, re

# ──────────────────────────────────────────────────────────────
# 1. CARGA Y LIMPIEZA DE DATA REAL
# ──────────────────────────────────────────────────────────────

CSV_PATH = r"C:\Users\Lenovo\Documents\imput\FACTURADORES2026.csv"

df = pd.read_csv(CSV_PATH, sep=";", decimal=",", encoding="utf-8")
df.columns = df.columns.str.strip()

# Normalizar monto (coma decimal → punto)
df["Monto_Linea_Neto"] = pd.to_numeric(
    df["Monto_Linea_Neto"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
).fillna(0)

df["Fecha_Contabilizacion"] = pd.to_datetime(df["Fecha_Contabilizacion"])
df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0)

# Flag cancelado: Y y C = anulado
df["Es_Cancelado"] = df["CANCELADO"].isin(["Y", "C"])

# Columna Cadena: si no existe en el CSV, usar "Sin Cadena"
if "Cadena" not in df.columns:
    df["Cadena"] = "Sin Cadena"
else:
    df["Cadena"] = df["Cadena"].fillna("Sin Cadena").str.strip()

# Columnas opcionales (vienen del query fusionado)
for col in ["Canal", "Region", "Clasificacion"]:
    if col not in df.columns:
        df[col] = "N/D"
    else:
        df[col] = df[col].fillna("N/D").str.strip()

# ──────────────────────────────────────────────────────────────
# 2. REGLA DE SEGMENTO_OPERATIVO (basada en Lineas_Prom/factura por Cadena)
# ──────────────────────────────────────────────────────────────
# Cortes definidos desde el análisis exploratorio real:
#   AA → Lineas_Prom >= 13  (alta complejidad por factura)
#   A  → Lineas_Prom 8–12.9 (complejidad media)
#   B  → Lineas_Prom < 8    (facturas cortas, alta frecuencia)

lineas_por_cadena = (
    df[~df["Es_Cancelado"]]
    .groupby(["Numero_Factura", "Cadena"])
    .size()
    .reset_index(name="Lineas")
    .groupby("Cadena")["Lineas"]
    .mean()
    .rename("Lineas_Prom_Cadena")
)

def asignar_segmento(lineas_prom):
    if lineas_prom >= 13: return "AA"
    if lineas_prom >= 8:  return "A"
    return "B"

mapa_segmento = lineas_por_cadena.apply(asignar_segmento).to_dict()
df["Segmento_Operativo"] = df["Cadena"].map(mapa_segmento).fillna("B")

# ──────────────────────────────────────────────────────────────
# 3. DÍAS HÁBILES DEL PERÍODO (L–V, excluye Sáb/Dom)
# ──────────────────────────────────────────────────────────────

DIAS_HABILES_SET = set(
    d for d in pd.date_range(df["Fecha_Contabilizacion"].min(),
                             df["Fecha_Contabilizacion"].max())
    if d.weekday() < 5
)
DIAS_HABILES_N = len(DIAS_HABILES_SET)

FECHA_INICIO = df["Fecha_Contabilizacion"].min().strftime("%d/%m/%Y")
FECHA_FIN    = df["Fecha_Contabilizacion"].max().strftime("%d/%m/%Y")

# Solo registros hábiles y no cancelados para métricas de productividad
df_activos = df[
    (~df["Es_Cancelado"]) &
    (df["Fecha_Contabilizacion"].isin(DIAS_HABILES_SET))
].copy()

# ──────────────────────────────────────────────────────────────
# 4. PALETA DE COLORES DINÁMICA POR FACTURADOR
# ──────────────────────────────────────────────────────────────

PALETA = [
    "#1E3A5F","#00A99D","#7C3AED","#F4A261","#E63946",
    "#2563EB","#16A34A","#D97706","#9333EA","#0891B2",
    "#DC2626","#65A30D","#7C3AED","#C2410C"
]

facturadores_lista = sorted(df["Facturador"].unique())

def iniciales(nombre):
    partes = nombre.strip().split()
    if len(partes) >= 2:
        return (partes[0][0] + partes[1][0]).upper()
    return nombre[:2].upper()

COLOR_MAP  = {f: PALETA[i % len(PALETA)] for i, f in enumerate(facturadores_lista)}
AVATAR_MAP = {f: iniciales(f) for f in facturadores_lista}

# ──────────────────────────────────────────────────────────────
# 5. MÉTRICAS AGREGADAS POR FACTURADOR
# ──────────────────────────────────────────────────────────────

resumen = df_activos.groupby("Facturador").agg(
    Total_Lineas   = ("SKU",            "count"),
    Total_Facturas = ("Numero_Factura", "nunique"),
    Monto_Total    = ("Monto_Linea_Neto","sum"),
).reset_index()

resumen["Lineas_por_Dia"]     = (resumen["Total_Lineas"]   / DIAS_HABILES_N).round(1)
resumen["Facturas_por_Dia"]   = (resumen["Total_Facturas"] / DIAS_HABILES_N).round(1)
resumen["Lineas_por_Factura"] = (resumen["Total_Lineas"]   / resumen["Total_Facturas"]).round(1)

# Tasa cancelados (Y+C / total líneas del facturador)
tasa_cancel = df.groupby("Facturador")["Es_Cancelado"].mean().mul(100).round(2).reset_index()
tasa_cancel.columns = ["Facturador", "Tasa_Cancelados_Pct"]
resumen = resumen.merge(tasa_cancel, on="Facturador", how="left")

# Mix Segmento_Operativo (% líneas por AA/A/B)
mix_seg = (
    df_activos.groupby(["Facturador", "Segmento_Operativo"])
    .size().reset_index(name="N_Lineas")
)
total_f = mix_seg.groupby("Facturador")["N_Lineas"].transform("sum")
mix_seg["Pct"] = (mix_seg["N_Lineas"] / total_f * 100).round(1)
mix_pivot = mix_seg.pivot(index="Facturador", columns="Segmento_Operativo", values="Pct").fillna(0).reset_index()
for col in ["AA", "A", "B"]:
    if col not in mix_pivot.columns:
        mix_pivot[col] = 0.0

resumen = resumen.merge(mix_pivot, on="Facturador", how="left")
resumen = resumen.sort_values("Lineas_por_Dia", ascending=False).reset_index(drop=True)
resumen["Seg_Predominante"] = resumen[["AA","A","B"]].idxmax(axis=1)

# Tendencia diaria (solo días hábiles)
tendencia = (
    df_activos.groupby(["Facturador", "Fecha_Contabilizacion"])
    .size().reset_index(name="Lineas")
    .sort_values("Fecha_Contabilizacion")
)

# ──────────────────────────────────────────────────────────────
# 5b. CATEGORIZACIÓN CONCENTRACIÓN WAL-MART
# ──────────────────────────────────────────────────────────────
# Regla: % de líneas activas del facturador que son Wal-Mart
#   Especialista WM  >= 70%
#   Mixto WM          30% - 69%
#   Sin / Mínimo WM  < 30%

wm_mix = (
    df_activos.assign(Es_WM=df_activos["Cadena"].str.upper().str.contains("WAL-MART|WALMART", na=False))
    .groupby("Facturador")["Es_WM"]
    .mean()
    .mul(100)
    .round(1)
    .reset_index()
)
wm_mix.columns = ["Facturador", "Pct_WalMart"]

def cat_wm(pct):
    if pct >= 70: return ("Esp. WM",  "#1D4ED8", "🔵")
    if pct >= 30: return ("Mix WM",   "#D97706", "🟡")
    return              ("Sin WM",   "#64748B", "⚪")

wm_mix["Cat_WM_Label"], wm_mix["Cat_WM_Color"], wm_mix["Cat_WM_Icon"] = zip(
    *wm_mix["Pct_WalMart"].apply(cat_wm)
)
resumen = resumen.merge(wm_mix, on="Facturador", how="left")
resumen["Pct_WalMart"]  = resumen["Pct_WalMart"].fillna(0)
resumen["Cat_WM_Label"] = resumen["Cat_WM_Label"].fillna("Sin WM")
resumen["Cat_WM_Color"] = resumen["Cat_WM_Color"].fillna("#64748B")
resumen["Cat_WM_Icon"]  = resumen["Cat_WM_Icon"].fillna("⚪")

# ──────────────────────────────────────────────────────────────
# 6. KPIs GLOBALES
# ──────────────────────────────────────────────────────────────

# Excluir facturadores con < 3 días activos del benchmark (evita distorsión)
dias_activos = df_activos.groupby("Facturador")["Fecha_Contabilizacion"].nunique()
facturadores_activos = dias_activos[dias_activos >= 3].index
resumen_bench = resumen[resumen["Facturador"].isin(facturadores_activos)]

prom_equipo_lpd  = resumen_bench["Lineas_por_Dia"].mean().round(1)
prom_equipo_fpd  = resumen_bench["Facturas_por_Dia"].mean().round(1)
top_row          = resumen_bench.iloc[0]
top_facturador   = top_row["Facturador"]
top_val          = top_row["Lineas_por_Dia"]
top_seg          = top_row["Seg_Predominante"]

tasa_cancel_global = df["Es_Cancelado"].mean() * 100

UMBRAL_ALTO    = prom_equipo_lpd * 1.15
UMBRAL_BAJO    = prom_equipo_lpd * 0.85
UMBRAL_CRITICO = prom_equipo_lpd * 0.70

def semaforo(val):
    if val >= UMBRAL_ALTO:    return "#2ECC71"
    if val >= UMBRAL_BAJO:    return "#3498DB"
    if val >= UMBRAL_CRITICO: return "#F4A261"
    return "#E63946"

def semaforo_bg(val):
    if val >= UMBRAL_ALTO:    return "#F0FDF4"
    if val >= UMBRAL_BAJO:    return "#EFF6FF"
    if val >= UMBRAL_CRITICO: return "#FFF8EE"
    return "#FEF2F2"

resumen["Color_Bar"]   = resumen["Lineas_por_Dia"].apply(semaforo)
resumen["Color_Point"] = resumen["Facturador"].map(COLOR_MAP)
cancel_color = "#E63946" if tasa_cancel_global > 3 else "#F4A261" if tasa_cancel_global > 1 else "#2ECC71"

# ──────────────────────────────────────────────────────────────
# 7. GRÁFICOS PLOTLY
# ──────────────────────────────────────────────────────────────

LAYOUT_BASE = dict(
    plot_bgcolor="#FAFBFC", paper_bgcolor="white",
    font=dict(family="DM Sans, Segoe UI, sans-serif", color="#334155"),
    hoverlabel=dict(bgcolor="#1E3A5F", bordercolor="#00A99D", font=dict(color="white", size=12)),
)

# G1 — Ranking barras horizontales
fig_g1 = go.Figure()
fig_g1.add_vline(
    x=prom_equipo_lpd, line_dash="dash", line_color="#F4A261", line_width=2,
    annotation_text=f"Prom. {prom_equipo_lpd:.1f}",
    annotation_position="top right",
    annotation_font=dict(color="#F4A261", size=11)
)
for _, row in resumen.sort_values("Lineas_por_Dia").iterrows():
    fig_g1.add_trace(go.Bar(
        y=[row["Facturador"]],
        x=[row["Lineas_por_Dia"]],
        orientation="h",
        marker_color=row["Color_Bar"],
        text=f" {row['Lineas_por_Dia']:.1f}",
        textposition="outside",
        textfont=dict(size=11, color="#1E3A5F"),
        showlegend=False,
        hovertemplate=(
            f"<b>{row['Facturador']}</b><br>"
            f"Líneas/día: {row['Lineas_por_Dia']:.1f}<br>"
            f"Facturas/día: {row['Facturas_por_Dia']:.1f}<br>"
            f"Líneas/factura: {row['Lineas_por_Factura']:.1f}<br>"
            f"Seg. predominante: {row['Seg_Predominante']}<br>"
            f"Cancelados: {row['Tasa_Cancelados_Pct']:.1f}%<extra></extra>"
        )
    ))
fig_g1.update_layout(
    **LAYOUT_BASE,
    margin=dict(l=0, r=60, t=40, b=10),
    title=dict(text="<b>G1 · Ranking Líneas / Día Hábil</b>", font=dict(size=13, color="#1E3A5F"), x=0),
    xaxis=dict(showgrid=True, gridcolor="#EEF1F5", title="Líneas/día hábil"),
    yaxis=dict(showgrid=False),
    height=max(280, len(resumen) * 36 + 60),
)

# G2 — Stacked 100% Segmento_Operativo
fig_g2 = go.Figure()
colores_seg = {"AA": "#1E3A5F", "A": "#00A99D", "B": "#B0BEC5"}
for seg in ["AA", "A", "B"]:
    fig_g2.add_trace(go.Bar(
        name=f"Seg. {seg}",
        x=resumen["Facturador"],
        y=resumen[seg],
        marker_color=colores_seg[seg],
        hovertemplate=f"<b>%{{x}}</b><br>Seg. {seg}: %{{y:.1f}}%<extra></extra>",
    ))
layout_g2 = {**LAYOUT_BASE, "margin": dict(l=10, r=10, t=40, b=120)}
fig_g2.update_layout(
    **layout_g2,
    barmode="stack",
    title=dict(text="<b>G2 · Mix Segmento Operativo por Facturador</b>", font=dict(size=13, color="#1E3A5F"), x=0),
    legend=dict(orientation="h", y=-0.38, x=0, font=dict(size=11)),
    yaxis=dict(title="% líneas", ticksuffix="%", range=[0, 105]),
    xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
    height=360,
)

# G3 — Tendencia diaria con filtro JS (Top 5 por defecto)
# El orden de trazas debe coincidir con G3_TRACE_ORDER para el JS
top5_names = list(resumen.head(5)["Facturador"])
g3_trace_order = facturadores_lista  # orden alfabético = orden de trazas en Plotly

fig_g3 = go.Figure()
for nombre in facturadores_lista:
    sub = tendencia[tendencia["Facturador"] == nombre]
    if sub.empty:
        # traza vacía para mantener índice consistente
        fig_g3.add_trace(go.Scatter(
            x=[], y=[], mode="lines+markers", name=nombre,
            line=dict(color=COLOR_MAP[nombre], width=2),
            marker=dict(size=4), visible=False,
            hovertemplate="%{x|%d %b}<br>Líneas: %{y}<extra>" + nombre + "</extra>"
        ))
        continue
    visible = True if nombre in top5_names else False
    fig_g3.add_trace(go.Scatter(
        x=sub["Fecha_Contabilizacion"], y=sub["Lineas"],
        mode="lines+markers", name=nombre,
        line=dict(color=COLOR_MAP[nombre], width=2),
        marker=dict(size=4),
        visible=visible,
        hovertemplate="%{x|%d %b}<br>Líneas: %{y}<extra>" + nombre + "</extra>"
    ))
fig_g3.add_hline(
    y=prom_equipo_lpd, line_dash="dot", line_color="#F4A261", line_width=1.5,
    annotation_text=f"Prom/día {prom_equipo_lpd:.1f}",
    annotation_position="top left",
    annotation_font=dict(color="#F4A261", size=10)
)
fig_g3.update_layout(
    **LAYOUT_BASE,
    margin=dict(l=0, r=10, t=40, b=20),
    title=dict(text="<b>G3 · Tendencia Diaria de Líneas (L–V)</b>", font=dict(size=13, color="#1E3A5F"), x=0),
    showlegend=False,
    xaxis=dict(showgrid=True, gridcolor="#EEF1F5"),
    yaxis=dict(showgrid=True, gridcolor="#EEF1F5", title="Líneas procesadas"),
    height=310,
    hovermode="x unified",
)

# Datos para el JS del filtro
import json as _json
G3_FACTURADORES_JS = _json.dumps(facturadores_lista)
G3_TOP5_JS         = _json.dumps(top5_names)
G3_COLORS_JS       = _json.dumps([COLOR_MAP[n] for n in facturadores_lista])

# G4 — Scatter velocidad vs complejidad
fig_g4 = go.Figure()
fig_g4.add_vline(x=resumen["Lineas_por_Factura"].mean(),
                  line_dash="dot", line_color="#CBD5E1", line_width=1)
fig_g4.add_hline(y=resumen["Facturas_por_Dia"].mean(),
                  line_dash="dot", line_color="#CBD5E1", line_width=1)
for _, row in resumen.iterrows():
    size = max(14, min(40, row["Monto_Total"] / (resumen["Monto_Total"].max() / 35)))
    fig_g4.add_trace(go.Scatter(
        x=[row["Lineas_por_Factura"]],
        y=[row["Facturas_por_Dia"]],
        mode="markers+text",
        marker=dict(color=row["Color_Point"], size=size,
                    opacity=0.85, line=dict(color="white", width=1.5)),
        text=[AVATAR_MAP.get(row["Facturador"], "??")],
        textposition="middle center",
        textfont=dict(color="white", size=9, family="Arial Black"),
        showlegend=False,
        hovertemplate=(
            f"<b>{row['Facturador']}</b><br>"
            f"Líneas/factura: {row['Lineas_por_Factura']:.1f}<br>"
            f"Facturas/día: {row['Facturas_por_Dia']:.1f}<br>"
            f"Monto total: ${row['Monto_Total']:,.0f}<br>"
            f"Seg. predominante: {row['Seg_Predominante']}<extra></extra>"
        )
    ))
fig_g4.update_layout(
    **LAYOUT_BASE,
    margin=dict(l=0, r=10, t=40, b=10),
    title=dict(text="<b>G4 · Velocidad vs Complejidad</b>", font=dict(size=13, color="#1E3A5F"), x=0),
    xaxis=dict(title="Líneas por Factura (Complejidad →)", showgrid=True, gridcolor="#EEF1F5"),
    yaxis=dict(title="Facturas / Día Hábil (Velocidad →)", showgrid=True, gridcolor="#EEF1F5"),
    height=310,
    annotations=[
        dict(x=0.98, y=0.98, xref="paper", yref="paper",
             text="⭐ Alta complejidad + velocidad",
             showarrow=False, font=dict(size=9, color="#94a3b8"), xanchor="right"),
        dict(x=0.02, y=0.02, xref="paper", yref="paper",
             text="⚠ Revisar",
             showarrow=False, font=dict(size=9, color="#E63946"), xanchor="left"),
    ]
)

G1_JSON = fig_g1.to_json()
G2_JSON = fig_g2.to_json()
G3_JSON = fig_g3.to_json()
G4_JSON = fig_g4.to_json()

# ──────────────────────────────────────────────────────────────
# 8. TABLA RESUMEN HTML
# ──────────────────────────────────────────────────────────────

tabla_filas = ""
for _, row in resumen.iterrows():
    bg      = semaforo_bg(row["Lineas_por_Dia"])
    color   = COLOR_MAP.get(row["Facturador"], "#94A3B8")
    avatar  = AVATAR_MAP.get(row["Facturador"], "??")
    c_style = "color:#E63946;font-weight:700;" if row["Tasa_Cancelados_Pct"] > 3 else ""
    # días activos reales
    dias_fact = int(dias_activos.get(row["Facturador"], 0))
    tabla_filas += f"""
    <tr style="background:{bg};">
      <td style="padding:8px 12px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:28px;height:28px;border-radius:50%;background:{color};
               display:flex;align-items:center;justify-content:center;
               color:white;font-size:9px;font-weight:700;flex-shrink:0">{avatar}</div>
          <div style="display:flex;flex-direction:column;gap:2px;">
            <span style="font-weight:600;color:#1E3A5F">{row['Facturador']}</span>
            <span style="font-size:9px;font-weight:600;color:{row['Cat_WM_Color']}">{row['Cat_WM_Icon']} {row['Cat_WM_Label']} ({row['Pct_WalMart']:.0f}%)</span>
          </div>
        </div>
      </td>
      <td style="text-align:center;font-weight:700;font-size:15px;color:#1E3A5F">{row['Lineas_por_Dia']:.1f}</td>
      <td style="text-align:center">{row['Facturas_por_Dia']:.1f}</td>
      <td style="text-align:center">{row['Lineas_por_Factura']:.1f}</td>
      <td style="text-align:center">${row['Monto_Total']:,.0f}</td>
      <td style="text-align:center;{c_style}">{row['Tasa_Cancelados_Pct']:.1f}%</td>
      <td style="text-align:center">{dias_fact} días</td>
      <td style="text-align:center;">
        <div style="display:flex;gap:2px;justify-content:center">
          <span style="background:#1E3A5F;color:white;border-radius:3px;padding:1px 6px;font-size:10px">{row['AA']:.0f}%</span>
          <span style="background:#00A99D;color:white;border-radius:3px;padding:1px 6px;font-size:10px">{row['A']:.0f}%</span>
          <span style="background:#94a3b8;color:white;border-radius:3px;padding:1px 6px;font-size:10px">{row['B']:.0f}%</span>
        </div>
      </td>
    </tr>"""

# Cadenas por segmento (para sidebar)
cadenas_aa = ", ".join(sorted([c for c,s in mapa_segmento.items() if s=="AA"]))
cadenas_a  = ", ".join(sorted([c for c,s in mapa_segmento.items() if s=="A"]))
cadenas_b  = ", ".join(sorted([c for c,s in mapa_segmento.items() if s=="B"]))

# Sidebar facturadores
sidebar_facts = "".join(
    f'<div class="fact-row">'
    f'<div class="avatar" style="background:{COLOR_MAP[n]}">{AVATAR_MAP[n]}</div>'
    f'<span class="fact-name">{n}</span>'
    f'</div>'
    for n in facturadores_lista
)

# ──────────────────────────────────────────────────────────────
# 9. HTML FINAL
# ──────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dashboard Facturadores — TRT FOOD</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --navy:  #1E3A5F; --teal:  #00A99D; --red:   #E63946;
    --amber: #F4A261; --green: #2ECC71; --blue:  #3498DB;
    --bg:    #EEF1F6; --card:  #FFFFFF; --border:#E2E8F0;
    --text:  #334155; --muted: #94A3B8;
  }}
  body {{ font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }}
  /* HEADER */
  .header {{ background:var(--navy); color:white; padding:14px 28px;
    display:flex; align-items:center; justify-content:space-between;
    position:sticky; top:0; z-index:100; box-shadow:0 2px 12px rgba(30,58,95,.35); }}
  .header-left {{ display:flex; align-items:center; gap:14px; }}
  .logo-box {{ width:42px; height:42px; background:var(--teal); border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-family:'Space Mono',monospace; font-size:12px; font-weight:700; color:white; }}
  .header h1 {{ font-size:15px; font-weight:700; }}
  .header-sub {{ font-size:11px; color:#7EB8D4; margin-top:2px; }}
  .period-chip {{ background:rgba(0,169,157,.25); border:1px solid var(--teal);
    color:#7EE8E2; padding:5px 14px; border-radius:20px; font-size:12px; font-weight:600; }}
  /* LAYOUT */
  .main {{ display:flex; min-height:calc(100vh - 70px); }}
  .sidebar {{ width:205px; flex-shrink:0; background:var(--card);
    border-right:1px solid var(--border); padding:18px 12px;
    position:sticky; top:70px; height:calc(100vh - 70px); overflow-y:auto; }}
  .content {{ flex:1; padding:18px; display:flex; flex-direction:column; gap:14px; overflow:hidden; }}
  /* SIDEBAR */
  .sidebar-section {{ margin-bottom:18px; }}
  .sidebar-label {{ font-size:10px; font-weight:700; color:var(--muted);
    text-transform:uppercase; letter-spacing:1px; margin-bottom:7px; }}
  .filter-card {{ background:#F8FAFC; border:1px solid var(--border); border-radius:8px;
    padding:7px 10px; font-size:11px; color:var(--navy); font-weight:500; margin-bottom:5px; }}
  .seg-pills {{ display:flex; gap:3px; flex-wrap:wrap; }}
  .seg-pill {{ padding:4px 8px; border-radius:6px; font-size:10px; font-weight:700;
    cursor:pointer; border:none; transition:opacity .2s; }}
  .pill-all {{ background:#F4A261; color:white; }}
  .pill-aa  {{ background:#1E3A5F; color:white; }}
  .pill-a   {{ background:#00A99D; color:white; }}
  .pill-b   {{ background:#94A3B8; color:white; }}
  .day-checks {{ display:flex; flex-direction:column; gap:4px; }}
  .day-check {{ display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text);
    padding:2px 4px; border-radius:4px; }}
  .day-check input {{ accent-color:var(--teal); }}
  .day-check.disabled {{ opacity:.38; text-decoration:line-through; }}
  .fact-list {{ display:flex; flex-direction:column; gap:4px; max-height:220px; overflow-y:auto; }}
  .fact-row {{ display:flex; align-items:center; gap:7px; padding:3px 6px;
    border-radius:6px; cursor:pointer; transition:background .15s; }}
  .fact-row:hover {{ background:#F1F5F9; }}
  .avatar {{ width:22px; height:22px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:8px; font-weight:700; color:white; flex-shrink:0; }}
  .fact-name {{ font-size:11px; font-weight:500; color:var(--navy); }}
  .seg-legend {{ margin-top:10px; }}
  .seg-item {{ display:flex; align-items:flex-start; gap:6px; font-size:10px;
    color:var(--text); margin-bottom:6px; line-height:1.4; }}
  .seg-badge {{ flex-shrink:0; width:20px; height:20px; border-radius:4px;
    display:flex; align-items:center; justify-content:center;
    font-size:9px; font-weight:700; color:white; }}
  .tip-box {{ background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px;
    padding:8px 10px; font-size:10px; color:#3B82F6; line-height:1.6; margin-top:10px; }}
  /* KPI */
  .kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }}
  .kpi-card {{ background:var(--card); border-radius:12px; padding:14px 16px;
    border-left:4px solid var(--teal); box-shadow:0 1px 5px rgba(0,0,0,.06); position:relative; }}
  .kpi-label {{ font-size:10px; color:var(--muted); font-weight:500; line-height:1.4; margin-bottom:7px; }}
  .kpi-value {{ font-size:26px; font-weight:700; color:var(--navy); line-height:1;
    font-family:'Space Mono',monospace; }}
  .kpi-sub {{ font-size:10px; color:var(--muted); margin-top:4px; }}
  .kpi-delta {{ font-size:11px; font-weight:600; margin-top:4px; }}
  .delta-up {{ color:#2ECC71; }} .delta-down {{ color:#E63946; }}
  .kpi-badge {{ position:absolute; top:10px; right:10px; font-size:9px; font-weight:700;
    padding:2px 7px; border-radius:10px; }}
  .badge-green {{ background:#DCFCE7; color:#16A34A; }}
  .badge-red   {{ background:#FEE2E2; color:#E63946; }}
  .badge-blue  {{ background:#DBEAFE; color:#2563EB; }}
  /* CHARTS */
  .chart-row {{ display:grid; gap:12px; }}
  .row-2-3 {{ grid-template-columns:1.4fr 1fr; }}
  .row-1-1 {{ grid-template-columns:1fr 1fr; }}
  .chart-card {{ background:var(--card); border-radius:12px; padding:16px;
    box-shadow:0 1px 5px rgba(0,0,0,.06); }}
  /* TABLE */
  .table-card {{ background:var(--card); border-radius:12px;
    box-shadow:0 1px 5px rgba(0,0,0,.06); overflow:hidden; }}
  .table-header {{ padding:12px 18px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; }}
  .table-title {{ font-size:13px; font-weight:700; color:var(--navy); }}
  .table-sub {{ font-size:11px; color:var(--muted); }}
  .data-table {{ width:100%; border-collapse:collapse; }}
  .data-table th {{ background:#F8FAFC; padding:8px 10px; font-size:10px; font-weight:700;
    color:var(--muted); text-transform:uppercase; letter-spacing:.6px;
    border-bottom:1px solid var(--border); text-align:center; }}
  .data-table th:first-child {{ text-align:left; }}
  .data-table td {{ padding:0; border-bottom:1px solid #F1F5F9; font-size:12px; }}
  .data-table tr:last-child td {{ border-bottom:none; }}
  /* LEGEND */
  .legend-row {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; }}
  .legend-item {{ display:flex; align-items:center; gap:5px; font-size:11px; color:var(--muted); }}
  .leg-dot {{ width:10px; height:10px; border-radius:50%; }}
  ::-webkit-scrollbar {{ width:4px; }} ::-webkit-scrollbar-thumb {{ background:#CBD5E1; border-radius:3px; }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo-box">TRT</div>
    <div>
      <h1>Eficiencia de Facturadores &mdash; TRT FOOD</h1>
      <div class="header-sub">SAP HANA · TRT_PROD · OINV + INV1 + OUSR + OCRD · Solo días hábiles L&ndash;V</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <div class="period-chip">{FECHA_INICIO} &ndash; {FECHA_FIN}</div>
    <div style="font-size:11px;color:#a8c4e0;">&#x1F4CA; {DIAS_HABILES_N} días hábiles · {len(df_activos):,} líneas activas</div>
  </div>
</div>

<div class="main">
  <div class="sidebar">

    <div class="sidebar-section">
      <div class="sidebar-label">&#9881; Período</div>
      <div class="filter-card">&#x1F4C5; {FECHA_INICIO} &ndash; {FECHA_FIN}</div>
      <div class="filter-card" style="margin-top:4px">&#x1F4CB; {DIAS_HABILES_N} días hábiles</div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">Segmento Operativo</div>
      <div class="seg-pills">
        <button class="seg-pill pill-all">Todos</button>
        <button class="seg-pill pill-aa">AA</button>
        <button class="seg-pill pill-a">A</button>
        <button class="seg-pill pill-b">B</button>
      </div>
      <div class="seg-legend" style="margin-top:8px">
        <div class="seg-item">
          <div class="seg-badge" style="background:#1E3A5F">AA</div>
          <span style="font-size:9px;color:#64748b">&ge;13 líneas/fact<br>{cadenas_aa}</span>
        </div>
        <div class="seg-item">
          <div class="seg-badge" style="background:#00A99D">A</div>
          <span style="font-size:9px;color:#64748b">8–12 líneas/fact<br>{cadenas_a}</span>
        </div>
        <div class="seg-item">
          <div class="seg-badge" style="background:#94A3B8">B</div>
          <span style="font-size:9px;color:#64748b">&lt;8 líneas/fact<br>{cadenas_b}</span>
        </div>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">Día semana</div>
      <div class="day-checks">
        <label class="day-check"><input type="checkbox" checked> Lunes</label>
        <label class="day-check"><input type="checkbox" checked> Martes</label>
        <label class="day-check"><input type="checkbox" checked> Miércoles</label>
        <label class="day-check"><input type="checkbox" checked> Jueves</label>
        <label class="day-check"><input type="checkbox" checked> Viernes</label>
        <div class="day-check disabled">&#x26D4; Sábado</div>
        <div class="day-check disabled">&#x26D4; Domingo</div>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">Facturador ({len(facturadores_lista)})</div>
      <div class="fact-list">{sidebar_facts}</div>
    </div>

    <div class="tip-box">
      <strong>&#x1F4A1;</strong> Segmento derivado de líneas promedio por factura por cadena.<br>
      Umbral semáforo: &plusmn;15% y &minus;30% del promedio del equipo.
    </div>
  </div>

  <div class="content">

    <div class="kpi-grid">
      <div class="kpi-card" style="border-left-color:var(--teal)">
        <div class="kpi-badge badge-blue">Equipo</div>
        <div class="kpi-label">Líneas / Día Hábil<br><small>Promedio equipo</small></div>
        <div class="kpi-value">{prom_equipo_lpd:.0f}</div>
        <div class="kpi-delta delta-up">&#9650; Benchmark del período</div>
      </div>
      <div class="kpi-card" style="border-left-color:var(--blue)">
        <div class="kpi-badge badge-blue">Equipo</div>
        <div class="kpi-label">Facturas / Día Hábil<br><small>Promedio equipo</small></div>
        <div class="kpi-value">{prom_equipo_fpd:.0f}</div>
        <div class="kpi-delta delta-up">&#9650; Base de comparación</div>
      </div>
      <div class="kpi-card" style="border-left-color:#16A34A">
        <div class="kpi-badge badge-green">&#x1F3C6; Top</div>
        <div class="kpi-label">Mejor Facturador<br><small>del período</small></div>
        <div class="kpi-value" style="font-size:15px;margin-top:4px">{top_facturador}</div>
        <div class="kpi-sub" style="color:var(--teal);font-weight:600">{top_val:.0f} líneas/día · Seg. {top_seg}</div>
      </div>
      <div class="kpi-card" style="border-left-color:{cancel_color}">
        <div class="kpi-badge" style="background:#FEF2F2;color:{cancel_color};font-weight:700">{"⚠" if tasa_cancel_global > 3 else "✓"} {tasa_cancel_global:.1f}%</div>
        <div class="kpi-label">Tasa Anulados (Y+C)<br><small>Umbral: 3%</small></div>
        <div class="kpi-value" style="color:{cancel_color}">{tasa_cancel_global:.1f}%</div>
        <div class="kpi-delta {'delta-down' if tasa_cancel_global > 3 else 'delta-up'}">{"▲ Por encima del umbral" if tasa_cancel_global > 3 else "✓ Dentro del umbral"}</div>
      </div>
      <div class="kpi-card" style="border-left-color:var(--muted)">
        <div class="kpi-label">Días Hábiles<br><small>en el período</small></div>
        <div class="kpi-value">{DIAS_HABILES_N}</div>
        <div class="kpi-sub">{FECHA_INICIO} &ndash; {FECHA_FIN}</div>
      </div>
    </div>

    <div class="chart-row row-2-3">
      <div class="chart-card">
        <div id="g1" style="width:100%"></div>
        <div class="legend-row" style="margin-top:8px">
          <div class="legend-item"><div class="leg-dot" style="background:#2ECC71"></div>&ge;+15% prom.</div>
          <div class="legend-item"><div class="leg-dot" style="background:#3498DB"></div>&plusmn;15% prom.</div>
          <div class="legend-item"><div class="leg-dot" style="background:#F4A261"></div>&lt;&minus;15%</div>
          <div class="legend-item"><div class="leg-dot" style="background:#E63946"></div>&lt;&minus;30% &#9888;</div>
        </div>
      </div>
      <div class="chart-card">
        <div id="g2" style="width:100%"></div>
        <div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:6px;
             padding:7px 9px;margin-top:8px;font-size:10px;color:#92400E;line-height:1.5">
          <strong>&#x1F4A1;</strong> Segmento derivado de líneas promedio reales por cadena.
          Facturadores con alto % AA procesan facturas más complejas — su benchmark de líneas/día es naturalmente menor.
        </div>
      </div>
    </div>

    <div class="chart-row row-1-1">
      <div class="chart-card">
        <!-- G3 HEADER + FILTRO VENDEDOR -->
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:8px;flex-wrap:wrap;">
          <div style="font-size:13px;font-weight:700;color:#1E3A5F;">G3 &middot; Tendencia Diaria de L&iacute;neas (L&ndash;V)</div>
          <div style="display:flex;gap:5px;align-items:center;flex-wrap:wrap;">
            <button onclick="g3SetTop5()" class="g3btn g3btn-primary" id="btn-top5">Top 5</button>
            <button onclick="g3SetAll()"  class="g3btn" id="btn-all">Todos</button>
            <button onclick="g3Clear()"   class="g3btn g3btn-danger" id="btn-none">Limpiar</button>
          </div>
        </div>
        <!-- CHECKBOXES VENDEDOR -->
        <div id="g3-filters" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;"></div>
        <div id="g3" style="width:100%"></div>
      </div>
      <div class="chart-card">
        <div id="g4" style="width:100%"></div>
        <div class="legend-row" style="margin-top:8px">
          <div class="legend-item"><div class="leg-dot" style="background:#1E3A5F"></div>Seg. AA</div>
          <div class="legend-item"><div class="leg-dot" style="background:#00A99D"></div>Seg. A</div>
          <div class="legend-item"><div class="leg-dot" style="background:#94A3B8"></div>Seg. B</div>
          <div style="font-size:10px;color:var(--muted);margin-left:auto">Tamaño &prop; Monto total $</div>
        </div>
      </div>
    </div>

    <div class="table-card">
      <div class="table-header">
        <div>
          <div class="table-title">Resumen por Facturador &mdash; Data real SAP HANA</div>
          <div class="table-sub">Verde &ge;+15% · Azul &plusmn;15% · &Aacute;mbar &lt;&minus;15% · Rojo &lt;&minus;30% del promedio equipo</div>
        </div>
        <div style="font-size:11px;color:var(--muted)">
          Benchmark: <strong style="color:var(--navy)">{prom_equipo_lpd:.1f} líneas/día</strong>
          &nbsp;|&nbsp; Excluye facturadores con &lt;3 días activos del promedio
        </div>
      </div>
      <table class="data-table">
        <thead>
          <tr>
            <th style="text-align:left">Facturador</th>
            <th>Líneas/Día</th>
            <th>Facturas/Día</th>
            <th>Líneas/Factura</th>
            <th>Monto Total</th>
            <th>% Anulados</th>
            <th>Días Activos</th>
            <th>Mix Seg. (AA / A / B)</th>
          </tr>
        </thead>
        <tbody>{tabla_filas}</tbody>
      </table>
    </div>

    <div style="text-align:center;font-size:11px;color:var(--muted);padding:6px 0">
      &#x1F4CA; Fuente: SAP HANA · TRT_PROD · {len(df):,} registros totales ·
      {len(df_activos):,} líneas activas (excl. Y+C y fines de semana) ·
      {len(facturadores_lista)} facturadores
    </div>

  </div>
</div>

<style>
  .g3btn {{
    font-size:10px; font-weight:600; padding:4px 10px; border-radius:6px;
    border:1px solid #E2E8F0; background:#F8FAFC; color:#475569;
    cursor:pointer; transition:all .15s; font-family:'DM Sans',sans-serif;
  }}
  .g3btn:hover {{ background:#E2E8F0; }}
  .g3btn-primary {{ background:#1E3A5F; color:white; border-color:#1E3A5F; }}
  .g3btn-primary:hover {{ background:#162d4a; }}
  .g3btn-danger {{ background:#FEF2F2; color:#E63946; border-color:#FECACA; }}
  .g3btn-danger:hover {{ background:#FEE2E2; }}
  .g3-check-label {{
    display:flex; align-items:center; gap:5px; font-size:11px; font-weight:500;
    color:#334155; padding:3px 8px; border-radius:6px; cursor:pointer;
    border:1px solid #E2E8F0; background:#F8FAFC; transition:all .15s;
    user-select:none;
  }}
  .g3-check-label:hover {{ background:#EFF6FF; }}
  .g3-check-label.active {{ border-color: var(--c); background: color-mix(in srgb, var(--c) 12%, white); }}
  .g3-dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
</style>
<script>
  const cfg = {{responsive:true, displayModeBar:false}};
  Plotly.newPlot('g1', {G1_JSON}, cfg);
  Plotly.newPlot('g2', {G2_JSON}, cfg);
  Plotly.newPlot('g3', {G3_JSON}, cfg);
  Plotly.newPlot('g4', {G4_JSON}, cfg);

  // ── G3 FILTRO VENDEDOR ──────────────────────────────────────
  const G3_FACTS  = {G3_FACTURADORES_JS};
  const G3_TOP5   = {G3_TOP5_JS};
  const G3_COLORS = {G3_COLORS_JS};

  // Estado actual de visibilidad (sincronizado con Plotly)
  let g3Visible = {{}};
  G3_FACTS.forEach((f, i) => {{
    g3Visible[f] = G3_TOP5.includes(f);
  }});

  // Renderizar checkboxes
  function buildG3Filters() {{
    const container = document.getElementById('g3-filters');
    container.innerHTML = '';
    G3_FACTS.forEach((nombre, idx) => {{
      const color = G3_COLORS[idx];
      const isActive = g3Visible[nombre];
      const label = document.createElement('label');
      label.className = 'g3-check-label' + (isActive ? ' active' : '');
      label.style.setProperty('--c', color);
      label.innerHTML = `
        <input type="checkbox" ${{isActive ? 'checked' : ''}} style="display:none"
               onchange="g3Toggle('${{nombre}}', ${{idx}}, this.checked, '${{color}}')">
        <div class="g3-dot" style="background:${{color}}"></div>
        <span>${{nombre}}</span>`;
      container.appendChild(label);
    }});
  }}

  function g3Toggle(nombre, idx, checked, color) {{
    g3Visible[nombre] = checked;
    Plotly.restyle('g3', {{visible: checked}}, [idx]);
    buildG3Filters();
  }}

  function g3SetTop5() {{
    G3_FACTS.forEach((f, i) => {{
      g3Visible[f] = G3_TOP5.includes(f);
    }});
    const visArr = G3_FACTS.map(f => g3Visible[f]);
    Plotly.restyle('g3', {{visible: visArr}}, G3_FACTS.map((_, i) => i));
    buildG3Filters();
  }}

  function g3SetAll() {{
    G3_FACTS.forEach(f => g3Visible[f] = true);
    const visArr = G3_FACTS.map(() => true);
    Plotly.restyle('g3', {{visible: visArr}}, G3_FACTS.map((_, i) => i));
    buildG3Filters();
  }}

  function g3Clear() {{
    G3_FACTS.forEach(f => g3Visible[f] = false);
    const visArr = G3_FACTS.map(() => false);
    Plotly.restyle('g3', {{visible: visArr}}, G3_FACTS.map((_, i) => i));
    buildG3Filters();
  }}

  // Init
  buildG3Filters();
</script>
</body>
</html>"""

OUTPUT = r"C:\Users\Lenovo\Documents\imput\dashboard_facturadores.html"
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Dashboard generado: {OUTPUT}")
print(f"   Registros totales   : {len(df):,}")
print(f"   Líneas activas      : {len(df_activos):,}")
print(f"   Facturadores        : {len(facturadores_lista)}")
print(f"   Días hábiles        : {DIAS_HABILES_N}")
print(f"   Período             : {FECHA_INICIO} → {FECHA_FIN}")
print(f"   Benchmark equipo    : {prom_equipo_lpd:.1f} líneas/día")
print(f"   Tasa cancelados     : {tasa_cancel_global:.2f}%")
print(f"\n   Segmento Operativo:")
for seg in ["AA","A","B"]:
    cads = [c for c,s in mapa_segmento.items() if s==seg]
    print(f"     {seg}: {', '.join(sorted(cads))}")
