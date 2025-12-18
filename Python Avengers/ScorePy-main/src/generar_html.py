import base64
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.analisis_datos import (
    analisis_promedio_por_metodo,
    analisis_distractores,
    analisis_promedio_por_rango_horas,
    analisis_motivacion_vs_promedio,
    analisis_recursos_estudio_popularidad,
    analisis_promedio_por_frecuencia_repaso,
    analisis_sentimientos_vs_promedio,
)


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _plot_bar(series_or_df, title, xlabel='', ylabel='Promedio', rotate=30):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    if hasattr(series_or_df, 'columns') and series_or_df.shape[1] >= 1:
        data = series_or_df.iloc[:, 0]
    else:
        data = series_or_df
    data.plot(kind='bar', color='#155dff', edgecolor='black', ax=ax)
    ax.set_title(title, fontsize=13, weight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for label in ax.get_xticklabels():
        label.set_rotation(rotate)
        label.set_ha('right')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_pie(series, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    colores = sns.color_palette('Set2', n_colors=len(series))
    series.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax, colors=colores, pctdistance=0.75, textprops={'fontsize':9})
    ax.set_ylabel('')
    ax.set_title(title, fontsize=13, weight='bold')
    fig.tight_layout()
    return fig


def _plot_box(df, x, y, title):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=df, x=x, y=y, palette='Set3', ax=ax)
    ax.set_title(title, fontsize=13, weight='bold')
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def _obs_from_df(df, entidad, metrica='promedio'):
    if df is None or len(df) == 0:
        return f"No hay datos suficientes para analizar {entidad.lower()}."
    try:
        col_prom = None
        for c in df.columns:
            if 'promedio' in c.lower():
                col_prom = c
                break
        if col_prom is None:
            return f"No se encontró columna de promedio en el análisis de {entidad}."
        orden = df.sort_values(col_prom, ascending=False)
        top = orden.iloc[0]
        bottom = orden.iloc[-1]
        return (
            f"El mayor valor en {entidad} es '{orden.index[0]}' con {top[col_prom]:.2f}. \n"
            f"Promedio total: {orden[col_prom].mean():.2f}. "
            f"El menor valor en {entidad} es '{orden.index[-1]}' con {bottom[col_prom]:.2f}. \n"
            f"Diferencia: {top[col_prom]-bottom[col_prom]:.2f}."
        )
    except Exception:
        return f"Observación automática no disponible para {entidad}."


def _val_to_str(val, decimals=2, as_int=False):
    """Format a value to string, replacing NaN/None with '0'."""
    try:
        if pd.isna(val):
            return "0"
        if as_int:
            return str(int(val))
        return f"{float(val):.{decimals}f}"
    except Exception:
        try:
            return str(val)
        except Exception:
            return "0"


def generar_reporte_html(df: pd.DataFrame, ruta_salida: str, incluir: list[str] | None = None, nombre_html: str | None = None):
    """Genera un HTML con gráficos y observaciones para los análisis disponibles."""
    catalogo = []


    a0 = analisis_promedio_por_rango_horas(df)
    if not a0.empty:
        fig0 = _plot_bar(a0[['promedio']], 'Relación Horas de Estudio y Promedio', xlabel='Rango')
        try:
            if 'promedio' in a0.columns:
                series_prom = a0['promedio']
            else:
                series_prom = a0.iloc[:, 0]
            rows = [(str(idx), _val_to_str(val, decimals=2, as_int=False)) for idx, val in series_prom.items()]
        except Exception:
            rows = []

        catalogo.append({
            'clave':'horas', 'titulo':'Relación Horas de Estudio y Promedio', 'fig': fig0,
            'obs': '',
            'detail_headers': ('Horas Estudiadas', 'Promedio'),
            'detail_rows': rows,
        })


    try:
        dist_metodos = df['metodo_estudio'].value_counts()
        if dist_metodos.shape[0] > 0:
            figm = _plot_pie(dist_metodos, 'Distribución de Métodos de Estudio')
            try:
                if 'escala_promedio' in df.columns:
                    prom_por_metodo = df.groupby('metodo_estudio')['escala_promedio'].mean()
                    rows = [(str(idx), _val_to_str(val, decimals=2, as_int=False)) for idx, val in prom_por_metodo.items()]
                    value_header = 'Promedio'
                else:
                    rows = [(str(idx), _val_to_str(val, as_int=True)) for idx, val in dist_metodos.items()]
                    value_header = 'Estudiantes'
            except Exception:
                rows = []
                value_header = 'Valor'

            catalogo.append({
                'clave':'metodos', 'titulo':'Distribución de Métodos de Estudio', 'fig': figm,
                'obs': '',
                'detail_headers': ('Metodo Educativo', value_header),
                'detail_rows': rows,
            })
    except Exception:
        pass


    a1 = analisis_promedio_por_metodo(df)
    if not a1.empty:
        try:
            rows = [(str(idx), _val_to_str(val, decimals=2, as_int=False)) for idx, val in a1['promedio'].items()]
        except Exception:
            rows = []
        catalogo.append({
            'clave':'promedio_metodo', 'titulo':'Promedio Académico por Método de Estudio',
            'fig': _plot_bar(a1[['promedio']], 'Promedio por Método', xlabel='Método'),
            'obs': '',
            'detail_headers': ('Metodo', 'Promedio'),
            'detail_rows': rows,
        })


    a2 = analisis_promedio_por_rango_horas(df)
    if not a2.empty:
        try:
            series_prom = a2['promedio'] if 'promedio' in a2.columns else a2.iloc[:,0]
            rows = [(str(idx), _val_to_str(val, decimals=2, as_int=False)) for idx, val in series_prom.items()]
        except Exception:
            rows = []
        catalogo.append({
            'clave':'rango_horas', 'titulo':'Promedio por Rango de Horas de Estudio',
            'fig': _plot_bar(a2[['promedio']], 'Rango de Horas vs Promedio', xlabel='Rango'),
            'obs': '',
            'detail_headers': ('Rango', 'Valor'),
            'detail_rows': rows,
        })


    a3 = analisis_distractores(df)
    if not a3.empty:
        try:
            series_prom = a3['promedio'] if 'promedio' in a3.columns else a3.iloc[:,0]
            rows = [(str(idx), _val_to_str(val, decimals=2, as_int=False)) for idx, val in series_prom.items()]
        except Exception:
            rows = []
        catalogo.append({
            'clave':'distractores', 'titulo':'Impacto de Distractores en el Promedio',
            'fig': _plot_bar(a3[['promedio']], 'Distractores vs Promedio', xlabel='Distractor'),
            'obs': '',
            'detail_headers': ('Distractor', 'Promedio'),
            'detail_rows': rows,
        })


    a4 = analisis_motivacion_vs_promedio(df)
    if not a4.empty:
        try:
            series_prom = a4['promedio'] if 'promedio' in a4.columns else a4.iloc[:,0]
            rows = [(str(idx), f"{float(val):.2f}") for idx, val in series_prom.items()]
        except Exception:
            rows = []
        catalogo.append({
            'clave':'motivacion', 'titulo':'Motivación vs Promedio',
            'fig': _plot_bar(a4[['promedio']], 'Motivación vs Promedio', xlabel='Motivación'),
            'obs': '',
            'detail_headers': ('Motivación', 'Promedio'),
            'detail_rows': rows,
        })


    a5 = analisis_recursos_estudio_popularidad(df)
    if not a5.empty:
        fig5 = _plot_bar(a5, 'Recursos de Estudio (Frecuencia)', xlabel='Recurso', ylabel='Estudiantes', rotate=25)
        top_recurso = a5['estudiantes'].idxmax()
        obs5 = f"" if not a5.empty else "Sin datos de recursos."
        try:
            lineas = []
            if 'estudiantes' in a5.columns:
                rows = [(str(idx), _val_to_str(val, as_int=True)) for idx, val in a5['estudiantes'].items()]
            else:
                serie = a5.iloc[:,0]
                rows = [(str(idx), _val_to_str(val, decimals=2, as_int=False)) for idx, val in serie.items()]
            detalle_rec = "Detalle por recurso:\n" + "\n".join([f"- {k}: {v}" for k, v in rows])
            obs5 = obs5 + "\n\n" + detalle_rec
        except Exception:
            pass
        catalogo.append({
            'clave':'recursos', 'titulo':'Frecuencia de Uso de Recursos de Estudio',
            'fig': fig5,
            'obs': obs5,
            'detail_headers': ('Recurso', 'Estudiantes'),
            'detail_rows': rows if 'rows' in locals() else [],
        })


    a6 = analisis_promedio_por_frecuencia_repaso(df)
    if not a6.empty:
        try:
            series_prom = a6['promedio'] if 'promedio' in a6.columns else a6.iloc[:,0]
            rows = [(str(idx), _val_to_str(val, decimals=2, as_int=False)) for idx, val in series_prom.items()]
        except Exception:
            rows = []
        catalogo.append({
            'clave':'repaso', 'titulo':'Frecuencia de Repaso vs Promedio',
            'fig': _plot_bar(a6[['promedio']], 'Repaso vs Promedio', xlabel='Frecuencia'),
            'obs': '',
            'detail_headers': ('Frecuencia', 'Promedio'),
            'detail_rows': rows,
        })


    a7 = analisis_sentimientos_vs_promedio(df)
    if not a7.empty:
        try:
            series_prom = a7['promedio'] if 'promedio' in a7.columns else a7.iloc[:,0]
            rows = [(str(idx), _val_to_str(val, decimals=2, as_int=False)) for idx, val in series_prom.items()]
        except Exception:
            rows = []
        catalogo.append({
            'clave':'sentimientos', 'titulo':'Sentimientos respecto al Estudio vs Promedio',
            'fig': _plot_bar(a7[['promedio']], 'Sentimientos vs Promedio', xlabel='Sentimiento'),
            'obs': '',
            'detail_headers': ('Sentimiento', 'Promedio'),
            'detail_rows': rows,
        })


    if incluir is not None:
        catalogo = [c for c in catalogo if c['clave'] in incluir]


    partes_html = [
        '<!DOCTYPE html>',
        '<html lang="es">',
        '<head>',
        '<meta charset="UTF-8" />',
        '<title>ScorePy - Analisis Educativo</title>',
        '<style>'
        'body{font-family:Arial,Helvetica,sans-serif;background:#111;color:#eee;margin:0;padding:20px;}'
        'h1{color:#4da3ff;text-align:center;margin-top:0;}'
        'h2{color:#fff;border-left:6px solid #155dff;padding-left:10px;margin-top:0;margin-bottom:14px;}'
        '.card{background:#1e1e1e;border:1px solid #333;padding:18px 22px 20px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,.4);margin:0 auto 35px;max-width:980px;}'
        'img{max-width:100%;height:auto;display:block;margin:6px auto 10px;border-radius:6px;border:1px solid #222;background:#000;}'
    '.obs{background:#162131;border-left:4px solid #155dff;padding:10px 14px;border-radius:6px;white-space:pre-wrap;line-height:1.38;font-size:14px;}'
    '.obs-header{font-weight:600;margin-bottom:8px;color:#dfeeff;}'
    '.details{width:100%;border-collapse:collapse;margin-top:6px;background:#0f1720;border-radius:6px;overflow:hidden;}'
    '.details th{background:#0b1220;padding:8px 10px;text-align:left;color:#9fbff6;font-size:13px;border-bottom:1px solid #1f2a35;}'
    '.details td{padding:8px 10px;border-bottom:1px solid #111;color:#cde0ff;font-size:13px;}'
    '.details tr:last-child td{border-bottom:0;}'
        '.meta{display:flex;gap:18px;flex-wrap:wrap;justify-content:center;margin:25px auto 30px;max-width:1100px;}'
        '.kpi{background:#1d2733;padding:12px 18px;border-radius:8px;font-size:14px;min-width:180px;text-align:center;border:1px solid #2c3b4a;}'
        'footer{margin-top:50px;font-size:12px;text-align:center;color:#888;}'
        '@page{size:A4 portrait;margin:14mm 14mm 16mm 14mm;}'
        '@media print{' 
            'body{background:#fff;color:#000;padding:0;}'
            'h1{color:#000;margin-bottom:10px;}'
            'h2{color:#000;border-left:6px solid #0b57d6;}'
            '.meta{page-break-after:always;break-after:page;}'
            '.card{background:#fff;color:#000;box-shadow:none;border:1px solid #555;page-break-inside:avoid;break-inside:avoid;page-break-after:always;break-after:page;}'
            '.card:last-of-type{page-break-after:auto;break-after:auto;}'
            'p.obs{background:#f2f6fb;color:#000;border-left:4px solid #0b57d6;}'
            'img{background:#fff;border:1px solid #666;}'
            'footer{page-break-before:always;}'
        '}'
        '</style>',
        '</head>',
        '<body>',
        '<h1>Reporte de Análisis de Estudio</h1>'
    ]


    try:
        partes_html.append('<div class="meta">')
        partes_html.append(f'<div class="kpi"><strong>Estudiantes</strong><br>{len(df)}</div>')
        partes_html.append(f'<div class="kpi"><strong>Promedio Global</strong><br>{df["escala_promedio"].mean():.2f}</div>')
        partes_html.append(f'<div class="kpi"><strong>Edad Promedio</strong><br>{df["edad_alumno"].mean():.1f}</div>')
        partes_html.append(f'<div class="kpi"><strong>Horas Promedio</strong><br>{df["horas_estudiadas"].mean():.1f}</div>')
        partes_html.append('</div>')
    except Exception:
        pass


    for bloque in catalogo:
        obs_html = ''
        headers = bloque.get('detail_headers')
        rows = bloque.get('detail_rows')
        if headers and rows:
            table_html = ['<table class="details">', '<thead><tr>']
            table_html.append(f"<th>{headers[0]}</th><th>{headers[1]}</th>")
            table_html.append('</tr></thead>')
            table_html.append('<tbody>')
            for k, v in rows:
                table_html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
            table_html.append('</tbody></table>')
            obs_html = ''.join(table_html)
        else:
            obs_text = str(bloque.get('obs', '') or '')
            if obs_text:
                obs_html = f"<div class='obs-header'>{obs_text}</div>"
        bloque['obs_html'] = obs_html

    for bloque in catalogo:
        b64 = _fig_to_base64(bloque['fig'])
        partes_html.append('<div class="card">')
        partes_html.append(f"<h2>{bloque['titulo']}</h2>")
        partes_html.append(f'<img src="data:image/png;base64,{b64}" alt="{bloque['clave']}" />')
        partes_html.append(f"<div class='obs'>{bloque.get('obs_html','')}</div>")
        partes_html.append('</div>')

    partes_html.append('<footer>ScorePy - Analisis Educativo</footer>')

    partes_html.append('</body></html>')


    if nombre_html:
        nombre_html = os.path.basename(nombre_html)
        if not nombre_html.lower().endswith('.html'):
            nombre_html += '.html'
        if os.path.isdir(ruta_salida):
            ruta_final = os.path.join(ruta_salida, nombre_html)
        else:
            ruta_final = os.path.join(os.path.dirname(ruta_salida), nombre_html)
    else:
        ruta_final = ruta_salida

    with open(ruta_final, 'w', encoding='utf-8') as f:
        f.write("\n".join(partes_html))

    print(f"Reporte HTML generado: {ruta_final}")
    try:
        if os.name == 'nt':
            os.startfile(ruta_final)
        else:
            import webbrowser
            webbrowser.open_new_tab('file://' + os.path.abspath(ruta_final))
    except Exception:
        pass


__all__ = ["generar_reporte_html"]
