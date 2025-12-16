from core.nlp import puntaje_ambiente

def calcular_riesgo(row, modelo_nlp):
    A = row["asistencia"]
    N = row["nota_promedio"]
    F = puntaje_ambiente(row["observaciones"], modelo_nlp)

    Rd = (
        0.25 * (1 - A / 100) +
        0.25 * max(0, 75 - N) / 100 +
        0.5  * (1 - F)
    )

    return round(Rd,3), round(F,3)
