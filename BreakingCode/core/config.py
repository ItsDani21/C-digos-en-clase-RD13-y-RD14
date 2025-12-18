import re

PALABRAS_POS = [
    "participativo","atento","motivación","esfuerzo","responsable","colaborador",
    "proactivo","excelente","comprometido","constante","interés","aprendizaje",
    "participativa", "colaboradora", "responsable", "motivado", "proactivo", 
    "esfuerzo", "destacado", "excelente", "atento", "comprometido", "activo",
    "puntual", "asistencia", "constante", "iniciativa", "liderazgo", "respetuoso",
    "organizado", "autónomo", "perseverante", "curioso", "profundo", "reflexivo",
    "empático", "gratitud", "balance", "adaptable", "eficiente", "sobresaliente",
    "ejemplar", "dedicación", "entusiasta", "creativo", "analítico", "metódico",
    "disciplinado", "focalizado", "resiliente", "innovador", "colaborativo",
    "comunicativo", "propositivo", "visionario", "ordenado", "preciso", "detallista",
    "solidario", "paciente", "tolerante", "flexible", "autocrítico", "mejora",
    "evolución", "crecimiento", "superación", "exigente", "calidad", "excelencia",
    "potencial", "talento", "habilidad", "capacidad", "inteligente", "ágil",
    "rápido", "eficaz", "productivo", "constructivo", "positivo", "optimista",
    "alegre", "energético", "dinámico", "vital", "entregado", "compromiso",
    "vocación", "pasión", "interés", "dedicación", "constancia", "regularidad",
    "confiable", "confianza", "seguridad", "autoestima", "autonomía", "independiente",
    "maduro", "serio", "formal", "educado", "cortés", "amable", "gentil",
    "considerado", "atento", "detallista", "cuidadoso", "meticuloso", "perfeccionista",
    "exhaustivo", "completo", "integral", "holístico", "equilibrado", "armónico"
]

PALABRAS_NEG = [
    "desinterés", "dificultades", "timido", "timida","falta","problemas","conflicto","apatía","incumplimiento","excusas",
    "derrotista","respeto","inapropiado","minimiza","ausencias","bajo rendimiento",
    "desinterés", "problemas", "conflicto", "desmotivado", "bajo rendimiento", 
    "distrae", "falta de motivación", "ausencia", "tardanza", "celular", "irrespetuoso",
    "evasión", "copia", "plagio", "desafiante", "aislamiento", "fatiga", "somnolencia",
    "negativa", "frustración", "ira", "falta materiales", "incapacidad", "disruptivo",
    "descuido", "promesas incumplidas", "justificaciones", "resistencia", "dificultad",
    "concentración", "problemas personales", "agresivo", "pasivo-agresivo", "abandono",
    "ausencia objetivos", "priorización equivocada", "historial conflictivo", "evasivo",
    "deterioro", "contacto visual", "postura cerrada", "estrés", "apatía", "indiferencia",
    "negligencia", "irresponsable", "inconstante", "inestable", "volátil", "impulsivo",
    "agresividad", "violencia", "acoso", "bullying", "marginación", "exclusión",
    "discriminación", "prejuicio", "intolerante", "inflexible", "rígido", "terco",
    "obstinado", "negación", "evasión", "engaño", "mentira", "fraude", "trampa",
    "chantaje", "manipulación", "egoísmo", "individualismo", "narcisismo", "arrogancia",
    "soberbia", "prepotencia", "autoritario", "dominante", "controlador", "posesivo",
    "celoso", "envidioso", "rencoroso", "vengativo", "resentimiento", "amargura",
    "pesimismo", "derrotista", "fatalista", "catastrófico", "ansiedad", "depresión",
    "desánimo", "desaliento", "desesperanza", "abandono", "deserción", "desistimiento",
    "renuncia", "fracaso", "reprobación", "suspensión", "expulsión", "sanción",
    "amonestación", "llamado atención", "advertencia", "alerta", "peligro", "riesgo",
    "vulnerabilidad", "fragilidad", "debilidad", "limitación", "deficiencia",
    "carencia", "necesidad", "dependencia", "inseguridad", "inestabilidad", "caos",
    "desorden", "confusión", "desorientación", "perdido", "desubicado", "inadaptado",
    "incomprendido", "solitario", "aislado", "marginado", "excluido", "rechazado",
    "ignorado", "invisible", "silenciado", "callado", "tímido", "retraído", "introvertido",
    "inhibido", "bloqueado", "paralizado", "estancado", "regresión", "retroceso",
    "decaimiento", "disminución", "reducción", "pérdida", "deterioro", "empeoramiento",
    "agravamiento", "complicación", "crisis", "emergencia", "urgencia", "prioritario",
    "intervención", "asistencia", "apoyo", "seguimiento", "monitoreo", "evaluación",
    "diagnóstico", "tratamiento", "terapia", "rehabilitación", "recuperación",
    "reinserción", "reincorporación", "reintegración", "normalización", "estabilización"
]

MODIFICADORES = [
    "muy", "extremadamente", "totalmente", "completamente", "absolutamente",
    "realmente", "verdaderamente", "genuinamente", "profundamente", "intensamente",
    "levemente", "ligeramente", "moderadamente", "parcialmente", "ocasionalmente",
    "frecuentemente", "constantemente", "siempre", "nunca", "jamás", "rara vez",
    "a veces", "generalmente", "habitualmente", "usualmente", "típicamente"
]

RIESGO_EXPR = [
    "riesgo de deserción", "posible abandono", "en observación", "necesita apoyo",
    "requiere atención", "situación vulnerable", "contexto complicado", "factores de riesgo",
    "indicadores alarmantes", "señales de alerta", "comportamiento preocupante",
    "rendimiento decreciente", "motivación en caída", "interés disminuido",
    "participación reducida", "compromiso menguante", "esfuerzo decreciente",
    "asistencia irregular", "puntualidad deficiente", "organización pobre",
    "planificación ausente", "metas indefinidas", "objetivos claros",
    "visión futura", "proyección académica", "expectativas realistas",
    "autoeficacia baja", "autoconcepto académico", "autoestima académica",
    "pertenencia escolar", "vinculación institucional", "identidad estudiantil"
]
PALABRAS_CIENCIA = [
    "experimento","cientifico","ciencia","laboratorio",
    "investigacion","quimica","biologia","fisica"
]

PALABRAS_NUMERO = [
    "logico","numerico","calculo","analisis",
    "matematico","razonamiento"
]

PALABRAS_SOCIAL = [
    "lider","equipo","comunicacion","debate",
    "argumenta","expresa","oratoria"
]
