import pandas as pd

provincias_por_region = {
    "Norte": [
        "MONTE CRISTI",
        "PUERTO PLATA",
        "SANTIAGO",
        "VALVERDE",
        "SANTIAGO RODRÍGUEZ",
        "LA VEGA",
        "ESPAILLAT",
        "HERMANAS MIRABAL",
        "DAJABÓN",
        "DUARTE",
        "MARÍA TRINIDAD SÁNCHEZ",
        "SAMANÁ",
    ],
    "Sur": [
        "SAN JUAN",
        "AZUA",
        "BARAHONA",
        "PEDERNALES",
        "INDEPENDENCIA",
        "BAORUCO",
        "SAN CRISTÓBAL",
        "PERAVIA",
        "MONSEÑOR NOUEL",
        "ELÍAS PIÑA",
        "SAN JOSÉ DE OCOA"
    ],
    "Este": [
        "LA ALTAGRACIA",
        "LA ROMANA",
        "SAN PEDRO DE MACORÍS",
        "SANCHEZ RAMÍREZ",
        "EL SEIBO",
       "SANTO DOMINGO",
        "DISTRITO NACIONAL",
        "MONTE PLATA",
        "HATO MAYOR",
    ]
}

df = pd.read_csv("Data_consumo_Edesur_2012-2024.csv")

def consumo_por_agno(df, agno):
    return df[df['Año'] == agno]

def consumo_por_region(df, region):
    region = "Ede" + region
    return df[df['Empresa Distribuidora'] == region]

def consumo_por_mes(df, mes):
    return df[df['Mes'] == mes]

def consumo_por_cliente(df, cliente):
    return df[df['Tipo Cliente'] == cliente]



def resumen_consumo_por_region(region, df=df):
    df = consumo_por_region(df, region)
    
    MESES = [
    "Enero", "Febrero", "Marzo", "Abril",
    "Mayo", "Junio", "Julio", "Agosto",
    "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]

    CLIENTES = ['Ayuntamiento', 'Comercial', 'Gobierno', 'Industrial', 'Residencial']
    
    df_meses = pd.DataFrame({"Mes":[], "Energia":[], "Potencia":[]})
    
    for i in range(1, 13):
        energia = consumo_por_mes(df, i)['Energía (gwh)'].mean()
        potencia = consumo_por_mes(df, i)['Potencia (mw)'].mean()
        df_mes_temporal = pd.DataFrame({"Mes":[MESES[i-1]], "Energia":[energia], "Potencia":[potencia]})
        df_meses = pd.concat([df_meses, df_mes_temporal], ignore_index=True)
    
    df_cliente = pd.DataFrame({"Cliente":[], "Energia":[], "Potencia":[]})
    
    for cliente in CLIENTES:
        energia = consumo_por_cliente(df, cliente)['Energía (gwh)'].mean()
        potencia = consumo_por_cliente(df, cliente)['Potencia (mw)'].mean()
        df_cliente_temporal = pd.DataFrame({"Cliente":[cliente], "Energia":[energia], "Potencia":[potencia]})
        df_cliente = pd.concat([df_cliente, df_cliente_temporal], ignore_index=True)
        
    return df_meses, df_cliente


def resumen_consumo_por_agno(agno, df=df):
    df = consumo_por_agno(df, agno)
    
    REGIONES = ['Norte', "Sur", "Este"]
    df_regiones = pd.DataFrame({"region":[], "Energia":[], "Potencia":[]})
    
    for region in REGIONES:
        energia = consumo_por_region(df, region)['Energía (gwh)'].mean()
        potencia = consumo_por_region(df, region)['Potencia (mw)'].mean()
        df_region_temporal = pd.DataFrame({"region":[region], "Energia":[energia], "Potencia":[potencia]})
        df_regiones = pd.concat([df_regiones, df_region_temporal], ignore_index=True)

  
    return df_regiones