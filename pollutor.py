import pandas as pd
import numpy as np
import json
import math
from datetime import datetime

def add_gauss(lon, lat, sigma):
    """ adiciona ruído gaussiano em pontos GPS """
    
    # Raio da Terra em metros
    R = 6378137.0

    dev_x = np.random.normal(loc=0.0, scale=sigma)
    dev_y = np.random.normal(loc=0.0, scale=sigma)

    delta_lat = (dev_y / R) * (180.0 / math.pi)
    delta_lon = (dev_x / (R * math.cos(math.pi * lat / 180.0))) * (180.0 / math.pi)

    new_lat = lat + delta_lat
    new_lon = lon + delta_lon

    return [new_lon, new_lat]

def mod_accuracy(df, pct: float, sigma: float):
    """ altera polyline adicionando ruido baseado no método CL-TSim
    pct: % de trajetórias afetadas no dataset
    sigma: desvio padrão em metros
    """
    print(f"Ruido gaussiano (sigma = {sigma}m) injetado em {pct*100}% das viagens.")
    print("Taxa de pontos distorcidos (r_t) selecionada aleatoriamente de {0, 0.2, 0.4, 0.6, 0.8} por trajetória...")

    df_mod = df.copy()

    polluted = int(len(df_mod) * pct)

    # Seleciona viagens que poluidas
    dev_index = np.random.choice(df_mod.index, polluted, replace=False)

    rt_options = [0.0, 0.2, 0.4, 0.6, 0.8]
    r_prop = [0, 0, 0, 0, 0]

    for idx in dev_index:
        poly_str = df_mod.at[idx, 'POLYLINE']

        if pd.notna(poly_str) and isinstance(poly_str, str) and poly_str != '[]':
            try:
                poly = json.loads(poly_str)
                num_points = len(poly)

                # Sorteia a taxa r_t para a trajetória atual
                r_t = np.random.choice(rt_options)

                # taxa escolhida
                if r_t == 0.0 or num_points == 0:
                    r_prop[0] += 1
                    continue
                elif r_t == 0.2:
                    r_prop[1] += 1
                elif r_t == 0.4:
                    r_prop[2] += 1
                elif r_t == 0.6:
                    r_prop[3] += 1
                elif r_t == 0.8:
                    r_prop[4] += 1

                # Calcula a quantidade de pontos que serão poluídos
                num_noisy_points = int(num_points * r_t)

                # Sorteia os índices dos pontos que serão poluídos
                noisy_indices = set(np.random.choice(range(num_points), num_noisy_points, replace=False))

                noisy_polyline = []

                for i, pt in enumerate(poly):
                    if i in noisy_indices:
                        lon, lat = pt[0], pt[1]
                        noisy_point = add_gauss(lon, lat, sigma)
                        noisy_polyline.append(noisy_point)
                    else:
                        noisy_polyline.append(pt)

                df_mod.at[idx, 'POLYLINE'] = json.dumps(noisy_polyline)

            except json.JSONDecodeError:
                continue

    print("\n--- Distribuição da Taxa de Poluição (r_t) ---")
    total_validas = sum(r_prop)
    for c, contagem in enumerate(r_prop):
        if total_validas > 0:
            porcentagem = (contagem / total_validas) * 100
        else:
            porcentagem = 0
        print(f"Taxa {rt_options[c]}: {contagem} trajetórias ({porcentagem:.2f}%)")

    return df_mod

def mod_completeness_trimming(df, pct: float, min_gap_ratio: float, max_gap_ratio: float):
    """
    Altera o polyline simulando a perda de dados nas extremidades da trajetória (Trimming).
    Remove um segmento sequencial de pontos apenas no INÍCIO (origem) ou no FIM (destino).

    pct: % de trajetórias afetadas no dataset
    min_gap_ratio: tamanho mínimo do corte (proporção da trajetória, ex: 0.05 para 5%)
    max_gap_ratio: tamanho máximo do corte (proporção da trajetória, ex: 0.15 para 15%)
    """
    print(f"Trajectory Trimming (corte nas extremidades) aplicado em {pct*100:.1f}% das viagens.")
    print(f"Tamanho do corte varia aleatoriamente de {min_gap_ratio*100:.1f}% a {max_gap_ratio*100:.1f}% dos pontos...")

    df_mod = df.copy()
    polluted = int(len(df_mod) * pct)

    # Seleciona as viagens que sofrerão o trimming
    dev_index = np.random.choice(df_mod.index, polluted, replace=False)

    trims_applied = 0
    skipped_short = 0

    for idx in dev_index:
        poly_str = df_mod.at[idx, 'POLYLINE']

        if pd.notna(poly_str) and isinstance(poly_str, str) and poly_str != '[]':
            try:
                poly = json.loads(poly_str)
                num_points = len(poly)

                # Trajetórias muito curtas são ignoradas
                if num_points < 10:
                    skipped_short += 1
                    continue

                # Sorteia a proporção do downsampling para a trajetória
                trim_ratio = np.random.uniform(min_gap_ratio, max_gap_ratio)

                trim_len = max(1, int(num_points * trim_ratio))

                if trim_len >= num_points - 2:
                    trim_len = num_points - 2

                if trim_len > 0:
                    # Sorteia se o corte será no INÍCIO (prefixo) ou no FIM (sufixo)
                    trim_position = np.random.choice(['prefix', 'suffix'])

                    if trim_position == 'prefix':
                        poly_trimmed = poly[trim_len:]
                    else:
                        poly_trimmed = poly[:-trim_len]

                    df_mod.at[idx, 'POLYLINE'] = json.dumps(poly_trimmed)
                    trims_applied += 1

            except json.JSONDecodeError:
                continue

    print("\n--- Relatório de Completude (Trimming) ---")
    print(f"Cortes aplicados com sucesso: {trims_applied} trajetórias")
    print(f"Trajetórias ignoradas por serem muito curtas: {skipped_short}")

    return df_mod

