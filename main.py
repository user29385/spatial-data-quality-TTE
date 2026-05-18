import argparse
from polluter import mod_accuracy, mod_completeness_trimming
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gerar datasets ruidosos a partir de um arquivo de entrada CSV."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Caminho para o arquivo CSV de entrada.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_path:
        porto = args.input_path
    else:
        porto = "data/train.csv"  # Caminho padrão para o dataset

    df_original = pd.read_csv(porto, sep=",")
    
    pct = 1
    sigma = 50 / 1.2533  # Desvio padrão do erro GPS em metros (50m/1.2533)

    """Acurácia posicional"""
    df_ruido = mod_accuracy(df_original, pct, sigma)
    out = 'taxi_porto_noisy.csv'
    df_ruido.to_csv(out, index=False)

    print(f"Dataset ruidoso salvo em: {out}")
    print(f"Total de linhas processadas: {len(df_ruido)}")

    """Completude"""
    df_ruido = mod_completeness_trimming(df_original, 1, 0.05, 0.15)
    out = 'completude_ds.csv'
    df_ruido.to_csv(out, index=False)
    print(f"Dataset de completude salvo em: {out}")


if __name__ == "__main__":
    main()
