import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

def extract_metrics(csv_file):
    df = pd.read_csv(csv_file)
    
    metrics = {}
    
    for col in ['semantic_similarity', 'answer_correctness', 'correct']:
        if col not in df.columns:
            continue
        
        values = df[col].dropna()
        
        if col == 'correct':
            values = values.astype(float)
        
        n = len(values)
        mean = values.mean()
        std = values.std()
        
        ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=stats.sem(values))
        
        metrics[col] = {
            'n': n,
            'média': round(mean, 4),
            'desvio_padrão': round(std, 4),
            'IC_95%': (round(ci_95[0], 4), round(ci_95[1], 4)),
            'mínimo': round(values.min(), 4),
            'máximo': round(values.max(), 4),
            'mediana': round(values.median(), 4),
        }
    
    acuracia = (df['correct'].sum() / len(df) * 100) if 'correct' in df.columns else None
    metrics['acurácia_%'] = round(acuracia, 2) if acuracia else None
    
    return metrics, df

def print_metrics(metrics):
    print("\n" + "="*80)
    print("MÉTRICAS RAGAS - RESUMO ESTATÍSTICO")
    print("="*80 + "\n")
    
    for metric_name, values in metrics.items():
        if metric_name == 'acurácia_%':
            print(f"Acurácia: {values}%\n")
            continue
        
        print(f"📊 {metric_name.upper()}")
        print(f"   N (amostras):        {values['n']}")
        print(f"   Média:               {values['média']}")
        print(f"   Desvio Padrão:       {values['desvio_padrão']}")
        print(f"   IC 95%:              [{values['IC_95%'][0]}, {values['IC_95%'][1]}]")
        print(f"   Mínimo:              {values['mínimo']}")
        print(f"   Máximo:              {values['máximo']}")
        print(f"   Mediana:             {values['mediana']}")
        print()

def save_metrics_json(metrics, output_file):
    import json
    
    metrics_clean = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_clean[key] = {k: v for k, v in value.items()}
        else:
            metrics_clean[key] = value
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_clean, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Métricas salvas em: {output_file}")

if __name__ == "__main__":
    csv_file = Path("/home/iris/LightRAG/experiments/results/eval_20260713_205000_ragas_results.csv")
    
    if not csv_file.exists():
        print(f"❌ Arquivo não encontrado: {csv_file}")
        exit(1)
    
    metrics, df = extract_metrics(csv_file)
    print_metrics(metrics)
    
    output_json = csv_file.parent / f"{csv_file.stem}_metrics.json"
    save_metrics_json(metrics, output_json)
    
    print(f"\n✓ Total de questões avaliadas: {len(df)}")
