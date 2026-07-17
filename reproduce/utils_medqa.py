import json
import os
import re
import csv
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score


def carregar_questoes_medqa(caminho_jsonl: str, limite: Optional[int] = None) -> List[Dict]:
    """
    Carrega questões do arquivo JSONL do MedQA-USMLE.
    
    Args:
        caminho_jsonl: Caminho para o arquivo test.jsonl
        limite: Número máximo de questões a carregar (None = todas)
    
    Returns:
        Lista de dicionários com as questões
    """
    questoes = []
    try:
        with open(caminho_jsonl, 'r', encoding='utf-8') as f:
            for idx, linha in enumerate(f):
                if limite and idx >= limite:
                    break
                try:
                    item = json.loads(linha.strip())
                    # Validar campos obrigatórios
                    if 'question' in item and 'options' in item and 'answer_idx' in item:
                        questoes.append(item)
                except json.JSONDecodeError as e:
                    print(f"Erro ao decodificar JSON na linha {idx + 1}: {e}")
                    continue
        print(f"Carregadas {len(questoes)} questões do arquivo {caminho_jsonl}")
        return questoes
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho_jsonl}")
        return []


def extrair_alternativa(resposta) -> str:
    """
    Extrai a letra da alternativa (A-E) da resposta do LLM.
    
    Args:
        resposta: String contendo a resposta do LLM (pode ser None, dict, list, etc)
    
    Returns:
        Letra da alternativa (A-E) ou vazio se não encontrado
    """
    # Tratar casos onde resposta é None ou não é string
    if resposta is None:
        return ""
    
    # Se for dict, tentar extrair 'response' ou 'answer'
    if isinstance(resposta, dict):
        resposta = resposta.get('response') or resposta.get('answer') or str(resposta)
    
    # Se for list, converter para string
    if isinstance(resposta, list):
        resposta = ' '.join(str(item) for item in resposta)
    
    # Converter para string se não for
    if not isinstance(resposta, str):
        resposta = str(resposta)
    
    resposta_limpa = resposta.strip().upper()
    
    # Se vazio após limpeza, retornar vazio
    if not resposta_limpa:
        return ""
    
    # Procurar por padrão de letra isolada
    match = re.search(r'\b([A-E])\b', resposta_limpa)
    if match:
        return match.group(1)
    
    # Procurar por "Answer: X" ou "answer: X"
    match = re.search(r'(?:answer|resposta)[:\s]+([A-E])', resposta_limpa, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Se nenhum padrão encontrado, retornar vazio
    return ""


def calcular_metricas(y_verdadeiro: List[str], y_predito: List[str]) -> Dict[str, float]:
    """
    Calcula acurácia e F1-score ponderado.
    
    Args:
        y_verdadeiro: Lista de respostas corretas (A-E)
        y_predito: Lista de respostas preditas (A-E)
    
    Returns:
        Dicionário com acurácia (%) e F1-score ponderado (%)
    """
    # Filtrar apenas respostas válidas (A-E)
    pares_validos = [
        (true, pred) for true, pred in zip(y_verdadeiro, y_predito)
        if pred in ['A', 'B', 'C', 'D', 'E']
    ]
    
    if not pares_validos:
        return {
            'acuracia': 0.0,
            'f1_score': 0.0,
            'questoes_validas': 0,
            'questoes_invalidas': len(y_verdadeiro)
        }
    
    y_true_valido = [p[0] for p in pares_validos]
    y_pred_valido = [p[1] for p in pares_validos]
    
    acuracia = accuracy_score(y_true_valido, y_pred_valido) * 100
    f1 = f1_score(y_true_valido, y_pred_valido, average='weighted', zero_division=0) * 100
    
    return {
        'acuracia': round(acuracia, 2),
        'f1_score': round(f1, 2),
        'questoes_validas': len(pares_validos),
        'questoes_invalidas': len(y_verdadeiro) - len(pares_validos)
    }


def salvar_checkpoint(resultados: List[Dict], caminho_checkpoint: str) -> None:
    """
    Salva resultados em um checkpoint JSON.
    
    Args:
        resultados: Lista de resultados de queries
        caminho_checkpoint: Caminho para salvar o checkpoint
    """
    try:
        os.makedirs(os.path.dirname(caminho_checkpoint), exist_ok=True)
        with open(caminho_checkpoint, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
        print(f"Checkpoint salvo em: {caminho_checkpoint}")
    except Exception as e:
        print(f"Erro ao salvar checkpoint: {e}")


def registrar_log(mensagem: str, caminho_log: str) -> None:
    """
    Registra mensagem em arquivo de log com timestamp.
    
    Args:
        mensagem: Mensagem a registrar
        caminho_log: Caminho para o arquivo de log
    """
    try:
        os.makedirs(os.path.dirname(caminho_log), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(caminho_log, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {mensagem}\n")
    except Exception as e:
        print(f"Erro ao registrar log: {e}")


def salvar_resultados_csv(resultados: List[Dict], caminho_csv: str) -> None:
    """
    Salva resultados em formato CSV para análise.
    
    Args:
        resultados: Lista de resultados com métricas
        caminho_csv: Caminho para salvar o CSV
    """
    try:
        os.makedirs(os.path.dirname(caminho_csv), exist_ok=True)
        
        if not resultados:
            print("Nenhum resultado para salvar em CSV")
            return
        
        # Extrair campos do primeiro resultado para definir cabeçalho
        campos = ['indice', 'pergunta', 'resposta_verdadeira', 'resposta_predita', 
                  'acerto', 'chamadas_api', 'tempo_ms']
        
        with open(caminho_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            
            for idx, resultado in enumerate(resultados):
                linha = {
                    'indice': idx + 1,
                    'pergunta': resultado.get('pergunta', '')[:100],  # Primeiros 100 chars
                    'resposta_verdadeira': resultado.get('resposta_verdadeira', ''),
                    'resposta_predita': resultado.get('resposta_predita', ''),
                    'acerto': 1 if resultado.get('acerto') else 0,
                    'chamadas_api': resultado.get('chamadas_api', 0),
                    'tempo_ms': resultado.get('tempo_ms', 0)
                }
                writer.writerow(linha)
        
        print(f"Resultados salvos em CSV: {caminho_csv}")
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")


def validar_estrutura_json(resultado: Dict) -> bool:
    """
    Valida se a estrutura JSON do resultado está correta.
    
    Args:
        resultado: Dicionário com resultado de query
    
    Returns:
        True se estrutura válida, False caso contrário
    """
    campos_obrigatorios = ['pergunta', 'resposta_verdadeira', 'resposta_predita', 'chamadas_api']
    return all(campo in resultado for campo in campos_obrigatorios)


def gerar_relatorio_metricas(resultados: List[Dict], caminho_relatorio: str) -> Dict:
    """
    Gera relatório completo de métricas.
    
    Args:
        resultados: Lista de resultados
        caminho_relatorio: Caminho para salvar o relatório
    
    Returns:
        Dicionário com métricas agregadas
    """
    if not resultados:
        return {}
    
    y_verdadeiro = [r.get('resposta_verdadeira', '') for r in resultados]
    y_predito = [r.get('resposta_predita', '') for r in resultados]
    
    metricas = calcular_metricas(y_verdadeiro, y_predito)
    
    # Adicionar métricas de eficiência
    total_chamadas_api = sum(r.get('chamadas_api', 0) for r in resultados)
    tempo_total_ms = sum(r.get('tempo_ms', 0) for r in resultados)
    tempo_medio_ms = tempo_total_ms / len(resultados) if resultados else 0
    
    relatorio = {
        'total_questoes': len(resultados),
        'acuracia_percentual': metricas.get('acuracia', 0),
        'f1_score_percentual': metricas.get('f1_score', 0),
        'questoes_validas': metricas.get('questoes_validas', 0),
        'questoes_invalidas': metricas.get('questoes_invalidas', 0),
        'total_chamadas_api': total_chamadas_api,
        'chamadas_api_por_query': round(total_chamadas_api / len(resultados), 2) if resultados else 0,
        'tempo_total_ms': tempo_total_ms,
        'tempo_medio_por_query_ms': round(tempo_medio_ms, 2),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        os.makedirs(os.path.dirname(caminho_relatorio), exist_ok=True)
        with open(caminho_relatorio, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)
        print(f"Relatório de métricas salvo em: {caminho_relatorio}")
    except Exception as e:
        print(f"Erro ao salvar relatório: {e}")
    
    return relatorio
