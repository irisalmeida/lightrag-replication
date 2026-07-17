#!/bin/bash
# Limpar cache de grafos e progresso

echo "Limpando cache..."
rm -rf ./working_dir_grafos/
rm -f progress_4graphs.json
rm -f stats_*.json

echo "✅ Cache limpo!"
echo "Diretórios/arquivos removidos:"
echo "  - working_dir_grafos/"
echo "  - progress_4graphs.json"
echo "  - stats_*.json"
