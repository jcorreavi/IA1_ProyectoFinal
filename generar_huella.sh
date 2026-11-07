#!/usr/bin/env bash
# Script rápido para generar una huella sintética con Anguli y guardarla en bandeja_agente
# Ejemplos:
#   ./generar_huella.sh --autorizada   # Genera huella autorizada (seed 1604, arch)
#   ./generar_huella.sh --intrusa      # Genera huella intrusa (seed aleatoria)
#   ./generar_huella.sh -s 1604 -c arch --impresion

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DIR/.venv/bin/python3" ]; then
    PY="$DIR/.venv/bin/python3"
else
    PY="python3"
fi

"$PY" "$DIR/generador_anguli.py" "$@"

