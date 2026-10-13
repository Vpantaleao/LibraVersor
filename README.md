# LibraVersor

Reconhecimento de gestos de Libras com visão computacional (Python 3.9–3.12).

## Instalação

```bash
git clone https://github.com/Vpantaleao/LibraVersor.git
cd LibraVersor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ou manualmente:

```bash
pip install mediapipe==0.10.14 opencv-contrib-python numpy scipy scikit-learn matplotlib sounddevice
```

Linux: `sudo apt install libportaudio2` (necessário para o `sounddevice`).

## Execução

```bash
python main.py   # TODO: confirmar o arquivo principal
```

## Gestos adicionados

1. Bom dia
2. Sim
3. Não
4. Dor
5. Febre
6. Muito
7. Pouco
8. Remédio
