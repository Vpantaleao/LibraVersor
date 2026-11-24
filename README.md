# LibraVersor
Projeto desenvolvido como trabalho da disciplina **Extensão IV**, ministrada pelo **Prof. Carlos Melo**, na **Universidade de Pernambuco (UPE)**.

---

Tradutor de **Libras em tempo real**: a webcam captura a mão, um modelo de aprendizado de máquina reconhece o gesto e o sistema mostra uma **legenda na tela** e **fala a palavra em voz alta** (offline).

O vocabulário atual é focado em **atendimento médico** (sintomas, dor, remédio, emergência).



## Sumário

1. [Visão geral](#1-visão-geral)
2. [Estrutura do repositório](#2-estrutura-do-repositório)
3. [Instalação](#3-instalação)
4. [Como usar](#4-como-usar)
5. [Vocabulário reconhecido](#5-vocabulário-reconhecido)
6. [Arquitetura dos módulos](#6-arquitetura-dos-módulos)
7. [Formato dos dados](#7-formato-dos-dados)
8. [Histórico resumido](#8-histórico-resumido)
9. [Limitações e melhorias futuras](#9-limitações-e-melhorias-futuras)
10. [Autores e contexto acadêmico](#10-autores-e-contexto-acadêmico)

---

## 1. Visão geral

| Item | Detalhe |
|---|---|
| Linguagem | Python 3.9–3.12 |
| Visão computacional | MediaPipe Hands (21 pontos da mão) + OpenCV |
| Classificador | Ensemble Random Forest + SVM (scikit-learn) |
| Entrada do modelo | Janela de 30 frames (~1 s) de landmarks da mão |
| Saída | Legenda sobre o vídeo + voz (pyttsx3) |
| Dados | 16 classes × 30 gravações = 480 sequências (`dados_gestos/`) |
| Contexto | Disciplina Extensão IV — Prof. Carlos Melo — Universidade de Pernambuco |
| Repositório | https://github.com/Vpantaleao/LibraVersor (branch `main`) |

### Fluxo geral

```
Webcam → MediaPipe (21 landmarks) → janela de 30 frames
      → features da sequência → Random Forest + SVM (média das probabilidades)
      → StreamRecognizer (regras de estabilidade) → palavra
      → legenda (Pillow/OpenCV) + voz (pyttsx3)
```

---

## 2. Estrutura do repositório

```
LibraVersor/
├── main.py                # Aplicação principal (webcam → legenda + voz)
├── libra_core.py          # Núcleo compartilhado: features, aumento de dados, modelo, reconhecimento
├── treinar_modelo.py      # Treina e valida o modelo a partir de dados_gestos/
├── coletor_dinamico.py    # Grava sequências de gestos (30 frames) em .npy
├── coletor_estatico.py    # Grava frames isolados em CSV (abordagem inicial, não usada pelo treino atual)
├── tts.py                 # Leitor de voz (thread própria, escolhe voz pt-BR)
├── modelo_gestos.p        # Modelo treinado (pickle, ~24 MB)
├── dados_gestos/<gesto>/  # Gravações: 0.npy … 29.npy, cada uma com shape (30, 63)
├── requirements.txt       # Dependências fixadas
└── README.md
```

---

## 3. Instalação

```bash
git clone https://github.com/Vpantaleao/LibraVersor.git
cd LibraVersor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ou instale manualmente:

```bash
pip install mediapipe==0.10.14 opencv-contrib-python numpy scipy scikit-learn matplotlib sounddevice pyttsx3 pillow
```

- **Linux:** `sudo apt install libportaudio2` (para `sounddevice`) e `sudo apt install espeak-ng` (voz do `pyttsx3`).
- **Windows:** para uma voz em português, adicione "Maria" em *Configurações > Hora e idioma > Fala*.
- **macOS:** a voz "Luciana" já vem instalada.
- O `main.py` também usa **Pillow** (desenho do texto com acentos); ele já está em `requirements.txt`.

---

## 4. Como usar

### 4.1 Executar o reconhecimento

```bash
python main.py [--camera 1] [--sem-voz] [--debug] [--confianca 0.6]
```

| Opção | Efeito |
|---|---|
| `--camera N` | Índice da câmera (padrão `0`) |
| `--sem-voz` | Inicia com a voz desligada |
| `--debug` | Mostra painel de diagnóstico (fps, estado, candidato, confiança) |
| `--confianca X` | Confiança mínima para aceitar um gesto (padrão `0.60`) |

**Teclas durante a execução:**

| Tecla | Ação |
|---|---|
| `Q` / `ESC` | Sair |
| `C` | Limpar legenda e reiniciar o reconhecedor |
| `V` | Ligar/desligar a voz |
| `D` | Mostrar/ocultar diagnóstico |
| `F` | Tela cheia |

A legenda some sozinha após 7 s sem palavra nova; a palavra mais recente fica destacada por 1,6 s.

### 4.2 Gravar um novo gesto

```bash
python coletor_dinamico.py
```

1. Digite o nome do gesto (vira o nome da pasta em `dados_gestos/`).
2. Pressione `ESPAÇO` na janela de vídeo.
3. O programa grava **30 repetições**; antes de cada uma há 1,5 s de preparação, e depois 30 frames (~1 s) são salvos em `dados_gestos/<gesto>/<n>.npy`.

Dicas: grave também a classe **`repouso`** (mão parada) — ela evita que a mão em descanso vire palavra. Varie levemente posição, distância e velocidade entre as repetições.

### 4.3 Treinar o modelo

```bash
python treinar_modelo.py            # valida e treina
python treinar_modelo.py --rapido   # pula a validação
```

- A **validação** separa o último terço das repetições de cada gesto (nunca vistas no treino), o que é mais honesto do que sortear amostras, já que repetições consecutivas são muito parecidas.
- Imprime acurácia e `classification_report` por classe.
- O modelo final é treinado com todas as gravações e salvo em `modelo_gestos.p`.
- Avisa se alguma classe tem menos de 20 gravações ou se falta a classe `repouso`.

> Sempre que adicionar um gesto novo, rode o treino de novo. Para exibir o nome com acento correto na legenda/voz, adicione-o também em `DISPLAY_NAMES` em `libra_core.py`.

---

## 5. Vocabulário reconhecido

15 gestos mais a classe técnica `repouso` (30 gravações cada em `dados_gestos/`):

1. Bom dia
2. Sim
3. Não
4. Dor 
5. Febre 
6. Muito 
7. Pouco 
8. Remédio 
9. Gripe 
10. Vômito 
11. (dor) de Cabeça 
12. Sentindo 
13. Emergência 
14. Obrigado 
15. Oi

`repouso` não é uma palavra: é a classe de "mão parada", usada apenas para evitar falsos positivos.

---

## 6. Arquitetura dos módulos

### 6.1 `libra_core.py` (núcleo)

Concentra tudo o que treino e execução precisam usar **de forma idêntica**, garantindo que o que foi treinado é exatamente o que roda ao vivo.

**Constantes principais**

| Constante | Valor | Significado |
|---|---|---|
| `SEQUENCE_LENGTH` | 30 | Frames por janela |
| `FEATURE_VERSION` | 2 | Versão das features; se mudar, o modelo antigo é recusado e é preciso retreinar |
| `ASPECT` | 9/16 | Correção de proporção para câmera 16:9 |
| `REST_LABELS` | `{"repouso","neutro"}` | Classes que nunca viram palavra |

**`extract_landmarks(hand_landmarks)`** — devolve 63 valores (21 pontos × x,y,z) relativos ao punho (ponto 0). Sem mão → vetor de zeros.

**`sequence_features(seq)`** — transforma a janela (30×63) em um vetor de features que:
- não depende da distância da mão à câmera (normaliza pelo tamanho da palma);
- não depende de quando o gesto começou na janela;
- usa distâncias entre pontos (robustas a inclinação e à mão usada).

Inclui média e desvio-padrão por frame, mínimos/máximos, médias por terços da janela (início/meio/fim), velocidade média, caminho percorrido pelas pontas dos dedos e fração de frames com mão detectada.

**`transform` / `augment`** — aumento de dados (só no treino): escala, rotação, espelhamento, ruído, deslocamento temporal, gesto mais rápido/lento e perda de frames pelo MediaPipe.

**`GestureEnsemble`** — Random Forest (300 árvores) + SVM (kernel RBF, `C=10`, com `StandardScaler`); a probabilidade final é a média das duas. `for_inference()` desativa o paralelismo para não criar threads a cada frame.

**`StreamRecognizer`** — transforma previsões contínuas em palavras confiáveis. Regras:
1. `repouso` e ausência de mão nunca geram palavra (apenas "rearmam" o sistema).
2. A palavra só é aceita se ficar **estável** por `stable_frames` (6), com confiança ≥ `conf_min` (0,60) e margem ≥ `margin_min` (0,15) sobre a 2ª colocada.
3. Após aceitar, há um **período refratário** de 15 frames.
4. A mesma palavra só se repete depois que a janela inteira volta a repouso/sem mão (8 frames) — evita repetir a "cauda" do gesto anterior.
5. Classifica a cada `stride`=2 frames e exige ao menos 12 frames com mão na janela.

Estados expostos para a interface: `aquecendo`, `sem_mao`, `repouso`, `reconhecendo`, `analisando`.

**`CaptionBuffer`** — guarda até 12 palavras e limpa tudo após 7 s sem novidade.

### 6.2 `main.py` (aplicação)

- Lê a webcam (1280×720), espelha a imagem e roda o MediaPipe (1 mão, confiança 0,7).
- `AsyncRecognizer` executa o reconhecimento em **outra thread** com fila de entrada (máx. 150 frames), de modo que o vídeo nunca espera o modelo.
- `draw_hud` desenha a interface com Pillow (o OpenCV não renderiza acentos): barra superior com status da mão e da voz, legenda centralizada com até 2 linhas, barra de tempo até a legenda sumir e painel de debug.
- `load_model` valida o pickle e a `feature_version` antes de iniciar.

### 6.3 `tts.py` (voz)

- Classe `Speaker`, em thread própria, **offline** (usa as vozes do sistema via `pyttsx3`).
- Prefere voz **pt-BR**, depois qualquer português, depois a voz padrão.
- Cria um motor novo a cada fala (contorna um bug conhecido do `pyttsx3` que só fala na primeira vez).
- Palavras que chegam enquanto outra é falada são unidas numa só fala; a fila descarta as mais antigas se passar de 8 itens.
- Se a voz falhar, o programa continua apenas com legenda.

### 6.4 `treinar_modelo.py` e coletores

- `treinar_modelo.py`: carrega `dados_gestos/<gesto>/<n>.npy` (ignora arquivos com shape diferente de `(30, 63)`), gera 6 cópias aumentadas por gravação e treina o ensemble. O arquivo salvo contém `model`, `labels_dict`, `feature_version` e `sequence_length`.
- `coletor_dinamico.py`: gravação de sequências (usado no pipeline atual).
- `coletor_estatico.py`: grava 100 frames isolados em `dataset_gestos.csv` (63 features + `label`). É da abordagem inicial (gestos estáticos) e **não é usado** pelo treino atual.

---

## 7. Formato dos dados

Cada arquivo `dados_gestos/<gesto>/<n>.npy` é um array NumPy `(30, 63)`:

- 30 frames consecutivos (~1 s);
- 63 = 21 landmarks × (x, y, z), com coordenadas **relativas ao punho**;
- frames sem mão detectada ficam como vetor de zeros.

---

## 8. Histórico resumido

1. Rastreamento inicial das mãos (MediaPipe).
2. Primeiro classificador Random Forest e primeira versão de gestos dinâmicos.
3. Trava por tempo para palavras detectadas; limite de 1 mão.
4. Inclusão de gestos 1–8, depois mais gestos (até 12 no README).
5. Última versão: melhoria no treinador e interface com legenda + leitor de voz (20/10/2026).

---

## 9. Limitações e melhorias futuras

**Observações importantes**

- `modelo_gestos.p` é um arquivo pickle: carregue apenas modelos de fonte confiável. Se der erro ao carregar (versão do scikit-learn diferente), rode `python treinar_modelo.py`.
- O `main.py` usa a API `mp.solutions` do `mediapipe==0.10.14`; versões mais novas podem exigir ajustes.
- O `dataset_gestos.csv` gerado pelo coletor estático está no `.gitignore`.

**Melhorias pendentes**

1. **Fontes:** o `main.py` procura `assets/fonts/DejaVuSans*.ttf`, mas a pasta `assets/` não está no repositório (o programa usa fontes do sistema). Basta adicionar as fontes.
2. **Código duplicado:** `extract_landmarks` e `SEQUENCE_LENGTH` aparecem em `coletor_dinamico.py` e em `libra_core.py`; os coletores poderiam importar do núcleo.
3. **Cobertura de dados:** 30 gravações por gesto, provavelmente de uma pessoa só; a precisão tende a cair com outras pessoas, iluminação e fundos. Gravar com mais pessoas é a melhoria de maior impacto.

---

## 10. Autores

- Daniel Bezerra
- João Barreto
- Maria Luísa
- Paulo Samuel
- Vinicyus Alves