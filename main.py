"""
LibraVersor - reconhecimento de gestos de Libras em tempo real.
Webcam -> legenda na tela + voz.

Teclas:  Q / ESC sair | C limpar legenda | V voz liga/desliga | D diagnóstico | F tela cheia
Uso:     python main.py [--camera 1] [--sem-voz] [--debug] [--confianca 0.6]
"""
import argparse
import os
import pickle
import queue
import sys
import threading
import time
import traceback

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import libra_core as lc
from tts import Speaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_gestos.p")
FONT_BOLD = os.path.join(BASE_DIR, "assets", "fonts", "DejaVuSans-Bold.ttf")
FONT_REGULAR = os.path.join(BASE_DIR, "assets", "fonts", "DejaVuSans.ttf")

WINDOW = "LibraVersor"
CAPTION_TIMEOUT = 7.0       # segundos sem palavra nova até a legenda sumir
HIGHLIGHT_SECONDS = 1.6     # a palavra mais recente fica destacada por este tempo
CAPTION_FONT_SIZE = 32      # tamanho das letras da legenda (para 1280x720; acompanha a janela)

ACCENT = (255, 196, 46)
GREEN = (76, 217, 130)
GRAY = (150, 158, 168)
WHITE = (255, 255, 255)


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit("Modelo não encontrado. Rode primeiro:  python treinar_modelo.py")
    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        ok = payload.get("feature_version") == lc.FEATURE_VERSION
    except Exception as exc:
        sys.exit(f"Não foi possível carregar o modelo ({type(exc).__name__}). "
                 "Rode:  python treinar_modelo.py")
    if not ok:
        sys.exit("O modelo foi treinado com uma versão antiga do projeto. "
                 "Rode:  python treinar_modelo.py")
    return payload["model"].for_inference(), payload["labels_dict"]


class AsyncRecognizer:
    """Roda o reconhecimento em outra thread: o vídeo nunca espera o modelo."""

    def __init__(self, recognizer):
        self.rec = recognizer
        self._in = queue.Queue(maxsize=150)
        self._words = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            kind, data = self._in.get()
            if kind == "stop":
                break
            try:
                if kind == "reset":
                    self.rec.reset()
                else:
                    word = self.rec.update(data)
                    if word:
                        self._words.put(word)
            except Exception:
                traceback.print_exc()

    def push(self, feats):
        try:
            self._in.put_nowait(("frame", feats))
        except queue.Full:
            pass

    def reset(self):
        try:
            while True:
                self._in.get_nowait()
        except queue.Empty:
            pass
        self._in.put(("reset", None))

    def pop_words(self):
        out = []
        try:
            while True:
                out.append(self._words.get_nowait())
        except queue.Empty:
            return out

    def stop(self):
        self._in.put(("stop", None))

    state = property(lambda self: self.rec.state)
    candidate = property(lambda self: self.rec.candidate)
    confidence = property(lambda self: self.rec.confidence)


# --------------------------------------------------------------------------
# Interface (Pillow desenha o texto: o OpenCV não suporta acentos)
# --------------------------------------------------------------------------
_fonts = {}


def font(size, bold=True):
    """Fonte embutida; se faltar, tenta fontes do sistema; por último a padrão do Pillow."""
    key = (size, bold)
    if key not in _fonts:
        names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold
                 else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Arial.ttf"])
        found = None
        for candidate in [FONT_BOLD if bold else FONT_REGULAR] + names:
            try:
                found = ImageFont.truetype(candidate, size)
                break
            except OSError:
                continue
        if found is None:
            try:
                found = ImageFont.load_default(size)       # Pillow >= 10.1: respeita o tamanho
            except TypeError:
                found = ImageFont.load_default()
        _fonts[key] = found
    return _fonts[key]


def panel(img, x0, y0, x1, y1, radius, fill, alpha):
    """Retângulo arredondado semitransparente."""
    w, h = max(int(x1 - x0), 1), max(int(y1 - y0), 1)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle(
        (0, 0, w - 1, h - 1), radius=radius, fill=(*fill, int(255 * alpha)))
    img.paste(overlay, (int(x0), int(y0)), overlay)


def draw_hud(img, ui):
    W, H = img.size
    k = min(W / 1280, H / 720)

    def px(v):
        return max(1, int(round(v * k)))

    d = ImageDraw.Draw(img)

    # ---- barra superior ----
    bar_h = px(64)
    panel(img, 0, 0, W, bar_h, 0, (14, 17, 23), 0.74)
    d.ellipse((px(20), bar_h // 2 - px(8), px(36), bar_h // 2 + px(8)), fill=ACCENT)
    f_title = font(px(30))
    d.text((px(48), bar_h // 2), "LibraVersor", font=f_title, fill=WHITE, anchor="lm")
    tw = d.textlength("LibraVersor", font=f_title)
    d.text((px(48) + tw + px(16), bar_h // 2 + px(2)), "Libras em texto e voz",
           font=font(px(17), False), fill=GRAY, anchor="lm")

    def pill(right, text, color):
        f = font(px(18), False)
        tw_ = d.textlength(text, font=f)
        pad, dot, h = px(14), px(10), px(34)
        w = int(tw_ + pad * 2 + dot + px(8))
        x0, y0 = right - w, (bar_h - h) // 2
        panel(img, x0, y0, right, y0 + h, h // 2, (38, 44, 54), 0.95)
        d.ellipse((x0 + pad, y0 + h // 2 - dot // 2, x0 + pad + dot, y0 + h // 2 + dot // 2), fill=color)
        d.text((x0 + pad + dot + px(8), y0 + h // 2), text, font=f, fill=(235, 238, 242), anchor="lm")
        return x0 - px(10)

    if not ui["voice_available"]:
        voice = ("Voz indisponível", GRAY)
    elif ui["voice_on"]:
        voice = ("Voz ligada", GREEN)
    else:
        voice = ("Voz desligada", GRAY)
    if not ui["hand"]:
        status = ("Mostre a mão à câmera", GRAY)
    elif ui["state"] == "aquecendo":
        status = ("Iniciando…", GRAY)
    elif ui["state"] in ("reconhecendo", "analisando"):
        status = ("Reconhecendo…", ACCENT)
    else:
        status = ("Pronto", GREEN)
    right = pill(W - px(18), *voice)
    pill(right, *status)

    # ---- legenda ----
    words, y1 = ui["words"], H - px(60)
    if words:
        size = px(CAPTION_FONT_SIZE)
        f = font(size)
        toks = list(words)
        toks[0] = toks[0][:1].upper() + toks[0][1:]
        space = d.textlength(" ", font=f)
        max_w = int(W * 0.88) - 2 * px(40)
        lines, cur, cur_w = [], [], 0
        for i, t in enumerate(toks):
            t_w = d.textlength(t, font=f)
            if cur and cur_w + space + t_w > max_w:
                lines.append(cur)
                cur, cur_w = [(i, t, t_w)], t_w
            else:
                cur_w += t_w if not cur else space + t_w
                cur.append((i, t, t_w))
        lines.append(cur)
        lines = lines[-2:]
        widths = [sum(t_w for _, _, t_w in ln) + space * (len(ln) - 1) for ln in lines]
        line_h = int(size * 1.3)
        box_w = int(max(widths)) + 2 * px(40)
        box_h = line_h * len(lines) + 2 * px(26)
        x0, y0 = (W - box_w) // 2, y1 - box_h
        panel(img, x0, y0, x0 + box_w, y1, px(28), (10, 12, 18), 0.80)
        y = y0 + px(26) + line_h // 2
        for ln, wd in zip(lines, widths):
            x = (W - wd) / 2
            for i, t, t_w in ln:
                is_last = i == len(toks) - 1
                color = ACCENT if (is_last and ui["age"] < HIGHLIGHT_SECONDS) else WHITE
                d.text((x, y), t, font=f, fill=color, anchor="lm")
                x += t_w + space
            y += line_h
        bar_w = int((box_w - 2 * px(40)) * max(0.0, 1 - ui["age"] / CAPTION_TIMEOUT))
        if bar_w > px(4):
            d.rounded_rectangle((x0 + px(40), y1 - px(17), x0 + px(40) + bar_w, y1 - px(13)),
                                radius=px(2), fill=(90, 98, 110))
    else:
        hint = "Faça um sinal com uma das mãos"
        f = font(px(26), False)
        box_w, box_h = int(d.textlength(hint, font=f) + px(56)), px(56)
        x0, y0 = (W - box_w) // 2, y1 - box_h
        panel(img, x0, y0, x0 + box_w, y1, box_h // 2, (10, 12, 18), 0.55)
        d.text((W // 2, (y0 + y1) // 2), hint, font=f, fill=(215, 220, 228), anchor="mm")

    # ---- diagnóstico (tecla D) ----
    if ui["debug"]:
        cand = lc.display_name(ui["candidate"]) if ui["candidate"] else "-"
        info = (f"{ui['fps']:.0f} fps | estado: {ui['state']} | candidato: {cand} | "
                f"confiança: {ui['confidence'] * 100:.0f}%")
        f = font(px(18), False)
        panel(img, px(14), bar_h + px(10), px(14) + d.textlength(info, font=f) + px(24),
              bar_h + px(48), px(10), (10, 12, 18), 0.75)
        d.text((px(26), bar_h + px(29)), info, font=f, fill=(200, 206, 214), anchor="lm")


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def run(args):
    if not (os.path.exists(FONT_BOLD) and os.path.exists(FONT_REGULAR)):
        print("[aviso] Fonte não encontrada em assets/fonts: usando fonte do sistema. "
              "Copie a pasta 'assets' do pacote para dentro do projeto.")
    model, labels = load_model()
    rec = AsyncRecognizer(lc.StreamRecognizer(model, labels, conf_min=args.confianca))
    speaker = Speaker(enabled=not args.sem_voz)
    print(speaker.message)
    captions = lc.CaptionBuffer(timeout=CAPTION_TIMEOUT)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit("Não consegui abrir a câmera. Tente outra com:  python main.py --camera 1")

    mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
    point_style = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=-1, circle_radius=3)
    line_style = mp_draw.DrawingSpec(color=(70, 210, 190), thickness=2)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1600, 900)
    fullscreen, debug, fps, last = False, args.debug, 0.0, time.time()

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("Falha ao ler a câmera.")
                break
            bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None
            rec.push(lc.extract_landmarks(hand))
            if hand is not None:
                mp_draw.draw_landmarks(rgb, hand, mp_hands.HAND_CONNECTIONS, point_style, line_style)

            now = time.time()
            for word in rec.pop_words():
                text = lc.display_name(word)
                captions.add(text, now)
                speaker.say(text)
            captions.prune(now)
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now

            img = Image.fromarray(rgb)
            draw_hud(img, {
                "words": captions.words(), "age": captions.last_age(now) or 0.0,
                "hand": hand is not None, "state": rec.state,
                "candidate": rec.candidate, "confidence": rec.confidence,
                "voice_on": speaker.enabled, "voice_available": speaker.available,
                "debug": debug, "fps": fps,
            })
            cv2.imshow(WINDOW, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                captions.clear()
                rec.reset()
            elif key == ord("v"):
                speaker.toggle()
            elif key == ord("d"):
                debug = not debug
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    rec.stop()
    speaker.close()
    cap.release()
    cv2.destroyAllWindows()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="LibraVersor - Libras em legenda e voz")
    ap.add_argument("--camera", type=int, default=0, help="índice da câmera (padrão 0)")
    ap.add_argument("--sem-voz", action="store_true", help="começa com a voz desligada")
    ap.add_argument("--debug", action="store_true", help="mostra o painel de diagnóstico")
    ap.add_argument("--confianca", type=float, default=0.60,
                    help="confiança mínima para aceitar um gesto (padrão 0.60)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())