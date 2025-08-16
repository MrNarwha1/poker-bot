"""
Poker Analyzer Overlay – Windows (PokerStars MVP)

New in this version:
- Windows screen capture (ROI) via `mss` running on a timer thread.
- Basic template matcher hooks for PokerStars: supports per-seat card crops, rank/suit templates, and pot/bet OCR via Tesseract.
- Range Editor: enter villain ranges (PokerStove-style) and simulate equity vs weighted ranges instead of random hands.
- Keeps manual entry as a fallback when detection is not yet calibrated.

DISCLAIMER: Many online poker sites restrict real-time tools/overlays. Use only where permitted. For PokerStars, check their latest third‑party tools policy and **do not** use during real‑money play if disallowed. This code is for educational/study purposes.

Dependencies (install with pip in your VS Code venv):
    pip install PySide6 numpy mss opencv-python pytesseract
    # optional
    pip install numba

Also install Tesseract OCR for Windows and note its install path:
- Download Windows installer (e.g., from UB Mannheim builds)
- Example path: C:\Program Files\Tesseract-OCR
- After installing: add that path to your System PATH or set TESSERACT_EXE in the UI.

Run:
    python poker_overlay.py

Pack to EXE (PyInstaller):
    pip install pyinstaller
    pyinstaller --noconfirm --onefile --windowed --name PokerOverlay \
        --add-data "templates;templates" poker_overlay.py

Folder layout suggestion:
    project/
      poker_overlay.py
      templates/  (put PNG templates for ranks/suits and optional UI icons)
         ranks/  (e.g., A.png, K.png, Q.png, ... , 2.png)
         suits/  (c.png, d.png, h.png, s.png)

Notes:
- The template matcher here is intentionally simple (single-scale normalized cross-correlation). Calibrate with good crops from your PokerStars theme and keep them small and crisp.
- The calibration panel lets you save/load a JSON profile with ROI + per-element sub-ROIs (hero cards, board slots, pot box).
- If detection isn't calibrated, the app still works with manual card/pot entry.
"""
from __future__ import annotations

import sys
import os
import json
import time
import random
import itertools
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from numba import int32
from numba.experimental import jitclass

import numpy as np

try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False

from PySide6 import QtCore, QtGui, QtWidgets

# ---- New imports for capture / OCR / CV ----
import mss
import cv2
import pytesseract

############################
# Card parsing / utilities #
############################
RANK_STR = "23456789TJQKA"
SUIT_STR = "cdhs"  # clubs, diamonds, hearts, spades
rank_map = {ch: i+2 for i, ch in enumerate(RANK_STR)}
rank_map.update({ch.lower(): i+2 for i, ch in enumerate(RANK_STR)})
suit_map = {ch: i for i, ch in enumerate(SUIT_STR)}
suit_map.update({ch.upper(): i for i, ch in enumerate(SUIT_STR)})
spec = [
    ('rank', int32), 
    ('suit', int32)
]

# @dataclass(frozen=True)
@jitclass(spec)
class Card: 
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    @staticmethod
    def from_str(s: str) -> 'Card':
        s = s.strip()
        if len(s) != 2:
            raise ValueError(f"Bad card string: {s}")
        r, u = s[0], s[1]
        if r not in rank_map or u not in suit_map:
            raise ValueError(f"Bad card characters: {s}")
        return Card(rank_map[r], suit_map[u])
    def __str__(self) -> str:
        return f"{RANK_STR[self.rank-2]}{SUIT_STR[self.suit]}"


def parse_cards(s: str) -> List[Card]:
    if not s.strip():
        return []
    parts = s.replace(',', ' ').split()
    return [Card.from_str(p) for p in parts]


def full_deck() -> List[Card]:
    return [Card(r, s) for r in range(2, 15) for s in range(4)]


def remove_cards(deck: List[Card], to_remove: List[Card]) -> List[Card]:
    remaining = deck.copy()
    for c in to_remove:
        remaining.remove(c)
    return remaining

###################################
# Hand ranking (5-card exact eval) #
###################################

def _evaluate_5_python(cards: Tuple[Card, Card, Card, Card, Card]) -> Tuple[int, Tuple[int, ...]]:
    ranks = sorted([c.rank for c in cards])
    suits = [c.suit for c in cards]
    counts = {r: ranks.count(r) for r in set(ranks)}
    unique = sorted(set(ranks))
    is_flush = len(set(suits)) == 1
    # Straight (wheel support)
    def straight_high(vals: List[int]) -> int:
        a = sorted(set(vals))
        if 14 in a:
            a.append(1)
        best = 0
        run = 1
        for i in range(1, len(a)):
            if a[i] == a[i-1] + 1:
                run += 1
                best = max(best, a[i]) if run >= 5 else best
            else:
                run = 1
        return best
    straight_top = straight_high(ranks)
    # Straight flush
    if is_flush:
        by_rank = sorted([c.rank for c in cards])
        sf_top = straight_high(by_rank)
        if sf_top:
            return (8, (sf_top,))
    # Quads
    if 4 in counts.values():
        quad = max([r for r, c in counts.items() if c == 4])
        kicker = max([r for r, c in counts.items() if c == 1])
        return (7, (quad, kicker))
    # Full house
    trips = sorted([r for r, c in counts.items() if c == 3], reverse=True)
    pairs = sorted([r for r, c in counts.items() if c == 2], reverse=True)
    if trips and (pairs or len(trips) >= 2):
        t = trips[0]
        p = pairs[0] if pairs else trips[1]
        return (6, (t, p))
    # Flush
    if is_flush:
        return (5, tuple(sorted(ranks, reverse=True)))
    # Straight
    if straight_top:
        return (4, (straight_top,))
    # Trips
    if trips:
        t = trips[0]
        kickers = sorted([r for r in ranks if r != t], reverse=True)[:2]
        return (3, (t, *kickers))
    # Two pair
    if len(pairs) >= 2:
        p1, p2 = pairs[:2]
        kicker = max([r for r in ranks if r != p1 and r != p2])
        return (2, (p1, p2, kicker))
    # One pair
    if len(pairs) == 1:
        p = pairs[0]
        kickers = sorted([r for r in ranks if r != p], reverse=True)[:3]
        return (1, (p, *kickers))
    # High card
    return (0, tuple(sorted(ranks, reverse=True)))

if NUMBA:
    _evaluate_5 = njit(_evaluate_5_python)  # type: ignore
else:
    _evaluate_5 = _evaluate_5_python


def evaluate_7(cards7: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    best = (-1, ())
    for combo in itertools.combinations(cards7, 5):
        score = _evaluate_5(combo)
        if score > best:
            best = score
    return best

############################
# Range parsing / sampling #
############################
# Supports basic PokerStove notation: e.g., "22+, A2s+, KTs+, QJs, AJo+"

RANKS_ORDER = "23456789TJQKA"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS_ORDER)}


def expand_range(token: str) -> List[Tuple[str, str]]:
    token = token.strip()
    hands: List[Tuple[str, str]] = []
    # Specific like 'AKs' or 'AQo'
    if len(token) in (2, 3):
        r1, r2 = token[0], token[1]
        suited_flag = token[2] if len(token) == 3 else ''
        if r1 == r2:  # pair like '77'
            hands += expand_pair(r1 + r2)
        else:
            if suited_flag == 's':
                hands += expand_nonpair(r1, r2, suited=True)
            elif suited_flag == 'o':
                hands += expand_nonpair(r1, r2, suited=False)
            else:  # unspecified: both suited and offsuit
                hands += expand_nonpair(r1, r2, suited=True)
                hands += expand_nonpair(r1, r2, suited=False)
        return hands
    # With +, e.g., 77+, ATs+, KQo+
    if token.endswith('+'):
        core = token[:-1]
        if len(core) == 2 and core[0] == core[1]:  # pairs like 77+
            start = core[0]
            idx = RANKS_ORDER.index(start)
            for r in RANKS_ORDER[idx:]:
                hands += expand_pair(r + r)
            return hands
        # non-pairs with s/o or unspecified
        if len(core) in (2,3):
            r1, r2 = core[0], core[1]
            flag = core[2] if len(core) == 3 else ''
            # Increase the kicker up to A
            start_idx = RANKS_ORDER.index(r2)
            for rlow in RANKS_ORDER[start_idx:]:
                if r1 == rlow:
                    continue
                if flag == 's':
                    hands += expand_nonpair(r1, rlow, suited=True)
                elif flag == 'o':
                    hands += expand_nonpair(r1, rlow, suited=False)
                else:
                    hands += expand_nonpair(r1, rlow, suited=True)
                    hands += expand_nonpair(r1, rlow, suited=False)
            return hands
    # Ranges like "A2s-A5s"
    if '-' in token:
        a, b = token.split('-', 1)
        if len(a) in (2,3) and len(b) in (2,3):
            r1a, r2a, fa = a[0], a[1], (a[2] if len(a)==3 else '')
            r1b, r2b, fb = b[0], b[1], (b[2] if len(b)==3 else '')
            if r1a != r1b or fa != fb:
                return []
            start = RANKS_ORDER.index(r2a)
            end = RANKS_ORDER.index(r2b)
            if start > end:
                start, end = end, start
            for idx in range(start, end+1):
                r2 = RANKS_ORDER[idx]
                if fa == 's':
                    hands += expand_nonpair(r1a, r2, suited=True)
                elif fa == 'o':
                    hands += expand_nonpair(r1a, r2, suited=False)
                else:
                    hands += expand_nonpair(r1a, r2, suited=True)
                    hands += expand_nonpair(r1a, r2, suited=False)
            return hands
    return hands


def expand_pair(pair: str) -> List[Tuple[str, str]]:
    r = pair[0]
    # All 6 pair combos regardless of suit
    suits = list(SUIT_STR)
    hands = []
    for i in range(4):
        for j in range(i+1,4):
            hands.append((r + suits[i], r + suits[j]))
    return hands


def expand_nonpair(r1: str, r2: str, suited: bool) -> List[Tuple[str, str]]:
    suits = list(SUIT_STR)
    hands: List[Tuple[str, str]] = []
    if suited:
        for s in suits:
            hands.append((r1 + s, r2 + s))
    else:
        for s1 in suits:
            for s2 in suits:
                if s1 == s2:
                    continue
                hands.append((r1 + s1, r2 + s2))
    return hands


def parse_range(range_text: str) -> List[Tuple[str, str]]:
    tokens = [t.strip() for t in range_text.replace(',', ' ').split() if t.strip()]
    combos: List[Tuple[str,str]] = []
    for t in tokens:
        combos += expand_range(t)
    # Deduplicate
    seen = set()
    uniq: List[Tuple[str,str]] = []
    for a,b in combos:
        key = tuple(sorted((a,b)))
        if key not in seen:
            seen.add(key); uniq.append((a,b))
    return uniq


def card_from_token(tok: str) -> Card:
    return Card.from_str(tok)


############################
# Monte Carlo (random/range)
############################

def simulate_equity_random(hole: List[Card], board: List[Card], n_opponents: int = 1, trials: int = 10000, rng: Optional[random.Random] = None) -> Tuple[float, float]:
    if rng is None:
        rng = random.Random()
    known = hole + board
    deck = full_deck()
    for c in known:
        try:
            deck.remove(c)
        except ValueError:
            raise ValueError(f"Duplicate card: {c}")
    wins = ties = 0
    for _ in range(trials):
        rng.shuffle(deck)
        draw_idx = 0
        board_drawn = list(board)
        for _j in range(5 - len(board)):
            board_drawn.append(deck[draw_idx]); draw_idx += 1
        opp_scores = []
        for _o in range(n_opponents):
            h = [deck[draw_idx], deck[draw_idx+1]]; draw_idx += 2
            opp_scores.append(evaluate_7(h + board_drawn))
        hero = evaluate_7(hole + board_drawn)
        best_opp = max(opp_scores)
        if hero > best_opp:
            wins += 1
        elif hero == best_opp:
            ties += 1
    total = float(trials)
    return wins/total, ties/total


def simulate_equity_vs_ranges(hole: List[Card], board: List[Card], ranges: List[List[Tuple[str,str]]], trials: int = 20000, rng: Optional[random.Random] = None) -> Tuple[float, float]:
    """Ranges: list of opponent ranges; each range is list of (c1,c2) string tokens like ('As','Kd').
    Sampling rejects combos that conflict with known cards or previously sampled opponent hands.
    """
    if rng is None:
        rng = random.Random()
    known = {str(c) for c in hole + board}
    wins = ties = 0
    for _ in range(trials):
        # Complete board from fresh deck minus known
        deck = full_deck()
        for c in hole + board:
            deck.remove(c)
        rng.shuffle(deck)
        draw_idx = 0
        board_drawn = list(board)
        for _j in range(5 - len(board)):
            board_drawn.append(deck[draw_idx]); draw_idx += 1
        # Sample hands from ranges
        used: set[str] = set(str(c) for c in hole + board)
        opp_scores = []
        ok = True
        sampled: List[List[Card]] = []
        for r in ranges:
            tries = 0
            while tries < 1000:
                c1t, c2t = r[rng.randrange(len(r))]
                c1 = card_from_token(c1t)
                c2 = card_from_token(c2t)
                s1, s2 = str(c1), str(c2)
                if s1 not in used and s2 not in used and s1 != s2:
                    used.add(s1); used.add(s2)
                    sampled.append([c1, c2])
                    break
                tries += 1
            if tries >= 1000:
                ok = False; break
        if not ok:
            continue
        for h in sampled:
            opp_scores.append(evaluate_7(h + board_drawn))
        hero = evaluate_7(hole + board_drawn)
        best_opp = max(opp_scores) if opp_scores else (-1, ())
        if hero > best_opp:
            wins += 1
        elif hero == best_opp:
            ties += 1
    total = float(trials)
    return wins/total, ties/total

############################
# Pot odds / EV helpers    #
############################

def pot_odds_to_call(pot_size: float, to_call: float) -> float:
    if to_call <= 0:
        return 0.0
    return to_call / (pot_size + to_call)


def decision_from_equity(equity: float, required: float) -> str:
    if equity > required + 1e-9:
        return "Call (+EV)"
    elif equity < required - 1e-9:
        return "Fold (-EV)"
    else:
        return "Indifferent (~EV)"

############################
# Calibration profile      #
############################
@dataclass
class Profile:
    roi: Tuple[int,int,int,int] = (100,100,800,450)
    hero_card_boxes: List[Tuple[int,int,int,int]] = None  # two boxes inside ROI
    board_boxes: List[Tuple[int,int,int,int]] = None      # five boxes inside ROI
    pot_box: Tuple[int,int,int,int] = None               # box inside ROI
    template_dir: str = "templates"
    tesseract_exe: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> 'Profile':
        data = json.loads(s)
        p = Profile()
        p.roi = tuple(data.get('roi', p.roi))
        p.hero_card_boxes = [tuple(t) for t in data.get('hero_card_boxes') or []]
        p.board_boxes = [tuple(t) for t in data.get('board_boxes') or []]
        pot = data.get('pot_box')
        p.pot_box = tuple(pot) if pot else None
        p.template_dir = data.get('template_dir', 'templates')
        p.tesseract_exe = data.get('tesseract_exe', '')
        return p

############################
# Template matcher / OCR   #
############################
class TemplateMatcher:
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.rank_tmpls: Dict[str, np.ndarray] = {}
        self.suit_tmpls: Dict[str, np.ndarray] = {}
        self.load_templates()

    def load_templates(self):
        ranks_dir = os.path.join(self.template_dir, 'ranks')
        suits_dir = os.path.join(self.template_dir, 'suits')
        def load_dir(d):
            m = {}
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.lower().endswith('.png'):
                        key = os.path.splitext(fn)[0]
                        img = cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            m[key] = img
            return m
        self.rank_tmpls = load_dir(ranks_dir)
        self.suit_tmpls = load_dir(suits_dir)

    def detect_rank_suit(self, img_gray: np.ndarray) -> Optional[Tuple[str,str,float]]:
        # Very simple: find best rank and best suit by max NCC score within the crop
        best_rank, br_score = None, -1
        best_suit, bs_score = None, -1
        for k, tmpl in self.rank_tmpls.items():
            if img_gray.shape[0] < tmpl.shape[0] or img_gray.shape[1] < tmpl.shape[1]:
                continue
            res = cv2.matchTemplate(img_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > br_score:
                br_score, best_rank = score, k
        for k, tmpl in self.suit_tmpls.items():
            if img_gray.shape[0] < tmpl.shape[0] or img_gray.shape[1] < tmpl.shape[1]:
                continue
            res = cv2.matchTemplate(img_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > bs_score:
                bs_score, best_suit = score, k
        if best_rank and best_suit:
            return best_rank.upper(), best_suit.lower(), min(br_score, bs_score)
        return None

    def card_from_crop(self, crop_bgr: np.ndarray) -> Optional[Card]:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        # optional binarization
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        det = self.detect_rank_suit(gray)
        if det is None:
            return None
        r, s, conf = det
        try:
            return Card.from_str(r + s)
        except Exception:
            return None

class OCRReader:
    def __init__(self, tesseract_exe: str = ""):
        if tesseract_exe:
            pytesseract.pytesseract.tesseract_cmd = tesseract_exe
    def read_number(self, crop_bgr: np.ndarray) -> Optional[float]:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        cfg = "--psm 7 -c tessedit_char_whitelist=$0123456789."
        txt = pytesseract.image_to_string(gray, config=cfg).strip()
        txt = txt.replace('$', '')
        try:
            return float(txt)
        except Exception:
            return None

############################
# Overlay UI & capture     #
############################
class DraggableROI(QtWidgets.QFrame):
    roiChanged = QtCore.Signal(QtCore.QRect)
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.SizeAllCursor))
        self.setStyleSheet("background: rgba(0, 150, 255, 0.10); border: 2px solid rgba(0,150,255,0.9); border-radius: 8px;")
        self.resize(800, 450)
        self._drag = False
        self._drag_pos = QtCore.QPoint()
    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = True
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()
    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        if self._drag:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            self.roiChanged.emit(self.geometry())
            e.accept()
    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = False
            self.roiChanged.emit(self.geometry())
            e.accept()

class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poker Analyzer Overlay (Windows / PokerStars)")
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_AlwaysStackOnTop)

        self.roi = DraggableROI(); self.roi.setParent(self); self.roi.move(100, 100)

        # Control panel
        self.panel = QtWidgets.QFrame(self)
        self.panel.setStyleSheet("background: rgba(20,20,28,0.95); color: #eaeaea; border-radius: 10px; border: 1px solid #333;")
        self.panel_layout = QtWidgets.QGridLayout(self.panel)
        row = 0
        self.panel_layout.addWidget(QtWidgets.QLabel("Hole (e.g., Ah Kh)"), row, 0)
        self.hole_edit = QtWidgets.QLineEdit(); self.panel_layout.addWidget(self.hole_edit, row, 1); row += 1
        self.panel_layout.addWidget(QtWidgets.QLabel("Board"), row, 0)
        self.board_edit = QtWidgets.QLineEdit(); self.panel_layout.addWidget(self.board_edit, row, 1); row += 1
        self.panel_layout.addWidget(QtWidgets.QLabel("Opponents"), row, 0)
        self.opps_spin = QtWidgets.QSpinBox(); self.opps_spin.setRange(0, 9); self.opps_spin.setValue(1)
        self.panel_layout.addWidget(self.opps_spin, row, 1); row += 1
        self.panel_layout.addWidget(QtWidgets.QLabel("Pot ($)"), row, 0)
        self.pot_edit = QtWidgets.QDoubleSpinBox(); self.pot_edit.setRange(0, 1e9); self.pot_edit.setDecimals(2)
        self.panel_layout.addWidget(self.pot_edit, row, 1); row += 1
        self.panel_layout.addWidget(QtWidgets.QLabel("To call ($)"), row, 0)
        self.call_edit = QtWidgets.QDoubleSpinBox(); self.call_edit.setRange(0, 1e9); self.call_edit.setDecimals(2)
        self.panel_layout.addWidget(self.call_edit, row, 1); row += 1
        self.panel_layout.addWidget(QtWidgets.QLabel("Trials"), row, 0)
        self.trials_spin = QtWidgets.QSpinBox(); self.trials_spin.setRange(100, 500000); self.trials_spin.setValue(10000)
        self.panel_layout.addWidget(self.trials_spin, row, 1); row += 1

        # Range Editor UI
        self.panel_layout.addWidget(QtWidgets.QLabel("Villain Ranges (comma/space separated)"), row, 0)
        self.ranges_edit = QtWidgets.QPlainTextEdit()
        self.ranges_edit.setPlaceholderText(
            "e.g., Player1: 22+, A2s+, KTs+\nPlayer2: 77+, AJo+, KQo"
        )
        self.ranges_edit.setFixedHeight(70)
        self.panel_layout.addWidget(self.ranges_edit, row, 1)
        row += 1

# Calibration controls
        self.calib_btn = QtWidgets.QPushButton("Calibration Wizard")
        self.panel_layout.addWidget(self.calib_btn, row, 0)
        self.calib_btn.clicked.connect(self.open_calibration)

        self.run_btn = QtWidgets.QPushButton("Run / Update")
        self.panel_layout.addWidget(self.run_btn, row, 1); row += 1

        # Stats label
        self.stats = QtWidgets.QLabel("–")
        self.stats.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.stats.setStyleSheet("font: 12pt 'Segoe UI';")
        self.panel_layout.addWidget(self.stats, row, 0, 1, 2)

        # Move/close
        self.close_btn = QtWidgets.QPushButton("×"); self.close_btn.setFixedSize(24,24)
        self.close_btn.setStyleSheet("background:#b33; color:white; border:none; border-radius:12px; font-weight:bold;")
        self.panel_layout.addWidget(self.close_btn, 0, 2)
        self.close_btn.clicked.connect(self.close)
        self._panel_drag = False; self._panel_drag_pos = QtCore.QPoint()
        self.panel.mousePressEvent = self._panel_mouse_press
        self.panel.mouseMoveEvent = self._panel_mouse_move
        self.panel.mouseReleaseEvent = self._panel_mouse_release

        # Capture & detection
        self.profile = Profile()
        self.tmatcher = TemplateMatcher(self.profile.template_dir)
        self.ocr = OCRReader(self.profile.tesseract_exe)
        self.mss_inst = mss.mss()
        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.timeout.connect(self.poll_frame)
        self.frame_timer.start(200)  # 5 FPS

        self.run_btn.clicked.connect(self.compute_update)
        self.roi.roiChanged.connect(lambda _: self.update_panel_geometry())
        self.update_panel_geometry()
        self.resize(1400, 900)

    # Panel dragging
    def _panel_mouse_press(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._panel_drag = True
            self._panel_drag_pos = e.globalPosition().toPoint() - self.panel.frameGeometry().topLeft()
            e.accept()
    def _panel_mouse_move(self, e: QtGui.QMouseEvent):
        if self._panel_drag:
            self.panel.move(e.globalPosition().toPoint() - self._panel_drag_pos)
            e.accept()
    def _panel_mouse_release(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._panel_drag = False
            e.accept()

    def update_panel_geometry(self):
        r = self.roi.geometry()
        desired = QtCore.QPoint(r.x() + r.width() + 16, r.y())
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        w, h = 420, 360
        x = min(max(desired.x(), screen.left()), screen.right() - w)
        y = min(max(desired.y(), screen.top()), screen.bottom() - h)
        self.panel.setGeometry(x, y, w, h)

    def grab_roi_bgr(self) -> Optional[np.ndarray]:
        r = self.roi.geometry()
        bbox = {"top": r.y(), "left": r.x(), "width": r.width(), "height": r.height()}
        try:
            img = np.asarray(self.mss_inst.grab(bbox))  # BGRA
            bgr = img[...,:3][:, :, ::-1]  # to BGR
            return bgr
        except Exception:
            return None

    def poll_frame(self):
        try:
            frame = self.grab_roi_bgr()
            if frame is None:
                return
            # If calibrated, attempt detection
            hero, board, pot_val = None, None, None
            if self.profile.hero_card_boxes and self.profile.board_boxes:
                hero = []
                for box in self.profile.hero_card_boxes:
                    x,y,w,h = box
                    crop = frame[y:y+h, x:x+w]
                    c = self.tmatcher.card_from_crop(crop)
                    if c:
                        hero.append(c)
                board = []
                for box in self.profile.board_boxes:
                    x,y,w,h = box
                    crop = frame[y:y+h, x:x+w]
                    c = self.tmatcher.card_from_crop(crop)
                    if c:
                        board.append(c)
            if self.profile.pot_box:
                x,y,w,h = self.profile.pot_box
                crop = frame[y:y+h, x:x+w]
                pot_val = self.ocr.read_number(crop)
        except Exception as e:
            import traceback; traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "poll_frame failed", str(e))
        # Push into UI (non-destructive: only fill if detected)
        try:
            if hero and len(hero) == 2:
                self.hole_edit.setText(f"{hero[0]} {hero[1]}")
            if board and 0 < len(board) <= 5:
                self.board_edit.setText(' '.join(str(c) for c in board))
            if pot_val is not None:
                self.pot_edit.setValue(float(pot_val))
        except Exception:
            pass

    def open_calibration(self):
        dlg = CalibrationDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.profile = dlg.get_profile()
            # reload helpers
            self.tmatcher = TemplateMatcher(self.profile.template_dir)
            self.ocr = OCRReader(self.profile.tesseract_exe)

    def compute_update(self):
        try:
            hole = parse_cards(self.hole_edit.text())
            board = parse_cards(self.board_edit.text())
            n_opps = int(self.opps_spin.value())
            trials = int(self.trials_spin.value())
            # Ranges parsing (each line = one opponent)
            ranges_text = [ln.strip() for ln in self.ranges_edit.toPlainText().splitlines() if ln.strip()]
            if ranges_text:
                ranges = [parse_range(ln.split(':',1)[-1]) for ln in ranges_text]
                win, tie = simulate_equity_vs_ranges(hole, board, ranges, trials)
            else:
                win, tie = simulate_equity_random(hole, board, n_opps, trials)
            equity = win + tie / (max(1, n_opps) + 1)
            pot = float(self.pot_edit.value())
            to_call = float(self.call_edit.value())
            req = pot_odds_to_call(pot, to_call)
            decision = decision_from_equity(equity, req)
            text = (
                f"Equity (win): {win*100:.2f}%"
                f"Equity (tie): {tie*100:.2f}%"
                f"Effective equity: {equity*100:.2f}%"
                f"Pot odds req.: {req*100:.2f}%"
                f"Decision: {decision}"
            )
            self.stats.setText(text)
        except Exception as e:
            import traceback; traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "compute_update failed", str(e))
            self.stats.setText(f"Error: {e}")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(0,0,0,40))

# ---- Calibration Dialog ----
class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Wizard")
        self.setMinimumSize(520, 500)
        layout = QtWidgets.QFormLayout(self)
        self.template_dir = QtWidgets.QLineEdit("templates")
        self.tesseract_exe = QtWidgets.QLineEdit("")
        self.load_btn = QtWidgets.QPushButton("Load Profile…")
        self.save_btn = QtWidgets.QPushButton("Save Profile…")
        self.pick_tess = QtWidgets.QPushButton("Browse…")
        self.pick_tess.clicked.connect(self._pick_tesseract)
        hl = QtWidgets.QHBoxLayout(); hl.addWidget(self.tesseract_exe); hl.addWidget(self.pick_tess)
        layout.addRow("Template dir", self.template_dir)
        layout.addRow("Tesseract exe", hl)
        self.hero_boxes_edit = QtWidgets.QLineEdit("")
        self.board_boxes_edit = QtWidgets.QLineEdit("")
        self.pot_box_edit = QtWidgets.QLineEdit("")
        layout.addRow("Hero card boxes [x,y,w,h; …]", self.hero_boxes_edit)
        layout.addRow("Board boxes [x,y,w,h; …]", self.board_boxes_edit)
        layout.addRow("Pot box [x,y,w,h]", self.pot_box_edit)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addRow(self.load_btn, self.save_btn)
        layout.addRow(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.load_btn.clicked.connect(self._load)
        self.save_btn.clicked.connect(self._save)

    def _pick_tesseract(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select tesseract.exe", "C:/Program Files", "Executables (*.exe)")
        if path:
            self.tesseract_exe.setText(path)

    def _parse_boxes(self, s: str) -> List[Tuple[int,int,int,int]]:
        # Format: x,y,w,h; x,y,w,h; ...
        out = []
        for seg in s.split(';'):
            seg = seg.strip()
            if not seg:
                continue
            parts = [int(p.strip()) for p in seg.split(',')]
            if len(parts) != 4:
                raise ValueError("Box must be x,y,w,h")
            out.append(tuple(parts))
        return out

    def get_profile(self) -> Profile:
        p = Profile()
        # ROI from parent ROI
        if isinstance(self.parent(), OverlayWindow):
            r = self.parent().roi.geometry()
            p.roi = (r.x(), r.y(), r.width(), r.height())
        p.template_dir = self.template_dir.text().strip() or 'templates'
        p.tesseract_exe = self.tesseract_exe.text().strip()
        p.hero_card_boxes = self._parse_boxes(self.hero_boxes_edit.text()) if self.hero_boxes_edit.text().strip() else []
        p.board_boxes = self._parse_boxes(self.board_boxes_edit.text()) if self.board_boxes_edit.text().strip() else []
        p.pot_box = None
        if self.pot_box_edit.text().strip():
            vals = [int(v.strip()) for v in self.pot_box_edit.text().split(',')]
            if len(vals) == 4:
                p.pot_box = tuple(vals)
        return p

    def _load(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load profile", "", "JSON (*.json)")
        if not fn:
            return
        with open(fn, 'r', encoding='utf-8') as f:
            p = Profile.from_json(f.read())
        self.template_dir.setText(p.template_dir)
        self.tesseract_exe.setText(p.tesseract_exe or "")
        self.hero_boxes_edit.setText('; '.join(','.join(map(str,b)) for b in (p.hero_card_boxes or [])))
        self.board_boxes_edit.setText('; '.join(','.join(map(str,b)) for b in (p.board_boxes or [])))
        if p.pot_box:
            self.pot_box_edit.setText(','.join(map(str,p.pot_box)))

    def _save(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save profile", "pokerstars_profile.json", "JSON (*.json)")
        if not fn:
            return
        p = self.get_profile()
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(p.to_json())

############################
# Entry point              #
############################

def main():
    app = QtWidgets.QApplication(sys.argv)
    # Workaround: ensure transparent window shows on Windows
    # app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    w = OverlayWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
