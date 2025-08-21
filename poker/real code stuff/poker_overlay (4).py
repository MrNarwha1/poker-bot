
from __future__ import annotations

import sys
import os
import json
import time
import random
import itertools
import re
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict

import numpy as np

# ---- NUMBA (optional) ----------------------------------------------------
try:
    from numba import int32, njit
    from numba.experimental import jitclass
    NUMBA = True
except Exception:
    NUMBA = False
    def njit(x): return x
    def jitclass(spec):
        def wrap(cls): return cls
        return wrap
    from numpy import int32  # type: ignore

# ---- GUI / capture / OCR / CV -------------------------------------------
from PySide6 import QtCore, QtGui, QtWidgets
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
    ('suit', int32),
]

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
    out = [Card.from_str(p) for p in parts]
    # reject dupes
    seen = set()
    for c in out:
        k = (c.rank, c.suit)
        if k in seen:
            raise ValueError("Duplicate card entered.")
        seen.add(k)
    return out

def full_deck() -> List[Card]:
    return [Card(r, s) for r in range(2, 15) for s in range(4)]

def _same_card(a: Card, b: Card) -> bool:
    return (a.rank == b.rank) and (a.suit == b.suit)

def remove_cards(deck: List[Card], to_remove: List[Card]) -> List[Card]:
    out: List[Card] = []
    for d in deck:
        found = False
        for r in to_remove:
            if _same_card(d, r):
                found = True
                break
        if not found:
            out.append(d)
    return out

def _has_duplicate_cards(cards: List[Card]) -> bool:
    n = len(cards)
    for i in range(n):
        for j in range(i + 1, n):
            if _same_card(cards[i], cards[j]):
                return True
    return False

###################################
# Hand ranking (5-card exact eval)#
###################################
def _evaluate_5_python(cards: Tuple) -> Tuple[int, np.ndarray]:
    rank_counts = np.zeros(15, dtype=np.int32)
    suit_counts = np.zeros(4, dtype=np.int32)
    card_ranks = np.empty(5, dtype=np.int32)
    for i in range(5):
        r = cards[i].rank
        s = cards[i].suit
        card_ranks[i] = r
        rank_counts[r] += 1
        suit_counts[s] += 1

    # sort card_ranks descending
    for i in range(4):
        max_idx = i
        for j in range(i + 1, 5):
            if card_ranks[j] > card_ranks[max_idx]:
                max_idx = j
        if max_idx != i:
            tmp = card_ranks[i]; card_ranks[i] = card_ranks[max_idx]; card_ranks[max_idx] = tmp

    is_flush = any(suit_counts[s] == 5 for s in range(4))

    is_straight = False
    straight_high = 0
    for start in range(14, 5 - 1, -1):
        ok = True
        for k in range(5):
            if rank_counts[start - k] == 0:
                ok = False; break
        if ok:
            is_straight = True
            straight_high = start
            break
    if not is_straight:
        if (rank_counts[14] > 0 and rank_counts[5] > 0 and
            rank_counts[4] > 0 and rank_counts[3] > 0 and
            rank_counts[2] > 0):
            is_straight = True
            straight_high = 5

    quads = 0
    trips = 0
    pairs = np.zeros(2, dtype=np.int32); npairs = 0
    singles = np.zeros(5, dtype=np.int32); nsingles = 0

    for r in range(14, 1, -1):
        c = rank_counts[r]
        if c == 4: quads = r
        elif c == 3: trips = r
        elif c == 2:
            if npairs < 2:
                pairs[npairs] = r; npairs += 1
        elif c == 1:
            singles[nsingles] = r; nsingles += 1

    kick = np.zeros(5, dtype=np.int32)

    if is_straight and is_flush:
        kick[0] = straight_high
        return 8, kick
    if quads:
        kick[0] = quads
        if nsingles > 0: kick[1] = singles[0]
        return 7, kick
    if trips and npairs >= 1:
        kick[0] = trips; kick[1] = pairs[0]
        return 6, kick
    if is_flush:
        for i in range(5): kick[i] = card_ranks[i]
        return 5, kick
    if is_straight:
        kick[0] = straight_high
        return 4, kick
    if trips:
        kick[0] = trips
        if nsingles >= 1: kick[1] = singles[0]
        if nsingles >= 2: kick[2] = singles[1]
        return 3, kick
    if npairs == 2:
        kick[0] = pairs[0]; kick[1] = pairs[1]
        if nsingles >= 1: kick[2] = singles[0]
        return 2, kick
    if npairs == 1:
        kick[0] = pairs[0]
        if nsingles >= 1: kick[1] = singles[0]
        if nsingles >= 2: kick[2] = singles[1]
        if nsingles >= 3: kick[3] = singles[2]
        return 1, kick

    for i in range(5): kick[i] = card_ranks[i]
    return 0, kick

_evaluate_5 = njit(_evaluate_5_python) if NUMBA else _evaluate_5_python

def evaluate_7(cards7: List[Card]) -> Tuple[int, np.ndarray]:
    best_rank: int = -1
    best_vals: Optional[np.ndarray] = None
    for combo in itertools.combinations(cards7, 5):
        r, v = _evaluate_5(combo)
        cur = (int(r), tuple(int(x) for x in v))
        prev = (int(best_rank), tuple(int(x) for x in best_vals) if best_vals is not None else ())
        if cur > prev:
            best_rank, best_vals = int(r), v
    return best_rank, (best_vals if best_vals is not None else np.array([], dtype=np.int32))

############################
# Range parsing / sampling #
############################
RANKS_ORDER = "23456789TJQKA"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS_ORDER)}

def expand_pair(pair: str) -> List[Tuple[str, str]]:
    r = pair[0]
    suits = list(SUIT_STR)
    hands = []
    for i in range(4):
        for j in range(i + 1, 4):
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
                if s1 == s2: continue
                hands.append((r1 + s1, r2 + s2))
    return hands

def expand_range(token: str) -> List[Tuple[str, str]]:
    token = token.strip()
    hands: List[Tuple[str, str]] = []
    if len(token) in (2, 3):
        r1, r2 = token[0], token[1]
        suited_flag = token[2] if len(token) == 3 else ''
        if r1 == r2:
            hands += expand_pair(r1 + r2)
        else:
            if suited_flag == 's':
                hands += expand_nonpair(r1, r2, suited=True)
            elif suited_flag == 'o':
                hands += expand_nonpair(r1, r2, suited=False)
            else:
                hands += expand_nonpair(r1, r2, suited=True)
                hands += expand_nonpair(r1, r2, suited=False)
        return hands
    if token.endswith('+'):
        core = token[:-1]
        if len(core) == 2 and core[0] == core[1]:
            start = core[0]
            idx = RANKS_ORDER.index(start)
            for r in RANKS_ORDER[idx:]:
                hands += expand_pair(r + r)
            return hands
        if len(core) in (2, 3):
            r1, r2 = core[0], core[1]
            flag = core[2] if len(core) == 3 else ''
            start_idx = RANKS_ORDER.index(r2)
            for rlow in RANKS_ORDER[start_idx:]:
                if r1 == rlow: continue
                if flag == 's': hands += expand_nonpair(r1, rlow, suited=True)
                elif flag == 'o': hands += expand_nonpair(r1, rlow, suited=False)
                else:
                    hands += expand_nonpair(r1, rlow, suited=True)
                    hands += expand_nonpair(r1, rlow, suited=False)
            return hands
    if '-' in token:
        a, b = token.split('-', 1)
        if len(a) in (2, 3) and len(b) in (2, 3):
            r1a, r2a, fa = a[0], a[1], (a[2] if len(a) == 3 else '')
            r1b, r2b, fb = b[0], b[1], (b[2] if len(b) == 3 else '')
            if r1a != r1b or fa != fb:
                return []
            start = RANKS_ORDER.index(r2a)
            end = RANKS_ORDER.index(r2b)
            if start > end: start, end = end, start
            for idx in range(start, end + 1):
                r2 = RANKS_ORDER[idx]
                if fa == 's': hands += expand_nonpair(r1a, r2, suited=True)
                elif fa == 'o': hands += expand_nonpair(r1a, r2, suited=False)
                else:
                    hands += expand_nonpair(r1a, r2, suited=True)
                    hands += expand_nonpair(r1a, r2, suited=False)
            return hands
    return hands

def parse_range(range_text: str) -> List[Tuple[str, str]]:
    tokens = [t.strip() for t in range_text.replace(',', ' ').split() if t.strip()]
    combos: List[Tuple[str, str]] = []
    for t in tokens: combos += expand_range(t)
    seen = set(); uniq: List[Tuple[str, str]] = []
    for a, b in combos:
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key); uniq.append((a, b))
    return uniq

def card_from_token(tok: str) -> Card:
    return Card.from_str(tok)

############################
# Monte Carlo
############################
def _cmp_score(score: Tuple[int, np.ndarray]) -> Tuple[int, Tuple[int, ...]]:
    r, v = score
    return (int(r), tuple(int(x) for x in v))

def simulate_equity_random(hole: List[Card], board: List[Card], n_opponents: int = 1, trials: int = 10000, rng: Optional[random.Random] = None) -> Tuple[float, float]:
    if rng is None: rng = random.Random()
    known = hole + board
    if _has_duplicate_cards(known):
        raise ValueError("Duplicate card in input.")
    deck = remove_cards(full_deck(), known)

    wins = ties = 0
    for _ in range(trials):
        rng.shuffle(deck)
        draw_idx = 0
        board_drawn = list(board)
        for _j in range(5 - len(board)):
            board_drawn.append(deck[draw_idx]); draw_idx += 1
        opp_scores = []
        for _o in range(n_opponents):
            h = [deck[draw_idx], deck[draw_idx + 1]]; draw_idx += 2
            opp_scores.append(evaluate_7(h + board_drawn))
        hero = evaluate_7(hole + board_drawn)
        best_opp = max(opp_scores, key=_cmp_score) if opp_scores else (-1, np.array([], dtype=np.int32))
        if _cmp_score(hero) > _cmp_score(best_opp): wins += 1
        elif _cmp_score(hero) == _cmp_score(best_opp): ties += 1
    total = float(trials)
    return wins / total, ties / total

def simulate_equity_vs_ranges(hole: List[Card], board: List[Card], ranges: List[List[Tuple[str, str]]], trials: int = 20000, rng: Optional[random.Random] = None) -> Tuple[float, float]:
    if rng is None: rng = random.Random()
    if _has_duplicate_cards(hole + board):
        raise ValueError("Duplicate card in input.")
    wins = ties = 0
    for _ in range(trials):
        deck = remove_cards(full_deck(), hole + board)
        rng.shuffle(deck)
        draw_idx = 0
        board_drawn = list(board)
        for _j in range(5 - len(board)):
            board_drawn.append(deck[draw_idx]); draw_idx += 1
        used: set[Tuple[int, int]] = set((c.rank, c.suit) for c in hole + board)
        opp_scores = []
        ok = True
        sampled: List[List[Card]] = []
        for r in ranges:
            tries = 0
            while tries < 1000:
                c1t, c2t = r[rng.randrange(len(r))]
                c1 = card_from_token(c1t); c2 = card_from_token(c2t)
                k1 = (c1.rank, c1.suit); k2 = (c2.rank, c2.suit)
                if k1 not in used and k2 not in used and k1 != k2:
                    used.add(k1); used.add(k2)
                    sampled.append([c1, c2]); break
                tries += 1
            if tries >= 1000:
                ok = False; break
        if not ok: continue
        for h in sampled:
            opp_scores.append(evaluate_7(h + board_drawn))
        hero = evaluate_7(hole + board_drawn)
        best_opp = max(opp_scores, key=_cmp_score) if opp_scores else (-1, np.array([], dtype=np.int32))
        if _cmp_score(hero) > _cmp_score(best_opp): wins += 1
        elif _cmp_score(hero) == _cmp_score(best_opp): ties += 1
    total = float(trials)
    return wins / total, ties / total

############################
# Pot odds / EV helpers    #
############################
def pot_odds_to_call(pot_size: float, to_call: float) -> float:
    if to_call <= 0: return 0.0
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
class CardBox:
    rank_box: Tuple[int, int, int, int]
    suit_box: Tuple[int, int, int, int]

@dataclass
class Profile:
    roi: Tuple[int, int, int, int] = (100, 100, 800, 450)
    hero_boxes: List[CardBox] = field(default_factory=list)
    board_boxes: List[CardBox] = field(default_factory=list)
    pot_box: Optional[Tuple[int, int, int, int]] = None
    template_dir: str = "templates"
    tesseract_exe: str = ""
    rank_psm: int = 7
    suit_min_score: float = 0.6
    auto_debug_on_fail: bool = True
    
    

    def to_json(self) -> str:
        d = asdict(self)
        d['hero_boxes'] = [asdict(cb) for cb in self.hero_boxes]
        d['board_boxes'] = [asdict(cb) for cb in self.board_boxes]
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> 'Profile':
        data = json.loads(s)
        p = Profile()
        p.roi = tuple(data.get('roi', p.roi))
        p.hero_boxes = [CardBox(tuple(cb['rank_box']), tuple(cb['suit_box'])) for cb in data.get('hero_boxes', [])]
        p.board_boxes = [CardBox(tuple(cb['rank_box']), tuple(cb['suit_box'])) for cb in data.get('board_boxes', [])]
        pot = data.get('pot_box')
        p.pot_box = tuple(pot) if pot else None
        p.template_dir = data.get('template_dir', 'templates')
        p.tesseract_exe = data.get('tesseract_exe', '')
        p.rank_psm = int(data.get('rank_psm', 7))
        p.suit_min_score = float(data.get('suit_min_score', 0.6))
        p.auto_debug_on_fail = bool(data.get('auto_debug_on_fail', True))
        
        return p

############################
# Debug saver              #
############################
class DebugSaver:
    def __init__(self, root_dir: str):
        self.set_dir(root_dir)

    def set_dir(self, root_dir: str):
        self.root = root_dir or "debug_grabs"
        os.makedirs(self.root, exist_ok=True)

    def save(self, name: str, img: np.ndarray):
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.root, f"{ts}_{name}.png")
        try: cv2.imwrite(path, img)
        except Exception: pass

############################
# Template matcher / OCR   #
############################
class TemplateMatcher:
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.suit_tmpls: Dict[str, np.ndarray] = {}
        self.load_templates()

    def load_templates(self, show_warning: bool = False):
        """Load grayscale suit templates from template_dir/suits.
        If show_warning=True, pop up a dialog if nothing was loaded."""
        m: Dict[str, np.ndarray] = {}
        suits_dir = os.path.join(self.template_dir, "suits")
        if os.path.isdir(suits_dir):
            for fn in os.listdir(suits_dir):
                path = os.path.join(suits_dir, fn)
                if not os.path.isfile(path): 
                    continue
                key = os.path.splitext(fn)[0].lower()
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    m[key] = img
        self.suit_tmpls = m

        # only show popup when explicitly asked
        if show_warning:
            if not self.suit_tmpls:
                QtWidgets.QMessageBox.warning(
                    None, "Suit Templates Missing",
                    f"No suit templates found in:\n{os.path.join(self.template_dir,'suits')}"
                )
            else:
                QtWidgets.QMessageBox.information(
                    None, "Suit Templates Loaded",
                    f"Loaded {len(self.suit_tmpls)} templates: {', '.join(self.suit_tmpls.keys())}"
                )

    
    def detect_suit(self, gray_roi: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.suit_tmpls:
            return None, -1.0

        best_suit, best_score = None, -1.0

        # normalize ROI a bit to improve contrast
        roi = cv2.equalizeHist(gray_roi)

        # try a few template scales around 60%..140%
        scales = [0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.4]
        for k, tmpl0 in self.suit_tmpls.items():
            for s in scales:
                h = max(6, int(tmpl0.shape[0] * s))
                w = max(6, int(tmpl0.shape[1] * s))
                tmpl = cv2.resize(tmpl0, (w, h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
                if roi.shape[0] < h or roi.shape[1] < w:
                    continue
                res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                score = float(res.max())
                if score > best_score:
                    best_score, best_suit = score, k

        # normalize names to c/d/h/s
        if best_suit in ('club', 'clubs'): best_suit = 'c'
        if best_suit in ('diamond', 'diamonds'): best_suit = 'd'
        if best_suit in ('heart', 'hearts'): best_suit = 'h'
        if best_suit in ('spade', 'spades'): best_suit = 's'
        if best_suit and best_suit not in 'cdhs':
            best_suit = best_suit[0] if best_suit[0] in 'cdhs' else None

        return best_suit, best_score





class OCRReader:
    def __init__(self, tesseract_exe: str = "", psm: int = 7):
        if tesseract_exe:
            pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        self.psm = psm  # kept, but we’ll also try a couple of best‑known PSMs

    def _normalize_rank_char(self, ch: str) -> Optional[str]:
        """Map common OCR confusions to a valid rank in AKQJT98765432."""
        ch = ch.strip().upper()
        if not ch:
            return None
        # Direct hits
        if ch in "AKQJT98765432":
            return ch
        # Common confusions
        if ch in {"1", "7", "I", "|", "L"}:  # T often read as 1/7/I
            return "T"
        if ch == "S":            # stylized '5'
            return "5"
        if ch == "B":            # sometimes reads a decorated '8'
            return "8"
        if ch == "Z":            # rare, treat as '2'
            return "2"
        if ch == "O":            # very rare; we do not have 'O', ignore
            return None
        return None

    def _binarize_variants(self, gray: np.ndarray) -> list[np.ndarray]:
        # OTSU (light & inverted), and adaptive. We’ll also return the raw upscaled.
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(otsu)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 6
        )
        return [gray, otsu, inv, adaptive]

    def read_rank(self, crop_bgr: np.ndarray) -> Tuple[Optional[str], float, np.ndarray]:
        """
        Robust single‑char OCR for rank.
        Strategy:
          - Convert to gray, equalize, upscale 2–3× (tesseract likes ~40–80px tall glyphs)
          - Try several binarizations + both polarities
          - Try PSM 10 (single char) then 8 (single word)
          - Map common misreads to valid ranks
        Returns: (rank_char or None, rough_confidence [0..1], best_image_used)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0, np.zeros((1,1), np.uint8)

        gray0 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray0 = cv2.equalizeHist(gray0)

        # Upscale (keep aspect). Aim for ~60 px height if smaller.
        h, w = gray0.shape
        target_h = 60 if h < 60 else h
        scale = max(2.0, target_h / max(1, h)) if h < 60 else 1.8  # small crops get more boost
        up = cv2.resize(
            gray0, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC
        )

        # Light denoise
        up = cv2.GaussianBlur(up, (3, 3), 0)

        candidates = self._binarize_variants(up)
        # Also push a gently eroded and dilated version to handle ink weight
        kernel = np.ones((2,2), np.uint8)
        candidates += [cv2.erode(candidates[1], kernel, 1), cv2.dilate(candidates[1], kernel, 1)]

        cfgs = [
            "--oem 3 --psm 10 -c tessedit_char_whitelist=AKQJT98765432",
            "--oem 3 --psm 8  -c tessedit_char_whitelist=AKQJT98765432",
        ]

        best_rank, best_conf, best_img = None, 0.0, candidates[0]
        for img in candidates:
            for cfg in cfgs:
                txt = pytesseract.image_to_string(img, config=cfg)
                txt = txt.strip().replace(" ", "")
                if not txt:
                    continue
                # take the first usable char
                for ch in txt:
                    norm = self._normalize_rank_char(ch)
                    if norm:
                        # we don't have true confidences here; set rough tiers
                        conf = 0.92 if norm in "AKQJT" else 0.88
                        if conf > best_conf:
                            best_rank, best_conf, best_img = norm, conf, img
                        break
                if best_rank and best_conf >= 0.90:
                    # good enough; bail early
                    break
            if best_rank and best_conf >= 0.90:
                break

        # If still nothing, one last inverted try of the best candidate:
        if best_rank is None:
            inv = cv2.bitwise_not(candidates[0])
            txt = pytesseract.image_to_string(inv, config=cfgs[0]).strip().replace(" ", "")
            for ch in txt:
                norm = self._normalize_rank_char(ch)
                if norm:
                    best_rank, best_conf, best_img = norm, 0.80, inv
                    break

        return best_rank, best_conf, best_img


############################
# Preprocessing helpers    #
############################
def preprocess_gray(bgr: np.ndarray, return_all: bool=False):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    if return_all:
        return [otsu, inv, adaptive, gray]
    return (otsu, gray, inv, adaptive)

############################
# Overlay UI & capture     #
############################
class DraggableROI(QtWidgets.QFrame):
    roiChanged = QtCore.Signal(QtCore.QRect)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.SizeAllCursor))
        self.setStyleSheet("background: rgba(255, 0, 0, 0.08); border: 2px solid rgba(255,0,0,0.8); border-radius: 8px;")
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
            self.roiChanged.emit(self.geometry()); e.accept()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = False
            self.roiChanged.emit(self.geometry()); e.accept()

# ---------- Profile Manager (quick load/save) ------------------------------
class ProfileManager(QtWidgets.QWidget):
    """Quick-select profiles by name from a folder (default ./profiles)."""
    profileSelected = QtCore.Signal(str)  # emits file path

    def __init__(self, parent=None, folder: Optional[str] = None):
        super().__init__(parent)
        self.folder = folder or os.path.join(os.getcwd(), "profiles")
        os.makedirs(self.folder, exist_ok=True)

        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(True)
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_load = QtWidgets.QPushButton("Load")
        self.btn_del  = QtWidgets.QPushButton("Delete")
        lay.addWidget(QtWidgets.QLabel("Profile:"))
        lay.addWidget(self.combo, 1)
        lay.addWidget(self.btn_save)
        lay.addWidget(self.btn_load)
        lay.addWidget(self.btn_del)
        self.btn_save.clicked.connect(self._save_clicked)
        self.btn_load.clicked.connect(self._load_clicked)
        self.btn_del.clicked.connect(self._delete_clicked)
        self.refresh()

    def _name_to_path(self, name: str) -> str:
        safe = re.sub(r'[^a-zA-Z0-9_\- ]+', '', name).strip() or "profile"
        return os.path.join(self.folder, f"{safe}.json")

    def refresh(self):
        names = []
        for fn in os.listdir(self.folder):
            if fn.lower().endswith(".json"):
                names.append(os.path.splitext(fn)[0])
        names.sort()
        self.combo.clear()
        self.combo.addItems(names)

    def _save_clicked(self):
        name = self.combo.currentText().strip() or "profile"
        self.profileSelected.emit(self._name_to_path(name))

    def _load_clicked(self):
        name = self.combo.currentText().strip()
        if not name:
            QtWidgets.QMessageBox.information(self, "Load Profile", "Type or select a profile name first.")
            return
        path = self._name_to_path(name)
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Load Profile", f"No profile named '{name}' found.")
            return
        self.profileSelected.emit(path)

    def _delete_clicked(self):
        name = self.combo.currentText().strip()
        if not name: return
        path = self._name_to_path(name)
        if os.path.isfile(path):
            os.remove(path)
            self.refresh()

# ---------- Overlay main window -------------------------------------------
class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poker Analyzer Overlay")
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_AlwaysStackOnTop)

        #


        self._draw_labels_on_roi = False
        self._last_vis = {
            "hero": [],   # list of dicts: {"rank": "A", "suit": "h", "rank_box": (x,y,w,h), "suit_box":(...)}
            "board": [],  # same structure for 5 board cards
            "pot": None   # float or None
        }



        self.roi = DraggableROI(); self.roi.setParent(self); self.roi.move(100, 100)

        # ---- Right control panel (sidebar layout) ----
        self.panel = QtWidgets.QFrame(self)
        # after: self.panel = QtWidgets.QFrame(self)
        self.panel.setObjectName("panel")

        # Solid matte-black panel with light-gray text
        self.panel.setStyleSheet("""
        QFrame#panel {
            background-color: #0f0f0f;       /* matte black */
            color: #e6e6e6;                  /* light gray text */
            border: 1px solid #222;
            border-radius: 10px;
        }
        QLabel { color: #e6e6e6; }

        QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #1a1a1a;
            color: #e6e6e6;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 4px;
        }
        QListWidget {
            background-color: #141414;
            color: #e6e6e6;
            border: 1px solid #333;
            border-radius: 6px;
        }

        QPushButton {
            background-color: #262626;
            color: #e6e6e6;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 6px 10px;
        }
        QPushButton:hover { background-color: #2e2e2e; }
        QPushButton:pressed { background-color: #1f1f1f; }

        QSplitter::handle {
            background-color: #222;
        }


        QGroupBox {
            color: #eaeaea;       /* light gray titles like "Hero1", "Board1" */
            border: 1px solid #333;
            border-radius: 6px;
            margin-top: 6px;
            padding-top: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        QFormLayout > QLabel {
            color: #eaeaea;       /* labels like "Crop" and "Reads" */
        }



        """)

        self.panel_layout = QtWidgets.QVBoxLayout(self.panel); self.panel_layout.setContentsMargins(12,12,12,12)

        # Top bar: title + close
        topbar = QtWidgets.QHBoxLayout()
        self.title_lbl = QtWidgets.QLabel("Poker Overlay")
        self.title_lbl.setStyleSheet("font-weight:600; font-size:16px;")
        self.close_btn = QtWidgets.QPushButton("×"); self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet("background:#b33; color:white; border:none; border-radius:12px; font-weight:bold;")
        self.close_btn.clicked.connect(self.close)
        topbar.addWidget(self.title_lbl); topbar.addStretch(1); topbar.addWidget(self.close_btn)
        self.panel_layout.addLayout(topbar)

        # Profile manager row
        self.prof_mgr = ProfileManager(self)
        self.prof_mgr.profileSelected.connect(self._profile_action)
        self.panel_layout.addWidget(self.prof_mgr)

        # Split: nav list + stacked pages
        split = QtWidgets.QSplitter()
        split.setOrientation(QtCore.Qt.Horizontal)

        # Nav
        self.nav = QtWidgets.QListWidget()
        self.nav.addItems(["Core", "Ranges", "Capture", "Visual"])

        self.nav.setFixedWidth(140)
        self.nav.currentRowChanged.connect(self._nav_changed)
        split.addWidget(self.nav)

        # Pages
        self.pages = QtWidgets.QStackedWidget()
        split.addWidget(self.pages)
        split.setStretchFactor(1, 1)

        # Build pages
        self._build_core_page()
        self._build_ranges_page()
        self._build_capture_page()
        self._build_visual_page()

        

        self.panel_layout.addWidget(split, 1)

        # Stats
        self.stats = QtWidgets.QLabel("–")
        self.stats.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.panel_layout.addWidget(self.stats)

        # make draggable
        self._panel_drag = False; self._panel_drag_pos = QtCore.QPoint()
        self.panel.mousePressEvent = self._panel_mouse_press
        self.panel.mouseMoveEvent = self._panel_mouse_move
        self.panel.mouseReleaseEvent = self._panel_mouse_release

        # Capture & detection
        self.profile = Profile()
        self.tmatcher = TemplateMatcher(self.profile.template_dir)
        self.ocr = OCRReader(self.profile.tesseract_exe, psm=self.profile.rank_psm)
        self.debug = DebugSaver("debug_grabs")
        self.mss_inst = mss.mss()

        # Timers
        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.timeout.connect(self.poll_frame)
        self.frame_timer.start(350)  # was 200ms -> 350ms to reduce load

        self.compute_timer = QtCore.QTimer(self)
        self.compute_timer.timeout.connect(self.compute_update)

    
        self._computing = False                 # guard against overlapping Monte Carlos
        self._debug_sec_epoch = int(time.time())
        self._debug_saves_this_sec = 0          # rate-limit debug PNGs
        self._last_pot_ocr_t = 0.0              # reduce how often we OCR the pot



        # Hooks
        self._last_inputs: Optional[Tuple] = None
        self.update_panel_geometry()
        self.resize(1500, 950)
        self.nav.setCurrentRow(0)  # default page

    def _debug_save(self, name: str, img: np.ndarray):
        # Only save if the UI checkbox is ON and the profile allows auto-fail dumps
        if not (self.debug_cb.isChecked() and self.profile.auto_debug_on_fail):
            return
        t = time.time()
        sec = int(t)
        if sec != self._debug_sec_epoch:
            self._debug_sec_epoch = sec
            self._debug_saves_this_sec = 0
        # Cap to 8 saves/sec total to avoid disk thrash
        if self._debug_saves_this_sec >= 8:
            return
        self._debug_saves_this_sec += 1
        self.debug.save(name, img)  # NOTE: pass bare name (no .png)



    def _build_visual_page(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)

        # controls
        top = QtWidgets.QHBoxLayout()
        self.draw_labels_cb = QtWidgets.QCheckBox("Draw labels on ROI")
        self.draw_labels_cb.toggled.connect(self._set_draw_labels)
        top.addWidget(self.draw_labels_cb)
        top.addStretch(1)
        v.addLayout(top)

        # grid of previews
        grid = QtWidgets.QGridLayout()
        v.addLayout(grid, 1)

        def make_card_widgets(title):
            box = QtWidgets.QGroupBox(title)
            fl = QtWidgets.QFormLayout(box)
            rank_img = QtWidgets.QLabel(); rank_img.setFixedSize(90, 70); rank_img.setFrameShape(QtWidgets.QFrame.Box)
            suit_img = QtWidgets.QLabel(); suit_img.setFixedSize(90, 70); suit_img.setFrameShape(QtWidgets.QFrame.Box)
            status   = QtWidgets.QLabel("—")
            status.setStyleSheet("color: #bbb;")
            status.setMinimumWidth(90)
            fl.addRow("Rank", rank_img)
            fl.addRow("Suit", suit_img)
            fl.addRow("Reads", status)
            return box, rank_img, suit_img, status

        self._vis = {
            "hero": [],
            "board": [],
            "pot_img": QtWidgets.QLabel(),
            "pot_status": QtWidgets.QLabel("—")
        }

        # Hero 1/2
        for i in (1, 2):
            gb, rimg, simg, stat = make_card_widgets(f"Hero{i}")
            self._vis["hero"].append({"rimg": rimg, "simg": simg, "stat": stat})
            grid.addWidget(gb, 0, i-1)

        # Board 1..5
        row = 1
        col = 0
        for i in range(1, 6):
            gb, rimg, simg, stat = make_card_widgets(f"Board{i}")
            self._vis["board"].append({"rimg": rimg, "simg": simg, "stat": stat})
            grid.addWidget(gb, row + (i-1)//3, (i-1)%3)
        # Pot preview
        pot_box = QtWidgets.QGroupBox("Pot")
        fl = QtWidgets.QFormLayout(pot_box)
        self._vis["pot_img"].setFixedSize(160, 70)
        self._vis["pot_img"].setFrameShape(QtWidgets.QFrame.Box)
        fl.addRow("Crop", self._vis["pot_img"])
        fl.addRow("Reads", self._vis["pot_status"])
        grid.addWidget(pot_box, 0, 2)

        self.pages.addWidget(page)



    def _set_draw_labels(self, on: bool):
        self._draw_labels_on_roi = bool(on)
        self.update()  # trigger repaint

    def _to_qpix(self, bgr: np.ndarray, max_w: int, max_h: int) -> QtGui.QPixmap:
        if bgr is None or bgr.size == 0:
            return QtGui.QPixmap()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg)
        pm = pm.scaled(max_w, max_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        return pm



    # ---- Build individual pages --------------------------------------
    def _build_core_page(self):
        page = QtWidgets.QWidget(); lay = QtWidgets.QFormLayout(page)

        self.hole_edit = QtWidgets.QLineEdit()
        self.board_edit = QtWidgets.QLineEdit()
        self.opps_spin = QtWidgets.QSpinBox(); self.opps_spin.setRange(0, 9); self.opps_spin.setValue(1)
        self.pot_edit = QtWidgets.QDoubleSpinBox(); self.pot_edit.setRange(0, 1e9); self.pot_edit.setDecimals(2)
        self.call_edit = QtWidgets.QDoubleSpinBox(); self.call_edit.setRange(0, 1e9); self.call_edit.setDecimals(2)
        self.trials_spin = QtWidgets.QSpinBox(); self.trials_spin.setRange(100, 500000); self.trials_spin.setValue(10000)

        self.run_btn = QtWidgets.QPushButton("Run / Update (Ctrl+R)"); self.run_btn.clicked.connect(lambda: self.compute_update(force=True))
        self.auto_cb = QtWidgets.QCheckBox("Auto update (Ctrl+Shift+A)"); self.auto_cb.setChecked(False)
        self.auto_ms = QtWidgets.QSpinBox(); self.auto_ms.setRange(100, 10000); self.auto_ms.setSingleStep(100); self.auto_ms.setValue(1000)
        self.auto_ms.valueChanged.connect(lambda ms: self.compute_timer.setInterval(ms))
        self.auto_cb.toggled.connect(lambda on: (self.compute_timer.start(self.auto_ms.value()) if on else self.compute_timer.stop()))

        lay.addRow("Hole (e.g., Ah Kh)", self.hole_edit)
        lay.addRow("Board", self.board_edit)
        lay.addRow("Opponents", self.opps_spin)
        lay.addRow("Pot ($)", self.pot_edit)
        lay.addRow("To call ($)", self.call_edit)
        lay.addRow("Trials", self.trials_spin)
        lay.addRow(self.run_btn)
        lay.addRow("Auto interval (ms)", self.auto_ms)
        lay.addRow(self.auto_cb)

        self.pages.addWidget(page)

        # recompute on edits
        self.hole_edit.textChanged.connect(self._mark_dirty)
        self.board_edit.textChanged.connect(self._mark_dirty)
        self.opps_spin.valueChanged.connect(self._mark_dirty)
        self.trials_spin.valueChanged.connect(self._mark_dirty)
        self.pot_edit.valueChanged.connect(self._mark_dirty)
        self.call_edit.valueChanged.connect(self._mark_dirty)

    def _build_ranges_page(self):
        page = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(page)
        v.addWidget(QtWidgets.QLabel("Villain Ranges (one per line; 'Player: 22+, A2s+, KTs+')"))
        self.ranges_edit = QtWidgets.QPlainTextEdit(); self.ranges_edit.setFixedHeight(150)
        v.addWidget(self.ranges_edit, 1)
        self.pages.addWidget(page)
        self.ranges_edit.textChanged.connect(self._mark_dirty)

    def _build_capture_page(self):
        page = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(page)
        self.calib_btn = QtWidgets.QPushButton("Calibration Wizard (Ctrl+K)")
        self.calib_btn.clicked.connect(self.open_calibration)

        self.debug_cb = QtWidgets.QCheckBox("Save debug screenshots")
        self.debug_dir_btn = QtWidgets.QPushButton("Pick debug dir…")
        self.debug_dir_btn.clicked.connect(self._pick_debug_dir)

        form.addRow(self.calib_btn)
        form.addRow(self.debug_cb, self.debug_dir_btn)
        self.pages.addWidget(page)


    # ---- Navigation / panel geometry / dragging ----------------------
    def _nav_changed(self, idx: int):
        self.pages.setCurrentIndex(idx)

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
        w, h = 560, 520
        x = min(max(desired.x(), screen.left()), screen.right() - w)
        y = min(max(desired.y(), screen.top()), screen.bottom() - h)
        self.panel.setGeometry(x, y, w, h)

    

    
    # ---- Debug / dirs -------------------------------------------------
    def _pick_debug_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select debug directory", "")
        if path:
            self.debug.set_dir(path)

    # ---- Quick profile actions from ProfileManager --------------------
    def _profile_action(self, path: str):
        # Determine whether this came from save or load by checking sender button text isn't provided;
        # we just inspect file existence pre/post.
        sender = self.sender()
        if isinstance(sender, ProfileManager):
            # Not needed.
            pass
        # If the file exists -> LOAD. Else -> SAVE (using current profile fields).
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    p = Profile.from_json(f.read())
                self._apply_loaded_profile(p)
                QtWidgets.QMessageBox.information(self, "Profile Loaded", os.path.basename(path))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Profile Load Failed", str(e))
        else:
            try:
                # sync current ROI into profile
                r = self.roi.geometry()
                self.profile.roi = (r.x(), r.y(), r.width(), r.height())
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.profile.to_json())
                self.prof_mgr.refresh()
                QtWidgets.QMessageBox.information(self, "Profile Saved", os.path.basename(path))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Profile Save Failed", str(e))

    def _apply_loaded_profile(self, p: Profile):
        self.profile = p
        # Apply ROI geometry to the draggable ROI widget
        if p.roi and len(p.roi) == 4:
            x, y, w, h = p.roi
            self.roi.setGeometry(x, y, w, h)
            self.update_panel_geometry()
        # Rebuild detectors according to the loaded profile
        self.tmatcher = TemplateMatcher(p.template_dir)
        self.ocr = OCRReader(p.tesseract_exe, psm=p.rank_psm)
        
        
        # Force recompute
        self._mark_dirty()
        self.compute_update(force=True)

    # ---- Computation / capture ---------------------------------------
    def grab_roi_bgr(self) -> Optional[np.ndarray]:
        r = self.roi.geometry()
        bbox = {"top": r.y(), "left": r.x(), "width": r.width(), "height": r.height()}
        try:
            img = np.asarray(self.mss_inst.grab(bbox))  # BGRA
            bgr = img[..., :3]
            return bgr
        except Exception:
            return None

    def _detect_card_from_boxes(self, frame: np.ndarray, box: CardBox) -> Tuple[Optional[Card], float]:
        def crop_pad(img, x, y, w, h, pad=3):
            H, W = img.shape[:2]
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
            return img[y0:y1, x0:x1]

        x,y,w,h = box.rank_box
        rank_crop = crop_pad(frame, x,y,w,h, pad=3)
        x2,y2,w2,h2 = box.suit_box
        suit_crop = crop_pad(frame, x2,y2,w2,h2, pad=2)

        if rank_crop.size == 0 or suit_crop.size == 0:
            return None, 0.0

        rank, rconf, bin_rank = self.ocr.read_rank(rank_crop)
        suit_gray = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
        suit, sconf = self.tmatcher.detect_suit(suit_gray)

        # Debug on fail (rate-limited & gated)
        if suit is None or sconf < self.profile.suit_min_score:
            self._debug_save("fail_rank", rank_crop)
            self._debug_save("fail_suit", suit_crop)
            self._debug_save("fail_rank_bin", bin_rank)
            return None, 0.0
        if not rank:
            self._debug_save("fail_rank", rank_crop)
            self._debug_save("fail_suit_hit", suit_crop)
            return None, 0.0

        try:
            c = Card.from_str(rank + suit)
            return c, min(max(rconf, 0.0), 1.0) * min(max(sconf, 0.0), 1.0)
        except Exception:
            return None, 0.0


    def poll_frame(self):
        try:
            frame = self.grab_roi_bgr()
            if frame is None:
                return

            # --- optional debug grab ---
            if self.debug_cb.isChecked():
                self.debug.save("roi", frame)

            # --- run detections for this frame ---
            cur_hero, cur_board, cur_pot = [], [], None

            for cb in self.profile.hero_boxes:
                c, _ = self._detect_card_from_boxes(frame, cb)
                if c:
                    cur_hero.append(str(c))

            for cb in self.profile.board_boxes:
                c, _ = self._detect_card_from_boxes(frame, cb)
                if c:
                    cur_board.append(str(c))

            if self.profile.pot_box:
                x, y, w, h = self.profile.pot_box
                crop = frame[y:y + h, x:x + w]
                if self.debug_cb.isChecked():
                    self.debug.save("pot_raw", crop)
                    for idx, var in enumerate(preprocess_gray(crop, return_all=True)):
                        self.debug.save(f"pot_pre_{idx}", var)
                cur_pot = self._ocr_money(crop)

        except Exception as e:
            import traceback; traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "poll_frame failed", str(e))
            return

        # --------- debounce / stability buffers ----------
        try:
            from collections import deque
            import statistics

            # create buffers lazily (so you can just paste this function)
            if not hasattr(self, "_buf_hole"):
                self._buf_hole = deque(maxlen=5)   # each item: tuple(str,str) or ()
            if not hasattr(self, "_buf_board"):
                self._buf_board = deque(maxlen=5)  # each item: tuple of 0..5 strs
            if not hasattr(self, "_buf_pot"):
                self._buf_pot = deque(maxlen=7)    # each item: float or None

            # normalize current reads for buffering
            hole_tuple = tuple(cur_hero) if len(cur_hero) == 2 else tuple()
            board_tuple = tuple(cur_board) if 0 < len(cur_board) <= 5 else tuple()

            self._buf_hole.append(hole_tuple)
            self._buf_board.append(board_tuple)
            self._buf_pot.append(cur_pot if isinstance(cur_pot, (int, float)) else None)

            # helpers
            def most_common_nonempty(seq):
                from collections import Counter
                nonempty = [s for s in seq if s]  # filter () / None
                if not nonempty:
                    return None, 0
                c = Counter(nonempty).most_common(1)[0]
                return c[0], c[1]

            # require at least this many repeats in the buffer
            REQ_HOLE = 2
            REQ_BOARD = 2

            stable_hole, n_hole = most_common_nonempty(list(self._buf_hole))
            stable_board, n_board = most_common_nonempty(list(self._buf_board))

            # pot: rolling median of available numbers (ignore None)
            pot_vals = [p for p in self._buf_pot if isinstance(p, (int, float))]
            stable_pot = statistics.median(pot_vals) if len(pot_vals) >= 3 else (pot_vals[-1] if pot_vals else None)

            changed = False

            # ---- update hole when stable ----
            if stable_hole and n_hole >= REQ_HOLE and len(stable_hole) == 2:
                new_hole_text = f"{stable_hole[0]} {stable_hole[1]}"
                if self.hole_edit.text() != new_hole_text:
                    self.hole_edit.setText(new_hole_text)
                    changed = True

            # ---- update board when stable ----
            if stable_board and n_board >= REQ_BOARD:
                new_board_text = " ".join(stable_board)
                if self.board_edit.text() != new_board_text:
                    self.board_edit.setText(new_board_text)
                    changed = True

            # ---- update pot when we have a smoothed value ----
            if stable_pot is not None:
                # avoid thrashing for tiny OCR jitter
                current = float(self.pot_edit.value())
                if abs(current - float(stable_pot)) >= 0.25:  # tolerance
                    self.pot_edit.setValue(float(stable_pot))
                    changed = True

            if changed:
                self._mark_dirty()

        except Exception:
            # never crash the polling loop for UI update issues
            pass


    def _ocr_money(self, crop_bgr: np.ndarray) -> Optional[float]:
        # Only OCR at most ~2 times/sec
        now = time.time()
        if now - self._last_pot_ocr_t < 0.5:
            return None
        self._last_pot_ocr_t = now

        if crop_bgr is None or crop_bgr.size == 0:
            return None
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        h, w = gray.shape
        scale = 2.0 if h < 60 else 1.5
        up = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        up = cv2.GaussianBlur(up, (3,3), 0)

        variants = []
        _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        variants.append(cv2.bitwise_not(otsu))
        variants.append(cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 31, 6))

        cfgs = [
            "--oem 3 --psm 7 -c tessedit_char_whitelist=$0123456789.,",
            "--oem 3 --psm 6 -c tessedit_char_whitelist=$0123456789.,",
        ]
        best_val, best_len = None, -1
        for img in variants:
            for cfg in cfgs:
                txt = pytesseract.image_to_string(img, config=cfg).strip()
                if not txt:
                    continue
                txt = txt.replace(" ", "").replace(",", "").replace("$", "")
                # choose the longest numeric chunk we can parse
                m_all = re.findall(r"\d+(?:\.\d{1,2})?", txt)
                for m in m_all:
                    try:
                        val = float(m)
                        if len(m) > best_len:
                            best_val, best_len = val, len(m)
                    except Exception:
                        pass
            if best_val is not None:
                break
        return best_val



    def _collect_inputs(self) -> Tuple:
        return (
            self.hole_edit.text().strip(),
            self.board_edit.text().strip(),
            int(self.opps_spin.value()),
            int(self.trials_spin.value()),
            float(self.pot_edit.value()),
            float(self.call_edit.value()),
            self.ranges_edit.toPlainText().strip(),
        )

    def _mark_dirty(self):
        self._last_inputs = None

    def open_calibration(self):
        dlg = CalibrationDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.profile = dlg.get_profile()
            self.tmatcher = TemplateMatcher(self.profile.template_dir)
            self.ocr = OCRReader(self.profile.tesseract_exe, psm=self.profile.rank_psm)
            
            self._mark_dirty()

    def compute_update(self, force: bool = False):
        if self._computing:  # skip if a run is in progress
            return
        self._computing = True
        try:
            current_inputs = self._collect_inputs()
            if not force and self._last_inputs == current_inputs:
                return

            try:
                hole = parse_cards(self.hole_edit.text())
                board = parse_cards(self.board_edit.text())
            except Exception as pe:
                self.stats.setText(f"Input error: {pe}")
                self._last_inputs = None
                return

            if len(hole) != 2:
                self.stats.setText("Waiting for capture… (need 2 hero cards)")
                self._last_inputs = None
                return

            if _has_duplicate_cards(hole + board):
                self.stats.setText("Duplicate cards detected; waiting for clean capture…")
                self._last_inputs = None
                return

            n_opps = int(self.opps_spin.value())
            trials = int(self.trials_spin.value())

            ranges_text = [ln.strip() for ln in self.ranges_edit.toPlainText().splitlines() if ln.strip()]
            if ranges_text:
                ranges = [parse_range(ln.split(':', 1)[-1]) for ln in ranges_text]
                if any(len(r) == 0 for r in ranges):
                    self.stats.setText("Range parse error in one of the lines.")
                    self._last_inputs = None
                    return
                win, tie = simulate_equity_vs_ranges(hole, board, ranges, trials)
            else:
                win, tie = simulate_equity_random(hole, board, n_opps, trials)

            equity = win + (tie / max(1, (n_opps + 1)))
            pot = float(self.pot_edit.value())
            to_call = float(self.call_edit.value())
            req = pot_odds_to_call(pot, to_call)
            decision = decision_from_equity(equity, req)

            text = (
                f"Equity (win): {win*100:.2f}%\n"
                f"Equity (tie): {tie*100:.2f}%\n"
                f"Effective equity: {equity*100:.2f}%\n"
                f"Pot odds req.: {req*100:.2f}%\n"
                f"Decision: {decision}"
            )
            self.stats.setText(text)
            self._last_inputs = current_inputs
        except Exception as e:
            import traceback; traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "compute_update failed", str(e))
            self.stats.setText(f"Error: {e}")
        finally:
            self._computing = False


    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        # draw nothing; panel is separate, ROI frame draws itself
        pass

# ---- Calibration Dialog (drag & drop) ----
class BoxItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, rect: QtCore.QRectF, label: str, color: QtGui.QColor):
        super().__init__(rect)
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemIsSelectable |
                      QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.color = color
        self.label = label
        self.handle_size = 8.0
        self.setZValue(10)

    def paint(self, painter: QtGui.QPainter, option, widget=None):
        pen = QtGui.QPen(self.color, 2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(self.color.red(), self.color.green(), self.color.blue(), 50))
        painter.drawRect(self.rect())
        painter.setPen(QtGui.QPen(QtGui.QColor("white")))
        painter.drawText(self.rect().adjusted(2,2,-2,-2), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, self.label)
        r = self.rect()
        for p in [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]:
            painter.fillRect(QtCore.QRectF(p.x()-self.handle_size/2, p.y()-self.handle_size/2, self.handle_size, self.handle_size), QtGui.QBrush(self.color.darker()))

class CalibrationCanvas(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.bg_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self.mode = 'idle'
        self.draw_color = QtGui.QColor(255, 60, 60)  # RED for unlabeled
        self.start_pos: Optional[QtCore.QPointF] = None
        self.current_item: Optional[BoxItem] = None

    def set_background(self, bgr_img: np.ndarray):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.scene.clear()
        self.bg_item = self.scene.addPixmap(pix)
        self.bg_item.setZValue(0)
        self.setSceneRect(QtCore.QRectF(0,0,w,h))

    def add_box(self, rect: QtCore.QRectF, label: str, color: QtGui.QColor) -> BoxItem:
        item = BoxItem(rect, label, color)
        self.scene.addItem(item)
        return item

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton and (e.modifiers() & QtCore.Qt.ControlModifier):
            self.mode = 'draw'
            self.start_pos = self.mapToScene(e.position().toPoint())
            self.current_item = self.add_box(QtCore.QRectF(self.start_pos, self.start_pos), "Unlabeled", self.draw_color)
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.mode == 'draw' and self.current_item and self.start_pos is not None:
            cur = self.mapToScene(e.position().toPoint())
            rect = QtCore.QRectF(self.start_pos, cur).normalized()
            self.current_item.setRect(rect)
            e.accept()
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton and self.mode == 'draw':
            self.mode = 'idle'
            self.start_pos = None
            e.accept()
        else:
            super().mouseReleaseEvent(e)

class CalibrationDialog(QtWidgets.QDialog):
    HELP_TEXT = (
        "Instructions:\n"
        "1) Move/resize the main ROI in the overlay before opening this wizard.\n"
        "2) Click 'Snapshot ROI' to capture a still frame.\n"
        "3) Hold CTRL and drag to draw boxes on the canvas.\n"
        "   - Label the selected box as: Hero Rank/Suit #1/#2, Board Rank/Suit #1..#5, or Pot.\n"
        "4) Save/Load profiles as JSON.\n"
        "5) Set Tesseract path if needed, template folder for suits, thresholds, etc."
    )

    VALID_LABELS = [
        "Hero1 Rank", "Hero1 Suit", "Hero2 Rank", "Hero2 Suit",
        "Board1 Rank", "Board1 Suit", "Board2 Rank", "Board2 Suit",
        "Board3 Rank", "Board3 Suit", "Board4 Rank", "Board4 Suit", "Board5 Rank", "Board5 Suit",
        "Pot"
    ]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Wizard (drag boxes)")
        self.setMinimumSize(980, 720)

        layout = QtWidgets.QVBoxLayout(self)

        # Top controls
        top = QtWidgets.QHBoxLayout()
        self.template_dir = QtWidgets.QLineEdit("templates")
        self.tesseract_exe = QtWidgets.QLineEdit("")
        self.browse_tess = QtWidgets.QPushButton("Browse Tesseract…")
        self.browse_tmpl = QtWidgets.QPushButton("Browse Templates…")
        top.addWidget(QtWidgets.QLabel("Templates:")); top.addWidget(self.template_dir)
        top.addWidget(self.browse_tmpl)
        top.addSpacing(16)
        top.addWidget(QtWidgets.QLabel("tesseract.exe:")); top.addWidget(self.tesseract_exe)
        top.addWidget(self.browse_tess)
        layout.addLayout(top)

        # Options
        opts = QtWidgets.QHBoxLayout()
        self.rank_psm = QtWidgets.QSpinBox(); self.rank_psm.setRange(6, 13); self.rank_psm.setValue(7)
        self.suit_thresh = QtWidgets.QDoubleSpinBox(); self.suit_thresh.setRange(0.0, 1.0); self.suit_thresh.setSingleStep(0.05); self.suit_thresh.setValue(0.6)
        self.auto_fail_cb = QtWidgets.QCheckBox("Auto debug on detection fail"); self.auto_fail_cb.setChecked(True)
        opts.addWidget(QtWidgets.QLabel("Rank OCR PSM")); opts.addWidget(self.rank_psm)
        opts.addSpacing(16)
        opts.addWidget(QtWidgets.QLabel("Suit min score")); opts.addWidget(self.suit_thresh)
        opts.addSpacing(16)
        opts.addWidget(self.auto_fail_cb)
        layout.addLayout(opts)

        # Canvas + toolbox
        mid = QtWidgets.QHBoxLayout()
        self.canvas = CalibrationCanvas(self)
        mid.addWidget(self.canvas, 2)

        toolbox = QtWidgets.QGroupBox("Boxes")
        tb = QtWidgets.QVBoxLayout(toolbox)
        self.snapshot_btn = QtWidgets.QPushButton("Snapshot ROI")
        tb.addWidget(self.snapshot_btn)
        tb.addWidget(QtWidgets.QLabel(self.HELP_TEXT))
        tb.addSpacing(8)

        self.list_boxes = QtWidgets.QListWidget()
        tb.addWidget(QtWidgets.QLabel("Select a box and assign label:"))
        tb.addWidget(self.list_boxes)
        self.label_combo = QtWidgets.QComboBox()
        self.label_combo.addItems(self.VALID_LABELS)
        self.assign_btn = QtWidgets.QPushButton("Assign Label to Selected")
        tb.addWidget(self.label_combo); tb.addWidget(self.assign_btn)

        self.load_btn = QtWidgets.QPushButton("Load Profile…")
        self.save_btn = QtWidgets.QPushButton("Save Profile…")
        tb.addSpacing(8); tb.addWidget(self.load_btn); tb.addWidget(self.save_btn)

        mid.addWidget(toolbox, 1)
        layout.addLayout(mid)

        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(self.buttons)

        # Wire
        self.buttons.accepted.connect(self._accept_with_validation)
        self.buttons.rejected.connect(self.reject)
        self.browse_tess.clicked.connect(self._pick_tesseract)
        self.browse_tmpl.clicked.connect(self._pick_tmpl)
        self.snapshot_btn.clicked.connect(self._snapshot)
        self.assign_btn.clicked.connect(self._assign_label)
        self.load_btn.clicked.connect(self._load)
        self.save_btn.clicked.connect(self._save)
        self.list_boxes.itemSelectionChanged.connect(self._sync_selection)

        # Internal
        self._snapshot_img: Optional[np.ndarray] = None
        self._box_items: Dict[str, BoxItem] = {}

        # Prefill from parent profile
        if isinstance(parent, OverlayWindow):
            prof: Profile = parent.profile
            self.template_dir.setText(prof.template_dir)
            self.tesseract_exe.setText(prof.tesseract_exe or "")
            self.rank_psm.setValue(prof.rank_psm)
            self.suit_thresh.setValue(prof.suit_min_score)
            self.auto_fail_cb.setChecked(prof.auto_debug_on_fail)
            self._snapshot()

            if prof.hero_boxes:
                for i, cb in enumerate(prof.hero_boxes, 1):
                    it1 = self.canvas.add_box(QtCore.QRectF(*cb.rank_box), f"Hero{i} Rank", QtGui.QColor(0,200,255))
                    self._box_items[f"Hero{i} Rank"] = it1; self._add_to_list(f"Hero{i} Rank", it1.rect())
                    it2 = self.canvas.add_box(QtCore.QRectF(*cb.suit_box), f"Hero{i} Suit", QtGui.QColor(255,140,0))
                    self._box_items[f"Hero{i} Suit"] = it2; self._add_to_list(f"Hero{i} Suit", it2.rect())
            if prof.board_boxes:
                for i, cb in enumerate(prof.board_boxes, 1):
                    it1 = self.canvas.add_box(QtCore.QRectF(*cb.rank_box), f"Board{i} Rank", QtGui.QColor(0,200,255))
                    self._box_items[f"Board{i} Rank"] = it1; self._add_to_list(f"Board{i} Rank", it1.rect())
                    it2 = self.canvas.add_box(QtCore.QRectF(*cb.suit_box), f"Board{i} Suit", QtGui.QColor(255,140,0))
                    self._box_items[f"Board{i} Suit"] = it2; self._add_to_list(f"Board{i} Suit", it2.rect())
            if prof.pot_box:
                it = self.canvas.add_box(QtCore.QRectF(*prof.pot_box), "Pot", QtGui.QColor(100,255,100))
                self._box_items["Pot"] = it; self._add_to_list("Pot", it.rect())

    def _add_to_list(self, label: str, rect: QtCore.QRectF):
        item = QtWidgets.QListWidgetItem(f"{label}: {int(rect.x())},{int(rect.y())},{int(rect.width())},{int(rect.height())}")
        item.setData(QtCore.Qt.UserRole, label)
        self.list_boxes.addItem(item)

    def _sync_selection(self):
        items = self.list_boxes.selectedItems()
        if not items: return
        label = items[0].data(QtCore.Qt.UserRole)
        for it in self.canvas.scene.items():
            if isinstance(it, BoxItem):
                it.setSelected(it.label == label)

    def _snapshot(self):
        if not isinstance(self.parent(), OverlayWindow):
            return
        parent: OverlayWindow = self.parent()
        img = parent.grab_roi_bgr()
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Snapshot", "Could not grab ROI. Move ROI into visible area.")
            return
        self._snapshot_img = img
        self.canvas.set_background(img)

    def _apply_label_color(self, label: str) -> QtGui.QColor:
        if label.endswith("Rank"):
            return QtGui.QColor(0, 200, 255)  # cyan
        if label.endswith("Suit"):
            return QtGui.QColor(255, 140, 0)  # orange
        if label == "Pot":
            return QtGui.QColor(100, 255, 100)  # green
        return QtGui.QColor(255, 60, 60)  # red for unlabeled/unknown

    def _assign_label(self):
        items = [it for it in self.canvas.scene.items() if isinstance(it, BoxItem) and it.isSelected()]
        if not items:
            QtWidgets.QMessageBox.information(self, "Assign", "Select a box (click) in the canvas first.")
            return
        label = self.label_combo.currentText()
        if label not in self.VALID_LABELS:
            QtWidgets.QMessageBox.warning(self, "Invalid label", "Pick a valid label from the dropdown.")
            return
        box: BoxItem = items[0]
        box.label = label
        box.color = self._apply_label_color(label)
        box.update()
        for i in range(self.list_boxes.count()):
            it = self.list_boxes.item(i)
            if it.data(QtCore.Qt.UserRole) == label:
                it.setText(f"{label}: {self._rect_to_str(box.rect())}")
                break
        else:
            self._add_to_list(label, box.rect())

    def _rect_to_str(self, r: QtCore.QRectF) -> str:
        return f"{int(r.x())},{int(r.y())},{int(r.width())},{int(r.height())}"

    def _pick_tesseract(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select tesseract.exe", "C:/Program Files", "Executables (*.exe)")
        if path:
            self.tesseract_exe.setText(path)

    def _pick_tmpl(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select templates directory", "")
        if path:
            self.template_dir.setText(path)

    def _collect_scene_rects(self) -> Dict[str, Tuple[int,int,int,int]]:
        rects: Dict[str, Tuple[int,int,int,int]] = {}
        for it in self.canvas.scene.items():
            if isinstance(it, BoxItem):
                r = it.rect().toRect()
                rects[it.label] = (r.x(), r.y(), r.width(), r.height())
        return rects

    def _find_incomplete_pairs(self) -> List[str]:
        rects = self._collect_scene_rects()
        missing: List[str] = []
        for i in (1,2):
            has_rank = (f"Hero{i} Rank" in rects); has_suit = (f"Hero{i} Suit" in rects)
            if has_rank and not has_suit: missing.append(f"Hero{i} Rank has no matching Hero{i} Suit")
            if has_suit and not has_rank: missing.append(f"Hero{i} Suit has no matching Hero{i} Rank")
        for i in (1,2,3,4,5):
            has_rank = (f"Board{i} Rank" in rects); has_suit = (f"Board{i} Suit" in rects)
            if has_rank and not has_suit: missing.append(f"Board{i} Rank has no matching Board{i} Suit")
            if has_suit and not has_rank: missing.append(f"Board{i} Suit has no matching Board{i} Rank")
        return missing

    def _parse_boxes_from_scene(self) -> Tuple[List[CardBox], List[CardBox], Optional[Tuple[int,int,int,int]]]:
        rects = self._collect_scene_rects()
        hero, board = [], []
        for idx in (1,2):
            r = rects.get(f"Hero{idx} Rank"); s = rects.get(f"Hero{idx} Suit")
            if r and s:
                hero.append(CardBox(r, s))
        for idx in (1,2,3,4,5):
            r = rects.get(f"Board{idx} Rank"); s = rects.get(f"Board{idx} Suit")
            if r and s:
                board.append(CardBox(r, s))
        pot = rects.get("Pot")
        return hero, board, pot

    def get_profile(self) -> Profile:
        p = Profile()
        if isinstance(self.parent(), OverlayWindow):
            r = self.parent().roi.geometry()
            p.roi = (r.x(), r.y(), r.width(), r.height())
        p.template_dir = self.template_dir.text().strip() or 'templates'
        p.tesseract_exe = self.tesseract_exe.text().strip()
        p.rank_psm = int(self.rank_psm.value())
        p.suit_min_score = float(self.suit_thresh.value())
        p.auto_debug_on_fail = bool(self.auto_fail_cb.isChecked())
        hero, board, pot = self._parse_boxes_from_scene()
        p.hero_boxes = hero; p.board_boxes = board; p.pot_box = pot
        return p

    def _accept_with_validation(self):
        # block if any unlabeled box exists OR any incomplete pairs
        for it in self.canvas.scene.items():
            if isinstance(it, BoxItem) and it.label == "Unlabeled":
                QtWidgets.QMessageBox.warning(
                    self, "Unlabeled boxes found",
                    "Please assign labels to all boxes before saving/closing.\nUnlabeled boxes are shown in red."
                )
                return
        missing = self._find_incomplete_pairs()
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Missing rank/suit pairs",
                "Each rank box must have a matching suit box (and vice versa).\n\n"
                + "\n".join("• " + m for m in missing)
            )
            return

        # <-- ADD THIS SNIPPET
        profile = self.get_profile()
        tm = TemplateMatcher(profile.template_dir)
        tm.load_templates(show_warning=True)
        # -----------------------

        self.accept()


    def _load(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load profile", "", "JSON (*.json)")
        if not fn: return
        with open(fn, 'r', encoding='utf-8') as f:
            p = Profile.from_json(f.read())
        self.template_dir.setText(p.template_dir)
        self.tesseract_exe.setText(p.tesseract_exe or "")
        self.rank_psm.setValue(p.rank_psm)
        self.suit_thresh.setValue(p.suit_min_score)
        self.auto_fail_cb.setChecked(p.auto_debug_on_fail)
        self._snapshot()
        self.list_boxes.clear(); self._box_items.clear()
        def add_lbl(rect, label, color):
            if rect:
                it = self.canvas.add_box(QtCore.QRectF(*rect), label, color)
                self._box_items[label] = it; self._add_to_list(label, it.rect())
        for i, cb in enumerate(p.hero_boxes, 1):
            add_lbl(cb.rank_box, f"Hero{i} Rank", QtGui.QColor(0,200,255))
            add_lbl(cb.suit_box, f"Hero{i} Suit", QtGui.QColor(255,140,0))
        for i, cb in enumerate(p.board_boxes, 1):
            add_lbl(cb.rank_box, f"Board{i} Rank", QtGui.QColor(0,200,255))
            add_lbl(cb.suit_box, f"Board{i} Suit", QtGui.QColor(255,140,0))
        if p.pot_box:
            add_lbl(p.pot_box, "Pot", QtGui.QColor(100,255,100))

    def _save(self):
        # Block saving if ANY unlabeled boxes or incomplete pairs remain
        for it in self.canvas.scene.items():
            if isinstance(it, BoxItem) and it.label == "Unlabeled":
                QtWidgets.QMessageBox.warning(self, "Unlabeled boxes found",
                    "Please assign labels to all boxes before saving.\nUnlabeled boxes are shown in red.")
                return
        missing = self._find_incomplete_pairs()
        if missing:
            QtWidgets.QMessageBox.warning(self, "Missing rank/suit pairs",
                "Each rank box must have a matching suit box (and vice versa).\n\n" + "\n".join("• "+m for m in missing))
            return

        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save profile", "poker_profile.json", "JSON (*.json)")
        if not fn: return
        p = self.get_profile()
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(p.to_json())
        QtWidgets.QMessageBox.information(self, "Saved", f"Profile saved to:\n{fn}")

############################
# Entry point              #
############################
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = OverlayWindow(); w.show()
    if w.profile.tesseract_exe and not os.path.isfile(w.profile.tesseract_exe):
        QtWidgets.QMessageBox.warning(w, "Tesseract", "Configured tesseract.exe path does not exist. Open Calibration Wizard to set it.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
