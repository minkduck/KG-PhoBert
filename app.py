import os
import gc
# Must be set BEFORE importing transformers to prevent the
# tensorflow → keras → cv2 → numpy-2.x crash that hides AutoTokenizer.
os.environ.setdefault("USE_TF", "0")
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import re
import rdflib
from rdflib import Graph, Namespace
from rdflib.namespace import RDF
from pyvi import ViTokenizer

# ==========================================
# 0. Remote Storage Constants & Downloader
# ==========================================
HF_RESOLVE_URL = "https://huggingface.co/MinkDuck/KG-PhoBertv2/resolve/main"


def ensure_file_exists(relative_path: str) -> None:
    """
    Download *relative_path* from Hugging Face if it is not present locally.

    Skips silently when the file already exists so there is no start-up
    overhead on subsequent runs.
    """
    if os.path.exists(relative_path):
        return

    url = f"{HF_RESOLVE_URL}/{relative_path.replace(os.sep, '/')}"
    os.makedirs(os.path.dirname(relative_path) or ".", exist_ok=True)

    with st.spinner(f"Downloading `{relative_path}` from Hugging Face … (first run only)"):
        try:
            torch.hub.download_url_to_file(url, relative_path)
            st.info(f"✅ Downloaded `{relative_path}` successfully.")
        except Exception as exc:
            st.error(f"❌ Failed to download `{relative_path}`: {exc}")
            raise


# ==========================================
# 1. Config and Global Mappings
# ==========================================
st.set_page_config(page_title="Hybrid Ontology-NLP Emotion Classifier", layout="wide")

DOMAIN_CLASSES = {
    "VSFC":       ["Negative", "Neutral", "Positive"],
    "VSMEC":      ["Anger", "Disgust", "Enjoyment", "Fear", "Other", "Sadness", "Surprise"],
    "VSFC-Ekman": ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"],
}

# (domain, base_model_type) → (filename, fusion_type)
FILE_MAPPING = {
    ("VSFC",       "Baseline"): ("phobert_baseline_m1.pth",    "none"),
    ("VSFC",       "RawGate"):  ("phobert_gate_m3.pth",         "gate"),
    ("VSMEC",      "Baseline"): ("vsmec_baseline_m1.pth",       "none"),
    ("VSMEC",      "RawGate"):  ("vsmec_gate_m3.pth",           "gate"),
    ("VSFC-Ekman", "Baseline"): ("vsfc_ekman_baseline_m1.pth",  "none"),
    ("VSFC-Ekman", "RawGate"):  ("vsfc_ekman_raw_gate_m3.pth",  "gate"),
}

# ──────────────────────────────────────────────────────────────────────────────
# Module-level single-model slot — only ONE PhoBERT model in RAM at a time.
# Streamlit keeps module globals alive across reruns (same process lifetime).
# ──────────────────────────────────────────────────────────────────────────────
_model_slot: dict = {"key": None, "model": None}


def _evict_model() -> None:
    """Delete the currently cached model and free RAM immediately."""
    if _model_slot["model"] is not None:
        del _model_slot["model"]
        _model_slot["model"] = None
        _model_slot["key"]   = None
        gc.collect()
        torch.cuda.empty_cache()


def get_model(domain: str, model_type: str,
              ont_input_dim: int = 24) -> "nn.Module":
    """
    Return the requested model, loading it on demand and evicting the
    previously cached model first.  Baseline+Rule reuses Baseline weights.

    Only ONE model is in RAM at any point — safe within Streamlit Cloud’s
    1 GB limit.
    """
    base_type = "Baseline" if model_type == "Baseline+Rule" else model_type
    key = (domain, base_type, ont_input_dim)

    if _model_slot["key"] == key:          # cache hit — free
        return _model_slot["model"]

    _evict_model()                         # evict previous model

    filename, fusion = FILE_MAPPING[(domain, base_type)]
    rel_path = os.path.join("model", filename)

    # Download weight file if not already on disk
    ensure_file_exists(rel_path)

    num_labels = len(DOMAIN_CLASSES[domain])
    model = PhoBERT_Fusion_V2(
        n_classes=num_labels,
        fusion_type=fusion,
        ontology_dim=None,
        ont_input_dim=ont_input_dim,
    )

    if os.path.exists(rel_path):
        try:
            ckpt = torch.load(rel_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, nn.Module):
                model = ckpt
            else:
                sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
                # Filter out keys not present in this architecture
                # (e.g. 'pooler.*' from checkpoints saved with add_pooling_layer=True)
                own_keys  = set(model.state_dict().keys())
                filtered  = {k: v for k, v in sd.items() if k in own_keys}
                dropped   = set(sd.keys()) - own_keys
                if dropped:
                    print(f"\u2139\ufe0f  Ignored {len(dropped)} unexpected key(s) "
                          f"(e.g. {sorted(dropped)[:3]})")    # e.g. pooler.*
                model.load_state_dict(filtered, strict=True)
            print(f"\u2705 Loaded {filename}")
        except Exception as exc:
            st.warning(f"\u26a0\ufe0f Could not load {rel_path}: {exc}")

    model.eval()
    _model_slot["key"]   = key
    _model_slot["model"] = model
    return model

# ==========================================
# 2. OntologyEngine  (shared core)
# ==========================================
class OntologyEngine:
    """
    Ported from vsmec-12-phobertv2.ipynb (v12.2) and vsfc-10-phobertv2.ipynb (v12.3).
    Strict diacritic matching (no unidecode on keys), greedy longest-match MWE,
    ViTokenizer segmentation. Polarity Injection for VSFC polarity dimension.
    """

    ALLEMOTIONS   = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    ALLAPPRAISALS = [
        "GoalObstructionAppraisal", "PleasantnessAppraisal", "UnpleasantnessAppraisal",
        "DangerousnessAppraisal", "LossAppraisal", "SuddennessAppraisal",
        "LowPredictabilityAppraisal", "UnpredictabilityAppraisal", "UnfairnessAppraisal",
    ]
    INTENSITY_CLASSES = ["HighIntensity", "MediumIntensity", "LowIntensity"]
    POLARITY_CLASSES  = ["PositivePolarity", "NegativePolarity", "NeutralPolarity"]

    EMOTIONMAPPING = {
        "Enjoyment": "Happiness", "Happiness": "Happiness",
        "Anger": "Anger", "Disgust": "Disgust",
        "Fear": "Fear", "Sadness": "Sadness", "Surprise": "Surprise",
        "NeutralEmotion": "Neutral", "Neutral": "Neutral",
        "Other": "Neutral", "NegationCue": "NegationCue",
        "Joy": "Happiness",  # VSFC-Ekman label mapping
    }

    IMPLIED_POLARITY = {
        "Anger": "NegativePolarity", "Disgust": "NegativePolarity",
        "Fear": "NegativePolarity",  "Sadness": "NegativePolarity",
        "Happiness": "PositivePolarity", "Enjoyment": "PositivePolarity",
        "Surprise": "PositivePolarity",
        "UnpleasantnessAppraisal": "NegativePolarity",
        "GoalObstructionAppraisal": "NegativePolarity",
        "LossAppraisal": "NegativePolarity",
        "UnfairnessAppraisal": "NegativePolarity",
        "DangerousnessAppraisal": "NegativePolarity",
        "PleasantnessAppraisal": "PositivePolarity",
    }

    AMBIGUOUS_TERMS = {
        "hay", "chả", "cơ", "ngờ", "ý", "quá", "lại",
        "tôi", "tao", "tớ", "mình", "bạn", "nó", "hắn", "anh", "chị", "em",
        "là", "thì", "mà", "bị", "được", "cái", "con", "người",
    }
    VALID_DOMAINS = {"GeneralDomain", "SocialDomain", "EducationDomain"}

    # --- VSFC polarity injection sets ---
    POS_EMOTIONS = {"Happiness", "Enjoyment", "Surprise"}
    NEG_EMOTIONS = {"Anger", "Disgust", "Fear", "Sadness"}

    def __init__(self, rdfpath,
                 defaultconf=0.9,
                 negation_attenuation=0.4,
                 alpha_similarity=0.2,
                 verbose=False):
        self.EKMAN = Namespace("http://example.org/ekman-ontology#")
        self.defaultconf          = float(defaultconf)
        self.negation_attenuation = float(negation_attenuation)
        self.alpha_similarity     = float(alpha_similarity)
        self.verbose              = verbose

        self.g = Graph()
        if os.path.exists(rdfpath):
            self.g.parse(rdfpath)
            if verbose:
                print(f"✅ OntologyEngine: loaded {len(self.g)} triples from {rdfpath}")
        else:
            print(f"⚠️ OntologyEngine: RDF file not found: {rdfpath}")

        self.sim_matrix = self._build_similarity_matrix()
        self.lexiconmap = self._build_strict_lexicon()

        self.mwe_max_len = 1
        for k in self.lexiconmap:
            self.mwe_max_len = max(self.mwe_max_len, len(k.split()))

        if verbose:
            print(f"   📚 Lexicon: {len(self.lexiconmap)} entries, max MWE length: {self.mwe_max_len}")

    # ---- helpers -----------------------------------------------------------
    def _norm(self, s):
        return re.sub(r"\s+", " ", str(s).lower().strip().replace("_", " ")).strip() if s else ""

    def _local(self, uri):
        u = str(uri)
        return u.split("#")[-1] if "#" in u else u.split("/")[-1]

    # ---- similarity matrix -------------------------------------------------
    def _build_similarity_matrix(self):
        n = len(self.ALLEMOTIONS)
        mat = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        try:
            q = (
                "PREFIX ekman: <http://example.org/ekman-ontology#> "
                "SELECT ?score ?e1 ?e2 WHERE { "
                "  ?edge a ekman:EmotionSimilarityEdge ; "
                "        ekman:similarityScore ?score ; "
                "        ekman:linksEmotion ?e1 ; "
                "        ekman:linksEmotion ?e2 . }"
            )
            for row in self.g.query(q):
                e1 = self.EMOTIONMAPPING.get(self._local(row.e1))
                e2 = self.EMOTIONMAPPING.get(self._local(row.e2))
                if e1 in self.ALLEMOTIONS and e2 in self.ALLEMOTIONS:
                    i, j = self.ALLEMOTIONS.index(e1), self.ALLEMOTIONS.index(e2)
                    v = float(row.score)
                    mat[i, j] = mat[j, i] = max(mat[i, j], v)
        except Exception:
            pass
        s = mat.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return mat / s

    # ---- strict lexicon (NO unidecode on keys) ------------------------------
    def _build_strict_lexicon(self):
        EK = self.EKMAN
        lookup = {}
        mix_cache, sim_cache = {}, {}

        # 1. Cache emotion mixtures
        q_mix = (
            "PREFIX ekman: <http://example.org/ekman-ontology#> "
            "SELECT ?m ?e ?w WHERE { "
            "  ?m a ekman:EmotionMixture ; ekman:hasComponent ?c . "
            "  ?c ekman:componentEmotion ?e ; ekman:componentWeight ?w . }"
        )
        for r in self.g.query(q_mix):
            e = self.EMOTIONMAPPING.get(self._local(r.e))
            if e in self.ALLEMOTIONS:
                d = mix_cache.setdefault(r.m, {})
                d[e] = d.get(e, 0.0) + float(r.w)

        # 2. Cache direct emotion categories
        for s, _, o in self.g.triples((None, EK.hasEmotionCategory, None)):
            e = self.EMOTIONMAPPING.get(self._local(o))
            if e in self.ALLEMOTIONS:
                sim_cache[s] = [e]

        # 3. Type-based fallback
        for s in self.g.subjects(RDF.type, None):
            if s in mix_cache or s in sim_cache:
                continue
            ts = list(self.g.objects(s, RDF.type))
            if EK.NegationCue in ts:
                sim_cache[s] = ["NegationCue"]
            else:
                for t in ts:
                    m = self.EMOTIONMAPPING.get(self._local(t))
                    if m in self.ALLEMOTIONS:
                        sim_cache[s] = [m]
                        break

        # 4. Build entries via Evocation query (STRICT: no unidecode on primary key)
        q_evoc = (
            "PREFIX ekman: <http://example.org/ekman-ontology#> "
            "SELECT ?word ?stim ?ev ?base ?type ?dom WHERE { "
            "  ?evoc a ekman:Evocation ; "
            "        ekman:fromLexiconEntry ?lex ; "
            "        ekman:toStimulus ?stim . "
            "  ?lex ekman:hasContent ?word . "
            "  OPTIONAL { ?evoc ekman:confidenceScore ?ev } "
            "  OPTIONAL { ?evoc ekman:evidenceType ?type } "
            "  OPTIONAL { ?lex ekman:baseIntensityScore ?base } "
            "  OPTIONAL { ?lex ekman:belongsToDomain ?dom } }"
        )
        for r in self.g.query(q_evoc):
            if r.dom and self._local(r.dom) not in self.VALID_DOMAINS:
                continue
            w = self._norm(r.word)
            if len(w) < 2 or w in self.AMBIGUOUS_TERMS:
                continue

            # Resolve emotions
            emos = {}
            if r.stim in mix_cache:
                raw = mix_cache[r.stim]
                tot = sum(raw.values())
                if tot > 0:
                    emos = {k: v / tot for k, v in raw.items()}
            elif r.stim in sim_cache:
                lst = sim_cache[r.stim]
                emos = {k: 1.0 for k in lst}
            if not emos:
                continue

            apps, ints, pols = self._infer_attributes(r.stim, emos.keys())
            entry_score = float(r.ev or self.defaultconf) * float(r.base or 1.0)

            entry = {
                "emotions":    emos,
                "score":       entry_score,
                "appraisals":  list(apps),
                "intensities": list(ints),
                "polarities":  list(pols),
                "isnegation":  "NegationCue" in emos,
            }
            # PRIMARY: exact diacritic key — no unidecode applied
            lookup.setdefault(w, []).append(entry)

        return lookup

    # ---- attribute inference -----------------------------------------------
    def _infer_attributes(self, s, emos):
        EK = self.EKMAN
        a, i, p = set(), set(), set()

        def collect(node):
            for o in self.g.objects(node, EK.evokesAppraisal):
                a.add(self._local(o))
            for o in self.g.objects(node, EK.hasIntensity):
                i.add(self._local(o))
            for o in self.g.objects(node, EK.hasPolarity):
                p.add(self._local(o))

        collect(s)
        for t in self.g.objects(s, RDF.type):
            collect(t)

        if not p:
            for e in emos:
                if e in self.IMPLIED_POLARITY:
                    p.add(self.IMPLIED_POLARITY[e])
        if not p:
            for app in a:
                if app in self.IMPLIED_POLARITY:
                    p.add(self.IMPLIED_POLARITY[app])

        return a, i, p

    # ---- main inference entry point ----------------------------------------
    def getvector(self, text, debug=False):
        """
        Returns (vector, tokens, hit_trace) when debug=True, else just vector.

        Vector layout (24d total):
            [0:7]   emotion scores (ALLEMOTIONS order)
            [7:16]  appraisal scores (ALLAPPRAISALS order)
            [16:19] intensity flags (INTENSITY_CLASSES order)
            [19:22] polarity scores (POLARITY_CLASSES order)  ← idx 19=Pos, 20=Neg, 21=Neu
            [22:24] negation flag + reserved

        hit_trace entries:
            {"phrase": str, "emotions": dict[str,float], "polarities": list[str],
             "appraisals": list[str], "score": float}
        """
        raw_tokens = ViTokenizer.tokenize(str(text).lower()).split()
        flat = [self._norm(t) for t in raw_tokens if self._norm(t)]

        ve = np.zeros(len(self.ALLEMOTIONS),    dtype=np.float32)
        va = np.zeros(len(self.ALLAPPRAISALS),  dtype=np.float32)
        vi = np.zeros(len(self.INTENSITY_CLASSES), dtype=np.float32)
        vp = np.zeros(len(self.POLARITY_CLASSES),  dtype=np.float32)
        vn = np.zeros(2, dtype=np.float32)

        idx, n = 0, len(flat)
        hit_trace = []

        while idx < n:
            match, match_len = None, 0
            for L in range(min(self.mwe_max_len, n - idx), 0, -1):
                phrase = " ".join(flat[idx: idx + L])
                if phrase in self.lexiconmap:
                    match, match_len = self.lexiconmap[phrase], L
                    break

            if match is None:
                idx += 1
                continue

            phrase_str = " ".join(flat[idx: idx + match_len])

            if debug:
                ent0 = match[0]
                hit_trace.append({
                    "phrase":     phrase_str,
                    "emotions":   ent0["emotions"],
                    "polarities": ent0["polarities"],
                    "appraisals": ent0["appraisals"],
                    "score":      round(ent0["score"], 4),
                })

            for ent in match:
                if ent["isnegation"]:
                    vn[0] = 1.0
                    idx += match_len
                    continue

                sc = ent["score"]
                if vn[0] > 0:
                    sc *= self.negation_attenuation

                for e, w in ent["emotions"].items():
                    if e in self.ALLEMOTIONS:
                        ve[self.ALLEMOTIONS.index(e)] += sc * w

                # VSFC polarity injection: aggregate emotion scores into polarity dims
                pos_sc = sum(w for e, w in ent["emotions"].items() if e in self.POS_EMOTIONS)
                neg_sc = sum(w for e, w in ent["emotions"].items() if e in self.NEG_EMOTIONS)
                neu_sc = ent["emotions"].get("Neutral", 0.0)
                if pos_sc > 0: vp[0] += sc * pos_sc
                if neg_sc > 0: vp[1] += sc * neg_sc
                if neu_sc > 0: vp[2] += sc * neu_sc
                for x in ent["polarities"]:
                    if x in self.POLARITY_CLASSES:
                        vp[self.POLARITY_CLASSES.index(x)] += sc

                for x in ent["appraisals"]:
                    if x in self.ALLAPPRAISALS:
                        va[self.ALLAPPRAISALS.index(x)] += sc
                for x in ent["intensities"]:
                    if x in self.INTENSITY_CLASSES:
                        vi[self.INTENSITY_CLASSES.index(x)] = 1.0

            idx += match_len

        if self.alpha_similarity > 0:
            ve = np.clip(ve + self.alpha_similarity * (ve @ self.sim_matrix), 0, None)

        fin = np.concatenate([
            np.tanh(ve), np.tanh(va), vi, np.tanh(vp), vn
        ]).astype(np.float32)

        return (fin, flat, hit_trace) if debug else fin


# ==========================================
# 3. Model Definition — PhoBERT_Fusion_V2
# ==========================================
# ONT_INPUT_DIM is detected at runtime from the engine; fallback = 24
_ONT_INPUT_DIM_DEFAULT = 24


class PhoBERT_Fusion_V2(nn.Module):
    """
    Exact architecture used in both training notebooks.

    fusion_type:
        'none'   – Baseline (PhoBERT pooled CLS → fc)
        'gate'   – RawGate  (PhoBERT ⊗ sigmoid-gated ontology → concat → fc)
        'concat' – RawConcat (PhoBERT | ontology → fc)

    ontology_dim:
        None  → Raw mode: ont_adapter = Identity (keeps ont_input_dim as-is)
        int   → Dense mode: Linear(ont_input_dim → ontology_dim) + BN + ReLU + Dropout
    """

    def __init__(self, n_classes, fusion_type="none", ontology_dim=None,
                 ont_input_dim=_ONT_INPUT_DIM_DEFAULT):
        super().__init__()
        self.fusion_type = fusion_type

        # Backbone — add_pooling_layer=False avoids pooler.* keys that are
        # absent in checkpoints saved directly from RobertaModel without pooler.
        self.phobert = AutoModel.from_pretrained(
            "vinai/phobert-base-v2", add_pooling_layer=False
        )
        self.dropout  = nn.Dropout(p=0.3)

        # Ontology adapter
        if ontology_dim:
            self.ont_adapter = nn.Sequential(
                nn.Linear(ont_input_dim, ontology_dim),
                nn.BatchNorm1d(ontology_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.curr_dim = ontology_dim
        else:
            self.ont_adapter = nn.Identity()
            self.curr_dim = ont_input_dim

        # Gating mechanism
        if self.fusion_type == "gate":
            self.gate = nn.Linear(768, self.curr_dim)

        # Classifier head
        inp_dim = 768 + self.curr_dim if fusion_type != "none" else 768
        self.fc = nn.Linear(inp_dim, n_classes)

    def forward(self, input_ids, attention_mask, ontology_features=None):
        # 1. Raw [CLS] token from sequence output (no pooler transformation)
        #    seq_out shape: (batch, seq_len, 768)
        seq_out  = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        text_emb = self.dropout(seq_out[:, 0, :])   # (batch, 768)

        # 2. Fusion logic
        if self.fusion_type == "none":
            return self.fc(text_emb)

        # Ontology adapter
        ont_emb = self.ont_adapter(ontology_features)

        if self.fusion_type == "gate":
            # Sigmoid gate learned from PhoBERT embedding
            gate_values = torch.sigmoid(self.gate(text_emb))
            ont_emb = ont_emb * gate_values

        # Concat and classify
        combined = torch.cat([text_emb, ont_emb], dim=1)
        return self.fc(combined)


# ==========================================
# 4. Resource Loading (Cached)
# ==========================================
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


@st.cache_resource
def load_ontology_engine():
    """Load and cache the OntologyEngine (builds lexicon once, ~10–30 s)."""
    # Ensure the primary RDF file is present (downloads from HF if missing)
    ensure_file_exists("ekman6_4_9.rdf")

    rdf_candidates = ["ekman6_4_9.rdf", "ekman6_4_8.rdf"]
    rdfpath = next((p for p in rdf_candidates if os.path.exists(p)), None)
    if rdfpath is None:
        st.warning("⚠️ RDF ontology file not found. XAI features will be disabled.")
        return None
    return OntologyEngine(rdfpath, verbose=True)


# load_models() removed — replaced by the lazy get_model() above.
# Models are now loaded on demand, one at a time, to stay within 1 GB RAM.


# ==========================================
# 5. Rule Layer — Neuro-Symbolic Reasoning
# ==========================================
def apply_rules(initial_label: str, probs: dict, kg_vector: np.ndarray,
                hit_trace: list, classes: list, domain: str,
                ont_threshold: float = 0.4, conf_limit: float = 0.85
                ) -> tuple[str, bool, str | None]:
    """
    KG-based posterior rule adjustment.

    VSFC  (3-label: Negative / Neutral / Positive)
    ─────────────────────────────────────────────
      Signal: kg_vector[19] = PositivePolarity, kg_vector[20] = NegativePolarity
      Rule  : If model is uncertain (confidence < conf_limit) and Neutral is predicted,
              override with the dominant polarity signal if it exceeds ont_threshold.

    VSMEC (7-label)
    ───────────────────────────────────────────────────────────
      Signal : first 7 dims of kg_vector = emotion scores (ALLEMOTIONS order)
      Mapping: Ontology Happiness → "Enjoyment",  Neutral → "Other"
      Rule   : If model predicts "Other" OR confidence < conf_limit,
               and max ontology emotion score > ont_threshold,
               override with mapped ontology label (if valid in domain classes).

    VSFC-Ekman (7-label)
    ──────────────────────────────────────────────────────────
      Signal : hit_trace dominant emotion
      Rule   : If dominant KG emotion is a negative one (Disgust/Anger/Sadness/Fear)
               and confidence < conf_limit, force "Sadness" (or available negative label).
    """
    confidence   = probs.get(initial_label, 0.0)
    final_label  = initial_label
    rule_applied = False
    rule_name    = None

    # ── VSFC: Polarity-vector override ──────────────────────────────────────
    if domain == "VSFC" and len(kg_vector) >= 22:
        pos_signal = float(kg_vector[19])
        neg_signal = float(kg_vector[20])

        if initial_label == "Neutral" and confidence < conf_limit:
            if neg_signal > ont_threshold and neg_signal >= pos_signal:
                final_label, rule_applied, rule_name = "Negative", True, "Rule_VSFC_Neg_Polarity"
            elif pos_signal > ont_threshold and pos_signal > neg_signal:
                final_label, rule_applied, rule_name = "Positive", True, "Rule_VSFC_Pos_Polarity"

    # ── VSMEC: Emotion-vector override ──────────────────────────────────────
    elif domain == "VSMEC" and len(kg_vector) >= 7:
        ALLEMOTIONS = OntologyEngine.ALLEMOTIONS
        ONT_TO_VSMEC = {"Happiness": "Enjoyment", "Neutral": "Other"}

        ont_emotions  = kg_vector[0:7]
        max_ont_idx   = int(np.argmax(ont_emotions))
        max_ont_score = float(ont_emotions[max_ont_idx])
        ont_emo_name  = ALLEMOTIONS[max_ont_idx]
        mapped_label  = ONT_TO_VSMEC.get(ont_emo_name, ont_emo_name)

        if (initial_label == "Other" or confidence < conf_limit) and max_ont_score > ont_threshold:
            if mapped_label in classes:
                final_label  = mapped_label
                rule_applied = True
                rule_name    = f"Rule_VSMEC_KG_{ont_emo_name}"

    # ── VSFC-Ekman: Negative emotion override ───────────────────────────────
    elif domain == "VSFC-Ekman":
        # Aggregate emotions from hit_trace
        emotion_totals: dict[str, float] = {}
        for h in hit_trace:
            for emo, w in h["emotions"].items():
                emotion_totals[emo] = emotion_totals.get(emo, 0.0) + w * h["score"]

        if emotion_totals and confidence < conf_limit:
            dominant_emo = max(emotion_totals, key=emotion_totals.get)
            neg_emos = {"Disgust", "Anger", "Sadness", "Fear"}

            if dominant_emo in neg_emos:
                # Try to override with the matching label in VSFC-Ekman classes
                EKMAN_LABEL_MAP = {
                    "Disgust": "Disgust", "Anger": "Anger",
                    "Sadness": "Sadness", "Fear": "Fear",
                }
                target = EKMAN_LABEL_MAP.get(dominant_emo)
                if target and target in classes and initial_label not in neg_emos:
                    final_label  = target
                    rule_applied = True
                    rule_name    = f"Rule_EkmanNeg_{dominant_emo}"

            elif dominant_emo == "Happiness":
                if "Joy" in classes and initial_label not in {"Joy"}:
                    final_label  = "Joy"
                    rule_applied = True
                    rule_name    = "Rule_EkmanJoy"

    return final_label, rule_applied, rule_name


# ==========================================
# 6. Inference Function
# ==========================================
def predict_with_model(text, domain, model_type, model, tokenizer, ontology_engine):
    """Run inference using the pre-loaded *model* (caller's responsibility to
    obtain the correct model via get_model() before calling this function)."""
    classes = DOMAIN_CLASSES.get(domain, [])

    # --- KG vector + hit_trace via OntologyEngine ---
    kg_vector = np.zeros(_ONT_INPUT_DIM_DEFAULT, dtype=np.float32)
    hit_trace = []
    if ontology_engine is not None and text.strip():
        try:
            kg_vector, _, hit_trace = ontology_engine.getvector(text, debug=True)
        except Exception as e:
            print(f"OntologyEngine error: {e}")

    if not text.strip() or model is None:
        return {
            "initial_label": classes[0] if classes else "Unknown",
            "final_label":   classes[0] if classes else "Unknown",
            "probs":         {c: 1.0 / len(classes) for c in classes},
            "hit_trace":     hit_trace,
            "kg_vector":     kg_vector,
            "rule_applied":  False,
            "rule_name":     None,
        }

    # --- PhoBERT tokenisation ---
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    input_ids      = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # --- Ontology features tensor ---
    ont_tensor = torch.tensor(kg_vector, dtype=torch.float).unsqueeze(0)

    # --- Forward pass ---
    with torch.no_grad():
        logits   = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         ontology_features=ont_tensor)
        probs_t  = nn.functional.softmax(logits, dim=1).squeeze()
        if probs_t.dim() == 0:
            probs_t = probs_t.unsqueeze(0)
        probs_np = probs_t.numpy()

    max_idx       = min(len(classes) - 1, int(np.argmax(probs_np)))
    initial_label = classes[max_idx]
    probs         = {c: float(probs_np[i]) if i < len(probs_np) else 0.0
                     for i, c in enumerate(classes)}

    final_label  = initial_label
    rule_applied = False
    rule_name    = None

    if model_type == "Baseline+Rule":
        final_label, rule_applied, rule_name = apply_rules(
            initial_label=initial_label,
            probs=probs,
            kg_vector=kg_vector,
            hit_trace=hit_trace,
            classes=classes,
            domain=domain,
        )

    return {
        "initial_label": initial_label,
        "final_label":   final_label,
        "probs":         probs,
        "hit_trace":     hit_trace,
        "kg_vector":     kg_vector,
        "rule_applied":  rule_applied,
        "rule_name":     rule_name,
    }


# ==========================================
# 7. XAI rendering helper
# ==========================================
def render_xai(result, model_type):
    """Render the Explanation (XAI) expander content."""

    # Rule info (Baseline+Rule only)
    if model_type == "Baseline+Rule":
        if result["rule_applied"]:
            st.info(
                f"**Rule Applied:** `{result['rule_name']}`\n\n"
                f"Baseline predicted **{result['initial_label']}** → "
                f"Rule adjusted to **{result['final_label']}**"
            )
        else:
            st.info("No adjustment rules were triggered. Baseline prediction retained.")

    hit_trace = result.get("hit_trace", [])
    if not hit_trace:
        st.write("*No ontology matches found for this text.*")
        return

    st.write("**Matched Ontology Phrases (Ekman KG):**")

    rows = []
    for h in hit_trace:
        dominant_emo = max(h["emotions"], key=h["emotions"].get) if h["emotions"] else "—"
        dominant_pol = h["polarities"][0] if h["polarities"] else "—"
        pol_map = {
            "PositivePolarity": "✅ Positive",
            "NegativePolarity": "❌ Negative",
            "NeutralPolarity":  "➖ Neutral",
        }
        dominant_pol = pol_map.get(dominant_pol, dominant_pol)
        emo_str = ", ".join(
            f"{e} ({w:.2f})" for e, w in sorted(h["emotions"].items(), key=lambda x: -x[1])
        )
        rows.append({
            "Matched Phrase": h["phrase"],
            "Emotions":       emo_str,
            "Polarity":       dominant_pol,
            "Confidence":     f"{h['score']:.3f}",
        })

    st.table(pd.DataFrame(rows))

    # KG polarity signal bar (useful for VSFC debugging)
    kg_vec = result.get("kg_vector")
    if kg_vec is not None and len(kg_vec) >= 22:
        pol_vals = {
            "PositivePolarity": float(kg_vec[19]),
            "NegativePolarity": float(kg_vec[20]),
            "NeutralPolarity":  float(kg_vec[21]),
        }
        if any(v > 0 for v in pol_vals.values()):
            st.write("**KG Polarity Signal (tanh-scaled):**")
            st.bar_chart(pd.DataFrame(
                list(pol_vals.items()), columns=["Polarity", "Score"]
            ).set_index("Polarity"))


# ==========================================
# 8. Streamlit UI
# ==========================================
def main():
    # ── Sidebar — global controls ────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        DOMAIN_DISPLAY = {
            "Education (VSFC)":            "VSFC",
            "Social (VSMEC)":              "VSMEC",
            "Education – 7 label (Ekman)": "VSFC-Ekman",
            "All domains":                 "All",
        }

        domain_label = st.selectbox(
            "Domain",
            list(DOMAIN_DISPLAY.keys()),
            key="sidebar_domain",
            help="Applied to both Single-model and Compare tabs.",
        )
        domain = DOMAIN_DISPLAY[domain_label]

        st.divider()

        # Tab selector — stored in session_state so it survives reruns.
        # This is the ONLY reliable way to prevent Streamlit from jumping
        # back to tab 0 after a form submission triggers a rerun.
        active_tab = st.radio(
            "View",
            ["Single model", "Compare models"],
            key="active_tab",
            horizontal=False,
        )

        st.divider()
        st.caption("Hybrid Ontology-NLP Emotion Classifier")

    st.title("Hybrid Ontology-NLP Emotion Classifier (Demo)")

    # ── Load lightweight resources once (cached) ─────────────────────────────
    # PhoBERT model weights are loaded lazily via get_model() at inference time.
    with st.spinner("Initializing tokenizer and ontology engine…"):
        tokenizer       = load_tokenizer()
        ontology_engine = load_ontology_engine()
        if ontology_engine is not None:
            try:
                _dim = len(ontology_engine.getvector("test"))
            except Exception:
                _dim = _ONT_INPUT_DIM_DEFAULT
        else:
            _dim = _ONT_INPUT_DIM_DEFAULT

    # ── Session-state — initialise once per browser session ──────────────────
    if "results_tab1" not in st.session_state:
        st.session_state.results_tab1 = None
    if "results_tab2" not in st.session_state:
        st.session_state.results_tab2 = None

    # ── Clear stale results when the domain changes ───────────────────────────
    if st.session_state.get("_last_domain") != domain:
        st.session_state.results_tab1 = None
        st.session_state.results_tab2 = None
        st.session_state["_last_domain"] = domain

    # =========================================================================
    # VIEW: Single model
    # =========================================================================
    if active_tab == "Single model":
        st.header("Single Model Inference")
        st.caption(f"Domain: **{domain_label}**")

        with st.form("form_tab1"):
            text_input = st.text_area(
                "Enter Vietnamese sentence:",
                help="Type the sentence you want to analyse.",
                key="text1",
            )
            model_type = st.selectbox(
                "Model",
                ["Baseline", "RawGate", "Baseline+Rule"],
                key="model1",
            )
            submitted1 = st.form_submit_button("Predict")

        # Run inference on submit — loads exactly ONE model on demand
        if submitted1:
            if not text_input.strip():
                st.warning("Please enter some text.")
                st.session_state.results_tab1 = None
            else:
                domains_to_run = (
                    ["VSFC", "VSMEC", "VSFC-Ekman"] if domain == "All" else [domain]
                )
                items = []
                for d in domains_to_run:
                    with st.spinner(
                        f"Loading {model_type} model for '{d}'… "
                        f"(stays cached until you switch model)"
                    ):
                        model = get_model(d, model_type, _dim)
                    result = predict_with_model(
                        text_input, d, model_type, model, tokenizer, ontology_engine
                    )
                    items.append({
                        "domain":     d,
                        "model_type": model_type,
                        "result":     result,
                        "multi":      domain == "All",
                    })
                st.session_state.results_tab1 = items

        # Render results unconditionally from session_state
        if st.session_state.results_tab1:
            for item in st.session_state.results_tab1:
                d      = item["domain"]
                mt     = item["model_type"]
                result = item["result"]

                if item["multi"]:
                    label_key = next(
                        (k for k, v in DOMAIN_DISPLAY.items() if v == d), d
                    )
                    st.subheader(f"📌 {label_key}")
                else:
                    st.subheader("Prediction Result")

                final_prob = result["probs"].get(result["final_label"], 0.0)
                st.success(
                    f"**Final Label:** {result['final_label']} "
                    f"(Probability: {final_prob:.4f})"
                )

                prob_df = (
                    pd.DataFrame(
                        list(result["probs"].items()),
                        columns=["Emotion", "Probability"],
                    ).set_index("Emotion")
                )
                st.bar_chart(prob_df)

                if mt != "Baseline":
                    with st.expander(
                        "Explanation (XAI)",
                        expanded=(not item["multi"]),
                    ):
                        render_xai(result, mt)

    # =========================================================================
    # VIEW: Compare models
    # =========================================================================
    else:
        st.header("Compare Models")

        with st.form("form_tab2"):
            text_input2 = st.text_area(
                "Enter Vietnamese sentence:",
                help="Type the sentence to evaluate across all three models.",
                key="text2",
            )
            submitted2 = st.form_submit_button("Compare")

        # Run comparison on submit — loads ONE model at a time, processes, evicts
        if submitted2:
            if not text_input2.strip():
                st.warning("Please enter some text.")
                st.session_state.results_tab2 = None
            else:
                items2 = []
                for d in ["VSFC", "VSMEC", "VSFC-Ekman"]:
                    # ── Step 1: Load Baseline, run Baseline + Baseline+Rule ──────────
                    with st.spinner(f"[{d}] Processing with Baseline… (evicts previous model)"):
                        model_b  = get_model(d, "Baseline", _dim)
                        res_base = predict_with_model(
                            text_input2, d, "Baseline",      model_b, tokenizer, ontology_engine
                        )
                        res_rule = predict_with_model(
                            text_input2, d, "Baseline+Rule", model_b, tokenizer, ontology_engine
                        )
                        del model_b                # drop local reference

                    # ── Step 2: Evict Baseline from slot, load RawGate ───────────────
                    _evict_model()
                    with st.spinner(f"[{d}] Processing with RawGate… (evicts Baseline)"):
                        model_g  = get_model(d, "RawGate", _dim)
                        res_gate = predict_with_model(
                            text_input2, d, "RawGate", model_g, tokenizer, ontology_engine
                        )
                        del model_g                # drop local reference

                    _evict_model()                 # free RawGate before next domain

                    items2.append({
                        "domain":   d,
                        "multi":    True,
                        "res_base": res_base,
                        "res_gate": res_gate,
                        "res_rule": res_rule,
                    })
                st.session_state.results_tab2 = items2

        # Render comparison results unconditionally from session_state
        if st.session_state.results_tab2:
            for item in st.session_state.results_tab2:
                d        = item["domain"]
                res_base = item["res_base"]
                res_gate = item["res_gate"]
                res_rule = item["res_rule"]

                if item["multi"]:
                    label_key = next(
                        (k for k, v in DOMAIN_DISPLAY.items() if v == d), d
                    )
                    st.subheader(f"📌 {label_key}")

                # Summary table
                data = []
                for name, res in [
                    ("Baseline",      res_base),
                    ("RawGate",       res_gate),
                    ("Baseline+Rule", res_rule),
                ]:
                    final_prob = res["probs"].get(res["final_label"], 0.0)
                    rule_str   = "No"
                    if name == "Baseline+Rule" and res["rule_applied"]:
                        rule_str = f"Yes ({res['rule_name']})"
                    data.append({
                        "Model":          name,
                        "Final Label":    res["final_label"],
                        "Probability":    f"{final_prob:.4f}",
                        "Rule Applied":   rule_str,
                        "KG Phrases Hit": len(res["hit_trace"]),
                    })

                st.write("### Summary Table")
                st.table(pd.DataFrame(data))

                # Class probability comparison chart
                st.write("### Class Probability Comparison")
                all_classes = DOMAIN_CLASSES[d]
                chart_data  = {"Class": all_classes}
                for name, res in [
                    ("Baseline",      res_base),
                    ("RawGate",       res_gate),
                    ("Baseline+Rule", res_rule),
                ]:
                    chart_data[name] = [res["probs"].get(c, 0.0) for c in all_classes]
                st.bar_chart(pd.DataFrame(chart_data).set_index("Class"))

                # XAI for KG-aware models
                st.write("### Explanation (XAI)")
                for name, res in [
                    ("RawGate",       res_gate),
                    ("Baseline+Rule", res_rule),
                ]:
                    with st.expander(f"{name} — XAI Detail", expanded=False):
                        render_xai(res, name)


if __name__ == "__main__":
    main()
