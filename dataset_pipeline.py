# dataset_pipeline_oss_final_debug_cov_dangling.py

import os, re, json, time, math, random, string, logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

import spacy
from unstructured.partition.docx import partition_docx
from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser, LanguageConfig
from llama_index.core import Document

from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import requests
from collections import defaultdict

from docling.document_converter import DocumentConverter,WordFormatOption,PdfFormatOption,HTMLFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions,PaginatedPipelineOptions
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from pydantic import BaseModel







# ------------------------- Tokenizer -------------------------
tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.encoding_for_model("gpt-4o"),
    max_tokens=1000,  
)
# -------------------------

# -------------------------
# Config
# -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

SYSTEM_PROMPT = (
    "You are a highly knowledgeable assistant specialized in domain-specific instructions. "
    "Always provide step-by-step, accurate, and context-aware answers. Never assume; rely only "
    "on given information. Avoid general knowledge unless explicitly asked. Use concise, formal language."
)


CONFIG = {
    # IO
    "RAW_DOCS_DIR": "/home/ubuntu/dataset_prep_pipeline/Documents",
    "OUT_DIR": "./dataset_out",
    "TRAIN_FILE": "train.jsonl",
    "VAL_FILE": "val.jsonl",
    "TEST_FILE": "test.jsonl",
    "SPLIT": (0.90, 0.05, 0.05),

    # LLM
    "LLM_ENDPOINT": "http://prod-llm.excelacomcloud.net:11434/api/generate",
    "LLM_MODEL_NAME": "llama3.2-vision:latest",
    "NUM_CTX_TOKENS": 8192,
    "ANSWER_MAX_TOKENS": 2048,
    "TEMPERATURE_QA": 0.25,      # slightly lower to improve format adherence
    "TEMPERATURE_PARA": 0.45,
    "N_BEST": 5,                 # try more candidates, pick the cleanest
    "REQUEST_DELAY": 0.5,
    "MAX_RETRIES": 3,

    # Chunking/budgets
    "SEMANTIC_SPLITTER_CONFIG": {
        "initial_threshold": 0.75,
        "appending_threshold": 0.7,
        "merging_threshold": 0.4,
        "max_chunk_size": 6000,  # chars
        "spacy_model": "en_core_web_md",
    },
    "MERGE_COUNT": 1,
    "CONTENT_BUDGET_TOKENS": 4000,

    # Validation/dedup
    "SEMANTIC_DEDUP_THRESH": 0.92,
    "KEEP_SYSTEM_IN_DATA": True,

    # Repair/drop
    "ENABLE_PLACEHOLDER_REPAIR": True,
    "ENABLE_DROP_ON_UNREPAIRED_PLACEHOLDER": True,

    # Coverage tags & rebalancing
    "ENABLE_COVERAGE_TAGS": True,
    "ENABLE_REBALANCE": True,
    "REBALANCE_TARGETS": {
        "state": {"error_or_validation": 0.20},
    },

    # Coverage/continuation thresholds
    "COVERAGE_MIN_JACCARD": 0.60,
    "CONTINUATION_UI_OVERLAP_MIN": 1,  # >=1 shared UI term across next-window step line → continuation
}

# Debug controls
DEBUG = {
    "SAVE_RAW_QA": True,     # dump raw LLM outputs per window
    "LOG_DROPS": True,       # print counters per drop reason
}
os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
DEBUG_DIR = os.path.join(CONFIG["OUT_DIR"], "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

# -------------------------
# Init
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
nlp = spacy.load(CONFIG["SEMANTIC_SPLITTER_CONFIG"]["spacy_model"], disable=["ner","tagger","lemmatizer"])
EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
PAIR_EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # for Q+A dedup

# drop counters
DROP_COUNTER = defaultdict(int)
def count_drop(reason: str):
    if DEBUG.get("LOG_DROPS"):
        DROP_COUNTER[reason] += 1

# -------------------------
# Helpers
# -------------------------
def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))

def clip_tokens_rough(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4
    return text if len(text) <= max_chars else text[:max_chars]

def post_with_retry(url: str, body: Dict[str, Any], tries=3, base_sleep=0.7):
    last = None
    for i in range(tries):
        try:
            r = requests.post(url, json=body, timeout=180)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if i < tries - 1:
                time.sleep(base_sleep * (2 ** i) + random.random() * 0.25)
            else:
                logging.error(f"POST failed after {tries} attempts: {e}")
    raise last

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s{2,}", " ", s.replace("\t"," ")).strip()

def strip_emphasis(s: str) -> str:
    return re.sub(r"^\*+|\*+$", "", s).strip()

def normalize_q_text(q: str) -> str:
    q = strip_emphasis(q)
    q = normalize_spaces(q)
    if q.endswith("**"):
        q = q[:-2].strip()
    return q

def normalize_answer_text(a: str) -> str:
    a = a.replace("appreciate option", "appropriate option")
    a = normalize_spaces(a)
    return a

def normalize_question_for_dedup(q: str) -> str:
    q = q.lower().strip()
    q = q.translate(str.maketrans("", "", string.punctuation))
    q = re.sub(r"\s+", " ", q)
    return q

# -------------------------
# Cleaning
# -------------------------
NOTE_REQUIRED = re.compile(r"\b(click|select|enter|must|required|ensure|choose|navigate|save|submit|delete|clone|search)\b", re.I)

def should_drop_line(line: str) -> bool:
    s = line.strip()
    if not s: return True
    if re.fullmatch(r"[.\-\*\•\s]+", s): return True
    if re.fullmatch(r"click\s*\.", s.lower()): return True
    if re.match(r"^note[:\-]", s.lower()) and not NOTE_REQUIRED.search(s): return True
    return False

def clean_text_block(text: str) -> str:
    lines = text.split("\n")
    kept = [ln for ln in lines if not should_drop_line(ln)]
    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(r"(?m)^\s*\d+\.\s*[\.\-:]?\s*$", "", cleaned)
    cleaned = re.sub(r"\.{2,}$", ".", cleaned)
    cleaned_lines = [ln for ln in cleaned.split("\n") if not re.fullmatch(r"[.\-\*\•\s]+", ln)]
    return "\n".join(cleaned_lines).strip()

# -------------------------
# Load & chunk
# -------------------------

def load_clean_text_from_docling(docx_path: str) -> str:
   
    # Define pipeline options (no need for PipelineType anymore)
    pdf_pipeline_options = PdfPipelineOptions(paginate=True)
    html_pipeline_options = PaginatedPipelineOptions()
    docx_pipeline_options = PaginatedPipelineOptions()
 
    # Create document converter
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.HTML: HTMLFormatOption(pipeline_options=html_pipeline_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=docx_pipeline_options),
        },
    )
 
    # Create a hybrid chunker
    chunker = HybridChunker(tokenizer=tokenizer,merge_peers=True)
 
    # Convert and extract document content
    result = converter.convert(docx_path).document
 
    # Chunk the result
    filtered_chunks = chunker.chunk(result)
 
    all_chunks_text = []

    for chunk in filtered_chunks:
        if chunk.text and chunk.text.strip():
            # default to empty string for headings
            headings = ""
            if hasattr(chunk, "meta") and getattr(chunk.meta, "headings", None):
                if chunk.meta.headings:  # non-empty list
                    headings = " > ".join(chunk.meta.headings)
            
            # create the combined string in a variable
            text_part = f"{headings}\n{chunk.text.strip()}" if headings else chunk.text.strip()
            
            # append to list
            all_chunks_text.append(text_part)

    
    return all_chunks_text

def load_clean_text_from_docx(docx_path: str) -> str:
    try:
        elements = partition_docx(file=docx_path)
        filtered = [
            el.text for el in elements
            if getattr(el, "category", None) in ["NarrativeText", "Title", "ListItem"]
            and getattr(el, "text", None)
        ]
        return clean_text_block("\n".join(filtered))
    except Exception as e:
        logging.error(f"DOCX processing failed for {docx_path}: {e}")
        return ""

def chunk_text_semantically(text: str):
    config = LanguageConfig(language="english", spacy_model=CONFIG["SEMANTIC_SPLITTER_CONFIG"]["spacy_model"])
    splitter = SemanticDoubleMergingSplitterNodeParser(
        language_config=config,
        initial_threshold=CONFIG["SEMANTIC_SPLITTER_CONFIG"]["initial_threshold"],
        appending_threshold=CONFIG["SEMANTIC_SPLITTER_CONFIG"]["appending_threshold"],
        merging_threshold=CONFIG["SEMANTIC_SPLITTER_CONFIG"]["merging_threshold"],
        max_chunk_size=CONFIG["SEMANTIC_SPLITTER_CONFIG"]["max_chunk_size"],
    )
    return splitter.get_nodes_from_documents([Document(text=text)])

def merge_chunks(chunks, merge_count: int) -> List[str]:
    out = []
    for i in range(0, len(chunks), merge_count):
        group = chunks[i:i+merge_count]
        joined = "\n\n".join(ch.text.strip() for ch in group if ch.text and ch.text.strip())
        if joined: out.append(joined)
    return out

# -------------------------
# Prompting
# -------------------------
QA_PROMPT = """You are a documentation extraction specialist.

Task:
Convert content1 into granular Q&A pairs. Use content2 ONLY to complete missing details from content1. Never create Q/A from content2 alone.

Important: Follow the EXACT format shown below.

Example:
Q: How do I create a user group?
A:
1. Go to Administration > Groups.
2. Click [CREATE GROUP].
3. Enter Group Name and Description.
4. Click [SAVE].

Hard Rules:
- Answers must be fully grounded in content1; use content2 only to fill exact missing labels/fields.
- Never output placeholders (“click .”, “enter required details”, “make necessary changes”). Use concrete UI labels only (e.g., [Save], Search icon). If a required label is missing, skip that step or omit the pair.
- Procedures must be numbered (1., 2., …). Lists of options must be plain bullets.
- Do not include next-section headings, notes, or references to external figures/tables.
- Stop exactly when the documented procedure ends. Do not end early after success messages if more steps exist.
- No meta comments or pipeline notes.


Output Format (JSON array of objects):
    [
    {{"question": "Question 1 text", "ans": "Answer 1 text"}},
    {{"question": "Question 2 text", "ans": "Answer 2 text"}}
    ]

    Requirements:
    - Return ONLY valid JSON array
    - Each object must have "question" and "ans" fields
    - Generate at least 3 question-answer pairs

content1:
{c1}

content2:
{c2}
"""

PARA_PROMPT = """You are a linguistic expert. Generate three semantically equivalent paraphrases of the question.

Constraints:
- Keep the exact meaning and domain context.
- Avoid generic forms like "Steps to...".
- Output only:
Q1: ...
Q2: ...
Q3: ...

Question:
{q}
"""

def query_llm(prompt: str, temperature: float, max_tokens: int) -> str:
    body = {
        "model": CONFIG["LLM_MODEL_NAME"],
        "prompt": prompt,
        "stream": False,
        "format": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question text"
                },
                "ans": {
                    "type": "string",
                    "description": "The answer text"
                }
            },
            "required": ["question", "ans"]
            }
        },
        "options": {"max_tokens": max_tokens, "temperature": temperature, "num_ctx": CONFIG["NUM_CTX_TOKENS"]},
    }
    r = post_with_retry(CONFIG["LLM_ENDPOINT"], body, tries=CONFIG["MAX_RETRIES"])
    return (r.json().get("response") or "").strip()

def score_output(qa_text: str) -> float:
    pairs = QA_PATTERN.findall(qa_text)
    n_pairs = len(pairs)
    step_counts = [len(re.findall(r"(?m)^\s*\d+\.", a)) for _, a in pairs]
    vague_pen = len(re.findall(r"\brequired details|relevant information|necessary changes|etc\.", qa_text.lower()))
    ui_hits = len(UI_TOKEN_RE.findall(qa_text))
    return (
        2.0 * n_pairs
        + 0.30 * sum(min(s, 6) for s in step_counts)
        + 0.15 * ui_hits
        - 1.0 * vague_pen
    )

def best_of_k(prompt: str, k: int) -> str:
    cands = []
    for _ in range(k):
        out = query_llm(prompt, CONFIG["TEMPERATURE_QA"], CONFIG["ANSWER_MAX_TOKENS"])
        cands.append((score_output(out), out))
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

# -------------------------
# Validators & repair
# -------------------------
QA_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?\s*Q\d*[:\-\.]?\s*(.*?)\s*\n\s*(?:\*\*)?\s*A\d*[:\-\.]?\s*(.*?)(?=\n\s*(?:\*\*)?\s*Q\d*[:\-\.]?|\Z)",
    re.IGNORECASE | re.DOTALL
)
PARA_LINE = re.compile(r"^\s*Q\d+\s*[:\-\.]\s*(.+)$", re.IGNORECASE | re.MULTILINE)

SUCCESS_TAIL_PATTERNS = [r"record saved", r"saved successfully", r"success", r"acknowledg(e|ment)", r"confirmation displayed"]
STEP_START_PATTERN = re.compile(r"(?m)^\s*(?:\d+\.|\-|\*)\s+\w+", re.I)
PLACEHOLDER_RE = re.compile(r"\b(click|press|select)\s+(?:\.|button\b\s*$)", re.I)

BUTTON_HINTS = ["Save", "Submit", "Add", "Back", "Delete", "Clone", "Search", "Cancel", "OK", "Next", "Finish", "Update"]

# unresolved UI markers & step parsing
ANGLE_IMAGE_RE = re.compile(r"<\s*image\s*\d+\s*>", re.I)
GENERIC_PLACEHOLDER_RE = re.compile(r"\b(click|press|select)\s+(?:here|this|there)\b", re.I)
STEP_LINE_RE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$", re.M)

# split helpers
NEUTRAL_PREFIX_RE = re.compile(r"\b(open|go to|navigate|from the listing|on the .* page)\b", re.I)
SUFFIX_COMMON_RE  = re.compile(r"\b(save|submit|confirm|ok)\b", re.I)

META_NOTE_RE = re.compile(r"(?:^|\n)\s*(note:)?\s*the extracted q&a pairs.*$", re.I)
VAGUE_PHRASES = [
    r"\benter the required details\b",
    r"\benter relevant information\b",
    r"\bmake the necessary changes\b",
    r"\bclick add\b$",
    r"\bthe application displays the result\b",
]
UI_TOKEN_RE = re.compile(r"\[(Save|Submit|Add|Back|Delete|Search|OK|Cancel|Next|Finish)\]|icon|button|drop-?down|field|tab", re.I)

def strip_meta_lines(a: str) -> str:
    return META_NOTE_RE.sub("", a).strip()

def has_external_refs(text: str) -> bool:
    return bool(re.search(r"refer to (the )?(table|figure|image)|below table|above figure", text, re.I))

def leakage_or_heading(text: str) -> bool:
    if re.search(r"(?m)^\s*(Section\s+\d+)", text): return True
    if re.search(r"(?m)^[A-Z][A-Z\s\-]{5,}$", text): return True
    if re.search(r"\*\*[^*]+\*\*", text): return True
    return False

def scrub_unresolved_ui_refs(text: str) -> str:
    """Remove angle-image tokens and 'click here/this/there' phrases, then trim."""
    text = ANGLE_IMAGE_RE.sub("", text)
    text = GENERIC_PLACEHOLDER_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def is_too_vague(a: str) -> bool:
    if any(re.search(p, a, re.I) for p in VAGUE_PHRASES):
        if not UI_TOKEN_RE.search(a):
            return True
    steps = re.findall(r"(?m)^\s*\d+\.\s+", a)
    if len(steps) == 1 and not re.search(r"\b(save|submit|exit|open|expand|search)\b", a, re.I):
        return True
    return False

def has_duplicate_terminal_actions(a: str) -> bool:
    lines = [ln.strip().lower() for ln in a.splitlines() if re.match(r"^\s*\d+\.", ln)]
    seen = set(); dups = 0
    for ln in lines:
        if ln in seen: dups += 1
        seen.add(ln)
    return dups >= 1

def misaligned(q: str, a: str) -> bool:
    # Allow minimal lexical overlap for short, UI-driven answers
    q_kws = set(re.findall(r"[a-z]{3,}", q.lower()))
    a_kws = set(re.findall(r"[a-z]{3,}", a.lower()))
    overlap = len(q_kws & a_kws)
    if UI_TOKEN_RE.search(a):  # if UI tokens exist, allow ≥1 overlap
        return overlap < 1
    return overlap < 2

def enforce_numbering(answer: str) -> str:
    lines = [ln.rstrip() for ln in answer.split("\n")]
    step_lines = [ln for ln in lines if re.match(r"^\s*(?:\d+\.\s+|\-\s+|\*\s+|\•\s+)", ln)]
    if step_lines and not any(re.match(r"^\s*\d+\.\s+", ln) for ln in step_lines):
        num = 1; out = []
        for ln in lines:
            if re.match(r"^\s*(?:\-\s+|\*\s+|\•\s+)", ln):
                out.append(re.sub(r"^\s*(?:\-\s+|\*\s+|\•\s+)", f"{num}. ", ln))
                num += 1
            else:
                out.append(ln)
        return "\n".join(out)
    return answer

def renumber_steps(answer: str) -> str:
    steps = [m.groups() for m in STEP_LINE_RE.finditer(answer)]
    if not steps:
        return answer
    n = 1
    out = []
    for _, body in steps:
        body = body.strip()
        if not body:
            continue
        out.append(f"{n}. {body}")
        n += 1
    return "\n".join(out)

def has_nonconsecutive_numbers(answer: str) -> bool:
    nums = [int(m.group(1)) for m in STEP_LINE_RE.finditer(answer)]
    if not nums:
        return False
    return nums != list(range(nums[0], nums[0] + len(nums)))

def looks_early_stopped(answer: str, following_src: str) -> bool:
    tail = answer.strip().lower()[-200:]
    if any(re.search(p, tail, re.I) for p in SUCCESS_TAIL_PATTERNS):
        if STEP_START_PATTERN.search(following_src[:800] if following_src else ""):
            return True
    return False

def extract_button_hints_from_context(ctx: str) -> List[str]:
    cand = set()
    for b in BUTTON_HINTS:
        if re.search(fr"\b{re.escape(b)}\b", ctx, re.I):
            cand.add(b)
    for m in re.finditer(r"\b[A-Z][A-Z0-9_]{2,}\b", ctx):
        cand.add(m.group(0).title())
    return list(cand)

def try_repair_placeholders(answer: str, context: str) -> str or None:
    if not PLACEHOLDER_RE.search(answer):
        return answer
    hints = extract_button_hints_from_context(context)
    for b in hints:
        repl = PLACEHOLDER_RE.sub(rf"\1 {b}", answer)
        if not PLACEHOLDER_RE.search(repl):
            return repl
    return None

def min_steps_required(q: str, a: str) -> bool:
    ql = q.lower()
    steps = len(re.findall(r"(?m)^\s*\d+\.", a))
    single_ok = re.search(r"\b(open|expand|collapse|search|save|submit|confirm|click\s+\[[^\]]+\])\b", a.lower())

    if re.search(r"\bcreate|add|new\b", ql):
        return steps >= 2 or bool(single_ok)
    if re.search(r"\bupdate|edit|modify\b", ql):
        return steps >= 2 or bool(single_ok)
    if re.search(r"\bdelete|remove|terminate|eliminate\b", ql):
        return steps >= 2 or bool(single_ok)
    return True

# Expand inline bullets inside a step into proper sub-bullets
def expand_inline_bullets(answer: str) -> str:
    new_lines = []
    for ln in answer.splitlines():
        if '*' in ln and re.search(r"\*\s+\S", ln):
            if ':' in ln:
                prefix, rest = ln.split(':', 1)
                parts = re.split(r"\s*\*\s+", rest)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 2:
                    new_lines.append(prefix.strip() + ":")
                    for p in parts:
                        new_lines.append("   - " + p)
                    continue
        new_lines.append(ln)
    return "\n".join(new_lines)

# Intent correction from answer text
def tag_intent_from_answer(a: str) -> str:
    s = a.lower()
    if re.search(r"\bdelete|remove\b", s): return "delete"
    if re.search(r"\bcreate|new\b", s): return "create"
    if re.search(r"\bupdate|edit|modify\b", s): return "update"
    if re.search(r"\bsearch|find|filter\b", s): return "search/filter"
    return "configure/ui"

# Pair-level dedup (Q+A together)
SEEN_PAIR_EMB = []
def is_pair_dup(q: str, a: str, thresh=0.95) -> bool:
    vec = np.array(PAIR_EMB.embed_query((q + " || " + a).lower()))
    if not SEEN_PAIR_EMB:
        SEEN_PAIR_EMB.append(vec); return False
    sims = cosine_similarity(np.vstack(SEEN_PAIR_EMB), vec.reshape(1,-1)).ravel()
    if sims.max() >= thresh:
        return True
    SEEN_PAIR_EMB.append(vec)
    return False

# Question-only dedup state
GLOBAL_Q_TEXT = set()
GLOBAL_Q_EMB = []
def is_semantic_dup(q_norm: str, emb_vec: np.ndarray, thresh: float) -> bool:
    if not GLOBAL_Q_EMB: return False
    sims = cosine_similarity(np.vstack(GLOBAL_Q_EMB), emb_vec.reshape(1,-1)).ravel()
    return bool((sims >= thresh).any())
def add_semantic(q_norm: str, emb_vec: np.ndarray):
    GLOBAL_Q_EMB.append(emb_vec)

# Coverage tags
def tag_coverage(q: str, a: str) -> Dict[str, str]:
    tags = {"intent": "configure/ui", "state": "happy_path", "ui_area": "detail_form"}
    ql, al = q.lower(), a.lower()
    if re.search(r"\bclone|copy|duplicate\b", ql): tags["intent"] = "clone/copy"
    elif re.search(r"\bdelete|remove\b", ql): tags["intent"] = "delete"
    elif re.search(r"\bupdate|edit|modify\b", ql): tags["intent"] = "update"
    elif re.search(r"\bcreate|add|new\b", ql): tags["intent"] = "create"
    elif re.search(r"\bsearch|filter|find\b", ql): tags["intent"] = "search/filter"

    if re.search(r"\berror|invalid|required|cannot be blank|denied|timeout|unsuccessful\b", al):
        tags["state"] = "error_or_validation"

    if re.search(r"\blisting\b", al): tags["ui_area"] = "listing"
    elif re.search(r"\bpop-?up|modal\b", al): tags["ui_area"] = "modal_popup"
    elif re.search(r"\btoolbar\b", al): tags["ui_area"] = "toolbar"
    elif re.search(r"\btab\b", al): tags["ui_area"] = "tabs"
    else: tags["ui_area"] = "detail_form"
    return tags

# -------------------------
# Coverage & continuation
# -------------------------
STEP_CAND_RE = re.compile(
    r'^\s*\d+\.\s+.+$|^\s*(Click|Select|Enter|Choose|Go to|Open|Press)\b.+',
    re.I | re.M
)


def extract_step_candidates(text: str) -> List[str]:
    cands = [re.sub(r'\s+', ' ', s.strip()) for s in STEP_CAND_RE.findall(text)]
    # normalize tuples if any
    norm = []
    for x in cands:
        if isinstance(x, tuple):
            s = " ".join([t for t in x if t]).strip()
            if s: norm.append(s)
        else:
            norm.append(x)
    # de-dup & keep only reasonable length
    out, seen = [], set()
    for s in norm:
        k = s.lower()
        if len(k) >= 10 and k not in seen:
            seen.add(k); out.append(s)
    return out

def normalize_step(s: str) -> str:
    s = s.lower()
    s = re.sub(r'^\s*\d+\.\s*', '', s)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# NEW: dangling-step detection (colon or incomplete tail phrases)
def has_dangling_step(answer: str) -> bool:
    last = answer.strip().splitlines()[-1].strip().lower()
    if last.endswith(":"):
        return True
    if re.search(r"\b(including|such as|for example|consists of|following)\b$", last):
        return True
    if last.endswith("..."):
        return True
    # very short last step (<= 3 words) often indicates truncation
    if re.match(r"^\s*\d+\.\s+\S+(?:\s+\S+){0,2}\s*$", answer.strip().splitlines()[-1]):
        return True
    return False

def coverage_ok(answer: str, source_span: str, min_jacc: float) -> bool:
    src_steps_raw = extract_step_candidates(source_span)
    src_steps = [normalize_step(x) for x in src_steps_raw]
    if not src_steps:
        return True  # nothing obvious to compare, don't block
    ans_steps_raw = re.findall(r'(?m)^\s*\d+\.\s+(.+)$', answer)
    ans_steps = [normalize_step(x) for x in ans_steps_raw]
    if not ans_steps:
        return False

    # Jaccard coverage
    inter = len(set(ans_steps) & set(src_steps))
    union = len(set(ans_steps) | set(src_steps))
    jacc = inter / max(1, union)

    # Stricter final-step checks
    last_src = src_steps[-1]
    last_ans = ans_steps[-1]

    # Reject if last answer step looks incomplete
    if len(last_ans.split()) < 4:
        return False
    if ans_steps_raw and ans_steps_raw[-1].strip().endswith(":"):
        return False
    if re.search(r"\b(including|such as|for example|consists of|following)\b$", ans_steps_raw[-1].strip().lower()):
        return False

    # Tail must match (substring either direction)
    tail_match = last_src[:30] in last_ans or last_ans[:30] in last_src

    return jacc >= min_jacc and tail_match

UI_TOKEN_IN_ANSWER_RE = re.compile(r'\[([^\]]+)\]|(?:icon|button|drop-?down|field|tab|section|pane)', re.I)

def extract_ui_terms_from_answer(answer: str) -> List[str]:
    terms = set()
    for m in re.findall(r'\[([^\]]+)\]', answer):
        t = m.strip()
        if t: terms.add(t.lower())
    for m in re.findall(r'\b(Save|Submit|Add|Back|Delete|Clone|Search|Cancel|OK|Next|Finish|Update)\b', answer, re.I):
        terms.add(m.lower())
    for m in re.findall(r'\b(icon|button|drop-?down|field|tab|section|pane)\b', answer, re.I):
        terms.add(m.lower())
    return list(terms)

def next_window_continuation(answer: str, next_window_text: str, min_overlap: int) -> bool:
    if not next_window_text:
        return False
    has_steps = bool(re.search(r'(?m)^\s*\d+\.\s+', next_window_text)) or bool(STEP_CAND_RE.search(next_window_text))
    if not has_steps:
        return False
    ui_terms = extract_ui_terms_from_answer(answer)
    if not ui_terms:
        return False
    overlap = 0
    lower_c3 = next_window_text.lower()
    for t in ui_terms:
        if t and t in lower_c3:
            overlap += 1
            if overlap >= min_overlap:
                return True
    return False

# -------------------------
# Extraction & paraphrase
# -------------------------
def paraphrase_question(q: str) -> List[str]:
    # Keep paraphrase lightweight or disable if dataset is small
    prompt = PARA_PROMPT.format(q=q.strip())
    try:
        text = query_llm(prompt, CONFIG["TEMPERATURE_PARA"], CONFIG["ANSWER_MAX_TOKENS"])
    except Exception as e:
        logging.error(f"Paraphrasing failed: {e}")
        return []
    cands = [m.strip() for m in PARA_LINE.findall(text)]
    out = []
    for c in cands:
        if not c: continue
        if not c.endswith("?") and not re.match(r"^(list|show|describe|identify|explain)\b", c.lower()):
            continue
        if len(c.split()) < 6: continue
        if re.match(r"^\s*steps to\b", c.lower()): continue
        out.append(c)
    return out[:1]  # reduce near-dupes; set [] to disable entirely

# Split mixed add/remove into separate atomic QAs while keeping neutral prefix/suffix
def split_add_remove(q: str, a: str) -> List[Tuple[str, str]]:
    lines = [ln for ln in a.splitlines() if re.match(r"^\s*\d+\.", ln)]
    prefix, add_steps, rem_steps, suffix = [], [], [], []
    seen_keyword = False

    for ln in lines:
        lower = ln.lower()
        is_add = re.search(r"\b(move right|add|associate|create parameter)\b", lower)
        is_rem = re.search(r"\bremove\b", lower)

        if not seen_keyword and (NEUTRAL_PREFIX_RE.search(lower) or (not is_add and not is_rem and not SUFFIX_COMMON_RE.search(lower))):
            prefix.append(ln)
            continue

        if is_add:
            add_steps.append(ln); seen_keyword = True; continue
        if is_rem:
            rem_steps.append(ln); seen_keyword = True; continue
        if SUFFIX_COMMON_RE.search(lower):
            suffix.append(ln); continue

        if seen_keyword:
            suffix.append(ln)

    def build(q_text, seq_lines):
        if not seq_lines:
            return None
        seq = []
        if prefix: seq.extend(prefix)
        seq.extend(seq_lines)
        if suffix: seq.extend(suffix)
        seq_text = "\n".join(seq)
        if STEP_LINE_RE.search(seq_text):
            seq_text = renumber_steps(seq_text)
        return (q_text, seq_text)

    outs = []
    if add_steps:
        q_add = re.sub(r"(add or remove|remove)", "add", q, flags=re.I)
        pair_add = build(q_add, add_steps)
        if pair_add: outs.append(pair_add)
    if rem_steps:
        q_rem = re.sub(r"(add or remove|add)", "remove", q, flags=re.I)
        pair_rem = build(q_rem, rem_steps)
        if pair_rem: outs.append(pair_rem)

    return outs if outs else [(q, renumber_steps(a) if STEP_LINE_RE.search(a) else a)]

# -------------------------
# Pair windows
# -------------------------
def pairwise_chunks(merged_chunks: List[str]) -> List[Tuple[int, str, str, str]]:
    """Return triples of context: (index, c1, c2, c3) where c3 is the 'next window' peek for continuation checks."""
    pairs = []
    for i in range(len(merged_chunks)):
        c1 = merged_chunks[i]
        c2 = merged_chunks[i+1] if i+1 < len(merged_chunks) else merged_chunks[i]
        c3 = merged_chunks[i+2] if i+2 < len(merged_chunks) else ""
        budget = CONFIG["CONTENT_BUDGET_TOKENS"]
        c1 = clip_tokens_rough(c1, min(budget//2, budget-1000))
        c2 = clip_tokens_rough(c2, min(budget//2, budget - approx_tokens(c1)))
        while approx_tokens(c1) + approx_tokens(c2) > budget and len(c2) > 1000:
            c2 = c2[:int(len(c2)*0.9)]
        pairs.append((i, c1, c2, c3))
    return pairs

# -------------------------
# Extraction pipeline
# -------------------------
def extract_pairs(qa_text: str, src_after: str, src_next: str, meta_base: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = []
    matches = QA_PATTERN.findall(qa_text)
    for q_raw, a_raw in matches:
        q = normalize_q_text(q_raw)
        a = normalize_answer_text(clean_text_block(a_raw))
        if not q or not a:
            count_drop("empty_q_or_a")
            continue

        # strip meta bleed & hard filters
        a = strip_meta_lines(a)
        if has_external_refs(a):
            count_drop("external_refs"); continue
        if leakage_or_heading(a):
            count_drop("heading_leak"); continue

        # scrub unresolved UI refs instead of dropping
        a = scrub_unresolved_ui_refs(a)

        # expand inline bullets, enforce numbering & normalize numbering
        a = expand_inline_bullets(a)
        a = enforce_numbering(a)
        if has_nonconsecutive_numbers(a):
            a = renumber_steps(a)

        # placeholder repair or drop
        if CONFIG["ENABLE_PLACEHOLDER_REPAIR"]:
            repaired = try_repair_placeholders(a, src_after)
            if repaired is None and CONFIG["ENABLE_DROP_ON_UNREPAIRED_PLACEHOLDER"]:
                count_drop("unrepaired_placeholder"); continue
            a = repaired or a

        # general validations
        if is_too_vague(a): count_drop("too_vague"); continue
        if misaligned(q, a): count_drop("misaligned"); continue
        if has_duplicate_terminal_actions(a): count_drop("dup_terminal"); continue
        if looks_early_stopped(a, src_after): count_drop("early_stop"); continue
        if not min_steps_required(q, a): count_drop("min_steps"); continue

        # reject trivial single-step "create parameter"
        if re.search(r"\bcreate parameter\b", a.lower()) and len(re.findall(r"(?m)^\s*\d+\.", a)) < 2:
            count_drop("trivial_create_parameter"); continue

        # MAY split into add/remove atomic pairs
        for q_use, a_use in split_add_remove(q, a):
            # renumber again post-split (safety)
            if has_nonconsecutive_numbers(a_use):
                a_use = renumber_steps(a_use)

            # enforce minimum steps after split (unless clearly single-action)
            if len(re.findall(r"(?m)^\s*\d+\.", a_use)) < 2 and not re.search(r"\bexpand|open|search|save|submit\b", a_use, re.I):
                count_drop("post_split_too_short"); continue

            # NEW: dangling-step guard
            if has_dangling_step(a_use):
                count_drop("dangling_step"); continue

            # Coverage check vs source span(s)
            source_span = meta_base.get("_c1", "") + "\n" + meta_base.get("_c2", "")
            if not coverage_ok(a_use, source_span, CONFIG["COVERAGE_MIN_JACCARD"]):
                source_span_ext = source_span + "\n" + (src_next or "")
                if not coverage_ok(a_use, source_span_ext, CONFIG["COVERAGE_MIN_JACCARD"]):
                    count_drop("coverage_fail"); continue

            # Next-window continuation (reject if likely continuation)
            if next_window_continuation(a_use, src_next, CONFIG["CONTINUATION_UI_OVERLAP_MIN"]):
                count_drop("continuation_next_window"); continue

            # question dedup
            q_norm = normalize_question_for_dedup(q_use)
            q_emb = np.array(EMB.embed_query(q_norm))
            if q_norm in GLOBAL_Q_TEXT: count_drop("q_text_exact_dup"); continue
            if is_semantic_dup(q_norm, q_emb, CONFIG["SEMANTIC_DEDUP_THRESH"]): count_drop("q_text_sem_dup"); continue

            # pair-level dedup
            if is_pair_dup(q_use, a_use): count_drop("pair_sem_dup"); continue

            # build messages/meta
            messages = []
            if CONFIG["KEEP_SYSTEM_IN_DATA"]:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.extend([
                {"role": "user", "content": q_use},
                {"role": "assistant", "content": a_use},
            ])

            meta = dict(meta_base)
            # remove internal coverage fields from meta
            meta.pop("_c1", None); meta.pop("_c2", None)
            if CONFIG["ENABLE_COVERAGE_TAGS"]:
                meta.update(tag_coverage(q_use, a_use))
                meta["intent"] = tag_intent_from_answer(a_use)  # override using answer text

            recs.append({"messages": messages, "meta": meta})
            GLOBAL_Q_TEXT.add(q_norm); add_semantic(q_norm, q_emb)

            # paraphrases (share validated/possibly-split answer)
            for pq in paraphrase_question(q_use):
                pqn = normalize_question_for_dedup(pq)
                if pqn in GLOBAL_Q_TEXT: count_drop("para_q_text_exact_dup"); continue
                pq_emb = np.array(EMB.embed_query(pqn))
                if is_semantic_dup(pqn, pq_emb, CONFIG["SEMANTIC_DEDUP_THRESH"]): count_drop("para_q_text_sem_dup"); continue
                if is_pair_dup(pq, a_use): count_drop("para_pair_sem_dup"); continue

                pm = []
                if CONFIG["KEEP_SYSTEM_IN_DATA"]:
                    pm.append({"role":"system","content":SYSTEM_PROMPT})
                pm.extend([{"role":"user","content": pq}, {"role":"assistant","content": a_use}])
                pm_meta = dict(meta)
                recs.append({"messages": pm, "meta": pm_meta})
                GLOBAL_Q_TEXT.add(pqn); add_semantic(pqn, pq_emb)

    return recs

# -------------------------
# Generation
# -------------------------
def generate_dataset_from_docx(docx_path: str) -> List[Dict[str, Any]]:
    # text = load_clean_text_from_docx(docx_path)
    # if not text: return []
    # nodes = chunk_text_semantically(text)
    # merged = merge_chunks(nodes, CONFIG["MERGE_COUNT"])
    # if not merged: return []

    merged=load_clean_text_from_docling(docx_path)

    records = []
    for idx, c1, c2, c3 in tqdm(pairwise_chunks(merged), desc=f"Q&A from {Path(docx_path).name}"):
        prompt = QA_PROMPT.format(c1=c1, c2=c2)
        if approx_tokens(prompt) > CONFIG["NUM_CTX_TOKENS"] - 512:
            c2 = ""
            prompt = QA_PROMPT.format(c1=clip_tokens_rough(c1, 2000), c2=c2)

        try:
            qa_text = best_of_k(prompt, CONFIG["N_BEST"]) if CONFIG["N_BEST"] > 1 \
                      else query_llm(prompt, CONFIG["TEMPERATURE_QA"], CONFIG["ANSWER_MAX_TOKENS"])
        except Exception as e:
            logging.error(f"LLM QA failed at {docx_path}[{idx}]: {e}")
            continue

        # DEBUG: dump raw LLM output
        if DEBUG.get("SAVE_RAW_QA"):
            bn = Path(docx_path).stem
            with open(os.path.join(DEBUG_DIR, f"qa_raw_{bn}_{idx}.txt"), "w", encoding="utf-8") as f:
                f.write(qa_text)

        src_after = c2[:1200] if c2 else ""
        src_next = c3[:1800] if c3 else ""
        meta_base = {
            "doc_path": str(docx_path),
            "window_index": idx,
            "chunk_pairing": [idx, min(idx+1, len(merged)-1)],
            # internal fields for coverage
            "_c1": c1,
            "_c2": c2,
        }

        recs = extract_pairs(qa_text, src_after, src_next, meta_base)
        records.extend(recs)
        time.sleep(CONFIG["REQUEST_DELAY"])

    return records

# -------------------------
# Rebalancing (optional)
# -------------------------
def rebalance_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not CONFIG.get("ENABLE_REBALANCE"): return records
    targets = CONFIG.get("REBALANCE_TARGETS", {})
    if not targets: return records

    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        meta = r.get("meta", {})
        for key, table in targets.items():
            val = meta.get(key, "__none__")
            counts[key][val] += 1

    total = len(records)
    augmented: List[Dict[str, Any]] = list(records)
    for key, table in targets.items():
        cur = counts[key]
        for val, min_ratio in table.items():
            cur_n = cur.get(val, 0)
            target_n = max(cur_n, int(min_ratio * total))
            if target_n > cur_n:
                cand = [r for r in records if r.get("meta", {}).get(key) == val]
                if not cand: continue
                need = target_n - cur_n
                for _ in range(need):
                    augmented.append(random.choice(cand))

    max_size = int(len(records) * 1.5)
    if len(augmented) > max_size:
        random.Random(SEED).shuffle(augmented)
        augmented = augmented[:max_size]

    logging.info(f"Rebalanced from {len(records)} → {len(augmented)} records")
    return augmented

# -------------------------
# Save & split
# -------------------------
def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def split_and_save(records: List[Dict[str, Any]]):
    random.Random(SEED).shuffle(records)
    n = len(records)
    tr, va, te = CONFIG["SPLIT"]

    # default counts
    n_train = int(n*tr); n_val = int(n*va)
    # ensure at least 1 val if feasible
    if n >= 3 and n_val == 0:
        n_val = 1
        n_train = max(1, n_train - 1)
    # ensure at least 1 test if feasible
    n_test = n - (n_train + n_val)
    if n >= 4 and n_test == 0:
        if n_train > n_val:
            n_train -= 1; n_test = 1
        else:
            n_val -= 1; n_test = 1

    train = records[:n_train]
    val = records[n_train:n_train+n_val]
    test = records[n_train+n_val:]

    write_jsonl(os.path.join(CONFIG["OUT_DIR"], CONFIG["TRAIN_FILE"]), train)
    write_jsonl(os.path.join(CONFIG["OUT_DIR"], CONFIG["VAL_FILE"]), val)
    write_jsonl(os.path.join(CONFIG["OUT_DIR"], CONFIG["TEST_FILE"]), test)
    logging.info(f"Saved: train={len(train)}, val={len(val)}, test={len(test)} in {CONFIG['OUT_DIR']}")

# -------------------------
# Entrypoint
# -------------------------
def main():
    all_records: List[Dict[str, Any]] = []
    for dirpath, _, files in os.walk(CONFIG["RAW_DOCS_DIR"]):
        for fn in files:
            if not fn.lower().endswith(".docx"): continue
            path = os.path.join(dirpath, fn)
            recs = generate_dataset_from_docx(path)
            all_records.extend(recs)

    if not all_records:
        logging.warning("No records generated.")
    else:
        all_records = rebalance_records(all_records)
        split_and_save(all_records)

        # quick stats
        qlens, alens = [], []
        for r in all_records:
            u = next((m["content"] for m in r["messages"] if m["role"]=="user"), "")
            a = next((m["content"] for m in r["messages"] if m["role"]=="assistant"), "")
            qlens.append(len(u.split())); alens.append(len(a.split()))
        logging.info(f"Total={len(all_records)} | avg_q_words={np.mean(qlens):.1f} | avg_a_words={np.mean(alens):.1f}")

    # print drop reasons
    if DEBUG.get("LOG_DROPS") and DROP_COUNTER:
        logging.info("Drop reasons summary:")
        for k, v in sorted(DROP_COUNTER.items(), key=lambda x: (-x[1], x[0])):
            logging.info(f"  {k}: {v}")


# -------------------------
# Format Conversion (conversation -> messages)
# -------------------------
INPUT_DIR = CONFIG["OUT_DIR"]                # reuse generated dataset_out
OUTPUT_DIR = os.path.join(CONFIG["OUT_DIR"], "converted")  # save here

def convert_format(old_json):
    """
    Convert old dataset format with 'conversation' to new format with 'messages'
    """
    if "conversation" not in old_json:  # already new format
        return old_json

    conversation = old_json["conversation"]
    messages = []

    for turn in conversation:
        role = turn["role"]
        content = turn["content"]

        if role == "system":
            messages.append({
                "role": "developer",
                "content": "You are a helpful assistant that explains your reasoning clearly, then gives a concise final answer."
            })
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({
                "role": "assistant",
                "name": "analysis",
                "content": f"(Domain reasoning)\n{content}"
            })
            messages.append({
                "role": "assistant",
                "name": "final",
                "content": content
            })

    return {"messages": messages}

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            old_json = json.loads(line)
            new_json = convert_format(old_json)
            outfile.write(json.dumps(new_json, ensure_ascii=False) + "\n")

def run_format_conversion():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in [CONFIG["TRAIN_FILE"], CONFIG["VAL_FILE"], CONFIG["TEST_FILE"]]:
        input_path = os.path.join(INPUT_DIR, split)
        output_path = os.path.join(OUTPUT_DIR, split)
        if os.path.exists(input_path):
            print(f"Converting {input_path} → {output_path}")
            process_file(input_path, output_path)
        else:
            print(f"⚠️ Skipping {split} (not found)")

# Hook into main
if __name__ == "__main__":
    main()
    run_format_conversion()
    print("✅ Conversion complete! Check folder:", OUTPUT_DIR)

