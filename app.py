import os
import re
import time
import hashlib
import datetime
from collections import defaultdict, deque
from urllib.parse import quote_plus

from flask import Flask, render_template, request, jsonify
from openai import OpenAI

import feedparser
import httpx
from readability import Document

# ----------------------------------------
# CONFIG
# ----------------------------------------
app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

MODEL = "gpt-4o-mini"  # économique et fiable

# Modèles rapides
TEMPLATES = {
    "": {"c": "", "o": "", "q": ""},
    "Choix fournisseur A/B": {
        "c": "PME industrielle, 25 salariés, chaîne d'assemblage, 2 sites en France.",
        "o": "Réduire coûts d'approvisionnement de 8% sans dégrader qualité ni délais.",
        "q": "Faut-il choisir le fournisseur A ou B pour la pièce critique P123 ?",
    },
    "Promo Black Friday": {
        "c": "E-commerce mode éco-responsable, 12k clients actifs, stock limité sur 5 références.",
        "o": "Booster CA de novembre sans dégrader la marge globale.",
        "q": "Doit-on lancer une promotion -30% sur la gamme accessoires ?",
    },
    "Feature SaaS": {
        "c": "SaaS B2B, 200 clients, ticket moyen 120€/mois, churn 4.5%.",
        "o": "Augmenter MRR de 20% en 3 mois.",
        "q": "Prioriser la feature 'Analytics avancés' ou 'Intégration Slack' ?",
    },
    "Politique télétravail": {
        "c": "ESN de 90 personnes, équipes projets réparties sur 3 villes. Clients principaux : PME industrielles. Risques : perte de talents et baisse de satisfaction.",
        "o": "Améliorer rétention et productivité d'équipe.",
        "q": "Faut-il imposer 2 jours au bureau ou rester full remote ?",
    },
}

# Prompt système renforcé
SYSTEM = """Tu es un assistant d'aide à la décision concis et structuré.
Rends EXACTEMENT 4 sections en MAJUSCULES et dans CET ordre : LOGOS, PATHOS, ETHOS, SYNTHÈSE.

Exigences minimales :
- LOGOS : ≥4 puces. Inclure au moins 2 KPI chiffrés (estimation si inconnu) et 2 options comparées.
- PATHOS : ≥3 puces. Adopter le point de vue client/utilisateur (objections, adoption, friction).
- ETHOS : ≥4 puces. Risques (juridique/RGPD, réputation, opérationnel) + parades concrètes.
- SYNTHÈSE : 1 recommandation UNIQUE + plan en 3 étapes (30/60/90 jours) avec responsables et métriques de succès.

Si un bloc 'CONTEXTE EXTERNE' est fourni, intègre les infos pertinentes et cite les sources en fin de SYNTHÈSE avec des marqueurs [1], [2], ...

Style : phrases courtes, actionnable, sans jargon. Évite les généralités creuses.
Ne rajoute pas d’intro/conclusion hors des 4 sections.
Langue : français.
"""

# ----------------------------------------
# UTILS GÉNÉRAUX
# ----------------------------------------
def estimate_cost_tokens(prompt_chars: int, reply_chars: int = 1600) -> dict:
    in_tokens = max(1, prompt_chars // 4)
    out_tokens = max(1, reply_chars // 4)
    total = in_tokens + out_tokens
    price_per_token = 0.0000015   # indicatif
    return {"in": in_tokens, "out": out_tokens, "total": total, "eur": round(total * price_per_token, 4)}

_cache = {}
def cache_key(c, o, q, ext, w):
    return hashlib.sha256((c+"\n"+o+"\n"+q+"\n"+(ext or "")+str(w)).encode()).hexdigest()

hits = defaultdict(deque)

PER_MIN = {
    "/api/analyze": 6,      # reste bas car lourd
    "/api/refine": 60,      # autorise 60 requêtes par minute
    "/api/research": 20,    # moyen
    "default": 40,
}

def allowed(key: str, per_min: int) -> bool:
    now = time.time()
    dq = hits[key]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= per_min:
        return False
    dq.append(now)
    return True

@app.before_request
def _rl():
    ip = request.headers.get("x-forwarded-for", request.remote_addr) or "local"
    path = request.path
    per_min = PER_MIN.get(path, PER_MIN["default"])
    # rate limit par couple ip+path
    key = f"{ip}:{path}"
    if not allowed(key, per_min=per_min):
        return jsonify({
            "ok": False,
            "error": "Trop de requêtes sur cette action. Réessayez dans ~1 minute.",
            "path": path,
            "limit_per_min": per_min
        }), 429

# Hints si brief pauvre
def missing_context_hints(ctx, obj, q):
    hints = []
    if len(re.findall(r"\d", ctx)) < 1:
        hints.append("Ajoutez au moins un chiffre (effectifs, budgets, délais).")
    if not re.search(r"(client|utilisat|marché|équipe|personnel|employé|PME)", ctx, re.I):
        hints.append("Décrivez vos clients, utilisateurs ou équipes.")
    if not re.search(r"(risque|contrainte|qualité|budget|stock|délai)", ctx, re.I):
        hints.append("Mentionnez un risque ou une contrainte clé.")
    return hints


# Vérification qualité sortie
def score_quality(text: str) -> dict:
    upper = text.upper()
    sections = ["LOGOS", "PATHOS", "ETHOS", "SYNTHÈSE"]
    ok_sections = all(s in upper for s in sections)

    def count_bullets(block: str) -> int:
        return len(re.findall(r"(^-|\n-\s)", block))

    def extract(section: str) -> str:
        m = re.search(rf"{section}\s*:?(.*?)(?=\n[A-ZÉÈÎÂÙÇ ]{{3,}}\s*:|\Z)", text, re.S)
        return (m.group(1) if m else "").strip()

    logos = extract("LOGOS")
    pathos = extract("PATHOS")
    ethos = extract("ETHOS")
    syn = extract("SYNTHÈSE")

    bullets = {
        "logos": count_bullets(logos),
        "pathos": count_bullets(pathos),
        "ethos": count_bullets(ethos),
        "synthese": count_bullets(syn),
    }
    kpi_nums = len(re.findall(r"\b\d+(\.\d+)?\s*%?|\b€\s*\d", logos))

    return {
        "ok_sections": ok_sections,
        "bullets": bullets,
        "kpi_in_logos": kpi_nums >= 2,
        "ok": ok_sections and bullets["logos"]>=4 and bullets["pathos"]>=3 and bullets["ethos"]>=4 and (("1." in syn) or bullets["synthese"]>=3) and kpi_nums>=2
    }

# ----------------------------------------
# RECHERCHE (Google News RSS)
# ----------------------------------------
def google_news_rss(query: str, max_items: int = 5, lang: str = "fr", country: str = "FR"):
    base = "https://news.google.com/rss/search?"
    q = f"q={quote_plus(query)}&hl={lang}&gl={country}&ceid={country}:{lang}"
    url = base + q
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:max_items]:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        summary = getattr(entry, "summary", "").strip()
        published = getattr(entry, "published", "") or getattr(entry, "updated", "")
        source = ""
        if "source" in entry and hasattr(entry.source, "title"):
            source = entry.source.title
        items.append({"title": title, "url": link, "summary": summary, "published": published, "source": source})
    return items

async def _fetch(url: str, timeout=10):
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client_http:
        r = await client_http.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text

def fetch_page_text(url: str) -> str:
    try:
        import asyncio
        html = asyncio.run(_fetch(url))
        doc = Document(html)
        content = doc.summary(html_partial=True)
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000]
    except Exception:
        return ""

def build_research_context(query: str, max_items: int = 5) -> dict:
    items = google_news_rss(query, max_items=max_items)
    sources = []
    blocks = []
    for i, it in enumerate(items, start=1):
        body = fetch_page_text(it["url"]) or it["summary"]
        if not body:
            continue
        published = it["published"] or ""
        meta = f"[{i}] {it['title']} — {it['source'] or 'Source inconnue'} — {published} — {it['url']}"
        snippet = (body or "").strip()[:800]
        sources.append(meta)
        blocks.append(f"[{i}] {it['title']} : {snippet}")

    context = ""
    if blocks:
        today = datetime.date.today().isoformat()
        context = f"CONTEXTE EXTERNE (recherche '{query}', {today})\n" + "\n\n".join(blocks)
    return {"context": context, "sources": sources}

# ----------------------------------------
# OPENAI
# ----------------------------------------
def call_openai(messages, tries=2):
    for i in range(tries):
        try:
            return client.chat.completions.create(
                model=MODEL, temperature=0.4, max_tokens=700, messages=messages, timeout=40
            )
        except Exception:
            if i == tries-1: raise
            time.sleep(0.8)

def ask_ai(contexte: str, objectif: str, question: str, weights: dict = None, external: str = "") -> str:
    if not client:
        return "⚠️ OPENAI_API_KEY non configurée"

    w = weights or {"cost":50,"time":50,"risk":50}
    extra = f"Priorités: Coût={w.get('cost',50)}/100, Délai={w.get('time',50)}/100, Risque={w.get('risk',50)}/100. Prends ces priorités en compte dans ta recommandation."
    ext_block = f"\n\n{external}\n\n" if external else ""

    # Cache
    key = cache_key(contexte, objectif, question, external, w)
    if key in _cache:
        return _cache[key]

    user_prompt = f"""
CONTEXTE :
{contexte}

OBJECTIF :
{objectif}

QUESTION :
{question}
{ext_block}
{extra}
"""

    resp = call_openai([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt},
    ])
    content = resp.choices[0].message.content.strip()

    # Vérification qualité + auto-fix
    q = score_quality(content)
    if not q["ok"]:
        fix_prompt = f"""Réécris le texte ci-dessous en respectant STRICTEMENT :
- LOGOS ≥4 puces, ≥2 KPI chiffrés
- PATHOS ≥3 puces
- ETHOS ≥4 puces
- SYNTHÈSE = 1 reco + plan 30/60/90 jours
NE CHANGE PAS le fond, améliore la forme et la précision.
TEXTE :
{content}"""
        resp2 = call_openai([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": fix_prompt},
        ])
        content = resp2.choices[0].message.content.strip()

    _cache[key] = content
    return content

# ----------------------------------------
# ROUTES
# ----------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    contexte = objectif = question = ""
    weights = {"cost": 50, "time": 50, "risk": 50}
    selected_tpl = ""
    use_search = False
    search_query = ""

    if request.method == "POST":
        # Application d'un modèle
        if "apply_tpl" in request.form and request.form.get("tpl") in TEMPLATES:
            selected_tpl = request.form.get("tpl")
            t = TEMPLATES[selected_tpl]
            contexte, objectif, question = t["c"], t["o"], t["q"]
        else:
            # Formulaire normal
            contexte = request.form.get("contexte", "").strip()
            objectif = request.form.get("objectif", "").strip()
            question = request.form.get("question", "").strip()
            weights["cost"] = int(request.form.get("weight_cost", 50))
            weights["time"] = int(request.form.get("weight_time", 50))
            weights["risk"] = int(request.form.get("weight_risk", 50))
            use_search = (request.form.get("use_search") == "on")
            search_query = request.form.get("search_query", "").strip()

            # Hints si brief pauvre
            hints = missing_context_hints(contexte, objectif, question)
            if len(contexte) < 40 or len(objectif) < 10 or len(question) < 10 or hints:
                result = "Brief insuffisant.\n" + "\n".join(f"- {h}" for h in hints)
            else:
                external = ""
                if use_search and search_query:
                    ctx = build_research_context(search_query, max_items=5)
                    external = ctx["context"]
                result = ask_ai(contexte, objectif, question, weights, external=external)

    prompt_chars = len(contexte) + len(objectif) + len(question) + 280
    cost = estimate_cost_tokens(prompt_chars)

    return render_template("index.html",
                           result=result,
                           contexte=contexte, objectif=objectif, question=question,
                           weights=weights,
                           tpl=selected_tpl, templates=TEMPLATES,
                           use_search=use_search, search_query=search_query,
                           cost=cost)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True, silent=False)
    contexte = (data.get("contexte") or "").strip()
    objectif = (data.get("objectif") or "").strip()
    question = (data.get("question") or "").strip()
    weights = data.get("weights", {})
    use_search = bool(data.get("use_search"))
    search_query = (data.get("search_query") or "").strip()

    hints = missing_context_hints(contexte, objectif, question)
    if len(contexte) < 40 or len(objectif) < 10 or len(question) < 10 or hints:
        return jsonify({
            "ok": False,
            "error": "Brief insuffisant ou contexte invalide",
            "hints": hints,
            "context_length": len(contexte),
            "objectif_length": len(objectif),
            "question_length": len(question)
        }), 400

    external = ""
    if use_search and search_query:
        ctx = build_research_context(search_query, max_items=5)
        external = ctx["context"]

    prompt_chars = len(contexte) + len(objectif) + len(question) + 280
    cost = estimate_cost_tokens(prompt_chars)

    result = ask_ai(contexte, objectif, question, weights, external=external)
    return jsonify({"ok": True, "result": result, "cost": cost})

@app.route("/api/research", methods=["POST"])
def api_research():
    data = request.get_json(force=True, silent=False)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"ok": False, "error": "Paramètre 'query' manquant."}), 400
    ctx = build_research_context(query, max_items=5)
    return jsonify({"ok": True, "context": ctx["context"], "sources": ctx["sources"]})

@app.route("/api/refine", methods=["POST"])
def refine():
    data = request.get_json(force=True)
    text = data.get("text", "")
    mode = data.get("mode", "kpi")

    instructions = {
        "kpi": "Ajoute dans LOGOS au moins 3 KPI chiffrés supplémentaires, précise hypothèses.",
        "parades": "Dans ETHOS, liste 3 parades supplémentaires, concrètes et mesurables.",
        "plan": "Dans SYNTHÈSE, détaille chaque étape (responsable, métrique, délai exact).",
    }
    instr = instructions.get(mode, instructions["kpi"])

    prompt = f"Améliore le texte suivant en respectant STRICTEMENT le format 4 sections. {instr}\n\nTEXTE:\n{text}"
    r = call_openai([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return jsonify({"ok": True, "result": r.choices[0].message.content.strip()})


@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(debug=True)
