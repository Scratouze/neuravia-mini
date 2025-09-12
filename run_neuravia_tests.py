# run_neuravia_tests.py
# Tests complets de Neuravia Mini : /api/analyze, /api/refine, /api/research
# Prérequis: pip install requests

import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
import requests

BASE_URL = os.getenv("NEURAVIA_API_URL", "http://127.0.0.1:5000")
API_ANALYZE = f"{BASE_URL}/api/analyze"
API_REFINE  = f"{BASE_URL}/api/refine"
API_RESEARCH= f"{BASE_URL}/api/research"

OUT_DIR = "logs"

# -------------------------------------------------------------------
# Jeux de tests enrichis
# -------------------------------------------------------------------
TESTS: List[Dict[str, Optional[str]]] = [
    # 1) Décision fournisseur avec recherche
    {
        "name": "Choix fournisseur A/B",
        "contexte": ("PME industrielle de 25 salariés, 2 sites en France. Production de pièces mécaniques aéronautiques. "
                     "Volume annuel : 12 000 pièces P123. Délai critique 5 jours. A certifié ISO 9001, B -5% prix."),
        "objectif": "Réduire les coûts d'approvisionnement de 8% sans dégrader la qualité ni les délais.",
        "question": "Faut-il choisir le fournisseur A ou B pour la pièce P123 et comment négocier les conditions ?",
        "search": "fournisseur aéronautique prix qualité certification ISO 9001",
    },
    # 2) Politique télétravail (RH)
    {
        "name": "Politique télétravail",
        "contexte": ("ESN de 90 personnes réparties sur 3 villes, turnover 12%/an. "
                     "50% des collaborateurs demandent + de télétravail. Clients B2B banque/assurance."),
        "objectif": "Améliorer la rétention et la productivité des équipes en optimisant le mode de travail hybride.",
        "question": "Imposer 2 jours au bureau/semaine ou rester full remote ?",
        "search": "télétravail productivité full remote entreprise France 2025",
    },
    # 3) E-commerce Black Friday
    {
        "name": "E-commerce Black Friday",
        "contexte": ("Boutique e-commerce mode éco-responsable, 12 000 clients actifs, 5 refs en stock critique. "
                     "Marge actuelle: 42%. Budget marketing Q4: 15 000 €."),
        "objectif": "Augmenter le CA de novembre sans détruire la marge ni accroître le taux de retours.",
        "question": "Vaut-il mieux -30% sur accessoires ou une offre ciblée panier moyen ?",
        "search": "tendances e-commerce Black Friday France 2025",
    },
    # 4) SaaS (priorisation produit)
    {
        "name": "Choix de feature SaaS",
        "contexte": ("SaaS B2B facturation, 200 clients actifs. Ticket moyen 120€/mois. Churn 4.5%/mois. "
                     "Un concurrent a lancé une intégration Slack."),
        "objectif": "Augmenter MRR de 20% en 3 mois et réduire le churn sous 3%.",
        "question": "Prioriser 'Analytics avancés' ou 'Intégration Slack' ?",
        "search": "tendances SaaS intégrations analytics Slack churn 2025",
    },
    # 5) Gestion de crise / qualité
    {
        "name": "Crise qualité produit alimentaire",
        "contexte": ("PME agroalimentaire 150 employés, CA 12M€/an. Produit phare: barres protéinées GMS. "
                     "Rappel produit récent (lot contaminé). Perte client: 8%."),
        "objectif": "Limiter l'impact médiatique/commercial et corriger le problème qualité.",
        "question": "Communiquer proactivement ou attendre les résultats d'enquête ?",
        "search": "crise alimentaire rappel produit communication gestion",
    },
    # 6) CRM & RGPD
    {
        "name": "Choix logiciel CRM RGPD",
        "contexte": ("Retail 300 collaborateurs, 3M de clients en base CRM. Objectif 2025: conformité RGPD et meilleure CX."),
        "objectif": "Choisir un CRM conforme RGPD et scalable pour 3 ans.",
        "question": "Salesforce, Hubspot ou solution locale sur-mesure ?",
        "search": "CRM RGPD conformité solution France 2025",
    },
    # 7) Volontairement flou (doit renvoyer des hints en 400)
    {
        "name": "Cas volontairement flou",
        "contexte": "Entreprise en croissance.",
        "objectif": "Réduire les coûts.",
        "question": "Quel fournisseur choisir ?",
        "search": None,
    },
    # 8) Stress test (texte long)
    {
        "name": "Stress test texte long",
        "contexte": ("Groupe industriel 50 000 employés/25 pays. 320 références produits. CA 5 Md€. "
                     "Rotation stock 15%. 10% du CA en APAC. Migration SAP S/4HANA. Adoption IA générative."),
        "objectif": "Optimiser supply chain globale et réduire le BFR de 5% en 12 mois.",
        "question": "Internaliser la logistique ou rester avec le prestataire actuel ?",
        "search": "tendances supply chain 2025 SAP S4HANA IA",
    },
]

# Pondérations par défaut (peuvent être custom)
DEFAULT_WEIGHTS = {"cost": 50, "time": 50, "risk": 50}

# Pacing pour éviter 429 côté serveur
PAUSE_BETWEEN_CALLS = 0.6  # secondes

# -------------------------------------------------------------------
# Helpers HTTP avec retry/backoff (gère 429/5xx)
# -------------------------------------------------------------------
def request_json(method: str, url: str, json_payload: dict, timeout: int = 90, retries: int = 3):
    backoff = 0.8
    last_err_text = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.request(method, url, json=json_payload, timeout=timeout)
            # Essayer de parser JSON si content-type est JSON
            ct = r.headers.get("content-type", "")
            if "application/json" in ct:
                data = r.json()
            else:
                # Pas JSON → renvoyer le texte
                data = None
                last_err_text = r.text

            if r.status_code == 429:
                # Rate-limit : backoff et retry
                time.sleep(backoff)
                backoff *= 1.6
                continue

            return r.status_code, data if data is not None else {"raw": last_err_text}
        except requests.RequestException as e:
            last_err_text = str(e)
            time.sleep(backoff)
            backoff *= 1.6
    # si on sort de la boucle
    return None, {"raw": last_err_text or "Request failed"}

def has_changed(original: str, new_text: str) -> bool:
    if not original or not new_text:
        return False
    a = original.strip()
    b = new_text.strip()
    return b != a and len(b) >= len(a)  # simple heuristique

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
def test_analyze(test: Dict) -> Dict:
    payload = {
        "contexte": test["contexte"],
        "objectif": test["objectif"],
        "question": test["question"],
        "weights": DEFAULT_WEIGHTS,
    }
    if test.get("search"):
        payload["use_search"] = True
        payload["search_query"] = test["search"]

    status, data = request_json("POST", API_ANALYZE, payload, timeout=120, retries=4)
    ok = bool(status == 200 and isinstance(data, dict) and data.get("ok"))
    return {"ok": ok, "status": status, "data": data}

def test_refine(text: str, mode: str) -> Dict:
    payload = {"text": text, "mode": mode}
    status, data = request_json("POST", API_REFINE, payload, timeout=90, retries=4)
    ok = bool(status == 200 and isinstance(data, dict) and data.get("ok"))
    return {"ok": ok, "status": status, "data": data}

def test_research(query: str) -> Dict:
    payload = {"query": query}
    status, data = request_json("POST", API_RESEARCH, payload, timeout=60, retries=3)
    ok = bool(status == 200 and isinstance(data, dict) and data.get("ok"))
    return {"ok": ok, "status": status, "data": data}

# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = os.path.join(OUT_DIR, f"neuravia_tests_{ts}.md")
    out_json = os.path.join(OUT_DIR, f"neuravia_tests_{ts}.json")

    lines = []
    results = []

    def log_err_block(title: str, status, data):
        lines.append(f"- ❌ **{title}**")
        lines.append(f"  - Statut: {status}")
        if isinstance(data, dict):
            err = data.get("error")
            raw = data.get("raw")
            hints = data.get("hints")
            if err:
                lines.append(f"  - Erreur: `{err}`")
            if hints:
                if isinstance(hints, list) and hints:
                    lines.append("  - Hints:")
                    for h in hints:
                        lines.append(f"    - {h}")
            if raw and not err:
                # cas texte brut (ex: 429 textual)
                lines.append(f"  - Réponse brute: `{raw[:300]}`")
        else:
            lines.append(f"  - Payload non JSON: `{str(data)[:300]}`")
        lines.append("")

    header = f"# Neuravia Mini – Rapport complet ({ts})\n\nAPI: `{BASE_URL}`\n\n---\n"
    lines.append(header)

    for i, test in enumerate(TESTS, start=1):
        lines.append(f"## {i}. {test['name']}\n")
        lines.append("**Entrées**")
        lines.append(f"- Contexte : {test['contexte']}")
        lines.append(f"- Objectif : {test['objectif']}")
        lines.append(f"- Question : {test['question']}")
        if test.get("search"):
            lines.append(f"- Recherche : {test['search']}")
        lines.append("")

        # 1) Analyse
        res_analyze = test_analyze(test)
        if not res_analyze["ok"]:
            log_err_block("Analyse échouée", res_analyze["status"], res_analyze["data"])
            results.append({"test": test, "analyze": res_analyze})
            lines.append("\n---\n")
            time.sleep(PAUSE_BETWEEN_CALLS)
            continue

        data_analyze = res_analyze["data"]
        result_text = data_analyze.get("result", "") or ""
        cost = data_analyze.get("cost", {}) if isinstance(data_analyze, dict) else {}
        elapsed = data_analyze.get("elapsed_s", "?")

        lines.append("**Résultat analyse**")
        lines.append("```")
        lines.append(result_text)
        lines.append("```")
        lines.append(f"- ⏱️ Temps de réponse estimé: {elapsed}s")
        lines.append(f"- 💰 Coût estimé: {cost.get('total', '?')} tokens (~{cost.get('eur', '?')} €)\n")

        time.sleep(PAUSE_BETWEEN_CALLS)

        # 2) Raffinements (simulateurs des 3 boutons)
        refine_outcomes = {}
        for mode, label in [("kpi", "kpi"), ("parades", "parades"), ("plan", "plan")]:
            res_ref = test_refine(result_text, mode)
            if res_ref["ok"]:
                refined_text = res_ref["data"].get("result", "")
                changed = has_changed(result_text, refined_text)
                refine_outcomes[mode] = {"ok": True, "changed": changed, "len": len(refined_text)}
                lines.append(f"**Raffinement : {label}**")
                lines.append("```")
                lines.append(refined_text)
                lines.append("```")
                if changed:
                    lines.append(f"- ✅ Le texte a été enrichi ({len(refined_text)} caractères).")
                else:
                    lines.append(f"- ⚠️ Aucun changement détecté (vérifier le prompt refine côté serveur).")
                lines.append("")
                # Enchaîner les raffinements sur le texte le plus récent
                result_text = refined_text or result_text
            else:
                refine_outcomes[mode] = {"ok": False, "status": res_ref["status"], "err": res_ref["data"]}
                log_err_block(f"Raffinement {label}", res_ref["status"], res_ref["data"])

            time.sleep(PAUSE_BETWEEN_CALLS)

        # 3) Recherche directe (si applicable)
        research_block = None
        if test.get("search"):
            res_search = test_research(test["search"])
            if res_search["ok"]:
                ctx = res_search["data"].get("context", "")
                lines.append("**Résultat recherche directe**")
                lines.append("```")
                lines.append(ctx or "(contexte externe vide)")
                lines.append("```")
                lines.append("")
                research_block = {"ok": True, "len": len(ctx or "")}
            else:
                log_err_block("Recherche directe", res_search["status"], res_search["data"])
                research_block = {"ok": False, "status": res_search["status"], "err": res_search["data"]}

        results.append({
            "test": test,
            "analyze": res_analyze,
            "refine": refine_outcomes,
            "research": research_block,
        })

        lines.append("\n---\n")

    # Sauvegardes
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Rapport Markdown: {out_md}")
    print(f"✅ Données JSON:     {out_json}")
    print("Astuce: ajuste PAUSE_BETWEEN_CALLS si tu vois des 429.")

if __name__ == "__main__":
    main()
