#!/usr/bin/env python3
"""Controllo di qualita' di una tabella di metadati (CSV). Da eseguire presso l'ospedale.

Legge SOLO il CSV (e, se richiesto, controlla che i file immagine esistano).
Stampa SOLO conteggi aggregati: nessuna riga e nessun identificativo viene mostrato,
quindi il risultato si puo' condividere senza esporre dati dei pazienti.

Uso:
    python controllo_dataset.py metadati.csv
    python controllo_dataset.py metadati.csv --images cartella_immagini
    python controllo_dataset.py metadati.csv --out report.txt

Richiede solo Python 3 (nessuna libreria esterna).
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

REQUIRED = ["image_id", "patient_id", "lesion_id", "label", "label_source"]
LABELS = {"melanoma", "nevo", "altro_benigno", "altro_maligno", "ignoto"}
SOURCES = {"istologia", "consenso_esperti", "giudizio_singolo_clinico", "follow_up_stabile", "nessuna"}
SPLITS = {"train", "val", "test"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


def read_rows(path):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel
                return list(csv.DictReader(f, dialect=dialect))
        except UnicodeDecodeError:
            continue
    sys.exit("Impossibile leggere il file: codifica non riconosciuta.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--images", help="cartella con i file immagine (facoltativo)")
    ap.add_argument("--out", help="salva il report in un file di testo")
    a = ap.parse_args()

    rows = read_rows(a.csv)
    out = []
    p = out.append

    if not rows:
        sys.exit("Il file e' vuoto.")
    cols = list(rows[0].keys())
    missing_cols = [c for c in REQUIRED if c not in cols]
    p(f"Righe (immagini): {len(rows)}")
    if missing_cols:
        p(f"ERRORE: colonne indispensabili mancanti: {', '.join(missing_cols)}")
        finish(out, a.out)
        return

    g = lambda r, k: (r.get(k) or "").strip()

    # --- valori vuoti nelle colonne indispensabili
    for c in REQUIRED:
        n = sum(1 for r in rows if not g(r, c))
        if n:
            p(f"AVVISO: {n} righe con '{c}' vuoto")

    # --- duplicati e nomi file
    ids = Counter(g(r, "image_id") for r in rows)
    dup = sum(1 for v in ids.values() if v > 1)
    if dup:
        p(f"AVVISO: {dup} image_id duplicati")
    unsafe = sum(1 for r in rows if not SAFE_NAME.match(g(r, "image_id")))
    if unsafe:
        p(f"AVVISO: {unsafe} nomi file con caratteri insoliti (spazi, accenti, ecc.): verificare che non contengano dati personali")

    # --- conteggi principali
    patients = {g(r, "patient_id") for r in rows if g(r, "patient_id")}
    lesions = {g(r, "lesion_id") for r in rows if g(r, "lesion_id")}
    p(f"Pazienti distinti: {len(patients)}")
    p(f"Lesioni distinte: {len(lesions)}")

    # --- etichette (a livello di lesione, non di immagine)
    les_label = defaultdict(set)
    les_src = defaultdict(set)
    for r in rows:
        l = g(r, "lesion_id")
        les_label[l].add(g(r, "label"))
        les_src[l].add(g(r, "label_source"))
    conflicts = sum(1 for s in les_label.values() if len(s - {""}) > 1)
    if conflicts:
        p(f"AVVISO: {conflicts} lesioni con etichette in conflitto tra le immagini")
    lab_count = Counter(next(iter(s - {""}), "(vuota)") for s in les_label.values() if len(s - {""}) <= 1)
    p("Lesioni per etichetta: " + ", ".join(f"{k}={v}" for k, v in sorted(lab_count.items())))
    img_lab = Counter(g(r, "label") or "(vuota)" for r in rows)
    p("Immagini per etichetta: " + ", ".join(f"{k}={v}" for k, v in sorted(img_lab.items())))
    bad_lab = sum(1 for r in rows if g(r, "label") and g(r, "label") not in LABELS)
    if bad_lab:
        p(f"AVVISO: {bad_lab} righe con etichetta fuori dall'elenco proposto (puo' andare bene se usate un'altra classificazione)")
    src_count = Counter(g(r, "label_source") or "(vuota)" for r in rows)
    p("Immagini per fonte dell'etichetta: " + ", ".join(f"{k}={v}" for k, v in sorted(src_count.items())))
    bad_src = sum(1 for r in rows if g(r, "label_source") and g(r, "label_source") not in SOURCES)
    if bad_src:
        p(f"AVVISO: {bad_src} righe con fonte fuori dall'elenco proposto")

    # --- coerenza tra paziente e lesione
    les_pat = defaultdict(set)
    for r in rows:
        les_pat[g(r, "lesion_id")].add(g(r, "patient_id"))
    multi = sum(1 for s in les_pat.values() if len(s) > 1)
    if multi:
        p(f"ERRORE: {multi} lesioni associate a piu' di un paziente")

    # --- sbilanciamento
    mel = lab_count.get("melanoma", 0)
    if lesions:
        p(f"Lesioni con etichetta melanoma: {mel} ({100 * mel / len(lesions):.1f}%)")

    # --- divisione, se presente
    if "split" in cols and any(g(r, "split") for r in rows):
        pat_split = defaultdict(set)
        les_split = defaultdict(set)
        for r in rows:
            s = g(r, "split")
            if s:
                pat_split[g(r, "patient_id")].add(s)
                les_split[g(r, "lesion_id")].add(s)
        leak_p = sum(1 for s in pat_split.values() if len(s) > 1)
        leak_l = sum(1 for s in les_split.values() if len(s) > 1)
        p(f"Divisione: {dict(Counter(g(r, 'split') or '(vuota)' for r in rows))} immagini")
        if leak_p:
            p(f"ERRORE: {leak_p} pazienti compaiono in piu' di un insieme (contaminazione)")
        if leak_l:
            p(f"ERRORE: {leak_l} lesioni compaiono in piu' di un insieme (contaminazione)")
        if not leak_p and not leak_l:
            p("Divisione per paziente: OK (nessuna contaminazione)")
        bad_sp = sum(1 for r in rows if g(r, "split") and g(r, "split") not in SPLITS)
        if bad_sp:
            p(f"AVVISO: {bad_sp} righe con valore 'split' non valido")

    # --- distribuzione dei dispositivi
    if "device" in cols:
        dev = Counter(g(r, "device") or "(vuoto)" for r in rows)
        p("Immagini per dispositivo: " + ", ".join(f"{k}={v}" for k, v in sorted(dev.items())))
        # correlazione grezza dispositivo-etichetta (segnala possibili scorciatoie)
        tab = defaultdict(Counter)
        for r in rows:
            tab[g(r, "device") or "(vuoto)"][g(r, "label") or "(vuota)"] += 1
        for d, c in sorted(tab.items()):
            tot = sum(c.values())
            m = c.get("melanoma", 0)
            p(f"  dispositivo {d}: melanoma {100 * m / tot:.1f}% su {tot} immagini")

    # --- esistenza dei file
    if a.images:
        missing = sum(1 for r in rows if not os.path.exists(os.path.join(a.images, g(r, "image_id"))))
        p(f"File immagine mancanti: {missing} su {len(rows)}")

    finish(out, a.out)


def finish(out, path):
    text = "\n".join(out)
    print(text)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nReport salvato in {path}")


if __name__ == "__main__":
    main()
