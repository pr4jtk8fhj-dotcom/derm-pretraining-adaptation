# Sovrapposizione tra dati di pre-addestramento di PanDerm e coorte (MILK10k, HIBA, PROVe-AI)

Controllo del 2026-10-07, fatto leggendo il paper (versione PMC del Nature Medicine e arXiv 2410.15038v3) con uno
strumento di estrazione testo: le citazioni sotto vanno ricontrollate sul PDF prima di usarle nella proposta.

## Cosa dice il paper sul pre-addestramento
- "2,149,706 unlabeled multimodal skin images", "4 imaging modalities and 11 data sources".
- Composizione: 757,890 TBP tiles, 537,047 dermatopathology tiles, 460,328 clinical, 384,441 dermoscopic.
- Fonti elencate: MYM e HOP (TBP; 38,110 dermoscopiche), MMT (in-house: 316,399 dermoscopiche + 310,951 cliniche),
  ACEMID (patologia), NSSI (29,832 dermoscopiche, Brisbane Naevus Morphology Study), Edu1/Edu2 (cliniche, in-house),
  ISIC2024 ("352,034 tile images", sottoinsieme "stratified by institutions"), TCGA-SKCM, UAH89k.
- Nota aritmetica: 38,110 + 316,399 + 29,832 = 384,341, non 384,441 (differenza di 100 immagini, non spiegata nel testo letto).
- Sul rischio di contaminazione gli autori scrivono che SwAVDerm incorporava immagini di benchmark "such as ISIC and DermNet"
  e che la loro strategia "minimizes this risk". Non ho trovato una descrizione di un controllo esplicito dei duplicati.

## Le nostre tre sorgenti
| Sorgente | Nel pre-addestramento (per nome) | Altro uso nel paper |
|---|---|---|
| MILK10k (coll. 425) | non nominata | non nominata |
| HIBA (coll. 251) | non nominata | **valutazione downstream**: "curated from the HIBA data from the ISIC archive, containing 1,635 dermoscopic images" |
| PROVe-AI (coll. 218) | non nominata | non nominata |

## Aggiornamento del disegno (7/10/2026)
La coorte ora e' l'intero ISIC Archive filtrato, con sorgente = istituzione. Conseguenze per la scelta del test:
- escludere le istituzioni australiane (i dati in-house di pre-addestramento sono australiani: MYM, HOP, NSSI, ACEMID);
- le istituzioni che hanno contribuito a ISIC2024 potrebbero avere pazienti nel sottoinsieme TBP usato nel
  pre-addestramento: non verificabile, da dichiarare per ogni istituzione di test se nota;
- HIBA e' ammessa nel test ma e' un benchmark del paper di PanDerm.

## Conclusione per le tre collezioni della prima proposta (da riportare cosi')
- Nessuna delle tre sorgenti compare per nome tra i dati di pre-addestramento.
- NON verificabile: (1) se i dataset in-house (MMT, NSSI, MYM/HOP) contengano copie di immagini poi depositate su ISIC;
  (2) quali istituzioni compongano il sottoinsieme ISIC2024 usato (tile TBP, modalita' diversa dalla dermoscopia, ma
  stessi pazienti possibili se un'istituzione contribuisce a entrambi). Gli identificativi non sono collegabili.
- HIBA e' gia' stato usato dagli autori come benchmark: i loro numeri su HIBA non sono un test esterno per PanDerm
  rispetto alle scelte di sviluppo degli autori. Il checkpoint rilasciato e' quello pre-addestrato (non fine-tunato su HIBA).
- Limitazione da dichiarare: "provenienza del pre-addestramento verificata solo a livello di elenco dei dataset;
  sovrapposizione a livello di immagine non verificabile". DINOv2 (LVD-142M, immagini web) ha lo stesso limite.

Fonti: https://pmc.ncbi.nlm.nih.gov/articles/PMC12353815/ ; https://arxiv.org/html/2410.15038v3
