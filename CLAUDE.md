# Progetto: modello dermoscopico aperto da dati ISIC pubblici + studio di adattamento (PanDerm, DINOv2)

Tesi al Politecnico di Milano. Servono numeri VERI per l'application al CINECA Open Hackathon (con NVIDIA e
OpenACC; ibrido, 2-17 dicembre 2026, da fonti pubbliche). Scadenza dell'application: 12 ottobre 2026 (riferita
dall'utente il 7/10; la pagina ufficiale riportava il 2 ottobre). Requisiti letti in fonti pubbliche: almeno 3 membri
presenti per tutto l'evento, codice con licenza, priorita' agli affiliati CINECA.
La fase A va finita in tempo per mettere i numeri nella proposta prima del 12: se qualcosa non e' misurato,
nella proposta resta "planned", mai stimato. Il modulo di candidatura si compila per ULTIMO, dopo i risultati.
L'utente parla italiano: rispondi in italiano; codice e commenti possono essere in inglese.

Team: Gaia Arienti, Olimpia Cordeschi, Alessandra Fatone, Marco Tarchi (lavorano tutti insieme su tutto, niente
divisione per persona). Codice nostro: Apache-2.0 (LICENSE, NOTICE, THIRD_PARTY.md); dati, pesi, proposta, guida e
`archivio/` esclusi dal repository (.gitignore).

## Domanda
Quesito clinico: "questa lesione va segnalata per una valutazione clinica piu' attenta?".
I dati pubblici non hanno etichette di segnalazione: si misura un PROXY, benigno contro maligno con conferma istologica.
Scrivilo sempre cosi' ("proxy diagnostico"), mai "triage validato" o "validato clinicamente".

Riferimento: `proposta/proposal.md` (versione del team, 8/10/2026). Domande di ricerca:
- PRIMARIA: il pre-addestramento self-supervised continuato su immagini dermoscopiche pubbliche migliora la
  discriminazione ESTERNA rispetto a DINOv2 originale, a parita' di protocollo di adattamento?
  Contrasto primario prespecificato: `dinov2_cpt` meno `dinov2`, fine-tuning completo, 224 px, 100% delle etichette,
  differenza APPAIATA di AUROC esterna (`aggregate.py` -> `results/primary.md`). Tutto il resto e' secondario.
- Adattamento: come varia il beneficio tra frozen, LoRA e full, e con il 10/30/100% delle etichette.
- Risoluzione: 448 px migliora abbastanza da giustificare calcolo e memoria?
- Calcolo: quale strategia di esecuzione riduce tempo e risorse mantenendo il confronto scientificamente valido?
Tre encoder: DINOv2 ViT-L/14 originale, lo stesso dopo pre-addestramento continuato (`dinov2_cpt`, solo fase B),
PanDerm ViT-L/16 come riferimento esterno (le differenze da PanDerm NON sono attribuibili al solo pre-addestramento).
PanDerm e' CC-BY-NC-ND: nessuna versione adattata si condivide. L'addestramento da zero NON fa parte del lavoro.
Si misurano anche: immagini/s, % di GPU inattiva, picco di memoria allocata e riservata, ore di GPU.
Le misure su H100 (RunPod) NON vanno presentate come misure su Leonardo (A100 64 GB): etichettale sempre con l'hardware.

Due fasi:
- FASE A (ora, RunPod, prima del 12 ottobre): studio di adattamento a 224 px = risultati preliminari; misure che
  giustificano la richiesta: profilo dello script originale di PanDerm, pipeline dati (CPU vs cache vs DALI su GPU),
  costo e memoria a 448 px, e un TEST BREVE (max 45 min) del pre-addestramento continuato di DINOv2 (`ssl_dinov2.py`):
  solo img/s, memoria, andamento della loss. Non e' un modello finito: non valutarlo, non presentarlo come risultato.
- FASE B (hackathon su Leonardo, NON si esegue qui): B1 porting e profilo su A100; B2 pre-addestramento continuato
  scalato (1 GPU -> nodo da 4 -> eventualmente 2 nodi), strong e weak scaling; B3 pipeline dati (CPU, cache, DALI);
  B4 completare l'encoder e valutarlo (contrasto primario a 224 px, poi 448 px e la matrice delle frazioni).
  Il codice per B4 e' gia' pronto (`--backbone dinov2_cpt --cpt-ckpt <teacher_checkpoint.pth>`) e testato sui dati sintetici.
- Nella proposta la fase A va descritta per quello che e': su una H100 a noleggio (RunPod), non "in laboratorio".

## Macchina, tempo e denaro
- Pod RunPod con H100 SXM 80 GB, 3,49 $/h per GPU (listino del 27 settembre 2026, da riverificare all'avvio), al secondo.
- Finestra: 15 ore dall'avvio. Tetto di spesa: 100 $; obiettivo 90 $.
- Default 1 x H100 (15 h = circa 52 $). Se dopo il benchmark la matrice supera 8 ore su una GPU, proponi all'utente
  un pod 2 x H100 (circa 7 $/h, prezzo da verificare) con run indipendenti, una per GPU (`run_matrix.sh` con shard).
- Il pod acceso costa anche quando non fa nulla: spegnilo (o chiedi di spegnerlo) appena i lavori finiscono.
- All'avvio `python scripts/progress.py start`; a ogni fase `progress.py log`: ore trascorse, costo, fatto, manca.
  L'utente puo' non essere davanti allo schermo: PROGRESS.md e' come ritrova la situazione.

Cronoprogramma indicativo (ore dall'avvio):
| Ore | Cosa |
| 0:00-0:45 | `00_setup_pod.sh`, `preflight.sh` |
| 0:45-1:15 | `smoke/run_smoke.sh` (dati sintetici): TUTTI i passi devono passare prima dei dati veri |
| 1:15-3:00 | Dati: metadati, coorte per istituzione (conferma del test), download, cache 256/512, duplicati, partizione |
| 3:00-4:00 | Frozen (entrambi i backbone, tre frazioni) + profilo dello script ORIGINALE + benchmark 224 + stima |
| 4:00-5:00 | Ingegneria: `bench_loader.py`, benchmark DALI, benchmark 448, SDPA, profilo |
| 5:00-5:45 | Sweep del learning rate (stesso budget per ogni metodo) |
| 5:45-11:00 | Matrice a 224 (tetto 5,25 ore). In parallelo su CPU: set SSL (selezione, download, cache, duplicati del test) |
| 11:00-11:45 | Test breve SSL DINOv2 (1 GPU, max 45 min) |
| 11:45-13:15 | Analisi, tabelle, numeri per la proposta; poi spegnere il pod |
| 13:15-15:00 | Margine |
Priorita' se il tempo stringe: si taglia prima seed 2-3 dei bracci secondari, poi la frazione 10%, poi LoRA. NON si
tagliano: DINOv2 full 224 px 100% con 3 seed (riferimento del contrasto primario), il test SSL e il benchmark 448.

## Regole non negoziabili
- NUMERI VERI: ogni numero riportato deve venire da una run eseguita qui, con log, seed e configurazione salvati.
  Mai stimare, arrotondare a favore, inventare o completare risultati mancanti. Se una run fallisce, dillo.
  I conteggi presi dall'API ISIC il 7 ottobre 2026 sono preliminari: vanno sostituiti da quelli degli script.
  Gli output del test sintetico (`$WS/smoke`) NON sono risultati: non riportarli mai.
- Salva per ogni run: comando esatto, commit, versioni, seed, GPU, log, risultati JSON, ore di GPU, img/s, picco di memoria.
- SOLO DATI PUBBLICI su questa macchina. Nessun dato clinico dell'ospedale partner, mai.
- Non pubblicare ne' condividere pesi PanDerm modificati (checkpoint fine-tunati, adattatori LoRA, pesi fusi): licenza ND.
- Licenze delle immagini: CC-0, CC-BY e CC-BY-NC ACCETTATE dall'utente il 7 ottobre 2026 per il set etichettato
  (ricerca non commerciale). Il set di pre-addestramento usa SOLO CC-0 e CC-BY. Registra licenza e attribuzione.
- Non usare il dataset Kaggle "toriqulislam1/melanoma-skin-cancer-dataset" ne' altri mirror non ufficiali.
- Un metodo con piu' tuning o calcolo non va presentato come confronto a budget uguale.
- Mai lasciare che pip cambi torch: `00_setup_pod.sh` usa `logs/constraints.txt`; installa ogni pacchetto con `-c` su quel file.
- tmux (o nohup) per i lavori lunghi; dati e checkpoint su /workspace.
- Autonomia: procedi senza chiedere, tranne:
  (a) scelta delle istituzioni di test: proponi secondo la regola sotto, MOSTRA la tabella e aspetta conferma;
  (b) se la stima della matrice supera il tetto di tempo o di spesa: proponi tagli e aspetta conferma;
  (c) se una scelta cambia il protocollo (partizioni, soglia, metrica, iperparametri dopo lo sweep);
  (d) se una run fallisce due volte per la stessa causa.
  Se il test sintetico rivela un bug negli script, correggilo (e' codice nostro), rilancia il passo e annotalo in PROGRESS.md.

## Dati: intero ISIC Archive, sorgente = istituzione
1. `scripts/01_metadata.sh`: metadati di tutte le immagini dermoscopiche con istologia (set etichettato) e di tutte le
   dermoscopiche (pool di pre-addestramento). Conteggi preliminari dall'API (7/10/2026): 20.087 maligne e 19.245 benigne
   con istologia; 124.961 dermoscopiche in totale, di cui 19.951 CC-0 e 49.024 CC-BY. Da riverificare.
2. Filtri: diagnosis_1 in {Benign, Malignant}; indeterminate e ignote ESCLUSE (mai trattate come benigne).
3. Sorgente = istituzione, dal campo `attribution` (funzione `source_of` in build_cohort.py, usata anche dal set SSL).
   Compila `configs/source_map.csv` (attribution, source, country) da `attributions.csv` e rilancia: una riga solo per
   le istituzioni VERIFICATE (attribuzione riconciliata con i metadati). Le sorgenti non verificate e "Anonymous" sono
   solo sviluppo: `make_partitions.py --source-map` rifiuta un test non verificato.
4. Download per id ISIC, SHA-256 di ogni file; cache ridimensionate (lato corto 256 e 512).
5. Duplicati: hash esatto + dHash (<= 6 bit), anche tra istituzioni; guarda `pairs_sheet.jpg`. Etichette in conflitto ->
   cluster escluso. Cluster a cavallo tra test e sviluppo -> si tolgono le copie di sviluppo (il test resta intatto).
6. Raggruppamento: union-find su patient_id, lesion_id e cluster di duplicati. Dichiara per istituzione la base usata.
7. Regola per il test (fissata PRIMA di qualsiasi risultato): 2-3 istituzioni con copertura patient_id >= 90%,
   almeno 150 maligne e 150 benigne dopo i duplicati, NON australiane (i dati in-house di PanDerm sono australiani),
   preferendo istituzioni diverse tra loro per paese/dispositivo. HIBA e' ammessa ma e' un benchmark del paper di PanDerm: dichiararlo.
8. Sviluppo = tutte le altre istituzioni: validazione 15% dei gruppi, stratificata su (istituzione, etichetta), seed 0.
   Frazioni di training annidate 10% / 30% / 100% dei gruppi di train; la validazione non cambia.
9. `manifest.csv` (colonne: image_id, file, source, patient_id, lesion_id, group_id, label, label_source, license,
   attribution, diagnosis_detail, age_band, sex, body_site, split, train_frac) e `panderm.csv` per lo script originale.
   Controllo: `python controllo_dataset.py manifest.csv` (avvisi su label/label_source attesi).
10. Pool di pre-addestramento (`ssl_build_set.py`, `ssl_dedup_vs_test.py`): dermoscopiche CC-0/CC-BY; esclusi istituzioni
   di test, immagini di validazione e test, stesso paziente/lesione di validazione o test, copie e quasi-copie di
   validazione o test. Le immagini di training etichettate possono restare (senza etichette). Etichette mai usate.
11. Conteggi sempre separati: immagini, pazienti, lesioni, gruppi collegati, istituzioni (`subsets_table.md`,
   `ssl_set_summary.json`), piu' copertura del patient_id per istituzione.
Sovrapposizione col pre-addestramento di PanDerm: `docs/panderm_pretraining_overlap.md`. Riportala come limitazione.

## Protocollo di valutazione
- Soglia scelta sulla VALIDAZIONE per 95% di sensibilita', poi BLOCCATA e applicata al test. Obiettivo sperimentale,
  non requisito clinico.
- Sul test, per ogni istituzione e in totale: AUROC, sensibilita' e specificita' alla soglia, falsi negativi, frazione segnalata.
- IC 95%: bootstrap per paziente (per lesione dove manca: dichiararlo), 1.000 ricampionamenti. Confronti appaiati sulle
  stesse immagini. Variabilita' tra seed riportata a parte. Risultati per istituzione oltre al totale.
- Non significativo NON vuol dire equivalente. Il test non decide nulla: iperparametri, epoca, soglia e priorita'
  degli esperimenti vengono solo dalla validazione. Se la fase B riusa lo stesso test esterno, non e' piu' "intatto": dirlo.
- Pipeline casuali (DALI vs torchvision): controllo di distribuzione (`bench_loader.py`), non uguaglianza bit per bit.
  Tolleranze numeriche documentate per precisione (fp32, bf16) in `test_backbones.py`.
- Riferimenti di reporting: TRIPOD+AI e TRIPOD-Cluster.

## Modelli
- PanDerm: https://github.com/SiyuanYan1/PanDerm (`classification/`), checkpoint ViT-L/16
  https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view?usp=sharing (`panderm_ll_data6_checkpoint-499.pth`,
  `gdown`). Classe `PanDerm_Large_FT`, input 224, patch 16. Script originale `run_class_finetuning.py` (timm==0.9.16).
- Il repo non ha LoRA e `Attention.forward` usa `F.linear(x, self.qkv.weight, ...)`: `scripts/backbones.py` risolve con
  `LoRALinear` che espone `.weight` = peso fuso. `scripts/test_backbones.py` e' OBBLIGATORIO prima di ogni training
  (LoRA a zero = originale bit per bit; LoRA non nulla cambia l'uscita; gradienti solo sugli adattatori).
- Attenzione esplicita con `rel_pos_bias`: la versione fusa (SDPA con bias additivo) va validata con `test_backbones.py --sdpa`.
- 448 px: tabelle di posizione interpolate dal checkpoint a 224 (`test_backbones.py --img448`).
- Dopo il caricamento del checkpoint stampa e controlla chiavi mancanti e inattese.
- DINOv2: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')`; training ufficiale in `$WS/dinov2`, lanciato da
  `ssl_dinov2.py` senza modificarlo (dataset da lista di file, pesi convertiti e verificati con strict=True).
- Stesse impostazioni per entrambi: dati, partizione, 224 px, augmentation, ottimizzatore, passi, seed.

## Esperimenti, in questo ordine (comandi esatti in `scripts/README.md`)
1. Setup, `preflight.sh`, `smoke/run_smoke.sh` (tutti PASS), `test_backbones.py`.
2. Dati fino al manifesto congelato (con conferma delle istituzioni di test).
3. Frozen per entrambi i backbone e le tre frazioni (feature estratte una volta).
4. Profilo dello script ORIGINALE di PanDerm non modificato (1 epoca, immagini originali): img/s, % GPU inattiva, memoria.
   Benchmark di `train.py` (1 epoca, 100%) per {PanDerm, DINOv2} x {LoRA, full}; `estimate_matrix.py`; ore in PROGRESS.md.
5. Ingegneria (max 60 min, sono i "prima/dopo" della proposta): `bench_loader.py` a 224 e 448; `train.py --loader dali`
   contro torchvision sulle immagini originali; `--sdpa`; `--torch-profile` (o `profile_nsys.sh`); 448 px: benchmark
   1 epoca a frazione 0.1 per {PanDerm, DINOv2} x {LoRA, full} (`--grad-ckpt`/`--accum` se serve) e
   `estimate_matrix.py --bench runs/bench448`. Ogni ottimizzazione usata nella matrice deve avere il suo test di equivalenza.
6. Sweep del learning rate (`lr_sweep.sh`, frazione 30%, solo validazione); `select_lr.py` -> `configs/hparams_selected.json`.
7. Matrice: {LoRA r16, full} x {PanDerm, DINOv2} x {10%, 30%, 100%}, seed 1; poi seed 2 e 3 dando priorita' a
   DINOv2 full 224 px 100% (e' il braccio di riferimento del contrasto primario: target 3 seed), poi al resto del 100%.
   In parallelo su CPU: set SSL.
8. Test breve SSL DINOv2 (max 45 min) + `ssl_summarize.py`: immagini ORIGINALI/s (non ritagli), memoria, stabilita',
   loss. Serve a riempire la formula della proposta GPU-ore = g x M / (3600 x Qg).
9. Analisi (`aggregate.py`): tabella per istituzione di test, curve per frazione, confronti appaiati; `primary.md`
   resta "NOT AVAILABLE" in fase A (manca dinov2_cpt) ed e' giusto cosi'. `resource_report.py` per la tabella risorse.
10. Solo se avanzano almeno 45 minuti e c'e' un pod con 2 GPU: scalabilita' con `ddp_bench.py`.

## Cosa consegnare
- Tabella della coorte per istituzione; dimensioni di training/validazione/test/pool SSL; numero di duplicati.
- Profilo dello script originale e del nostro, prima e dopo le ottimizzazioni; costo e memoria a 448 px.
- Risultati per istituzione di test con IC, curve per frazione di dati, confronti appaiati, ore di GPU e costo reale.
- Test SSL: img/s, memoria, andamento della loss (o "non riuscito" con la causa).
- Tabella risorse (`logs/resource_report.json`): spazio per immagini/cache/feature/checkpoint, CPU, RAM, letture.
- Per la tabella "Laboratory evidence" della proposta: date, URL e commit del repository, GPU/CPU/versioni,
  controlli di correttezza fatti e mancanti, run e seed completati per configurazione.
- Tutto con GPU, versioni e commit. La proposta (`proposta/proposal.md`) ha i numeri tra [parentesi quadre] da riempire.

## File
- `scripts/`: tutto il codice; `scripts/README.md` = ordine dei comandi. `scripts/smoke/`: test su dati sintetici.
- `configs/`: `hparams.json` (iperparametri fissi; lr finali dallo sweep), `ssl_dinov2_vitl14.yaml`, `source_map.csv`.
- `controllo_dataset.py`: controllo del manifesto, solo conteggi aggregati.
- `docs/panderm_pretraining_overlap.md`: cosa dice il paper di PanDerm sui suoi dati di pre-addestramento.
- `proposta/proposal.md`, `GUIDA.md`: documenti per il team (non vanno nel repository).
- `archivio/`: file vecchi, non servono sul pod.
