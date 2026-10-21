# Correzioni dell'audit — 21 ottobre 2026

Base: commit `9a8a728`. Modifiche inizialmente implementate e verificate nella
copia `/Users/simonsa/Tarzan-fixes-2026-10-21`, poi integrate nel repository
originale `/Volumes/workplace/PersonalProjects/Tarzan` il 21 ottobre 2026.

La suite completa è stata rieseguita nell'originale dopo l'integrazione.
Gli input personali non sono stati modificati. Non sono stati eseguiti
invii email o deployment.

## Problemi iniziali risolti

| Problema | Correzione | Verifica |
| --- | --- | --- |
| Operazione nell'ultimo giorno contata due volte nel TWR | Un'unica valutazione terminale per data | Acquisto e vendita nell'ultimo giorno: rendimento +10%, coerente con il NAV |
| Riordino artificiale delle operazioni nello stesso giorno | Ordinamento stabile per data di negoziazione, senza spareggi per regolamento o hash | Costo residuo 1500 EUR e imposta stimata 130 EUR nel controesempio |
| Prezzi sintetici in valute minori sovrastimati 100 volte | Conversione in unità maggiori prima del cambio EUR | GBp, GBX, ZAc e ILa: seed, storia e flussi riconciliati |
| Posizioni frazionarie eliminate | Tolleranza numerica ridotta a `1e-9`, non una soglia minima d'investimento | Posizione di 0,005 unità conservata, inclusa vendita parziale |
| Ordini assenti, vuoti, malformati o tutti futuri trattati come analisi riuscita | Errore di ingestione e risultato fallito autorevole | Portfolio e planning indisponibili, newsletter ordinaria bloccata |
| Due processi autorizzati a inviare lo stesso evento | CAS esclusivo; identificativo di operazione remoto stabile nei retry e distinto tra chiamanti | Race tra thread e processi, percorso SMTP simulato e protocollo Apps Script reale eseguito sotto Node |
| Commissione del trasferimento usata come costo del titolo | Autorità condivisa per il costo d'acquisto; trasferimenti basati sul lordo trasferito | Costo residuo, plusvalenza e stima fiscale coerenti |
| Dividendi contati sia nei prezzi rettificati sia negli incassi | Serie price-only distinta dalla serie total-return | Dividendo compensato dal calo del prezzo: rendimento e P&L zero; storia indisponibile senza prova price-only |
| Cache storica e coda recente su scale di rettifica diverse | Verifica delle date comuni e refresh completo quando cambiano i prezzi storici | Cache fredda e calda identiche; refresh fallito conserva cache esplicitamente fallback, senza mescolare scale |
| Classificazione fiscale ripetuta per ogni vendita | Raggruppamento iniziale e classificazione una volta per identità | Conteggio delle chiamate e benchmark comparativo con risultato fiscale invariato |

I primi 17 casi di regressione fallivano prima delle correzioni.

## Ulteriori difetti corretti

- Quote e serie con valori infiniti potevano entrare nella valutazione.
- Un incasso nell'ultimo giorno poteva scomparire dal grafico P&L quando il
  prezzo era invariato: la contabilità delle serie dense ora arriva alla data
  finale anche quando il NAV analitico viene troncato per evitare code stale.
- Un trasferimento in uscita senza contante veniva interpretato come perdita;
  le commissioni sui trasferimenti in ingresso venivano invece ignorate.
  I flussi XIRR distinguono capitale trasferito al valore di mercato e gamba cash.
- Un errore nella configurazione del run lasciava esposto il risultato del run
  precedente: ora viene sempre registrato un nuovo risultato fallito.
- Un errore prima dell'orchestrazione, nella risoluzione degli input, impediva
  anche la notifica di fallimento: ora produce evidenza critica sanitizzata.
- Una quotazione live poteva riscrivere il precedente prezzo rettificato usando
  il prezzo non rettificato, annullando la correzione per dividendi.
- Le quotazioni live in valuta estera potevano essere copiate nella serie EUR
  senza conversione se i livelli erano simili: 101 USD diventavano 101 EUR
  anziché 95,95 EUR con cambio 0,95. Ora ogni endpoint usa il proprio cambio
  causale, conserva la provenienza FX e normalizza valute minori e bond per 100;
  un cambio mancante impedisce l'aggiornamento EUR.
- La risposta delle quotazioni ufficiali perdeva la valuta del provider.
  Ora la conserva, rifiuta valori non finiti e isola timestamp malformati senza
  interrompere il resto del batch.
- Una quotazione recente ma più vecchia dell'ultimo prezzo storico poteva
  sovrascrivere una chiusura più nuova: ora viene esclusa prima della selezione.
- La conversione FX cambiava il fuso orario dell'indice dell'asset, alterando
  le etichette delle sessioni usate dai consumatori giornalieri. Ora conserva
  l'indice originale dopo l'allineamento causale del cambio.
- La pulizia Apps Script eliminava claim corrotti e poteva riabilitare un evento
  già inviato: i record illeggibili sono conservati per riconciliazione manuale.
- Errori nel salvataggio dello stato dopo SMTP impedivano la finalizzazione delle
  evidenze: ora lo stato locale viene finalizzato senza reinvio automatico.
- Se il lock di processo non è disponibile, il claim locale viene rifiutato
  invece di proseguire senza esclusione reciproca.

La verifica di sicurezza mantiene l'autenticazione esistente, controlla il
rifiuto delle richieste anonime e richiede conferme verificabili prima di SMTP.
Non sono stati aperti listener né modificati servizi o credenziali remoti.

## Verifiche eseguite

- Suite originale: **1447 test passati**.
- Suite corretta, rieseguita nel repository originale: **1517 test passati**
  in 47,69 secondi prima del commit, nessun test saltato; 2 warning di deprecazione protobuf,
  non introdotti dalle modifiche.
- **70 test aggiuntivi**, oltre alle aspettative aggiornate per i comportamenti
  intenzionalmente corretti.
- Gruppo delivery/concorrenza, ripetuto anche nell'originale: **13 test passati
  per 5 esecuzioni consecutive**, compresa esclusione tra processi e verifica
  Apps Script sotto Node.
- Compilazione di `tarzan` e `scripts`: superata.
- `pip check`: nessun requisito incompatibile.
- `git diff --check`: superato.
- Ricerca di listener aperti, esposizioni wildcard e autenticazione indebolita:
  nessuna occorrenza nelle sorgenti versionate.
- Conversione della risposta Yahoo confrontata anche con l'implementazione
  `auto_adjust` della dipendenza yfinance installata.

Benchmark fiscale: mediana di 3 esecuzioni, stessi dati sintetici e stessa
macchina; non misura il tempo complessivo dell'applicazione.

| Righe ordini | Prima | Dopo |
| ---: | ---: | ---: |
| 200 | 14,281 ms | 0,322 ms |
| 1000 | 347,572 ms | 1,611 ms |
| 2000 | 1513,342 ms | 3,082 ms |

Ambiente verificato: Python 3.12.7, pandas 2.2.2, numpy 1.26.4, yfinance 1.1.0
e Node 22.14.0. Le dipendenze non sono state aggiornate o reinstallate.
La suite locale non sostituisce il job CI con il lock delle dipendenze.
I test usano input sintetici e servizi simulati: nessuna email reale è stata
inviata e nessun deployment è stato effettuato.

Per ripetere i test dal repository originale:

```sh
python -m pytest tarzan/tests -q -p no:cacheprovider
```

Il test Apps Script richiede `node` nel PATH. In alternativa impostare
`TARZAN_TEST_NODE` al percorso dell'eseguibile Node; senza Node quel solo test
viene saltato e va eseguito separatamente.

## Integrazione e deployment

Le correzioni sono già applicate al repository originale. Commit e push
sono stati autorizzati dall'utente; la pubblicazione su GitHub non aggiorna
automaticamente Apps Script. La patch
`/Users/simonsa/Tarzan-audit-fixes-2026-10-21.patch` conserva la versione
preparata sulla base `9a8a728`: non va riapplicata al repository già corretto.

Il nuovo protocollo di claim richiede anche il deployment della nuova versione
di `scripts/apps_script/Code.gs`. Pianificare l'aggiornamento coordinato di
client e Web App, sospendendo gli invii durante il passaggio. Non cancellare i
claim esistenti o le Script Properties. Le chiamate senza `operation_id`, o
senza conferma dello stesso identificativo nella risposta, vengono rifiutate:
una vecchia Web App non deve essere considerata compatibile con il nuovo client.

I claim locali richiedono file locking di processo (`fcntl`, disponibile su
macOS/Linux); se non disponibile, usare uno store remoto compatibile anziché
aggirare il controllo.

Restano da eseguire, con l'accesso appropriato: verifica CI delle dipendenze
bloccate e verifica del deployment reale.
L'audit e i test non costituiscono una garanzia di assenza di ulteriori bug.
