## REVIEW EDUCAZIONALE — Open Access

## Pulizia elettronica del residuo marcato nella colonscopia TC: cosa i radiologi devono sapere

Thomas Mang ¹*, Christian Bräuer ¹, Stefaan Gryspeerdt ², Martina Scharitzer ¹, Helmut Ringl ³ e Philippe Lefere ²

¹ Department of Biomedical Imaging and Image-guided Therapy, Medical University of Vienna, Währinger Gürtel 18-20, A-1090 Vienna, Austria
² Department of Radiology, AZ Delta, Bruggesteenweg 90, B-8800 Roeselare, Belgio
³ Department of Radiology, Danube Hospital Vienna, Langobardenstrasse 122, A-1220 Vienna, Austria

*Corrispondenza: thomas.mang@meduniwien.ac.at*

## Abstract

La colonscopia TC (CTC) è l'esame radiologico di scelta per la diagnosi di neoplasia colorettale. La marcatura fecale (*faecal tagging*) è considerata una parte obbligatoria della preparazione intestinale. Tuttavia, la mucosa colica, oscurata dal residuo marcato, non è accessibile alle viste endoluminali 3D e richiede una valutazione 2D dispendiosa in termini di tempo. Gli algoritmi software di pulizia elettronica (*electronic cleansing*, EC) possono superare questo limite sottraendo digitalmente il residuo marcato dal lume colico. Idealmente, ciò consente una valutazione endoluminale 3D senza soluzione di continuità. Nonostante questo beneficio, l'EC è una potenziale fonte di un'ampia gamma di artefatti. Un'EC accurata richiede un'adeguata tecnica d'esame CTC e una marcatura fecale appropriata. È stato dimostrato che il processo di sottrazione digitale influisce sulle caratteristiche morfologiche rilevanti sia dell'anatomia colica sia delle lesioni coliche, se sommerse dal residuo fecale. Questo articolo riassume i potenziali effetti dell'EC sull'imaging CTC, le conseguenze per la refertazione e la gestione del paziente, e le strategie per evitare le insidie (*pitfalls*). Inoltre, vengono mostrati i potenziali effetti negativi sulla refertazione clinica e sulla gestione del paziente, e vengono presentate tecniche di *problem solving*, nonché raccomandazioni per l'uso appropriato delle tecniche di EC. I radiologi che utilizzano l'EC dovrebbero avere familiarità con gli effetti correlati all'EC sulla dimensione del polipo e anche con le corrette tecniche di misurazione.

**Parole chiave:** colonscopia TC, colonscopia virtuale, pulizia elettronica, marcatura fecale, polipi colorettali.

## Punti chiave

- Il software di pulizia elettronica (EC) sottrae digitalmente il residuo marcato dal lume colico, per consentire una valutazione colica endoluminale senza soluzione di continuità.
- L'EC ha il potenziale di ridurre i tempi di lettura e di migliorare il rilevamento dei polipi.
- L'EC può influire sull'aspetto morfologico dell'anatomia e della patologia colica ed è una potenziale fonte di artefatti.
- Le insidie correlate all'EC possono essere evitate mediante la valutazione dei reperti sui corrispondenti dati di immagine non sottratti.

## Background

La colonscopia TC (CTC) è raccomandata come esame radiologico di scelta per la diagnosi di neoplasia colorettale. È indicata nei pazienti con colonscopia incompleta o controindicata e funge da opzione diagnostica per lo screening del carcinoma colorettale e degli adenomi. Si basa su una scansione TC a bassa dose, a strato sottile, del colon pulito e disteso sia in posizione supina sia prona. Per la preparazione intestinale, i pazienti si sottopongono a una combinazione di dieta a basso contenuto di fibre o di liquidi chiari, somministrazione orale di lassativi e marcatura fecale il giorno prima della CTC. La distensione colica con visualizzazione completa di tutti i segmenti colici è ottenuta con l'insufflazione di CO₂ o aria mediante un sottile catetere rettale flessibile. La valutazione dei dati viene eseguita utilizzando un software CTC dedicato, che consente la valutazione simultanea di viste endoluminali 3D e di immagini 2D multiplanari del colon. Oltre ai risultati di uno studio di screening americano, l'importanza delle valutazioni endoluminali 3D è stata evidenziata in uno studio che valutava la performance della CTC nel programma inglese di screening del carcinoma intestinale (*English Bowel Cancer Screening programme*). Tuttavia, è noto che la valutazione 3D sia più dispendiosa in termini di tempo rispetto alla sola interpretazione 2D.

Un colon accuratamente pulito e ben disteso è un prerequisito di un esame di alta qualità. La valutazione dei dati di immagine CTC può essere influenzata negativamente da feci e liquidi residui nel colon, con il residuo fecale che simula e/o oscura le lesioni coliche e il liquido residuo che oscura le lesioni sommerse.

La «marcatura fecale» aumenta significativamente l'attenuazione TC delle feci e dei liquidi residui. Ciò è ottenuto mediante somministrazione orale di mezzi di contrasto positivi, come iodio e/o bario, durante il processo di preparazione intestinale. Gli agenti di marcatura sono quindi somministrati al paziente di solito il giorno prima dell'esame oppure, al più tardi, 3 ore prima dell'esame.

Di conseguenza, feci e liquidi residui, marcati con mezzo di contrasto, possono essere facilmente identificati come residuo fecale per via dei loro elevati valori di attenuazione TC (iperdensi), rispetto alle lesioni intrinseche dei tessuti molli (attenuazione dei tessuti molli), e viceversa. Le lesioni dei tessuti molli sommerse all'interno di liquido marcato iperdenso possono essere facilmente raffigurate sulle immagini 2D come difetti di riempimento negativi (Fig. 1a).

È stato dimostrato che la marcatura fecale aumenta sia la sensibilità sia la specificità della CTC, ed è considerata una parte obbligatoria della preparazione intestinale. Oltre alla marcatura fecale, è stato dimostrato che le viste endoluminali 3D migliorano i tassi di rilevamento dei polipi, ed è stato raccomandato di incorporarle nelle strategie di valutazione della CTC. Tuttavia, anche dopo la marcatura fecale, la mucosa colica, oscurata dal residuo fecale marcato, non è accessibile alle viste endoluminali 3D (Fig. 1b). Ciò richiede ripetute interruzioni della navigazione endoluminale 3D (*fly-through*) per la valutazione 2D di ogni area o segmento colico sommerso. Questo è laborioso e dispendioso in termini di tempo.

## Pulizia elettronica

La pulizia elettronica (EC) è un'applicazione software, progettata per superare questo limite semplicemente sottraendo digitalmente il residuo marcato dalle immagini CTC. Questo processo di sottrazione si basa sugli elevati valori di attenuazione del residuo marcato e comprende le seguenti fasi. Tutti i voxel all'interno del lume colico con un valore di densità TC che superi un valore soglia dedicato verranno riconosciuti dall'algoritmo di EC come residuo fecale potenzialmente marcato. Questa soglia è preimpostata all'interno del software di pulizia elettronica applicato e non può essere regolata dal radiologo. Il residuo fecale, così come le lesioni intraluminali con un'attenuazione inferiore a 100 HU, non verrà riconosciuto come marcato e non verrà rimosso elettronicamente. Il residuo fecale marcato con un'attenuazione TC superiore alla soglia di 100 HU verrà quindi estratto digitalmente dai dataset, attribuendogli il valore di densità dell'aria.

Di conseguenza, la parete colica, ricoperta dal residuo fecale, diventa endoluminalmente visibile sia nelle immagini CTC 2D sia 3D (Fig. 1c, d). Consentendo la valutazione 3D della mucosa colica che altrimenti sarebbe oscurata, l'intera parete colica può essere valutata senza soluzione di continuità sulle immagini endoluminali 3D quando si impiega l'EC. Pertanto, il tempo di lettura del radiologo è ridotto, poiché vengono eliminati numerosi passaggi dispendiosi in termini di tempo dalla vista endoluminale 3D alle corrispondenti viste 2D in ogni singolo segmento sommerso.

Inoltre, la sottrazione digitale del residuo fecale marcato intraluminale dalle immagini CTC può eliminare potenziali fonti di diagnosi falsamente negative e falsamente positive, e quindi migliorare il rilevamento dei polipi. Rimuovendo digitalmente dal lume colico le feci residue che potrebbero essere scambiate per polipi, la specificità della valutazione dei dati può essere migliorata. Inoltre, i polipi ricoperti da residuo marcato solido o da liquido possono essere raffigurati più facilmente semplicemente utilizzando le immagini endoluminali 3D, dove è stato dimostrato che i polipi sono più cospicui rispetto alle viste planari 2D.

*Fig. 1 — Polipo sessile nel colon ascendente, sommerso interamente sotto liquido residuo marcato. (a) La vista assiale 2D mostra un polipo sessile (freccia) con un'attenuazione omogenea dei tessuti molli all'interno del liquido marcato iperdenso. (b) Nella vista endoluminale corrispondente, si osserva solo uno strato orizzontale di liquido (punte di freccia). (c) Dopo l'EC, il liquido marcato è sottratto dalla vista 2D. (d) Il polipo diventa visibile anche sulle viste endoluminali 3D e si presenta con una forma rotonda e una superficie liscia (freccia). Un artefatto lineare si verifica spesso all'interfaccia aria-liquido (punte di freccia). Esso aiuta a identificare le parti della parete colica che erano originariamente sommerse. Si noti che le pliche haustrali sono preservate.*

## Dettagli della procedura

In questa review iconografica (*pictorial review*), riportiamo le nostre osservazioni sull'uso clinico di un algoritmo di EC disponibile in commercio (Tagged Stool Subtraction, Syngo CT Colonography VB10, Siemens Healthcare), applicato su un database di dataset di pazienti CTC di screening validati con colonscopia, nonché nella pratica radiologica quotidiana. Descriviamo i potenziali effetti dell'EC sull'aspetto della morfologia anatomica intraluminale, nonché sulla morfologia e sulla dimensione delle lesioni coliche sommerse sotto il residuo marcato. Inoltre, dimostriamo i potenziali effetti negativi sulla refertazione clinica e sulla gestione del paziente, e descriviamo tecniche di *problem solving* nonché raccomandazioni per l'uso appropriato delle tecniche di EC.

Le osservazioni sull'EC presentate in questa review sono limitate all'algoritmo di EC utilizzato dagli autori. Tuttavia, poiché il principio di base dell'EC potrebbe essere applicato anche in altri approcci, non si può escludere che artefatti e variazioni nelle caratteristiche delle lesioni compaiano anche con altri algoritmi.

## Reperti clinici

Idealmente, dopo l'EC, solo le feci e i liquidi residui intraluminali marcati dovrebbero essere stati rimossi digitalmente dal colon. Le strutture coliche intrinseche, come la parete colica, le pliche semilunari e le lesioni tumorali del colon, non dovrebbero essere influenzate dall'EC, in segmenti con o senza residuo fecale marcato. Pertanto, queste strutture devono mantenere il loro tipico aspetto morfologico quando vengono applicati gli algoritmi di EC.

### Parete colica normale

Negli esami CTC con adeguata distensione colica, la parete del colon normale disteso è molto sottile, misurando meno di 2 mm. Tipicamente, dovrebbe essere appena percettibile sulle immagini CTC 2D e può essere meglio raffigurata con impostazioni della finestra addominale. La parete colica normale ha un'attenuazione dei tessuti molli sulle immagini 2D. Può mostrare un lieve enhancement dopo somministrazione di mezzo di contrasto endovenoso. Sulle immagini endoluminali 3D, il colon normale si presenta con una superficie liscia. I segmenti sommersi sono riconosciuti come uno strato orizzontale di liquido sulle viste endoluminali 3D, con un'attenuazione iperdensa sulle viste planari 2D.

Dopo la sottrazione digitale, i segmenti colici puliti elettronicamente si presentano spesso con una superficie lievemente più liscia sulle viste endoluminali 3D rispetto ai segmenti senza residuo fecale. Sulle viste 2D, la parete colica mantiene il suo normale spessore. Un artefatto lineare si verifica spesso all'interfaccia aria-liquido, sia sulle viste 2D sia su quelle endoluminali 3D. Questo artefatto può aiutare i radiologi a riconoscere le aree e i segmenti colici che sono stati puliti elettronicamente durante il processo di lettura (Fig. 1d).

### Lesioni coliche

Le lesioni coliche intrinseche dovrebbero avere le stesse caratteristiche morfologiche prima e dopo la pulizia elettronica. I criteri di imaging CTC di base, necessari per caratterizzare in modo sufficiente un difetto di riempimento colico rilevato, sono: la morfologia, correlata alla forma e alla superficie di un reperto; la struttura interna, che descrive l'attenuazione e l'omogeneità; e infine la mobilità, che mostra se un reperto è adeso o meno alla parete colica.

I polipi sessili sono difetti di riempimento emisferici, rotondi o ovali con una superficie liscia o lobulata, con una base uguale o maggiore dell'altezza della lesione (Fig. 1). I polipi colici originano dalla parete colica e ci si aspetta generalmente che mantengano la loro posizione intraluminale quando il paziente viene girato.

I polipi peduncolati presentano tipicamente una testa rotonda, ovale o lobulata. La testa del polipo è connessa alla parete colica da un peduncolo. Pertanto, la testa del polipo è mobile e seguirà sempre la gravità verso la parete colica declive (*dependent*).

Le lesioni non polipoidi, o cosiddette lesioni piatte, sono caratterizzate dalla loro bassa altezza (< 3 mm) rispetto alla larghezza. Alla CTC, si presentano come elevazioni a placca della parete colica, con una superficie liscia o nodulare.

Il carcinoma colorettale si presenta come una massa polipoide focale o un ispessimento semicircolare o anulare della parete colica, tipicamente con una breve estensione segmentaria (< 5 cm) e margini sporgenti (*overhanging edges*) con scalino (*shouldering*) al passaggio verso la mucosa normale adiacente. Sulle viste 2D con impostazioni della finestra addominale, i polipi colici e i carcinomi dimostrano un'attenuazione interna omogenea dei tessuti molli.

## L'importanza di una preparazione intestinale e di una marcatura fecale adeguate

È importante tenere sempre presente che l'EC è solo una manipolazione digitale dei dataset di immagini. Inoltre, la sua funzione di base si basa inevitabilmente sulla qualità della preparazione intestinale e, più specificamente, sull'attenuazione e sull'omogeneità della marcatura fecale. Il corretto funzionamento dell'EC richiede una marcatura omogenea del residuo fecale con un'elevata attenuazione TC, che sia al di sopra di una soglia di densità minima alla quale il software riconosce il residuo fecale come marcato. Negli esami con preparazione intestinale e/o marcatura fecale insufficiente, il corretto funzionamento dell'EC sarà compromesso e possono comparire artefatti nei dati di immagine. Pertanto, i dati di immagine sottratti dovrebbero essere interpretati con cautela.

### Marcatura insufficiente

Se la marcatura fecale non ha aumentato sufficientemente i valori di attenuazione del residuo fecale fino alla soglia minima dell'algoritmo di EC (per es. 100 HU o superiore), esso non verrà riconosciuto dall'algoritmo come marcato. Ciò può essere il risultato di una tempistica inappropriata della somministrazione degli agenti di marcatura, in modo che il mezzo di contrasto non sia arrivato nel colon oppure abbia già lasciato il colon al momento dell'esame. A seconda del regime di marcatura utilizzato, gli agenti di marcatura dovrebbero essere somministrati il pomeriggio o la sera prima dell'esame se questo è programmato al mattino, ma almeno 3 ore prima dell'esame. Inoltre, una marcatura insufficiente (*undertagging*) può verificarsi se la quantità di mezzo di contrasto somministrato è troppo piccola. Ciò può verificarsi in pazienti non aderenti alle prescrizioni della preparazione. Pertanto, i segmenti contenenti residuo «sottomarcato» non verranno puliti digitalmente (Fig. 2).

### Marcatura disomogenea / pulizia insufficiente

Se il residuo fecale contiene contenuti alimentari indigeribili o particelle di feci non marcate, esso appare disomogeneo sulle viste 2D dopo la marcatura fecale, con aree focali di alta densità e aree di bassa densità. L'algoritmo computerizzato, tuttavia, è in grado di sottrarre solo le parti iperdense marcate del residuo, ignorando le particelle ipodense non marcate. L'EC sarà quindi incompleta, con difetti di riempimento luminali bizzarri e irregolari sulle viste endoluminali 3D. Le immagini planari 2D non sottratte sono utili per distinguere tra residuo fecale sottratto in modo incompleto e reperti colici (Fig. 2).

### Particelle solitarie di feci non marcate o scarsamente marcate

Le particelle solitarie di feci non marcate o scarsamente marcate, circondate da liquido marcato, rimangono nel lume colico mentre il residuo fecale marcato viene sottratto. Le feci non marcate possono quindi simulare polipi colici sulle viste endoluminali 3D dopo l'EC ed eventualmente causare un reperto falsamente positivo. Tuttavia, a differenza dei veri polipi colici, il residuo si presenta spesso con un'inclusione d'aria o galleggia in una pozza di contrasto e, pertanto, non è adeso alla parete colica. Di conseguenza, mostrerà uno spostamento posizionale verso le sezioni declivi del segmento colico tra le acquisizioni TC prona e supina (Fig. 3).

### Depositi polipoidi persistenti di feci marcate

I piccoli depositi focali di feci marcate possono non essere sottratti dall'algoritmo. Ciò è intenzionale, affinché le lesioni piccole o piatte rivestite di materiale marcato non vengano mancate. Questo fenomeno, che è stato descritto come presente in una percentuale fino al 79 % di tutti i polipi piatti, può indicare la presenza di lesioni serrate. I depositi polipoidi persistenti di feci marcate sono facilmente riconosciuti come pseudolesioni per via della loro alta densità sulle viste 2D (Fig. 4). È pertanto importante valutare le corrispondenti viste 2D per valutare le caratteristiche di attenuazione di ciascun difetto di riempimento e per identificare le lesioni dei tessuti molli che sono ricoperte da uno strato di residuo marcato.

## Artefatti correlati al processo di sottrazione digitale

L'EC può essere la fonte di un'ampia gamma di artefatti di immagine che possono influire sull'aspetto intraluminale del colon. Alcuni di questi artefatti hanno il potenziale di ridurre l'aspetto e la cospicuità delle lesioni coliche, e altri possono simulare la presenza di lesioni coliche.

### Artefatti lineari

Dopo l'EC del liquido marcato, è spesso presente un artefatto lineare all'interfaccia aria-liquido. Esso può essere riconosciuto sia sulle immagini 3D sia 2D e non dovrebbe essere confuso con una vera struttura o lesione colica. I radiologi che valutano le immagini endoluminali 3D dopo l'EC possono utilizzare questo artefatto come marcatore per identificare le sezioni sommerse del colon che sono state pulite elettronicamente. Ciò può essere utile per la valutazione, prestando particolare attenzione nell'analisi dei reperti localizzati nei segmenti puliti elettronicamente (Fig. 1d). In caso di dubbio, le immagini sottratte e non sottratte devono essere confrontate.

### Pseudopolipi dovuti a effetti di volume parziale

Per via delle caratteristiche di immagine della TC, vi è sempre un effetto di volume parziale (*partial volume effect*) all'interfaccia tra parete colica, aria intraluminale e residuo marcato. Ciò può aumentare i valori di attenuazione, specificamente di quelle sezioni della parete colica e dell'aria che sono adiacenti al residuo marcato iperdenso. Ciò può avere un effetto negativo sugli algoritmi di sottrazione, conducendo ad artefatti che simulano lesioni polipoidi, specificamente se localizzati vicino alle pliche semilunari. Gli pseudopolipi correlati a effetti di volume parziale sono sempre localizzati all'interfaccia tra residuo marcato e aria. Si presentano tipicamente con una morfologia bizzarra più angolare sulle viste endoluminali 3D e planari 2D, atipica per le vere lesioni polipoidi. Le corrispondenti immagini 2D non sottratte sono diagnostiche e non mostreranno alcuna lesione polipoide corrispondente (Fig. 5). Inoltre, non vi sarà alcun reperto corrispondente nell'altra posizione di scansione.

### Artefatti dovuti al movimento del liquido residuo marcato

Il movimento intraluminale del liquido residuo marcato durante le scansioni CTC conduce ad artefatti all'interfaccia tra residuo marcato e aria. Idealmente, la superficie di un livello di liquido si presenta come uno strato lineare orizzontale liscio. Il movimento del paziente, la peristalsi intestinale, ma anche gli spostamenti di liquido dai segmenti superiori a quelli inferiori, condurranno a un movimento del liquido intraluminale con un aspetto ondulato, a bolle, o persino irregolare del livello di superficie sulle immagini CTC non sottratte. Gli artefatti da movimento del liquido possono presentarsi con un aspetto ondulato o a gradini (*stair-step*) al livello del liquido, o persino come strutture intraluminali bizzarre sulle immagini planari 2D ed endoluminali 3D. La parete colica non mostra artefatti correlati. Il movimento del liquido intraluminale compromette il processo di sottrazione, il che conduce ad artefatti di pulizia elettronica. Dopo l'EC, artefatti a griglia o bizzarri rimangono all'interfaccia aria-liquido (Fig. 6).

### Pseudopolipi dovuti a bolle di gas intrappolate

Le piccole bolle di gas, intrappolate nei segmenti colici completamente riempiti di liquido residuo marcato, possono simulare lesioni polipoidi sessili dopo l'EC. Ciò può accadere se lo strato di superficie della bolla di gas non viene sottratto e rimane all'interno del segmento colico. Sulle viste endoluminali 3D, esse si presentano come reperti polipoidi sessili, rotondi, con una superficie liscia. Sulle corrispondenti immagini 2D pulite elettronicamente, si osserva un reperto cistico riempito di gas con una parete sottile (Fig. 7). Poiché le bolle di gas risalgono all'interno del liquido, esse sono sempre localizzate nelle parti superiori dei segmenti sommersi, ma possono anche essere intrappolate all'interno di haustra profonde. Esse scompariranno quindi nell'altra posizione di scansione. Le viste 2D non sottratte identificheranno la bolla di gas.

## Artefatti su strutture anatomiche e lesioni coliche

Gli artefatti associati all'EC possono non solo simulare lesioni coliche. Essi possono anche causare la distorsione di strutture anatomiche e di lesioni coliche, e quindi comprometterne la corretta valutazione radiologica.

### Pliche semilunari

Le pliche semilunari sono strutture sottili, a forma di mezzaluna, con una superficie liscia e una densità dei tessuti molli. Nelle viste 2D non sottratte, sono meglio riconosciute con impostazioni della finestra ampie. Dopo l'EC, le pliche semilunari sottili, ricoperte di residuo marcato, possono essere distorte o scomparire interamente insieme al residuo (Fig. 8). Ciò può essere causato da una sottrazione erronea, risultante da uno pseudoenhancement delle pliche semilunari sottili che sono circondate su entrambi i lati da residuo marcato iperdenso.

### Parete colica

Analogamente alle pliche semilunari, la parete colica può essere influenzata dallo stesso meccanismo. Se due segmenti colici adiacenti, riempiti di residuo marcato, sono in contatto diretto, anche la parete colica può apparire distorta, con apparenti difetti della parete colica. La valutazione dei dati di immagine non sottratti è utile per verificare che la parete colica sia intatta.

### Morfologia del polipo

La corretta valutazione della morfologia di una lesione sulle immagini CTC è cruciale per classificare correttamente un reperto polipoide secondo le linee guida esistenti; per es., per determinare se un polipo mostra una morfologia peduncolata o sessile, è importante valutare la sua connessione alla parete colica. A seconda di quanto del polipo sia sommerso nel residuo marcato e se vi sia uno strato di liquido marcato tra la testa del polipo e la parete colica, l'aspetto tipico del polipo può essere significativamente influenzato dal processo di sottrazione. Gli artefatti dovuti all'EC possono rendere più difficile distinguere un polipo peduncolato da uno sessile, per es. se il peduncolo del polipo è raffigurato in modo incompleto o se la forma del polipo è distorta per via di variazioni sulla superficie della lesione (Fig. 9).

### Dimensione del polipo

È stato recentemente osservato che l'EC del residuo marcato può condurre a una significativa riduzione della dimensione dei polipi sommersi dal residuo marcato. Sulla base della nostra esperienza, la dimensione dei polipi ≥ 6 mm diminuiva del 4,1 % quando misurata in una finestra TC per il colon (W 1500 / L -150) e del 13,4 % quando misurata in una finestra addominale (W 400 / L 40) (Fig. 10). Una possibile spiegazione per la riduzione della dimensione del polipo dovuta all'EC è l'effetto di volume parziale, che risulta in un gradiente di attenuazione all'interfaccia contrasto-polipo. Quando si utilizza l'EC, il gradiente di attenuazione dal residuo marcato alla lesione colica deve essere trasformato in un gradiente dall'aria alla lesione colica. Attualmente, il nuovo gradiente/profilo di attenuazione all'interfaccia aria-polipo mostra un profilo più piatto rispetto a quello precedente all'EC, risultando quindi in un aspetto più piccolo dei polipi.

### Importanza della dimensione del polipo

I polipi adenomatosi sono potenziali lesioni precursori del carcinoma colorettale e sono pertanto una lesione bersaglio della CTC. Il rischio a 10 anni di carcinoma colorettale aumenta con la dimensione del polipo. Mentre è dello 0,08 % per i polipi diminutivi < 6 mm, aumenta rispettivamente allo 0,7 % e al 15,7 % per i polipi piccoli (6-9 mm) e grandi (≥ 10 mm). Pertanto, nella CTC, la dimensione è critica per la stima della rilevanza clinica di un polipo e per l'approccio terapeutico. Mentre le lesioni diminutive < 6 mm possono essere ignorate, vi è un consenso generale sul fatto che tutti i polipi ≥ 6 mm debbano essere refertati. Riguardo al trattamento, i grandi polipi ≥ 10 mm richiedono la resezione endoscopica, mentre la gestione dei piccoli polipi tra 6 e 10 mm è stata discussa in modo controverso. Mentre le linee guida europee richiedono la resezione colonscopica, vi è una crescente evidenza, da studi longitudinali sui polipi, che dovrebbe essere supportato un approccio meno invasivo che comprenda solo il follow-up.

Tenendo in considerazione queste considerazioni, la conoscenza dell'apparente riduzione della dimensione del polipo è di importanza cruciale nel determinare il corretto approccio terapeutico. Le apparenti riduzioni della dimensione dei polipi correlate all'EC possono essere la fonte di una errata classificazione delle lesioni in una categoria dimensionale scorretta e più piccola e, pertanto, condurre alla sottostima di lesioni di potenziale rilevanza clinica. Nel caso più sfavorevole, questa riduzione dimensionale correlata all'EC potrebbe far scomparire interamente i polipi insieme al residuo marcato (Fig. 11). Questo effetto è specificamente osservato con impostazioni della finestra TC strette.

Gli errori di misurazione correlati all'EC possono essere evitati misurando la dimensione del polipo sui dati di immagine TC non sottratti, utilizzando un'impostazione della finestra ampia. Ciò assicura la refertazione più accurata della dimensione del polipo e, pertanto, anche la procedura terapeutica appropriata per il paziente.

## Conclusione

Gli algoritmi computerizzati per la pulizia elettronica del residuo fecale marcato dal lume colico hanno il potenziale di migliorare la valutazione colica, in primo luogo consentendo una valutazione endoluminale 3D senza soluzione di continuità ed efficiente in termini di tempo dei segmenti colici riempiti di liquido residuo marcato, e in secondo luogo migliorando la cospicuità delle lesioni che sono sommerse dal residuo fecale marcato.

Tuttavia, la potenziale utilità di questa applicazione software è ancora limitata da una serie di artefatti che possono influire sulla valutazione e interpretazione colica, nonché sulla misurazione della dimensione delle lesioni coliche che sono sommerse sotto il residuo fecale marcato. Le future strategie per ridurre ulteriormente le insidie e gli artefatti correlati all'EC possono comprendere schemi di EC basati sulla capacità di decomposizione dei materiali della TC dual-energy.

I radiologi, nell'incorporare gli algoritmi di EC nel loro flusso di lavoro di routine, devono essere consapevoli dei potenziali effetti avversi associati al processo di sottrazione. I reperti colici sospetti che si incontrano sui dati di immagine sottratti digitalmente devono essere correlati con le corrispondenti viste non sottratte, per prevenire le insidie correlate all'EC. Per evitare la sottostima della dimensione del polipo, le misurazioni dei polipi sommersi dovrebbero essere eseguite sui dati di immagine non sottratti, in un'impostazione della finestra per il colon.

## Abbreviazioni

2D: bidimensionale; 3D: tridimensionale; TC: tomografia computerizzata; CTC: colonscopia TC; EC: pulizia elettronica (*electronic cleansing*).

## Conflitti di interesse

Gli autori dichiarano di non avere conflitti di interesse.
