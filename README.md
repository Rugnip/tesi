# Tesi

# Gestione dell’ambiente di sviluppo e configurazione del repository

Per lo sviluppo del progetto è stato utilizzato Visual Studio Code in modalità remota, sfruttando la connessione a un server remoto tramite l’estensione Remote - SSH. In questo modo è stato possibile lavorare direttamente sui file presenti sul server mantenendo la comodità dell’interfaccia di VS Code.

Successivamente, Visual Studio Code è stato configurato per integrarsi con GitHub, consentendo la gestione del versionamento e delle operazioni di push/pull direttamente dall’editor. È stato quindi creato un nuovo repository GitHub denominato “Tesi”, al quale è stata collegata la cartella di progetto presente sul server remoto.

All’interno del progetto è stata creata la cartella dataset/, nella quale sono stati caricati diversi file JSON di grandi dimensioni necessari per le analisi. Tuttavia, tali file superavano il limite massimo consentito da GitHub (100 MB per singolo file). Per evitare errori durante le operazioni di push e mantenere il repository leggero, la cartella dataset/ è stata inserita nel file .gitignore, permettendo così a Git di ignorare il tracciamento dei file JSON presenti al suo interno pur continuando a renderli disponibili nell’ambiente locale di sviluppo.

# Download file Json

Poiché i file JSON contenuti nella cartella dataset/ superano il limite massimo di 100 MB imposto da GitHub, è stato deciso di non includerli direttamente nel repository, ma di renderli disponibili tramite link esterni. In questo modo è possibile mantenere leggero il repository garantendo comunque l’accesso ai dati necessari per l’esecuzione del progetto.

I dataset possono essere scaricati dai seguenti link:

- australian_user_reviews.json: https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz

- australian_users_items.json: https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_users_items.json.gz

- bundle_data.json: https://mcauleylab.ucsd.edu/public_datasets/data/steam/bundle_data.json.gz

- steam_games.json: https://cseweb.ucsd.edu/~wckang/steam_games.json.gz

Questi file devono essere posizionati all’interno della cartella dataset/ per poter essere utilizzati correttamente dagli script del progetto.

# Procedura per la gestione del commit e dell’invio del progetto su GitHub

Durante la fase di configurazione del repository Git, è stato necessario gestire correttamente la presenza di file di grandi dimensioni e l’integrazione con il sistema di versionamento. La procedura adottata per effettuare il commit in modo corretto è stata la seguente:

# Configurazione dell’identità Git

Sono state impostate le informazioni dell’utente per permettere a Git di registrare correttamente i commit:

git config --global user.name "rugnip"
git config --global user.email "marino.ruggiero04@gamil.com"

# Esclusione della cartella dataset/ dal tracking di Git

Per evitare problemi con file superiori a 100 MB, è stato creato o aggiornato il file .gitignore includendo:

git add .gitignore
git commit -m "Ignorata la cartella dataset"

# Caricamento file

Preparazione dei file da caricare su GitHub
Per caricare i file eseguiamo i segunti passaggi: 

git add .
git commit -m "Commento commit"
git push origin main

