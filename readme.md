# Local RAG PDF QA

Ce projet Python implémente un système de questions-réponses (QA) localement sur le contenu de fichiers PDF. Il utilise une architecture **RAG (Retrieval-Augmented Generation)** en combinant les capacités de **LangChain**, **Ollama** et **ChromaDB**, vous permettant d'interroger vos documents sans dépendance à des services cloud externes.


## Fonctionnalités

* **Chargement PDF Facilité** : Charge et découpe automatiquement les fichiers PDF en *chunks* optimisés pour la recherche.
* **Indexation Vectorielle Locale** : Crée et gère une base de données vectorielle locale avec **ChromaDB** pour une recherche sémantique efficace.
* **Génération d'Embeddings Offline** : Utilise **Ollama** pour générer des *embeddings* localement, garantissant la confidentialité des données.
* **Chaîne RAG Robuste** : Implémente une chaîne RAG pour récupérer les informations pertinentes du PDF et générer des réponses cohérentes.
* **Exemples de Requêtes Intégrés** : Fournit des exemples de requêtes pour vous aider à démarrer rapidement.



## Comment ça marche

Le processus se déroule en plusieurs étapes :

1.  **Chargement et Découpage du PDF** : Le fichier PDF est chargé et son contenu est divisé en petits segments (chunks) pour faciliter l'indexation et la récupération.
2.  **Génération d'Embeddings** : Chaque chunk est converti en un vecteur numérique (*embedding*) à l'aide d'un modèle de langage hébergé localement par Ollama. Ces vecteurs capturent le sens sémantique du texte.
3.  **Indexation Vectorielle (ChromaDB)** : Les *embeddings* sont stockés dans une base de données vectorielle ChromaDB. Cette base de données permet une recherche rapide des chunks les plus similaires à une question donnée.
4.  **Chaîne RAG (LangChain)** :
    * Lorsqu'une question est posée, elle est également convertie en *embedding*.
    * ChromaDB est interrogé pour trouver les chunks de documents les plus pertinents par rapport à la question.
    * Ces chunks pertinents sont ensuite passés, avec la question originale, à un grand modèle de langage (LLM) via LangChain.
    * Le LLM utilise ces informations pour générer une réponse synthétisée.

---

## Structure du Projet

.
├── agent_local.py      # Script principal pour la chaîne RAG (recommandé)
├── rag_local.py        # Variante alternative du script principal
├── data/
│   └── livre.pdf       # Placez votre fichier PDF à indexer ici (nommé "livre.pdf" par défaut)
├── chroma_db/          # Base de données vectorielle ChromaDB générée automatiquement
├── .env                # Variables d'environnement (optionnel - pour la configuration de modèles Ollama, etc.)
└── requirements.txt    # Dépendances Python

## Prérequis

Avant de commencer, assurez-vous d'avoir les éléments suivants :

* **Python 3.10 ou supérieur** : Vérifiez votre version avec `python --version`.
* **Ollama installé et fonctionnel localement** :   
* Téléchargez et installez Ollama depuis [ollama.com](https://ollama.com/).
* Assurez-vous qu'un modèle de langage est téléchargé et disponible (par exemple, `ollama pull llama2` ou `ollama pull mistral`). Ce modèle sera utilisé pour la génération d'embeddings et les réponses.
* **Un fichier PDF** : Placez le fichier PDF que vous souhaitez interroger dans le dossier `data/` et nommez-le `livre.pdf`. Si votre fichier a un nom différent, vous devrez peut-être ajuster le script `agent_local.py` en conséquence.



## Installation

Suivez ces étapes pour configurer le projet :

1.  **Cloner le dépôt** :
    ```bash
    git clone <URL_DU_REPO>
    cd local_AI # ou le nom de votre dossier si différent
    ```
    *(Remplacez `<URL_DU_REPO>` par l'URL réelle de votre dépôt Git.)*

2.  **Installer les dépendances Python** :
    ```bash
    pip install -r requirements.txt
    ```

3.  **Placer votre fichier PDF** :
    Copiez votre fichier PDF (`.pdf`) dans le dossier `data/`. Assurez-vous qu'il est nommé `livre.pdf` par défaut.


## Mode d'utilisation

Une fois que tous les prérequis sont installés et le projet configuré :

1.  **Lancez le script principal** :
    Exécutez le script `agent_local.py` qui gérera l'indexation et la chaîne de questions-réponses.

    ```bash
    python agent_local.py
    ```

2.  **Interagissez avec le système** :
    Le script vous invitera à poser des questions. Tapez votre question et appuyez sur Entrée. Le système cherchera dans votre PDF et vous fournira une réponse basée sur son contenu.

    *(Des exemples de requêtes peuvent être intégrés directement dans le script pour faciliter les tests.)*



## Conseils & Dépannage

* **Performance Ollama** : La qualité et la vitesse des réponses dépendront du modèle Ollama que vous utilisez et des ressources de votre machine. Des modèles plus petits peuvent être plus rapides, tandis que des modèles plus grands peuvent fournir des réponses plus détaillées.
* **Nettoyage de la base de données** : Si vous modifiez le PDF ou rencontrez des problèmes d'indexation, vous pouvez supprimer le dossier `chroma_db/` pour forcer une nouvelle indexation lors de la prochaine exécution.
* **Erreurs de dépendances** : Si vous rencontrez des erreurs liées à des modules manquants, assurez-vous que toutes les dépendances de `requirements.txt` sont correctement installées.
* **Problèmes avec Ollama** : Vérifiez que le serveur Ollama est bien en cours d'exécution et que le modèle spécifié est téléchargé (`ollama list` pour voir les modèles disponibles).
