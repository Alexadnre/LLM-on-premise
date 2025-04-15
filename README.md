# Projet E4 : LLM On Premise

## Table des matières
   0. [Installation](#0-installation)


   1. [Introduction](#1-introduction)
      - [Contexte du projet](#11-contexte-du-projet)
      - [Objectifs du projet](#12-objectifs-du-projet)
      - [Enjeux et défis](#13-enjeux-et-défis)
   
2. [État de l'art et concepts théoriques](#état-de-lart-et-concepts-théoriques)
   - [Les modèles LLM (Large Language Models)](#21-les-modèles-llm-large-language-models)
   - [Introduction aux systèmes RAG](#22-introduction-aux-systèmes-rag)
   - [Focus sur DeepSeek et ses atouts](#23-focus-sur-deepseek-et-ses-atouts)
   - [Présentation de la technique d'embedding et de la base vectorielle (Chroma DB)](#24-présentation-de-la-technique-dembedding-et-de-la-base-vectorielle-chroma-db)
   
3. [Présentation du jeu de données](#présentation-du-jeu-de-données)
   - [Nature des données fournies par l'entreprise](#31-nature-des-données-fournies-par-lentreprise)
   
4. [Méthodologie et architecture du projet](#méthodologie-et-architecture-du-projet)
   - [Pipeline globale du projet](#41-pipeline-globale-du-projet)
   - [Pré-traitement et ingestion des documents](#42-pré-traitement-et-ingestion-des-documents)
   - [Génération et stockage des embeddings](#43-génération-et-stockage-des-embeddings)
   - [Mise en place du moteur de recherche sémantique](#44-mise-en-place-du-moteur-de-recherche-sémantique)
   - [Interaction avec DeepSeek pour la génération](#45-interaction-avec-deepseek-pour-la-génération)
   
5. [Implémentation technique](#implémentation-technique)
   - [Technologies et bibliothèques utilisées](#51-technologies-et-bibliothèques-utilisées)
   - [Structure des scripts et modules](#52-structure-des-scripts-et-modules)
   - [Exemples de prompts et de réponses](#53-exemples-de-prompts-et-de-réponses)
   
6. [Résultats et analyses](#résultats-et-analyses)
   - [Qualité des réponses](#61-qualité-des-réponses)
   - [Performances](#62-performances)
   - [Limites rencontrées](#63-limites-rencontrées)
   
7. [Améliorations possibles et perspectives](#améliorations-possibles-et-perspectives)
   - [Améliorations techniques](#71-améliorations-techniques)
   - [Exploration d'une piste : OpenAI](#72-Exploration d'une piste : OpenAI)
   - [Optimisation des paramètres](#73-Optimisation des paramètres)
   
8. [Conclusion](#conclusion)
   - [Bilan du projet](#81-bilan-du-projet)
   - [Enseignements tirés](#82-enseignements-tirés)
   - [Conclusion générale et recommandations](#83-conclusion-générale-et-recommandations)

---
## 0. Installation  

### 0.1 Prérequis  
Avant d’installer le projet, assure-toi d’avoir les outils suivants installés sur ton système :  
- **Git** : [Télécharger Git](https://git-scm.com/downloads)  
- **Python  3.13.2** : [Télécharger Python](https://www.python.org/downloads/)  

### 0.2 Cloner le dépôt  
Récupère le projet en clonant le dépôt Git avec la commande suivante :  

```sh
git clone https://github.com/Alexadnre/LLM-on-premise.git
cd LLM-on-premise
```
### 0.3 Création d'un environnement virtuel  
Il est recommandé d'utiliser un environnement virtuel pour isoler les dépendances du projet.  

Crée un environnement virtuel et active-le avec les commandes suivantes :  

#### **Sur Windows**  
```sh
python -m venv venv
venv\\Scripts\\activate
```
#### **Sur macOS/Linux**
```sh
python3 -m venv venv
source venv/bin/activate
```

### 0.4 Installation des dépendances  
Installe les dépendances nécessaires en fonction de la technologie utilisée :  
 
```sh
pip install -r requirements.txt
```


### 0.5 Configuration  
Si nécessaire, modifie le fichier `.env` et remplis le avec tes paramètres.
 

### 0.6 Lancer le projet  
Démarre l'application avec la commande :  
  
```sh
git checkout dev
streamlit run main.py
```

Le chatbot est maintenant installé et prêt à être utilisé !



## 1. Introduction

### 1.1. Contexte du projet

A une époque où les IA deviennent omniprésentes dans tous les projets en lien avec les technologies du numérique, les modèles de langages prennent de plus en plus de place et d’importance au sein de nombreuses entreprises, conscient de la pleine capacité de ces modèles pour résumer et récupérer des données. Bien que limité par leur incapacité à s’adapter à l’évolution des données puisqu’ils sont figés aux informations données lors de l'entraînement, ils représentent des outils puissants.

Dans le but de répondre à cette problématique, une approche récente a émergé, le RAG (Retrieval-Augmented Generation). Cette méthode combine la génération des LLM à un moteur de recherche sémantique, permettant d’aller chercher efficacement et dynamiquement des informations dans une base de connaissance. Le modèle s’appuyant sur des données actualisées par l’entreprise, toujours à l’ordre du jour. 

L’objectif est donc de développer une interface basée sur un LLM, capable d’interagir avec une base de données alimentée par les données de l’entreprise, et de procurer une réponse précise et rapide.


### 1.2. Objectifs du projet

- Mettre en place un système pour charger, découper et vectoriser les documents fournis par l’entreprise.
- Mettre en place une base vectorielle sémantique pour indexer efficacement les données.
- Interroger la base grâce au LLM.
- Garantir l’obtention de réponses pertinentes et fiables.


### 1.3. Enjeux et défis

La mise en place de ce système permettrait à l’entreprise de : 
- Faciliter l’accès à l’information au sein de celle-ci
- Améliorer la productivité en réduisant le temps passé à chercher les informations
- Optimiser la prise de décision
- Anticiper la montée en puissance de l’IA dans tous les secteurs dans les prochaines années


---

## 2. État de l'art et concepts théoriques

### 2.1. Les modèles LLM (Large Language Models)

Un LLM (Large Language Model) est un type de modèle d’intelligence artificielle basé sur un réseau de neurones. Il est conçu pour comprendre et manipuler le langage naturel.

Pour se faire, on entraîne ces modèles sur de grandes quantités de données textuelles, permettant ainsi à celui-ci d’acquérir une bonne compréhension du texte et d’instaurer des liens entre les mots et les idées présentes dans les documents pour faire des recherches plus efficacement.

Les tâches pouvant être réalisées par un LLM incluent par exemple : 

- La génération de textes à partir d’une requête
-L’extraction d’informations pertinentes dans un document
- Un dialogue fluide avec l’utilisateur

Bien que très puissants, les modèles LLM ont plusieurs limites importantes quand non couplées à un RAG en parallèle : 
 
- Mémoire figée, les documents sont chargés pour l’entraînement de celui-ci mais le modèle ne se met pas au goût du jour, c’est le même principe que pour des IA comme ChatGPT, qui est incapable d’accèder à des informations plus vieilles que 2022
- Hallucinations, terme utilisé pour décrire les réponses fausses ou inventées par l’IA, des textes qui paraissent plausibles mais qui ne sont pas nécessairement vrais pour autant. Cela est couplé par une absence de sourçage qui peut faire douter de la fiabilité des informations.

On va s’intéresser plus précisément dans ce projet à la génération de texte et l’extraction d’informations depuis les documents.
Les limites du LLM classiques seront contrebalancées par l’utilisation du RAG


### 2.2. Introduction aux systèmes RAG

Un RAG (Retrieval-Augmented Generation) est une approche qui combine 2 choses : 

- Génération d’un texte par un LLM
- Recherche d’informations dans une base de données

Le RAG vise à renforcer le modèle LLM en lui donnant accès à une base de données externe, pour fournir des réponses précises et vérifiables. 

**Fonction du RAG :** 

Recherche : Le moteur de recherche sémantique retrouve les documents contenant les passages les plus pertinents en rapport avec les informations demandées dans le prompt par l’utilisateur.
Augmentation : Les documents sont fournis pour donner du contexte au modèle.
- Génération : Le modèle génère une réponse en s’appuyant sur les informations trouvées, pour améliorer la précision de la réponse.

Le RAG présente de multiples avantages, il fournit des réponses précises s’appuyant sur des informations correctement sourcées, il donne accès à des données à jour, réduit les hallucinations de l’IA, et permet le réentraînement du LLM en cas de modification des informations.


### 2.3. Focus sur DeepSeek et ses atouts

Nous utiliserons dans ce projet le modèle de langage LLM DeepSeek.

Celui-ci est comparable à des modèles comme ChatGPT ou Mistral, mais se distingue néanmoins par des capacités avancées de compréhension et de raisonnement sur le langage naturel.
Il est performant dans le traitement de requêtes complexes, et la manipulation de documents volumineux. 
On s’en servira ici comme moteur LLM pour générer les réponses.
DeepSeek est par ailleurs open-source, et offre un excellent compromis entre performance et coût d’utilisation, ce qui n’est pas nécessairement le cas d'autres LLM tels que ChatGPT-4 par exemple. Il exploite correctement les larges chunks de textes via le moteur de recherche sémantique (que nous exécuterons via Chroma).
Le modèle génère des réponses synthétiques structurées et précises à partir des sources d’informations fournies.
Fonctionnant directement avec Ollama, cela facilite son intégration locale sans avoir à passer par un cloud, ce qui favorise le développement et garantit une certaine confidentialité.

Pour DeepSeek pour ce projet : 

DeepSeek reçoit en entrée la question de l’utilisateur, il utilise alors un prompt en internet pour générer une réponse contextualisée en s’appuyant sur les documents qui lui ont été fournis au préalable. 

### 2.4. Présentation de la technique d'embedding et de la base vectorielle (Chroma DB)

Dans le cadre de notre modèle, les différents documents de notre base de connaissances doivent être vectorisés afin d’optimiser le fonctionnement de celui-ci, cela se fait par la méthode d’embedding.

L’embedding est le processus de numérisation des différents documents sous la forme de vecteur de haute dimension. On cherche ici à capturer la notion de relation sémantique entre les éléments, ce qui permet de faire en sorte que les éléments ayant des concepts relativement similaires aient leur vecteurs équivalents proches entre eux. 

Dans le cadre des textes, l’embedding va transformer les différentes phrases en nombres réels.
Pour donner un exemple concret, les pommes et bananes seraient relativement proches en tant que vecteurs puisque ce sont tous deux des fruits.

Ce processus permet ainsi de trouver les informations les plus pertinentes dans la base de connaissances en utilisant les termes employés dans la question directement.

Dans le cadre de notre projet nous utilisons la librairie SentenceTransformer, plus précisément l’embedding est effectué en utilisant le modèle : all-miniLM-L6-v2

Tous ces vecteurs sont stockés dans une base de donnée vectorielle, on utilise ici Chroma DB.

---

## 3. Présentation du jeu de données

### 3.1. Nature des données fournies par l'entreprise

Dans le cadre de notre projet, l’on cherche à avoir un modèle qui peut marcher avec différents types de documents, cela se faisant par leur conversion en document pdf qui seraient ensuite traités.

Le volume de donnée variant selon l’utilisation faite du modèle par l’entreprise, qui peut être amené à rajouter des documents au besoin, d’où l’utilité de l’utilisation d’un LLM comparé à un réseau de neurone classique qui ne peut être entraîné qu'une fois sur un set de données fixe.

---

## 4. Méthodologie et architecture du projet

### 4.1. Pipeline globale du projet

[Créer dossier pour insérer image]

### 4.2. Pré-traitement et ingestion des documents

Afin de travailler sur les documents, il est nécessaire que ceux-ci soient tous au format pdf. On utilise pour la conversion la librairie subprocess. 
Une fois le texte correctement converti et utilisable par le programme, on cherche à segmenter les différents documents en des chunks (des bouts du documents d’origine).
Pour cette segmentation, on découpe en bout de 500 caractères, ce qui permet le plus souvent de correctement capturer le contexte de chaque passage du document tout en s’assurant que les extraits ne soient pas trop longs et ne compliquent pas le processus de recherche en ajoutant des informations parasites. De plus, on utilise un overlap (chevauchement) afin de s’assurer qu’on ne perd rien du contexte en coupant un extrait important au mauvais endroit. On s’assure par ailleurs de toujours couper en s’arrêtant à un point pour garantir qu’on finit toujours les phrases ce qui permet de conserver un maximum du contexte.


### 4.3. Génération et stockage des embeddings

Une fois les différents chunks créés, on les vectorise grâce à SentenceTransformer, puis on les stocke dans la base de donnée vectorielle créée avec ChromaDB.

### 4.4. Mise en place du moteur de recherche sémantique

Une fois cela fait, on procède à la recherche par similarité sémantique. On ne cherche pas ici à effectuer une recherche par mots-clés, mais bien à identifier quel document est pertinent vis-à-vis de la requête initiale en recherchant des liens sémantiques entre les termes utilisés dans la question et dans les documents.
Plus précisément, la recherche par similarité consiste à comparer les vecteurs de la requête avec les vecteurs composant la base de donnée vectorielle. 


### 4.5. Interaction avec DeepSeek pour la génération

On utilise un prompt dans l'intéraction avec DeepSeek de manière à filtrer les réponses considérés dans le cadre du projet comme non pertinente, notamment celles qui ne répondent pas à la question en utilisant la base de connaissance ou celles qui manquent de précision soit en étant vague soit trop courte. On cherche à faire comprendre à deepseek que l'on cherche la réponse la plus complète possible tout en éliminant toute possibilité d'informations externes qui ne nous serviraient pas vraiment.

---

## 5. Implémentation technique

### 5.1. Technologies et bibliothèques utilisées

Dans le cadre de ce projet nous utilisons un certain nombre de technologies et de librairie externe, pour ce qui est des librairies utilisées :  

Dans interface.py : 

Streamlit : Utilisé dans le cadre de l’affichage de l’interface, permet à l’utilisateur d'interagir avec notre IA directement pour lui poser des questions 
ChatOllama : Utilisé pour accéder au modèle DeepSeek, permet de le faire tourner localement.
Sentence transformer : Permet la vectorisation des différents chunks
ChromaDB : Permet de stocker les chunks vectorisés 
dotenv et os : Permettent de load des variables directement depuis un fichier en .env.
src.search_bdd : Permet d’effectuer la recherche sémantique à partir du modèle d’embeddings
src.bm25 : Permet d’effectuer la recherche.

Dans bm25.py : 

rank_bm25 : Permet d’effectuer le ranking bm25

Dans chunking.py :

re : Permet de split les strings pour les chunks
tqdm : Permet d’importer une barre de progression pour suivre le processus de chunking

Dans load_pdf.py :

langchain_community.document_loaders : Permet d’importer PyPDFDirectoryLoader pour importer les documents pdf.

En termes de technologies, nous utilisons dans un premier temps DeepSeek, puis OpenAI, pour ce qui est du modèle LLM comme pour les embeddings.


### 5.2. Structure des scripts et modules
Dans le cadre de ce projet nous utilisons un certain nombre de technologies et de librairie externe, pour ce qui est des librairies utilisées :  

Dans interface.py : 

Streamlit : Utilisé dans le cadre de l’affichage de l’interface, permet à l’utilisateur d'interagir avec notre IA directement pour lui poser des questions 
ChatOllama : Utilisé pour accéder au modèle DeepSeek, permet de le faire tourner localement.
Sentence transformer : Permet la vectorisation des différents chunks
ChromaDB : Permet de stocker les chunks vectorisés 
dotenv et os : Permettent de load des variables directement depuis un fichier en .env.

Dans bm25.py : 

rank_bm25 : Permet d’effectuer le ranking bm25

Dans chunking.py :

re : Permet de split les strings pour les chunks
tqdm : Permet d’importer une barre de progression pour suivre le processus de chunking

Dans load_pdf.py :

langchain_community.document_loaders : Permet d’importer PyPDFDirectoryLoader pour importer les documents pdf.

En termes de technologies, nous utilisons dans un premier temps DeepSeek, puis OpenAI, pour ce qui est du modèle LLM comme pour les embeddings.



Pour ce qui est du script principal à savoir main.py :

Dans un premier temps la fonction main import les fonctions des différents autres programmes

On load tous les éléments depuis le fichier env (que ce soit le modèle d’embedding, la database etc)

(insérer loading)

On procède alors au splitting et aux embeddings : 

(insérer splitting et embedding)

Puis on affiche l’interface pour l’utilisateur : 

(insérer dernière ligne du code)

On entre à présent plus dans le détail des différents scripts utilisés dans le main.

Interface.py s’occupe d’afficher l’interface, on initialise dans un premier temps le modèle 

(insérer initialisation Ollama)

On récupère le message.

(insérer récup message)

Une fois le message récupéré on effectue une recherche sémantique et une recherche lexical via bm25 

(insérer ce passage)

Pour ce qui est de bm25.py, on effectue une simple recherche lexical dans les documents en effectuant un système de ranking pour trouver les chunks les plus pertinents 

(insérer tout le code bm25)

Pour ce qui est de search_bdd.py, on fait de même mais avec une recherche sémantique.

(insérer tout le code de search_bdd)

Dans le cas où aucun contexte valide n’est trouvé, on ne répond pas à la question, sinon on soumet le prompt au LLM qui tente de retrouver la réponse dans les documents en s’en servant puis on l’affiche.


Pour ce qui est de load_pdf.py on utilise PyPDFDirectorLoader pour les load.

(insérer load_pdf.py intégralement) 

Pour ce qui est de chunking.py, on split les différents en pdf en plusieurs chunks en s’assurant de toujours s’arrêter sur une fin de phrase, on stock les chink au fur et à mesure jusqu’à avoir tout split. On applique un chevauchement entre les différents chunks pour s’assurer qu’on ne perde rien du contexte de chaque passage. 

(insérer split_text intégralement)

On utilise ensuite index_chunk pour stocker les chunks dans une base de données créée avec ChromaDB.

(insérer index_on_chroma.py)


### 5.3. Exemples de prompts et de réponses

Certains résultats obtenus en utilisant le modèle OpenAI : 

(insérer des résultats de réponse)


---

## 6. Résultats et analyses

### 6.1. Qualité des réponses

On constate que lorsque l’on utilise DeepSeek, les réponses ne sont pas très qualitatives, elles ne sont pas forcément très pertinentes, il répond parfois en utilisant des informations externes à la base de connaissance.
(insérer exemple réponse pas ouf)

(insérer exemple réponse pas lié à la bdc)

(insérer réponse où il parle littéralement chinois (ça j’y crois moyen))


### 6.2. Performances

 On constate un temps de réponse relativement élevé, néanmoins cela s’explique sans trop de problème par le fait que les modèles LLM consomment énormément de ressources, néanmoins cela est une nécessité puisqu’un modèle plus puissant permet une meilleure pertinence des résultats.

### 6.3. Limites rencontrées

Dans un premier temps l’un des documents pdf nous a posés quelques soucis dans la mesure où il n’était pas adapté à l’utilisation qu’il en était fait par notre LLM dans la mesure où la mise en page du pdf en question présentait des caractères qui n’étaient pas correctement reconnus par le programme.

(insérer erreur avec le pdf là)

De plus, DeepSeek s’est finalement révélé peu fiable, étant souvent sujet à des “hallucinations”, répondant à côté même quand les réponses étaient trouvables dans la base de connaissance, ce qui n’est pas le cas de son homologue par OpenAI.

De plus dans le cas où on ferait des recherches, même si un paragraphe contient l’information que l’on recherche, si un autre paragraphe moins pertinent comprend plus de fois le terme présenté dans la question, même si il n’y répond pas, il sera sélectionné, ce qui mène à des réponses bien moins pertinentes. C’est le problème de redondance.


---

## 7. Améliorations possibles et perspectives

### 7.1. Améliorations techniques

(A DEVELOPPER)

Faire en sorte que l’IA reformule la requête de l’utilisateur pour qu’elle soit plus interprétable par le struct rag
Utilisation d’un modèle plus performant (nécessité d’en tester un certain nombre)
Changement de paramètres (température etc)
Modification de l’interface pour la rendre plus user friendly
Utilisation d'un modèle différent de DeepSeek, pour vérifier sa pertinence


### 7.2. Exploration d'une piste : OpenAI

Dans le cadre de notre travail, nous nous sommes rendus compte que DeepSeek avait parfois des résultats décevants, ou même incorrects, nous avons donc décidés de nous intéresser à l'utilisation d'un autre modèle, celui d'OpenAI.
On observe des (en fait je sais pas quoi écrire faudrait que quelqu'un qui était spécifiquement sur cette partie du projet le fasse)


### 7.3. Optimisation des paramètres
_Espace pour rédiger_

---

## 8. Conclusion

### 8.1. Bilan du projet
_Espace pour rédiger_

### 8.2. Enseignements tirés
_Espace pour rédiger_

### 8.3. Conclusion générale et recommandations
_Espace pour rédiger_

---

