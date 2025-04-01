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
   - [Fonctionnement détaillé du système](#53-fonctionnement-détaillé-du-système)
   - [Exemples de prompts et de réponses](#54-exemples-de-prompts-et-de-réponses)
   
6. [Résultats et analyses](#résultats-et-analyses)
   - [Qualité des réponses](#61-qualité-des-réponses)
   - [Performances](#62-performances)
   - [Cas d'usage réussis](#63-cas-dusage-réussis)
   - [Limites rencontrées](#64-limites-rencontrées)
   
7. [Améliorations possibles et perspectives](#améliorations-possibles-et-perspectives)
   - [Améliorations techniques](#71-améliorations-techniques)
   - [Intégration avec outils de l'entreprise](#72-intégration-avec-outils-de-lentreprise)
   - [Perspectives d'évolution](#73-perspectives-dévolution)
   
8. [Conclusion](#conclusion)
   - [Bilan du projet](#81-bilan-du-projet)
   - [Enseignements tirés](#82-enseignements-tirés)
   - [Conclusion générale et recommandations](#83-conclusion-générale-et-recommandations)
   
9. [Annexes](#annexes)
   - [Code source simplifié](#91-code-source-simplifié)
   - [Captures d'écran](#92-captures-decran)
   - [Liste des documents traités](#93-liste-des-documents-traités)
   - [Glossaire](#94-glossaire)

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
_Espace pour rédiger_

---

## 5. Implémentation technique

### 5.1. Technologies et bibliothèques utilisées
_Espace pour rédiger_

### 5.2. Structure des scripts et modules
_Espace pour rédiger_

### 5.3. Fonctionnement détaillé du système
_Espace pour rédiger_

### 5.4. Exemples de prompts et de réponses
_Espace pour rédiger_

---

## 6. Résultats et analyses

### 6.1. Qualité des réponses
_Espace pour rédiger_

### 6.2. Performances
_Espace pour rédiger_

### 6.3. Cas d'usage réussis
_Espace pour rédiger_

### 6.4. Limites rencontrées
_Espace pour rédiger_

---

## 7. Améliorations possibles et perspectives

### 7.1. Améliorations techniques
_Espace pour rédiger_

### 7.2. Intégration avec outils de l'entreprise
_Espace pour rédiger_

### 7.3. Perspectives d'évolution
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

## 9. Annexes

### 9.1. Code source simplifié
_Espace pour rédiger_

### 9.2. Captures d'écran
_Espace pour rédiger_

### 9.3. Liste des documents traités
_Espace pour rédiger_

### 9.4. Glossaire
_Espace pour rédiger_
