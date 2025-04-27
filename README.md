# Projet E4 : LLM On Premise

## Sommaire
- [Installation](#installation)
- [Introduction](#introduction)

---

## Installation

### Prérequis
Avant d’installer le projet, assure-toi d’avoir les outils suivants installés sur ton système :
- **Git** : [Télécharger Git](https://git-scm.com/downloads)
- **Python 3.13.2** : [Télécharger Python](https://www.python.org/downloads/)

### Cloner le dépôt
Récupère le projet en clonant le dépôt Git avec la commande suivante :

```bash
git clone https://github.com/Alexadnre/LLM-on-premise.git
cd LLM-on-premise
```

### Création d'un environnement virtuel
Il est recommandé d'utiliser un environnement virtuel pour isoler les dépendances du projet.

Crée un environnement virtuel et active-le avec les commandes suivantes :

#### Sur Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Sur macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Installation des dépendances
Installe les dépendances nécessaires avec :

```bash
pip install -r requirements.txt
```

### Configuration
Si nécessaire, modifie le fichier `.env` pour configurer tes paramètres personnels.

### Lancer le projet
Démarre l'application :

```bash
git checkout DevOpenAi
streamlit run main.py
```

Le chatbot est maintenant installé et prêt à être utilisé !

---

## Introduction

### Contexte du projet

À une époque où les IA deviennent omniprésentes, les modèles de langage (LLM) jouent un rôle crucial pour résumer et extraire des données. Cependant, ils sont limités par leur incapacité à se mettre à jour avec de nouvelles informations après leur entraînement.

Pour répondre à ce problème, une approche récente a émergé : le RAG (Retrieval-Augmented Generation), qui combine un moteur de recherche sémantique avec un LLM pour fournir des réponses basées sur une base de connaissances dynamique et actualisée.

Le projet vise à développer une interface basée sur un LLM, capable d'interroger une base de données interne et de fournir des réponses précises, fiables et rapides.

### Objectifs du projet
- Charger, découper et vectoriser des documents fournis par l'entreprise.
- Créer une base vectorielle sémantique pour indexer efficacement les données.
- Interroger cette base via un LLM.
- Garantir des réponses pertinentes et sourcées.

### Enjeux et défis
- Faciliter l’accès rapide à l’information en entreprise.
- Améliorer la productivité en réduisant le temps de recherche.
- Optimiser la prise de décision.
- Anticiper l’évolution massive de l'IA dans les prochaines années.

---

