# Projet E4 : LLM On Premise

## Sommaire
- [Installation](#installation)
- [Utilisation avec Azure OpenAI](#utilisation-avec-azure-openai)
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
Modifie le fichier `.env` pour ajouter tes informations personnelles :
- `AZURE_OPENAI_ENDPOINT` : URL de ton endpoint Azure.
- `AZURE_OPENAI_API_KEY` : Clé d'API fournie par Azure.

Exemple de structure de `.env` :
```env
AZURE_OPENAI_ENDPOINT=https://votre-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=votre-cle-api
```

### Lancer le projet
Démarre l'application avec :

```bash
git checkout DevOpenAI
streamlit run main.py
```

Le chatbot utilisant Azure OpenAI est maintenant installé et prêt à être utilisé !

---

## Utilisation avec Azure OpenAI

Dans cette version, l'application utilise le modèle OpenAI déployé sur Azure pour générer les réponses.

Le modèle est initialisé comme suit :
```python
client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version='2025-01-01-preview',
        temperature = 0,
        streaming = True
        )
```

**Important** :
- Assure-toi que ton modèle OpenAI est correctement déployé sur ton instance Azure.
- Utilise la version d'API `2025-01-01-preview` ou ajuste selon tes besoins.

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

