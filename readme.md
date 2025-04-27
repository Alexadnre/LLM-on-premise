# Projet E4 : Optimisation de DeepSeek - Tests de Paramètres

## Sommaire
- [Installation](#installation)
- [Lancement des tests d'optimisation](#lancement-des-tests-doptimisation)
- [Evaluation automatique des réponses](#evaluation-automatique-des-reponses)

---

## Installation

### Prérequis
Assure-toi d'avoir les outils suivants installés :
- **Git** : [Télécharger Git](https://git-scm.com/downloads)
- **Python 3.13.2** : [Télécharger Python](https://www.python.org/downloads/)

### Cloner le dépôt
```bash
git clone https://github.com/Alexadnre/LLM-on-premise.git
cd LLM-on-premise
```

### Création d'un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

### Installation des dépendances
```bash
pip install -r requirements.txt
```

---

## Lancement des tests d'optimisation

### Objectif
Tester différentes combinaisons de paramètres (temperature, top_k, top_p, max_tokens) sur DeepSeek pour trouver la configuration optimale.

### Commande
Vérifie que tu es sur la bonne branche :
```bash
git checkout DevDeepSeekTuning
```

Puis lance les tests :
```bash
streamlit run run_experiments.py
```

### Fonctionnement
- Charge et split les documents de la base de connaissance.
- Effectue des recherches sémantiques et BM25.
- Interroge DeepSeek avec différentes questions.
- Enregistre toutes les réponses et les paramètres utilisés dans `resultats_tests.xlsx`.

---

## Evaluation automatique des réponses

### Objectif
Noter automatiquement chaque réponse générée sur 100 selon sa pertinence, clarté et cohérence.

### Commande
Depuis la racine du projet :
```bash
streamlit run add_auto_scores_to_excel.py
```

### Fonctionnement
- Lit chaque feuille du fichier Excel `resultats_tests.xlsx`.
- Pour chaque réponse, envoie une demande d'évaluation à DeepSeek.
- Remplit une colonne supplémentaire "Note (/100)" pour chaque réponse.

**Remarque** : les évaluations sont automatiques et peuvent être affinées en ajustant le prompt d'évaluation.

---



