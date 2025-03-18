# Projet E4 : LLM On Premise

## Table des matières
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

## 1. Introduction

### 1.1. Contexte du projet
_Espace pour rédiger_

### 1.2. Objectifs du projet
- Mise en place d'un système RAG
- Utilisation du modèle DeepSeek  
_Espace pour rédiger_

### 1.3. Enjeux et défis
- Complexité des données internes
- Attentes en termes de pertinence, fiabilité, rapidité des réponses
- Respect de la confidentialité des données  
_Espace pour rédiger_

---

## 2. État de l'art et concepts théoriques

### 2.1. Les modèles LLM (Large Language Models)
- Définition et évolution
- Limites d'un LLM "simple"  
_Espace pour rédiger_

### 2.2. Introduction aux systèmes RAG
- Principe : combinaison retrieval + generation
- Architecture générale : embeddings, base vectorielle, LLM  
_Espace pour rédiger_

### 2.3. Focus sur DeepSeek et ses atouts
- Particularités du modèle
- Pourquoi le choix de DeepSeek pour ce projet ?  
_Espace pour rédiger_

### 2.4. Présentation de la technique d'embedding et de la base vectorielle (Chroma DB)
- Comment représenter des documents ?
- Recherche de similarité  
_Espace pour rédiger_

---

## 3. Présentation du jeu de données

### 3.1. Nature des données fournies par l'entreprise
- Type
- Volume  
_Espace pour rédiger_

---

## 4. Méthodologie et architecture du projet

### 4.1. Pipeline globale du projet
_Espace pour rédiger_

### 4.2. Pré-traitement et ingestion des documents
_Espace pour rédiger_

### 4.3. Génération et stockage des embeddings
_Espace pour rédiger_

### 4.4. Mise en place du moteur de recherche sémantique
_Espace pour rédiger_

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
