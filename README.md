# 🚀 Projet RAG avec DeepSeek, Azure et LangChain

Ce projet implémente un **système RAG (Retrieval-Augmented Generation)** en connectant **DeepSeek** à une base de données sur **Azure** et en utilisant **LangChain** pour le traitement des documents.

## 📌 Objectifs du projet

- Mettre en place une **interface utilisateur** avec **Streamlit**  
- Déployer un **modèle DeepSeek local** pour répondre aux questions avec un contexte  
- Stocker les documents traités dans une **base de données Azure**  
- Préparer et **splitter les documents** avec **LangChain**  
- **Vectoriser** les documents avec **OpenAI Embeddings / DeepSeek / GraphRAG**  
- Implémenter une **retrieval function** efficace avec **LangChain**  
- Tester et optimiser les paramètres (taille des splits, nombre d'extraits, etc.)  
- Développer une **fonction de reward** pour affiner les réponses  

## 🛠️ Technologies utilisées

| Outil / Techno  | Rôle |
|-----------------|------|
| **Python** | LangChain, DeepSeek, OpenAI, Neo4j, NetworkX, ArangoDB |
| **Streamlit** | Interface utilisateur |
| **Azure** | Base de données |
| **LangChain** | Traitement et retrieval des documents |
| **Git / GitHub** | Versionnement et collaboration |

## 📂 Structure du projet  

```
📦 mon_projet
├── 📂 data                # Documents sources
├── 📂 notebooks           # Tests et prototypes
├── 📂 src                 # Code source principal
│   ├── main.py           # Lancement de l'application
│   ├── ui.py             # Interface Streamlit
│   ├── deepseek.py       # Configuration du modèle DeepSeek
│   ├── db_setup.py       # Setup de la base Azure
│   ├── document_split.py # Split des documents avec LangChain
│   ├── vectorization.py  # Vectorisation des textes
│   ├── retrieval.py      # Fonction de recherche (RAG)
│   ├── reward.py         # Fonction de reward (ajustement auto)
│   ├── config.py         # Configuration générale
├── .gitignore            # Fichiers à ignorer pour Git
├── requirements.txt      # Dépendances Python
├── README.md             # Documentation du projet
```

## 🚀 Installation et Setup

### 1️⃣ Cloner le projet

```bash
git clone https://github.com/Alexadnre/LLM-on-premise
cd Projet
```

### 2️⃣ Créer un environnement virtuel et installer les dépendances  

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Lancer l'application

```bash
streamlit run src/main.py
```

## 💡 Fonctionnalités principales  

✅ **Interface utilisateur avec Streamlit**  
✅ **Base de données Azure** pour stocker les documents  
✅ **DeepSeek local** pour répondre aux questions avec contexte  
✅ **Préparation et split des documents** via **LangChain**  
✅ **Vectorisation des documents** avec OpenAI / DeepSeek / GraphRAG  
✅ **Retrieval intelligent** pour trouver les passages pertinents  
✅ **Optimisation du modèle** en ajustant les paramètres  
✅ **Fonction de reward** pour améliorer la pertinence des réponses  

## 📅 TODO  

- [ ] Mettre en place l'interface utilisateur graphique avec Streamlit  
- [ ] Télécharger et configurer DeepSeek  
- [ ] Déployer la base de données sur Azure  
- [ ] Implémenter la vectorisation des documents  
- [ ] Tester différentes stratégies de split  
- [ ] Ajuster la retrieval function pour améliorer la précision  

## 🤝 Contribution

Les contributions sont les bienvenues ! Si vous souhaitez contribuer :  

1. **Fork** le projet  
2. **Créez une branche** pour votre fonctionnalité (`git checkout -b feature-nouvelle-fonction`)  
3. **Committez** vos modifications (`git commit -m "Ajout d'une nouvelle fonctionnalité"`)  
4. **Poussez** vers votre branche (`git push origin feature-nouvelle-fonction`)  
5. **Ouvrez une Pull Request**  

