# Gestion de Portefeuille à Revenus Flous

> Modélisation mathématique et résolution algorithmique d'un problème de gestion de portefeuille où les revenus des actifs sont des variables floues.

---

## 📋 Description

Ce projet s'inscrit dans le cadre du cours de **Probabilités et Raisonnement Probabiliste** (Master IA). Il propose un modèle mathématique complet basé sur la **théorie des ensembles flous** appliquée à la gestion de portefeuille financier, en s'inspirant du modèle de Markowitz étendu aux variables floues.

### Problématique

Dans la réalité financière, les rendements des actifs ne sont pas des valeurs précises mais des grandeurs incertaines. Plutôt que de les modéliser par des distributions probabilistes classiques, ce projet utilise des **nombres flous triangulaires** qui capturent naturellement l'imprécision de l'expert financier.

### Apports du modèle

- Représentation des rendements incertains par des **triplets flous** `(r_min, r_mod, r_max)`
- Calcul d'une **espérance floue** et d'une **variance floue** analytiques
- Paramètre **α (alpha-coupe)** permettant de régler le niveau de tolérance à l'incertitude
- Optimisation par `scipy.optimize` (méthode SLSQP) pour deux stratégies :
  - Portefeuille **minimum variance**
  - Portefeuille **maximum Sharpe flou**
- Tracé de la **frontière d'efficience floue**

---

## 🧮 Modèle Mathématique

| Concept | Formule |
|---|---|
| Nombre flou triangulaire | `r̃ = (r_min, r_mod, r_max)` |
| α-coupe | `[r̃]^α = [r_min + α(r_mod−r_min), r_max − α(r_max−r_mod)]` |
| Espérance floue | `Ẽ(r̃) = (r_min + 2·r_mod + r_max) / 4` |
| Variance floue | `σ̃²(r̃) = [(r_max−r_min)² + (r_mod−r_min)(r_max−r_mod)] / 12` |
| Variance portefeuille | `σ̃²(P) = wᵀ · Σ̃ · w` |
| Ratio de Sharpe flou | `S̃ = (Ẽ(P) − rf) / σ̃(P)` |

---

## 📁 Structure du projet

```
portefeuille-flou/
│
├── portefeuille_flou.py     # Script principal (modèle + optimisation + visualisation)
├── requirements.txt         # Dépendances Python
└── README.md                # Ce fichier
```

---

## ⚙️ Installation et lancement

### Prérequis

- Python **3.10** ou supérieur
- `pip` installé

Vérifiez votre version Python :

```bash
python --version
```

### Étape 1 — Cloner ou télécharger le projet

```bash
git clone https://github.com/votre-repo/portefeuille-flou.git
cd portefeuille-flou
```

### Étape 2 — Créer l'environnement virtuel

```bash
python -m venv venv
```

### Étape 3 — Activer l'environnement virtuel

**Windows :**
```bash
venv\Scripts\activate
```

**macOS / Linux :**
```bash
source venv/bin/activate
```

### Étape 4 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 5 — Lancer le script

```bash
python portefeuille_flou.py
```

### Désactiver l'environnement virtuel

```bash
deactivate
```

---

## 📊 Sorties générées

À l'exécution, le programme produit :

1. **Terminal** — tableau des α-coupes pour chaque actif et rapport complet des deux portefeuilles optimaux (poids, espérance, variance, Sharpe, indice de possibilité)
2. **Fichier image** `portefeuille_flou_resultats.png` — 4 graphiques :
   - Frontière d'efficience floue avec les portefeuilles optimaux
   - Allocations comparées (min variance vs max Sharpe)
   - Fonctions d'appartenance triangulaires des actifs
   - Évolution des α-coupes selon le niveau de confiance

---

## 🧰 Dépendances

| Bibliothèque | Version minimale | Usage |
|---|---|---|
| `numpy` | 1.26.0 | Calculs matriciels |
| `scipy` | 1.12.0 | Optimisation (SLSQP) |
| `matplotlib` | 3.8.0 | Visualisation |

---

## 👤 Auteur

**Kevin xxx**
Master Intelligence Artificielle

---

*Projet académique — Probabilités et Raisonnement Probabiliste*
