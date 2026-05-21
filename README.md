# Gestion de Portefeuille à Revenus Flous

> Modélisation mathématique et résolution algorithmique d'un problème de gestion de portefeuille où les revenus des actifs sont des **variables floues** (nombres triangulaires flous), avec une interface interactive en terminal.

---

## Table des matières

1. [Description du projet](#1-description-du-projet)
2. [Modèle mathématique](#2-modèle-mathématique)
3. [Architecture du code](#3-architecture-du-code)
4. [Installation](#4-installation)
5. [Lancement et utilisation](#5-lancement-et-utilisation)
6. [Détail de chaque module interactif](#6-détail-de-chaque-module-interactif)
7. [Dépendances](#7-dépendances)
8. [Auteur](#8-auteur)

---

## 1. Description du projet

Ce projet s'inscrit dans le cadre du cours de **Probabilités et Raisonnement Probabiliste** (Master IA).

### Problématique

Dans la réalité financière, les rendements des actifs ne sont jamais des valeurs précises — ils dépendent de conditions de marché incertaines. Les méthodes classiques (espérance/variance probabilistes) supposent que l'on dispose d'un historique suffisant pour estimer ces distributions.

Ce projet adopte une approche différente : on modélise l'**imprécision de l'expert financier** grâce à la **théorie des ensembles flous**. Chaque rendement est représenté par un triplet `(r_min, r_mod, r_max)` qui exprime respectivement le scénario pessimiste, le plus probable, et l'optimiste.

### Ce que fait le programme

- Représente chaque actif par un **nombre flou triangulaire**
- Calcule analytiquement **l'espérance floue** et la **variance floue**
- Construit la **matrice de covariance floue** à partir des corrélations entre actifs
- Résout le problème d'optimisation (Markowitz flou) via `scipy.optimize`
- Propose deux stratégies : **minimum variance** et **maximum ratio de Sharpe flou**
- Trace la **frontière d'efficience floue**
- Expose tout cela dans un **menu interactif en terminal** avec 5 modules

---

## 2. Modèle mathématique

### 2.1 Nombre flou triangulaire

Chaque actif `i` a un rendement modélisé par :

```
r̃ᵢ = (r_min, r_mod, r_max)
```

Sa **fonction d'appartenance** est :

```
         ⎧ (x - r_min) / (r_mod - r_min)   si x ∈ [r_min, r_mod]
μ(x)  =  ⎨ (r_max - x) / (r_max - r_mod)   si x ∈ [r_mod, r_max]
         ⎩ 0                                sinon
```

> Implémenté dans `membership(x, actif)` — ligne 112.

---

### 2.2 α-coupe

L'**α-coupe** est l'intervalle des valeurs dont le degré d'appartenance est ≥ α :

```
[r̃ᵢ]^α = [ r_min + α(r_mod − r_min) ,  r_max − α(r_max − r_mod) ]
```

| α | Signification |
|---|---|
| `0` | Intervalle flou complet — incertitude totale prise en compte |
| `0.5` | Niveau de confiance modéré (valeur par défaut) |
| `1` | Seule la valeur modale est retenue — certitude totale |

> Implémenté dans `alpha_coupe(actif, alpha)` — ligne 117.

---

### 2.3 Espérance floue

```
Ẽ(r̃ᵢ) = (r_min + 2 × r_mod + r_max) / 4
```

Cette formule est la **moyenne pondérée** des trois paramètres, avec un poids double accordé à la valeur modale (la plus plausible).

> Implémenté dans `esperance_floue(actif)` — ligne 121.

---

### 2.4 Variance floue

```
σ̃²(r̃ᵢ) = [ (r_max − r_min)² + (r_mod − r_min)(r_max − r_mod) ] / 12
```

Elle mesure l'**étalement** du nombre flou. Plus l'intervalle `[r_min, r_max]` est large, plus la variance est élevée.

> Implémenté dans `variance_floue(actif)` — ligne 125.

---

### 2.5 Matrice de covariance floue

```
COṼ(i, j) = ρᵢⱼ × σ̃ᵢ × σ̃ⱼ
```

où `ρᵢⱼ` est le coefficient de corrélation entre les actifs `i` et `j`.
La matrice complète `Σ̃` (N×N) est construite automatiquement.

> Implémenté dans `matrice_covariance_floue()` — ligne 132.

---

### 2.6 Métriques du portefeuille

```
Ẽ(P)  = Σᵢ wᵢ × Ẽ(r̃ᵢ)             # Espérance floue du portefeuille

σ̃²(P) = wᵀ × Σ̃ × w                 # Variance floue du portefeuille

S̃(P)  = (Ẽ(P) − rf) / σ̃(P)         # Ratio de Sharpe flou

π(P)  = Φ((Ẽ(P) − rf) / σ̃(P))      # Indice de possibilité (loi normale)
```

---

### 2.7 Problème d'optimisation

```
Minimiser   σ̃²(P) = wᵀ Σ̃ w

Sous :      Σ wᵢ = 1     (somme des poids = 100%)
            wᵢ ≥ 0        (pas de vente à découvert)
            Ẽ(P) ≥ E*    (rendement cible, optionnel)
```

Résolu par la méthode **SLSQP** (Sequential Least Squares Programming) de `scipy.optimize.minimize`.

---

## 3. Architecture du code

```
portefeuille_flou.py  (657 lignes)
│
├── Bloc 1  — Utilitaires terminal          (lignes  22–83)
│   ├── Couleurs ANSI (CYAN, GREEN, YELLOW, RED, BOLD…)
│   ├── saisir_float()  — lecture sécurisée d'un nombre réel
│   ├── saisir_int()    — lecture sécurisée d'un entier
│   └── pause()         — attente "Entrée pour continuer"
│
├── Bloc 2  — État global                   (lignes  86–105)
│   ├── ACTIFS       — dictionnaire des 4 actifs (triplets flous)
│   ├── CORRELATION  — matrice 4×4 de corrélations
│   ├── RF = 2.0     — taux sans risque (%)
│   └── ALPHA = 0.5  — niveau de confiance flou par défaut
│
├── Bloc 3  — Fonctions mathématiques floues (lignes 108–190)
│   ├── membership()              — fonction d'appartenance μ(x)
│   ├── alpha_coupe()             — intervalle [r_L^α, r_R^α]
│   ├── esperance_floue()         — Ẽ(r̃)
│   ├── variance_floue()          — σ̃²(r̃)
│   ├── ecart_type_flou()         — σ̃(r̃)
│   ├── matrice_covariance_floue()— Σ̃ (N×N)
│   ├── variance_portefeuille()   — wᵀΣ̃w
│   ├── esperance_portefeuille()  — wᵀẼ
│   ├── ratio_sharpe_flou()       — S̃(P)
│   ├── indice_possibilite()      — π(P)
│   ├── optimiser_min_variance()  — SLSQP min variance
│   ├── optimiser_max_sharpe()    — SLSQP max Sharpe
│   └── calculer_frontiere()      — balayage 60 points
│
├── Module [1] menu_actifs()       (lignes 196–284)
├── Module [2] menu_alpha()        (lignes 290–323)
├── Module [3] menu_optimisation() (lignes 329–424)
├── Module [4] menu_simulateur()   (lignes 430–480)
├── Module [5] menu_graphiques()   (lignes 486–613)
└── menu_principal()               (lignes 619–657)
```

---

## 4. Installation

### Prérequis

- Python **3.10** ou supérieur

```bash
python --version
```

### Étape 1 — Récupérer les fichiers

```bash
cd TCHEUMTCHOUA_KOAGNE_FRANCK_MASTER2_IA
```

### Étape 2 — Créer l'environnement virtuel (optionnel car déjà fait dans le projet)

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

### Étape 5 — Lancer le programme

```bash
python portefeuille_flou.py
```

### Désactiver l'environnement virtuel

```bash
deactivate
```

---

## 5. Lancement et utilisation

Au démarrage, le programme affiche le **menu principal** coloré en terminal :

```
╔══════════════════════════════════════════════════════════╗
║           PORTEFEUILLE À REVENUS FLOUS                   ║
║     Kevin xxx — Master Intelligence Artificielle         ║
╚══════════════════════════════════════════════════════════╝

  [1]  Gérer les actifs             (ajouter, modifier, supprimer)
  [2]  Simulateur α-coupe           (intervalles de confiance flous)
  [3]  Optimisation du portefeuille (min variance / max Sharpe / cible)
  [4]  Simulateur personnalisé      (tester vos propres poids)
  [5]  Visualisation                (front efficience, graphiques)
  [q]  Quitter
```

Entrez le numéro du module souhaité et appuyez sur `Entrée`.
Chaque sous-menu propose des invites guidées avec valeurs par défaut entre crochets — appuyez directement sur `Entrée` pour les accepter.

---

## 6. Détail de chaque module interactif

---

### Module 1 — Gérer les actifs

**Accès :** taper `1` au menu principal.

Ce module affiche le tableau complet des actifs avec leurs paramètres flous et leurs métriques calculées (espérance floue Ẽ, écart-type flou σ̃).

**Actions disponibles :**

| Touche | Action | Détail |
|---|---|---|
| `1` à `4` | Modifier un actif | Saisir de nouveaux `r_min`, `r_mod`, `r_max`. Laisser vide = conserver la valeur. Le programme vérifie que `r_min ≤ r_mod ≤ r_max`. |
| `a` | Ajouter un actif | Entrer un nom et les trois paramètres flous. La corrélation avec les actifs existants est initialisée à 0.1 (neutre). |
| `s` | Supprimer un actif | Sélectionner par numéro. Confirmation demandée. Minimum 2 actifs requis. |
| `r` | Réinitialiser | Remet les 4 actifs par défaut et la matrice de corrélation d'origine. |
| `q` | Retour au menu | — |

> Tous les modules suivants utilisent l'état courant des actifs — toute modification ici est immédiatement répercutée.

---

### Module 2 — Simulateur α-coupe

**Accès :** taper `2` au menu principal.

Permet de choisir interactivement un **niveau de confiance α ∈ [0, 1]** et d'observer l'effet sur les intervalles flous de chaque actif.

**Ce qui s'affiche :**

```
Actif            [r_L^α       r_R^α]   Largeur   Milieu
Action A         [ 5.000,    11.000]     6.000    8.000
Action B         [ 3.000,     8.000]     5.000    5.500
Obligation C     [ 4.500,     7.500]     3.000    6.000
Immobilier D     [ 6.500,    12.500]     6.000    9.500
```

À la fin, le programme propose de tracer le **graphique d'évolution des α-coupes** : pour chaque actif, la zone colorée montre comment l'intervalle `[r_L^α, r_R^α]` se rétrécit quand α augmente, jusqu'à se réduire à un point (la valeur modale) en α = 1.

---

### Module 3 — Optimisation du portefeuille

**Accès :** taper `3` au menu principal.

Le programme demande d'abord le **taux sans risque rf** (valeur par défaut : 2%), puis propose 4 stratégies :

| Option | Stratégie | Description |
|---|---|---|
| `1` | Minimum variance | Trouve les poids `w*` qui minimisent `σ̃²(P) = wᵀΣ̃w`. Portefeuille le moins risqué. |
| `2` | Maximum Sharpe flou | Maximise `S̃ = (Ẽ(P) − rf) / σ̃(P)`. Meilleur compromis rendement/risque. |
| `3` | Cible de rendement | L'utilisateur fixe un rendement cible `E*`. Le programme trouve les poids minimisant le risque sous contrainte `Ẽ(P) ≥ E*`. |
| `4` | Comparer les 3 | Calcule et affiche les 3 portefeuilles simultanément, puis propose un graphique comparatif (front d'efficience + histogramme des allocations). |

**Pour chaque portefeuille, le programme affiche :**

- Les poids `wᵢ` sous forme de barres textuelles (visualisation immédiate)
- L'espérance floue `Ẽ(P)`
- La variance floue `σ̃²(P)` et l'écart-type `σ̃(P)`
- Le ratio de Sharpe flou `S̃(P)`
- L'indice de possibilité `π(P)` — probabilité que le rendement dépasse `rf`

---

### Module 4 — Simulateur personnalisé

**Accès :** taper `4` au menu principal.

Permet de tester **vos propres pondérations** et d'observer instantanément les métriques résultantes.

**Déroulement :**

1. Le programme demande le poids (en %) de chaque actif, un par un, avec la valeur équipondérée comme suggestion.
2. Si la somme ≠ 100%, une **normalisation automatique** est appliquée (avec avertissement).
3. Les métriques `Ẽ(P)`, `σ̃²(P)`, `σ̃(P)`, ratio de Sharpe et indice de possibilité sont affichés.
4. Le programme calcule le **portefeuille optimal (Max Sharpe)** et compare son ratio de Sharpe au vôtre — il vous indique si votre allocation est proche ou loin de l'optimum.
5. Une boucle permet de tester plusieurs allocations à la suite sans revenir au menu.

---

### Module 5 — Visualisation

**Accès :** taper `5` au menu principal.

Propose 5 types de graphiques via `matplotlib` :

| Option | Graphique | Contenu |
|---|---|---|
| `1` | Front d'efficience flou | Courbe risque/rendement avec les deux portefeuilles optimaux positionnés dessus, et la droite du taux sans risque. |
| `2` | Fonctions d'appartenance | Les 4 courbes triangulaires μ(x) superposées, avec aire colorée sous chaque courbe. |
| `3` | α-coupes par actif | Zones colorées montrant l'évolution des intervalles flous selon α, avec une ligne verticale pour le niveau α saisi. |
| `4` | Allocation (camembert) | Deux diagrammes circulaires côte à côte — Min Variance et Max Sharpe. |
| `5` | Tableau de bord complet | Les 4 graphiques précédents sur une seule figure 2×2, **sauvegardée automatiquement** en `portefeuille_flou_resultats.png`. |

---

## 7. Dépendances

| Bibliothèque | Version min. | Rôle dans le projet |
|---|---|---|
| `numpy` | 1.26.0 | Calculs matriciels (covariance, produits wᵀΣw) |
| `scipy` | 1.12.0 | Optimisation SLSQP (`scipy.optimize.minimize`) et loi normale (`scipy.stats.norm`) |
| `matplotlib` | 3.8.0 | Tous les graphiques (front d'efficience, α-coupes, camemberts…) |

Toutes font partie de la bibliothèque standard scientifique Python — aucune dépendance externe supplémentaire.

---

## 8. Auteur

**TCHEUMTCHOUA KOAGNE FRANCK**
Master 2 Intelligence Artificielle


---

*Projet académique — Probabilités et Raisonnement Probabiliste*


GitHub : https://github.com/KevinKefra1/portefeuille_flou.git