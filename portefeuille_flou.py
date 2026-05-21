"""
=============================================================================
  GESTION DE PORTEFEUILLE À REVENUS FLOUS
  Modèle mathématique basé sur les nombres flous triangulaires
  Optimisation par scipy.optimize (Markowitz flou)
=============================================================================

Auteur  : Projet de probabilités & raisonnement probabiliste
Langage : Python 3.10+
Dépend. : numpy, scipy, matplotlib

Formules clés :
  - Nombre flou triangulaire : r̃ = (r_min, r_mod, r_max)
  - α-coupe : [r̃]^α = [r_min + α(r_mod-r_min), r_max - α(r_max-r_mod)]
  - Espérance floue : Ẽ(r̃) = (r_min + 2·r_mod + r_max) / 4
  - Variance floue  : σ̃²(r̃) = [(r_max-r_min)² + (r_mod-r_min)(r_max-r_mod)] / 12
  - Covariance floue : COṼ(i,j) ≈ ρ_ij · σ̃_i · σ̃_j
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# 1. DÉFINITION DES ACTIFS (nombres flous triangulaires)
# ─────────────────────────────────────────────────────────────────────────────

ACTIFS = {
    "Action A":    {"r_min": 2.0,  "r_mod": 8.0,  "r_max": 14.0},
    "Action B":    {"r_min": 1.0,  "r_mod": 5.0,  "r_max": 11.0},
    "Obligation C":{"r_min": 3.0,  "r_mod": 6.0,  "r_max":  9.0},
    "Immobilier D":{"r_min": 4.0,  "r_mod": 9.0,  "r_max": 16.0},
}

# Matrice de corrélation entre actifs (estimation)
CORRELATION = np.array([
    [1.00,  0.35, -0.10,  0.20],
    [0.35,  1.00,  0.05,  0.15],
    [-0.10, 0.05,  1.00, -0.05],
    [0.20,  0.15, -0.05,  1.00],
])

NOMS   = list(ACTIFS.keys())
N      = len(NOMS)
RF     = 2.0   # Taux sans risque (%)
ALPHA  = 0.5   # Niveau de confiance flou par défaut


# ─────────────────────────────────────────────────────────────────────────────
# 2. FONCTIONS MATHÉMATIQUES FLOUES
# ─────────────────────────────────────────────────────────────────────────────

def membership(x, actif: dict) -> float:
    """Fonction d'appartenance triangulaire μ(x) ∈ [0,1]."""
    a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
    if x <= a or x >= b:
        return 0.0
    elif x <= m:
        return (x - a) / (m - a)
    else:
        return (b - x) / (b - m)


def alpha_coupe(actif: dict, alpha: float) -> tuple:
    """
    α-coupe d'un nombre flou triangulaire.
    Retourne l'intervalle [r_L^α, r_R^α].
    """
    a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
    r_L = a + alpha * (m - a)
    r_R = b - alpha * (b - m)
    return (r_L, r_R)


def esperance_floue(actif: dict) -> float:
    """
    Espérance floue d'un nombre triangulaire.
    Ẽ(r̃) = (r_min + 2·r_mod + r_max) / 4
    """
    a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
    return (a + 2*m + b) / 4


def variance_floue(actif: dict) -> float:
    """
    Variance floue d'un nombre triangulaire.
    σ̃²(r̃) = [(r_max-r_min)² + (r_mod-r_min)(r_max-r_mod)] / 12
    """
    a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
    return ((b - a)**2 + (m - a)*(b - m)) / 12


def ecart_type_flou(actif: dict) -> float:
    """σ̃(r̃) = sqrt(σ̃²(r̃))"""
    return np.sqrt(variance_floue(actif))


def matrice_covariance_floue() -> np.ndarray:
    """
    Matrice de covariance floue Σ̃ de taille (N×N).
    COṼ(i,j) = ρ_ij · σ̃_i · σ̃_j
    """
    noms = NOMS
    sigma = np.array([ecart_type_flou(ACTIFS[n]) for n in noms])
    cov   = np.outer(sigma, sigma) * CORRELATION
    return cov


def variance_portefeuille(w: np.ndarray, cov: np.ndarray) -> float:
    """σ̃²(P) = wᵀ · Σ̃ · w"""
    return float(w @ cov @ w)


def esperance_portefeuille(w: np.ndarray) -> float:
    """Ẽ(P) = Σ wᵢ · Ẽ(r̃ᵢ)"""
    E = np.array([esperance_floue(ACTIFS[n]) for n in NOMS])
    return float(w @ E)


def ratio_sharpe_flou(w: np.ndarray, cov: np.ndarray, rf: float = RF) -> float:
    """
    Ratio de Sharpe flou.
    S̃ = (Ẽ(P) - rf) / σ̃(P)
    """
    ep = esperance_portefeuille(w)
    vp = variance_portefeuille(w, cov)
    return (ep - rf) / np.sqrt(vp) if vp > 0 else 0.0


def indice_possibilite(w: np.ndarray, cov: np.ndarray, seuil: float = RF) -> float:
    """
    Indice de possibilité π que le rendement dépasse le seuil.
    π = P(Ẽ(P) ≥ seuil) ≈ Φ((Ẽ(P) - seuil) / σ̃(P))
    """
    ep = esperance_portefeuille(w)
    sp = np.sqrt(variance_portefeuille(w, cov))
    return float(norm.cdf((ep - seuil) / sp)) if sp > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIMISATION — PORTEFEUILLE OPTIMAL (min variance)
# ─────────────────────────────────────────────────────────────────────────────

def optimiser_min_variance(cov: np.ndarray, E_cible: float | None = None):
    """
    Minimise la variance floue du portefeuille.

    Problème :
        min  wᵀ Σ̃ w
        s.t. Σ wᵢ = 1
             wᵢ ≥ 0
             Ẽ(P) ≥ E_cible  (si spécifié)

    Retourne : poids optimaux w*
    """
    E_vect = np.array([esperance_floue(ACTIFS[n]) for n in NOMS])

    # Fonction objectif
    def objectif(w):
        return variance_portefeuille(w, cov)

    def grad_objectif(w):
        return 2 * cov @ w

    # Contraintes
    contraintes = [{"type": "eq",
                    "fun": lambda w: np.sum(w) - 1,
                    "jac": lambda w: np.ones(N)}]

    if E_cible is not None:
        contraintes.append({
            "type": "ineq",
            "fun": lambda w: E_vect @ w - E_cible,
            "jac": lambda w: E_vect,
        })

    # Bornes : 0 ≤ wᵢ ≤ 1
    bornes = [(0.0, 1.0)] * N

    # Point de départ : équipondéré
    w0 = np.ones(N) / N

    resultat = minimize(
        objectif,
        w0,
        jac=grad_objectif,
        method="SLSQP",
        bounds=bornes,
        constraints=contraintes,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not resultat.success:
        print(f"  [Avertissement] Convergence : {resultat.message}")

    return resultat.x


def optimiser_max_sharpe(cov: np.ndarray):
    """
    Maximise le ratio de Sharpe flou.
    Équivalent à minimiser -S̃(w).
    """
    def neg_sharpe(w):
        return -ratio_sharpe_flou(w, cov)

    contraintes = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bornes  = [(0.0, 1.0)] * N
    w0      = np.ones(N) / N

    resultat = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bornes,
        constraints=contraintes,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return resultat.x


# ─────────────────────────────────────────────────────────────────────────────
# 4. FRONTIÈRE D'EFFICIENCE FLOUE
# ─────────────────────────────────────────────────────────────────────────────

def calculer_frontiere(cov: np.ndarray, nb_points: int = 60):
    """
    Calcule la frontière d'efficience floue en balayant
    les niveaux de rendement cible de Ẽ_min à Ẽ_max.

    Retourne : (risques, rendements, poids)
    """
    E_vect = np.array([esperance_floue(ACTIFS[n]) for n in NOMS])
    E_min  = E_vect.min() + 0.01
    E_max  = E_vect.max() - 0.01
    cibles = np.linspace(E_min, E_max, nb_points)

    risques     = []
    rendements  = []
    poids_liste = []

    for E_c in cibles:
        w = optimiser_min_variance(cov, E_cible=E_c)
        ep = esperance_portefeuille(w)
        vp = variance_portefeuille(w, cov)
        if vp >= 0:
            risques.append(np.sqrt(vp))
            rendements.append(ep)
            poids_liste.append(w)

    return np.array(risques), np.array(rendements), poids_liste


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYSE DES α-COUPES
# ─────────────────────────────────────────────────────────────────────────────

def analyser_alpha_coupes(alphas: list | None = None):
    """
    Calcule les intervalles α-coupés pour différents niveaux α
    et montre leur influence sur la variance du portefeuille.
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\n" + "─"*65)
    print(f"  ANALYSE DES α-COUPES")
    print("─"*65)
    print(f"  {'Actif':<16} {'α':<6} {'[r_L^α':>10} {'r_R^α]':>10}  Largeur")
    print("─"*65)

    for nom, actif in ACTIFS.items():
        for alpha in alphas:
            lo, hi = alpha_coupe(actif, alpha)
            print(f"  {nom:<16} {alpha:<6.2f} [{lo:>8.3f}, {hi:>8.3f}]  {hi-lo:>6.3f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

COULEURS = ["#1D9E75", "#378ADD", "#7F77DD", "#D85A30"]


def tracer_resultats(cov: np.ndarray, w_mv: np.ndarray, w_ms: np.ndarray):
    """Génère 4 graphiques : front d'efficience, allocation, fonctions d'appartenance, α-coupes."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Gestion de Portefeuille à Revenus Flous\n(Modèle de Markowitz Flou)",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── 4a. Front d'efficience ──────────────────────────────────────────────
    ax = axes[0, 0]
    risques, rendements, _ = calculer_frontiere(cov)
    ax.plot(risques, rendements, color="#1D9E75", lw=2.5, label="Front d'efficience flou")

    for w, label, marker, color in [
        (w_mv, "Min Variance", "o", "#378ADD"),
        (w_ms, "Max Sharpe",   "*", "#D85A30"),
    ]:
        ep = esperance_portefeuille(w)
        sp = np.sqrt(variance_portefeuille(w, cov))
        ax.scatter(sp, ep, s=120, marker=marker, color=color,
                   zorder=5, label=f"Portfolio {label}")

    ax.axhline(RF, color="gray", ls="--", lw=1, label=f"Taux sans risque {RF}%")
    ax.set_xlabel("Risque flou σ̃(P)  (%)")
    ax.set_ylabel("Espérance floue Ẽ(P)  (%)")
    ax.set_title("Frontière d'efficience floue")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 4b. Allocations ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    x  = np.arange(N)
    lar = 0.35
    b1 = ax.bar(x - lar/2, w_mv * 100, lar, label="Min Variance",
                color=COULEURS, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + lar/2, w_ms * 100, lar, label="Max Sharpe",
                color=COULEURS, alpha=0.45, edgecolor=COULEURS, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(NOMS, fontsize=9, rotation=10)
    ax.set_ylabel("Poids (%)")
    ax.set_title("Allocation optimale des deux portefeuilles")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── 4c. Fonctions d'appartenance ─────────────────────────────────────────
    ax = axes[1, 0]
    for (nom, actif), couleur in zip(ACTIFS.items(), COULEURS):
        a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
        xs = np.linspace(a - 0.5, b + 0.5, 300)
        ys = np.array([membership(x, actif) for x in xs])
        ax.plot(xs, ys, color=couleur, lw=2, label=nom)
        ax.fill_between(xs, ys, alpha=0.08, color=couleur)
        ax.axvline(m, color=couleur, ls=":", lw=1)

    ax.set_xlabel("Rendement (%)")
    ax.set_ylabel("Degré d'appartenance μ(x)")
    ax.set_title("Fonctions d'appartenance triangulaires")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.grid(True, alpha=0.3)

    # ── 4d. α-coupes ─────────────────────────────────────────────────────────
    ax = axes[1, 1]
    alphas_plot = np.linspace(0, 1, 50)
    for (nom, actif), couleur in zip(ACTIFS.items(), COULEURS):
        lo_arr = [alpha_coupe(actif, a)[0] for a in alphas_plot]
        hi_arr = [alpha_coupe(actif, a)[1] for a in alphas_plot]
        ax.fill_between(alphas_plot, lo_arr, hi_arr,
                        alpha=0.25, color=couleur)
        ax.plot(alphas_plot, lo_arr, color=couleur, lw=1.5)
        ax.plot(alphas_plot, hi_arr, color=couleur, lw=1.5, label=nom)

    ax.axvline(ALPHA, color="black", ls="--", lw=1.2,
               label=f"α = {ALPHA} (défaut)")
    ax.set_xlabel("Niveau α")
    ax.set_ylabel("Intervalle de rendement (%)")
    ax.set_title("Évolution des α-coupes par actif")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("portefeuille_flou_resultats.png", dpi=150, bbox_inches="tight")
    print("\n  Graphique sauvegardé → portefeuille_flou_resultats.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. RAPPORT TERMINAL
# ─────────────────────────────────────────────────────────────────────────────

def afficher_rapport(cov: np.ndarray, w_mv: np.ndarray, w_ms: np.ndarray):
    ligne = "═" * 65

    print("\n" + ligne)
    print("  RAPPORT — PORTEFEUILLE À REVENUS FLOUS")
    print(ligne)

    # Paramètres des actifs
    print(f"\n  {'Actif':<16} {'Ẽ (%)':>8} {'σ̃ (%)':>8} {'σ̃² ':>8} {'Asymétrie':>10}")
    print("  " + "─"*57)
    for nom, actif in ACTIFS.items():
        E = esperance_floue(actif)
        V = variance_floue(actif)
        S = ecart_type_flou(actif)
        asym = (actif["r_mod"] - actif["r_min"]) / (actif["r_max"] - actif["r_min"])
        print(f"  {nom:<16} {E:>8.3f} {S:>8.3f} {V:>8.3f} {asym:>10.3f}")

    print(f"\n  Matrice de covariance floue Σ̃ :")
    for row in cov:
        print("    " + "  ".join(f"{v:7.4f}" for v in row))

    # Portefeuille min variance
    ep_mv = esperance_portefeuille(w_mv)
    vp_mv = variance_portefeuille(w_mv, cov)
    sh_mv = ratio_sharpe_flou(w_mv, cov)
    pi_mv = indice_possibilite(w_mv, cov)

    print(f"\n  {'─'*57}")
    print(f"  PORTEFEUILLE MINIMUM VARIANCE")
    print(f"  {'─'*57}")
    for nom, wi in zip(NOMS, w_mv):
        bar = "█" * int(wi * 30)
        print(f"  {nom:<16} {wi*100:>6.2f}%  {bar}")
    print(f"\n  Espérance floue Ẽ(P)   = {ep_mv:.4f} %")
    print(f"  Variance floue σ̃²(P)  = {vp_mv:.4f}")
    print(f"  Écart-type flou σ̃(P)  = {np.sqrt(vp_mv):.4f} %")
    print(f"  Ratio de Sharpe flou  = {sh_mv:.4f}")
    print(f"  Indice de possibilité = {pi_mv:.4f}")

    # Portefeuille max Sharpe
    ep_ms = esperance_portefeuille(w_ms)
    vp_ms = variance_portefeuille(w_ms, cov)
    sh_ms = ratio_sharpe_flou(w_ms, cov)
    pi_ms = indice_possibilite(w_ms, cov)

    print(f"\n  {'─'*57}")
    print(f"  PORTEFEUILLE MAX SHARPE FLOU")
    print(f"  {'─'*57}")
    for nom, wi in zip(NOMS, w_ms):
        bar = "█" * int(wi * 30)
        print(f"  {nom:<16} {wi*100:>6.2f}%  {bar}")
    print(f"\n  Espérance floue Ẽ(P)   = {ep_ms:.4f} %")
    print(f"  Variance floue σ̃²(P)  = {vp_ms:.4f}")
    print(f"  Écart-type flou σ̃(P)  = {np.sqrt(vp_ms):.4f} %")
    print(f"  Ratio de Sharpe flou  = {sh_ms:.4f}")
    print(f"  Indice de possibilité = {pi_ms:.4f}")

    print("\n" + ligne + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n  Calcul de la matrice de covariance floue...")
    cov = matrice_covariance_floue()

    print("  Optimisation — Minimum Variance...")
    w_mv = optimiser_min_variance(cov)

    print("  Optimisation — Maximum Sharpe Flou...")
    w_ms = optimiser_max_sharpe(cov)

    # Analyse des α-coupes
    analyser_alpha_coupes(alphas=[0.0, 0.25, 0.5, 0.75, 1.0])

    # Rapport complet
    afficher_rapport(cov, w_mv, w_ms)

    # Graphiques
    tracer_resultats(cov, w_mv, w_ms)
