"""
=============================================================================
  GESTION DE PORTEFEUILLE À REVENUS FLOUS
  Modèle mathématique basé sur les nombres flous triangulaires
  Optimisation par scipy.optimize (Markowitz flou)
  Version interactive avec menu terminal
=============================================================================

Auteur  : Kevin xxx — Master Intelligence Artificielle
Langage : Python 3.10+
Dépend. : numpy, scipy, matplotlib
=============================================================================
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES TERMINAL
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def titre(texte):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {texte}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")

def sous_titre(texte):
    print(f"\n{BOLD}{YELLOW}  ▸ {texte}{RESET}")
    print(f"{DIM}  {'─'*55}{RESET}")

def ok(texte):    print(f"{GREEN}  ✔ {texte}{RESET}")
def info(texte):  print(f"{CYAN}  ℹ {texte}{RESET}")
def warn(texte):  print(f"{YELLOW}  ⚠ {texte}{RESET}")
def erreur(texte):print(f"{RED}  ✘ {texte}{RESET}")

def saisir_float(prompt, defaut=None, min_val=None, max_val=None):
    """Lecture sécurisée d'un float avec valeur par défaut et bornes."""
    while True:
        hint = f" [{defaut}]" if defaut is not None else ""
        try:
            raw = input(f"    {prompt}{hint} : ").strip()
            if raw == "" and defaut is not None:
                return float(defaut)
            val = float(raw)
            if min_val is not None and val < min_val:
                warn(f"Valeur minimum : {min_val}")
                continue
            if max_val is not None and val > max_val:
                warn(f"Valeur maximum : {max_val}")
                continue
            return val
        except ValueError:
            erreur("Veuillez entrer un nombre valide.")

def saisir_int(prompt, defaut=None, min_val=1, max_val=999):
    while True:
        hint = f" [{defaut}]" if defaut is not None else ""
        try:
            raw = input(f"    {prompt}{hint} : ").strip()
            if raw == "" and defaut is not None:
                return int(defaut)
            val = int(raw)
            if val < min_val or val > max_val:
                warn(f"Entrez une valeur entre {min_val} et {max_val}.")
                continue
            return val
        except ValueError:
            erreur("Veuillez entrer un entier valide.")

def pause():
    input(f"\n{DIM}  Appuyez sur Entrée pour continuer...{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAT GLOBAL DE L'APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

ACTIFS = {
    "Action A":    {"r_min": 2.0,  "r_mod": 8.0,  "r_max": 14.0},
    "Action B":    {"r_min": 1.0,  "r_mod": 5.0,  "r_max": 11.0},
    "Obligation C":{"r_min": 3.0,  "r_mod": 6.0,  "r_max":  9.0},
    "Immobilier D":{"r_min": 4.0,  "r_mod": 9.0,  "r_max": 16.0},
}

CORRELATION = np.array([
    [1.00,  0.35, -0.10,  0.20],
    [0.35,  1.00,  0.05,  0.15],
    [-0.10, 0.05,  1.00, -0.05],
    [0.20,  0.15, -0.05,  1.00],
])

RF    = 2.0
ALPHA = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS MATHÉMATIQUES FLOUES
# ─────────────────────────────────────────────────────────────────────────────

def membership(x, actif):
    a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
    if x <= a or x >= b: return 0.0
    return (x - a)/(m - a) if x <= m else (b - x)/(b - m)

def alpha_coupe(actif, alpha):
    a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
    return (a + alpha*(m - a), b - alpha*(b - m))

def esperance_floue(actif):
    a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
    return (a + 2*m + b) / 4

def variance_floue(actif):
    a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
    return ((b - a)**2 + (m - a)*(b - m)) / 12

def ecart_type_flou(actif):
    return np.sqrt(variance_floue(actif))

def matrice_covariance_floue():
    noms  = list(ACTIFS.keys())
    sigma = np.array([ecart_type_flou(ACTIFS[n]) for n in noms])
    return np.outer(sigma, sigma) * CORRELATION

def variance_portefeuille(w, cov):
    return float(w @ cov @ w)

def esperance_portefeuille(w):
    E = np.array([esperance_floue(ACTIFS[n]) for n in ACTIFS])
    return float(w @ E)

def ratio_sharpe_flou(w, cov, rf=None):
    rf = rf if rf is not None else RF
    ep = esperance_portefeuille(w)
    vp = variance_portefeuille(w, cov)
    return (ep - rf) / np.sqrt(vp) if vp > 0 else 0.0

def indice_possibilite(w, cov, seuil=None):
    seuil = seuil if seuil is not None else RF
    ep = esperance_portefeuille(w)
    sp = np.sqrt(variance_portefeuille(w, cov))
    return float(norm.cdf((ep - seuil) / sp)) if sp > 0 else 0.0

def optimiser_min_variance(cov, E_cible=None):
    N      = len(ACTIFS)
    E_vect = np.array([esperance_floue(ACTIFS[n]) for n in ACTIFS])
    contraintes = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if E_cible is not None:
        contraintes.append({"type": "ineq", "fun": lambda w: E_vect @ w - E_cible})
    res = minimize(lambda w: variance_portefeuille(w, cov),
                   np.ones(N)/N, method="SLSQP",
                   bounds=[(0, 1)]*N, constraints=contraintes,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x

def optimiser_max_sharpe(cov):
    N = len(ACTIFS)
    res = minimize(lambda w: -ratio_sharpe_flou(w, cov),
                   np.ones(N)/N, method="SLSQP",
                   bounds=[(0, 1)]*N,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x

def calculer_frontiere(cov, nb_points=60):
    E_vect = np.array([esperance_floue(ACTIFS[n]) for n in ACTIFS])
    cibles = np.linspace(E_vect.min()+0.01, E_vect.max()-0.01, nb_points)
    risques, rendements, poids_liste = [], [], []
    for E_c in cibles:
        w  = optimiser_min_variance(cov, E_cible=E_c)
        ep = esperance_portefeuille(w)
        vp = variance_portefeuille(w, cov)
        if vp >= 0:
            risques.append(np.sqrt(vp))
            rendements.append(ep)
            poids_liste.append(w)
    return np.array(risques), np.array(rendements), poids_liste


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — AFFICHER / MODIFIER LES ACTIFS
# ─────────────────────────────────────────────────────────────────────────────

def menu_actifs():
    while True:
        titre("GESTION DES ACTIFS")
        noms = list(ACTIFS.keys())
        print(f"  {BOLD}{'#':<4} {'Actif':<16} {'r_min':>8} {'r_mod':>8} {'r_max':>8} {'Ẽ':>8} {'σ̃':>8}{RESET}")
        print(f"  {'─'*58}")
        for i, (nom, a) in enumerate(ACTIFS.items(), 1):
            E = esperance_floue(a)
            S = ecart_type_flou(a)
            print(f"  {i:<4} {nom:<16} {a['r_min']:>8.2f} {a['r_mod']:>8.2f} "
                  f"{a['r_max']:>8.2f} {E:>8.3f} {S:>8.3f}")

        print(f"\n  {BOLD}Options :{RESET}")
        print("    [1-4] Modifier un actif    [a] Ajouter un actif")
        print("    [s]   Supprimer un actif   [r] Réinitialiser")
        print("    [q]   Retour au menu principal")

        choix = input(f"\n  {BOLD}Choix :{RESET} ").strip().lower()

        if choix == "q":
            break

        elif choix in [str(i) for i in range(1, len(ACTIFS)+1)]:
            idx  = int(choix) - 1
            nom  = noms[idx]
            actif = ACTIFS[nom]
            sous_titre(f"Modifier : {nom}")
            info("Laissez vide pour conserver la valeur actuelle.")
            r_min = saisir_float("r_min (rendement pessimiste %)", defaut=actif["r_min"])
            r_mod = saisir_float("r_mod (rendement modal %)",      defaut=actif["r_mod"], min_val=r_min)
            r_max = saisir_float("r_max (rendement optimiste %)",  defaut=actif["r_max"], min_val=r_mod)
            ACTIFS[nom] = {"r_min": r_min, "r_mod": r_mod, "r_max": r_max}
            ok(f"Actif « {nom} » mis à jour.")
            pause()

        elif choix == "a":
            global CORRELATION
            sous_titre("Ajouter un nouvel actif")
            nom = input("    Nom de l'actif : ").strip()
            if not nom:
                warn("Nom invalide.")
                continue
            if nom in ACTIFS:
                warn("Cet actif existe déjà.")
                continue
            r_min = saisir_float("r_min (rendement pessimiste %)", min_val=-50, max_val=100)
            r_mod = saisir_float("r_mod (rendement modal %)",      min_val=r_min, max_val=100)
            r_max = saisir_float("r_max (rendement optimiste %)",  min_val=r_mod, max_val=100)
            ACTIFS[nom] = {"r_min": r_min, "r_mod": r_mod, "r_max": r_max}
            n = len(ACTIFS)
            new_corr = np.ones((n, n)) * 0.1
            new_corr[:n-1, :n-1] = CORRELATION
            np.fill_diagonal(new_corr, 1.0)
            CORRELATION = new_corr
            ok(f"Actif « {nom} » ajouté (corrélation neutre = 0.1).")
            pause()

        elif choix == "s":
            global CORRELATION
            if len(ACTIFS) <= 2:
                warn("Il faut au moins 2 actifs.")
                continue
            idx = saisir_int("Numéro de l'actif à supprimer", min_val=1, max_val=len(ACTIFS))
            nom = noms[idx - 1]
            confirm = input(f"    Supprimer « {nom} » ? (o/n) : ").strip().lower()
            if confirm == "o":
                del ACTIFS[nom]
                n = len(ACTIFS)
                new_corr = np.delete(np.delete(CORRELATION, idx-1, 0), idx-1, 1)
                CORRELATION = new_corr
                ok(f"« {nom} » supprimé.")
            pause()

        elif choix == "r":
            global ACTIFS, CORRELATION
            ACTIFS = {
                "Action A":    {"r_min": 2.0,  "r_mod": 8.0,  "r_max": 14.0},
                "Action B":    {"r_min": 1.0,  "r_mod": 5.0,  "r_max": 11.0},
                "Obligation C":{"r_min": 3.0,  "r_mod": 6.0,  "r_max":  9.0},
                "Immobilier D":{"r_min": 4.0,  "r_mod": 9.0,  "r_max": 16.0},
            }
            CORRELATION = np.array([
                [1.00,  0.35, -0.10,  0.20],
                [0.35,  1.00,  0.05,  0.15],
                [-0.10, 0.05,  1.00, -0.05],
                [0.20,  0.15, -0.05,  1.00],
            ])
            ok("Actifs réinitialisés aux valeurs par défaut.")
            pause()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — SIMULATEUR α-COUPE
# ─────────────────────────────────────────────────────────────────────────────

def menu_alpha():
    titre("SIMULATEUR α-COUPE")
    info("L'α-coupe filtre l'intervalle de confiance flou.")
    info("α = 0 → incertitude totale  |  α = 1 → valeur modale seule\n")

    alpha = saisir_float("Choisissez un niveau α", defaut=0.5, min_val=0.0, max_val=1.0)

    sous_titre(f"Intervalles α-coupés pour α = {alpha:.2f}")
    print(f"\n  {BOLD}{'Actif':<16} {'[r_L^α':>10}  {'r_R^α]':>10}  {'Largeur':>8}  {'Milieu':>8}{RESET}")
    print(f"  {'─'*56}")
    for nom, actif in ACTIFS.items():
        lo, hi = alpha_coupe(actif, alpha)
        print(f"  {nom:<16} [{lo:>9.3f}, {hi:>9.3f}]  {hi-lo:>8.3f}  {(lo+hi)/2:>8.3f}")

    print()
    voir = input("  Tracer l'évolution des α-coupes ? (o/n) [o] : ").strip().lower()
    if voir != "n":
        COULEURS = ["#1D9E75", "#378ADD", "#7F77DD", "#D85A30"]
        alphas_plot = np.linspace(0, 1, 80)
        fig, ax = plt.subplots(figsize=(9, 5))
        for (nom, actif), col in zip(ACTIFS.items(), COULEURS):
            lo_arr = [alpha_coupe(actif, a)[0] for a in alphas_plot]
            hi_arr = [alpha_coupe(actif, a)[1] for a in alphas_plot]
            ax.fill_between(alphas_plot, lo_arr, hi_arr, alpha=0.18, color=col)
            ax.plot(alphas_plot, lo_arr, color=col, lw=1.8)
            ax.plot(alphas_plot, hi_arr, color=col, lw=1.8, label=nom)
        ax.axvline(alpha, color="black", ls="--", lw=1.5, label=f"α = {alpha:.2f}")
        ax.set_xlabel("Niveau α"); ax.set_ylabel("Rendement (%)")
        ax.set_title("Évolution des α-coupes par actif")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    pause()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — OPTIMISATION INTERACTIVE
# ─────────────────────────────────────────────────────────────────────────────

def menu_optimisation():
    global RF
    titre("OPTIMISATION DU PORTEFEUILLE")

    print(f"  {BOLD}Stratégie d'optimisation :{RESET}")
    print("    [1] Minimum variance floue")
    print("    [2] Maximum ratio de Sharpe flou")
    print("    [3] Cible de rendement personnalisée")
    print("    [4] Comparer les trois stratégies")

    choix = input(f"\n  {BOLD}Choix :{RESET} ").strip()
    if choix not in ["1","2","3","4"]:
        warn("Choix invalide."); pause(); return

    rf_new = saisir_float("Taux sans risque rf (%)", defaut=RF, min_val=0.0, max_val=20.0)
    RF = rf_new

    cov = matrice_covariance_floue()
    NOMS = list(ACTIFS.keys())
    COULEURS = ["#1D9E75", "#378ADD", "#7F77DD", "#D85A30"]

    def afficher_portefeuille(label, w):
        ep = esperance_portefeuille(w)
        vp = variance_portefeuille(w, cov)
        sh = ratio_sharpe_flou(w, cov, rf=RF)
        pi = indice_possibilite(w, cov, seuil=RF)
        sous_titre(label)
        print(f"  {BOLD}{'Actif':<16} {'Poids':>8}  Barre{RESET}")
        for nom, wi in zip(NOMS, w):
            barre = "█" * int(wi * 32)
            print(f"  {nom:<16} {wi*100:>7.2f}%  {GREEN}{barre}{RESET}")
        print(f"\n  {CYAN}Ẽ(P)             = {ep:.4f} %{RESET}")
        print(f"  {CYAN}σ̃²(P)            = {vp:.4f}{RESET}")
        print(f"  {CYAN}σ̃(P)             = {np.sqrt(vp):.4f} %{RESET}")
        print(f"  {CYAN}Ratio de Sharpe  = {sh:.4f}{RESET}")
        print(f"  {CYAN}Indice possib.   = {pi:.4f}{RESET}")

    if choix == "1":
        w = optimiser_min_variance(cov)
        afficher_portefeuille("Portefeuille Minimum Variance", w)

    elif choix == "2":
        w = optimiser_max_sharpe(cov)
        afficher_portefeuille("Portefeuille Maximum Sharpe Flou", w)

    elif choix == "3":
        E_vect = np.array([esperance_floue(ACTIFS[n]) for n in ACTIFS])
        info(f"Plage de rendements atteignables : [{E_vect.min():.2f}%, {E_vect.max():.2f}%]")
        E_c = saisir_float("Rendement cible (%)", defaut=round(E_vect.mean(),1),
                           min_val=E_vect.min(), max_val=E_vect.max())
        w = optimiser_min_variance(cov, E_cible=E_c)
        afficher_portefeuille(f"Portefeuille — Cible Ẽ ≥ {E_c:.2f}%", w)

    elif choix == "4":
        E_vect = np.array([esperance_floue(ACTIFS[n]) for n in ACTIFS])
        w_mv = optimiser_min_variance(cov)
        w_ms = optimiser_max_sharpe(cov)
        w_eq = np.ones(len(ACTIFS)) / len(ACTIFS)
        afficher_portefeuille("① Minimum Variance", w_mv)
        afficher_portefeuille("② Maximum Sharpe Flou", w_ms)
        afficher_portefeuille("③ Équipondéré (référence)", w_eq)

        # Graphique comparatif
        voir = input("\n  Afficher le graphique comparatif ? (o/n) [o] : ").strip().lower()
        if voir != "n":
            risques, rendements, _ = calculer_frontiere(cov)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            ax = axes[0]
            ax.plot(risques, rendements, color="#1D9E75", lw=2.5, label="Front d'efficience")
            for w, label, marker, col in [
                (w_mv, "Min Variance", "o", "#378ADD"),
                (w_ms, "Max Sharpe",   "*", "#D85A30"),
                (w_eq, "Équipondéré",  "s", "#7F77DD"),
            ]:
                ax.scatter(np.sqrt(variance_portefeuille(w, cov)),
                           esperance_portefeuille(w),
                           s=120, marker=marker, color=col, zorder=5, label=label)
            ax.axhline(RF, color="gray", ls="--", lw=1, label=f"rf = {RF}%")
            ax.set_xlabel("σ̃(P) (%)"); ax.set_ylabel("Ẽ(P) (%)")
            ax.set_title("Frontière d'efficience floue"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            ax2 = axes[1]
            x = np.arange(len(NOMS)); lar = 0.25
            for j, (w, label, col) in enumerate([
                (w_mv, "Min Variance", "#378ADD"),
                (w_ms, "Max Sharpe",   "#D85A30"),
                (w_eq, "Équipondéré",  "#7F77DD"),
            ]):
                ax2.bar(x + j*lar, w*100, lar, label=label, color=col, alpha=0.85)
            ax2.set_xticks(x + lar); ax2.set_xticklabels(NOMS, fontsize=9, rotation=10)
            ax2.set_ylabel("Poids (%)"); ax2.set_title("Allocations comparées")
            ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
            plt.tight_layout(); plt.show()

    pause()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — SIMULATEUR DE PORTEFEUILLE PERSONNALISÉ
# ─────────────────────────────────────────────────────────────────────────────

def menu_simulateur():
    titre("SIMULATEUR DE PORTEFEUILLE PERSONNALISÉ")
    NOMS = list(ACTIFS.keys())
    N    = len(NOMS)
    info("Entrez vos propres poids et observez les métriques en temps réel.")

    while True:
        sous_titre("Saisie des poids")
        poids = []
        for nom in NOMS:
            w = saisir_float(f"w({nom}) %", defaut=round(100/N, 1), min_val=0.0, max_val=100.0)
            poids.append(w / 100)

        total = sum(poids)
        if abs(total - 1.0) > 1e-3:
            warn(f"Somme des poids = {total*100:.1f}% ≠ 100%. Normalisation automatique.")
            poids = [p / total for p in poids]

        w   = np.array(poids)
        cov = matrice_covariance_floue()
        ep  = esperance_portefeuille(w)
        vp  = variance_portefeuille(w, cov)
        sh  = ratio_sharpe_flou(w, cov, rf=RF)
        pi  = indice_possibilite(w, cov, seuil=RF)

        sous_titre("Résultats")
        print(f"  {BOLD}{'Actif':<16} {'Poids':>8}  Barre{RESET}")
        for nom, wi in zip(NOMS, w):
            barre = "█" * int(wi * 32)
            print(f"  {nom:<16} {wi*100:>7.2f}%  {CYAN}{barre}{RESET}")
        print(f"\n  {GREEN}Ẽ(P)             = {ep:.4f} %{RESET}")
        print(f"  {GREEN}σ̃²(P)            = {vp:.4f}{RESET}")
        print(f"  {GREEN}σ̃(P)             = {np.sqrt(vp):.4f} %{RESET}")
        print(f"  {GREEN}Ratio de Sharpe  = {sh:.4f}{RESET}")
        print(f"  {GREEN}Indice possib.   = {pi:.4f}{RESET}")

        # Comparaison avec optimal
        w_opt = optimiser_max_sharpe(cov)
        sh_opt = ratio_sharpe_flou(w_opt, cov, rf=RF)
        gap = sh_opt - sh
        if gap > 0.01:
            warn(f"Le portefeuille optimal (Max Sharpe) aurait un ratio de {sh_opt:.4f} (+{gap:.4f}).")
        else:
            ok("Votre allocation est proche de l'optimum !")

        again = input("\n  Essayer une autre allocation ? (o/n) [o] : ").strip().lower()
        if again == "n":
            break

    pause()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — GRAPHIQUES COMPLETS
# ─────────────────────────────────────────────────────────────────────────────

def menu_graphiques():
    titre("VISUALISATION COMPLÈTE")
    print("  [1] Front d'efficience flou")
    print("  [2] Fonctions d'appartenance")
    print("  [3] α-coupes par actif")
    print("  [4] Allocation optimale (camembert)")
    print("  [5] Tableau de bord complet (4 graphiques)")

    choix = input(f"\n  {BOLD}Choix :{RESET} ").strip()
    cov   = matrice_covariance_floue()
    w_mv  = optimiser_min_variance(cov)
    w_ms  = optimiser_max_sharpe(cov)
    NOMS  = list(ACTIFS.keys())
    COULEURS = ["#1D9E75", "#378ADD", "#7F77DD", "#D85A30"]

    if choix == "1":
        risques, rendements, _ = calculer_frontiere(cov)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(risques, rendements, color="#1D9E75", lw=2.5, label="Front d'efficience flou")
        for w, label, marker, col in [
            (w_mv, "Min Variance", "o", "#378ADD"),
            (w_ms, "Max Sharpe",   "*", "#D85A30"),
        ]:
            ax.scatter(np.sqrt(variance_portefeuille(w, cov)), esperance_portefeuille(w),
                       s=140, marker=marker, color=col, zorder=5, label=label)
        ax.axhline(RF, color="gray", ls="--", lw=1, label=f"rf = {RF}%")
        ax.set_xlabel("Risque flou σ̃(P) (%)"); ax.set_ylabel("Espérance floue Ẽ(P) (%)")
        ax.set_title("Frontière d'efficience floue"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    elif choix == "2":
        fig, ax = plt.subplots(figsize=(9, 5))
        for (nom, actif), col in zip(ACTIFS.items(), COULEURS):
            a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
            xs = np.linspace(a - 1, b + 1, 300)
            ys = [membership(x, actif) for x in xs]
            ax.plot(xs, ys, color=col, lw=2.2, label=nom)
            ax.fill_between(xs, ys, alpha=0.1, color=col)
            ax.axvline(m, color=col, ls=":", lw=1)
        ax.set_xlabel("Rendement (%)"); ax.set_ylabel("μ(x)")
        ax.set_title("Fonctions d'appartenance triangulaires")
        ax.set_ylim(0, 1.15); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    elif choix == "3":
        alpha_input = saisir_float("Niveau α à afficher", defaut=ALPHA, min_val=0.0, max_val=1.0)
        fig, ax = plt.subplots(figsize=(9, 5))
        alphas_plot = np.linspace(0, 1, 80)
        for (nom, actif), col in zip(ACTIFS.items(), COULEURS):
            lo_arr = [alpha_coupe(actif, a)[0] for a in alphas_plot]
            hi_arr = [alpha_coupe(actif, a)[1] for a in alphas_plot]
            ax.fill_between(alphas_plot, lo_arr, hi_arr, alpha=0.18, color=col)
            ax.plot(alphas_plot, lo_arr, color=col, lw=1.8)
            ax.plot(alphas_plot, hi_arr, color=col, lw=1.8, label=nom)
        ax.axvline(alpha_input, color="black", ls="--", lw=1.5, label=f"α = {alpha_input:.2f}")
        ax.set_xlabel("Niveau α"); ax.set_ylabel("Rendement (%)")
        ax.set_title("Évolution des α-coupes"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    elif choix == "4":
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, w, label in [(axes[0], w_mv, "Min Variance"), (axes[1], w_ms, "Max Sharpe")]:
            wedges, texts, autotexts = ax.pie(
                w, labels=NOMS, colors=COULEURS,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
            for at in autotexts: at.set_fontsize(9)
            ax.set_title(f"Portefeuille {label}")
        plt.suptitle("Allocations optimales", fontweight="bold")
        plt.tight_layout(); plt.show()

    elif choix == "5":
        risques, rendements, _ = calculer_frontiere(cov)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Gestion de Portefeuille à Revenus Flous", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(risques, rendements, color="#1D9E75", lw=2.5, label="Front d'efficience")
        for w, label, marker, col in [(w_mv,"Min Variance","o","#378ADD"),(w_ms,"Max Sharpe","*","#D85A30")]:
            ax.scatter(np.sqrt(variance_portefeuille(w,cov)), esperance_portefeuille(w),
                       s=120, marker=marker, color=col, zorder=5, label=label)
        ax.axhline(RF, color="gray", ls="--", lw=1)
        ax.set_xlabel("σ̃(P) (%)"); ax.set_ylabel("Ẽ(P) (%)")
        ax.set_title("Frontière d'efficience floue"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        x = np.arange(len(NOMS)); lar = 0.35
        ax.bar(x-lar/2, w_mv*100, lar, color=COULEURS, alpha=0.85, label="Min Variance")
        ax.bar(x+lar/2, w_ms*100, lar, color=COULEURS, alpha=0.45,
               edgecolor=COULEURS, linewidth=1.5, label="Max Sharpe")
        ax.set_xticks(x); ax.set_xticklabels(NOMS, fontsize=9, rotation=10)
        ax.set_ylabel("Poids (%)"); ax.set_title("Allocations comparées")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

        ax = axes[1, 0]
        for (nom, actif), col in zip(ACTIFS.items(), COULEURS):
            a, m, b = actif["r_min"], actif["r_mod"], actif["r_max"]
            xs = np.linspace(a-1, b+1, 300)
            ys = [membership(x, actif) for x in xs]
            ax.plot(xs, ys, color=col, lw=2, label=nom)
            ax.fill_between(xs, ys, alpha=0.08, color=col)
        ax.set_xlabel("Rendement (%)"); ax.set_ylabel("μ(x)")
        ax.set_title("Fonctions d'appartenance"); ax.legend(fontsize=8)
        ax.set_ylim(0, 1.15); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        alphas_plot = np.linspace(0, 1, 80)
        for (nom, actif), col in zip(ACTIFS.items(), COULEURS):
            lo_arr = [alpha_coupe(actif, a)[0] for a in alphas_plot]
            hi_arr = [alpha_coupe(actif, a)[1] for a in alphas_plot]
            ax.fill_between(alphas_plot, lo_arr, hi_arr, alpha=0.18, color=col)
            ax.plot(alphas_plot, lo_arr, color=col, lw=1.5)
            ax.plot(alphas_plot, hi_arr, color=col, lw=1.5, label=nom)
        ax.axvline(ALPHA, color="black", ls="--", lw=1.2)
        ax.set_xlabel("Niveau α"); ax.set_ylabel("Rendement (%)")
        ax.set_title("α-coupes par actif"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = "portefeuille_flou_resultats.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        ok(f"Graphique sauvegardé → {path}")
        plt.show()

    else:
        warn("Choix invalide.")

    pause()


# ─────────────────────────────────────────────────────────────────────────────
# MENU PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def menu_principal():
    while True:
        print(f"\n{BOLD}{CYAN}{'╔' + '═'*58 + '╗'}{RESET}")
        print(f"{BOLD}{CYAN}║{'PORTEFEUILLE À REVENUS FLOUS':^58}║{RESET}")
        print(f"{BOLD}{CYAN}║{'Kevin xxx — Master Intelligence Artificielle':^58}║{RESET}")
        print(f"{BOLD}{CYAN}{'╚' + '═'*58 + '╝'}{RESET}")

        print(f"""
  {BOLD}[1]{RESET}  Gérer les actifs            {DIM}(ajouter, modifier, supprimer){RESET}
  {BOLD}[2]{RESET}  Simulateur α-coupe          {DIM}(intervalles de confiance flous){RESET}
  {BOLD}[3]{RESET}  Optimisation du portefeuille {DIM}(min variance / max Sharpe / cible){RESET}
  {BOLD}[4]{RESET}  Simulateur personnalisé      {DIM}(tester vos propres poids){RESET}
  {BOLD}[5]{RESET}  Visualisation               {DIM}(front efficience, graphiques){RESET}
  {BOLD}[q]{RESET}  Quitter
""")
        choix = input(f"  {BOLD}Votre choix :{RESET} ").strip().lower()

        if   choix == "1": menu_actifs()
        elif choix == "2": menu_alpha()
        elif choix == "3": menu_optimisation()
        elif choix == "4": menu_simulateur()
        elif choix == "5": menu_graphiques()
        elif choix == "q":
            print(f"\n  {GREEN}Au revoir !{RESET}\n")
            sys.exit(0)
        else:
            warn("Choix invalide, entrez un chiffre entre 1 et 5.")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        menu_principal()
    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Programme interrompu.{RESET}\n")
        sys.exit(0)