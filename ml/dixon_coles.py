"""
Dixon-Coles bivariate Poisson model para predicción de fútbol.

Cada equipo tiene fuerza de ataque α y defensa β. Para un partido E1 (local)
vs E2 (visitante):
    λ = α_E1 * β_E2 * γ        # goles esperados de E1
    μ = α_E2 * β_E1            # goles esperados de E2
    γ = home advantage (>= 1)

Los goles se modelan como Poisson casi-independientes, con corrección τ
para 0-0, 0-1, 1-0, 1-1 (Dixon-Coles 1997, eq. 9). Eso captura la
correlación negativa de scores bajos que un Poisson puro no modela.

MLE con SLSQP, restricción mean(α) = 1 para identificabilidad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


class DixonColesModel:
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.params_ = None
        self.teams_ = None
        self.fit_result_ = None

    @staticmethod
    def _unpack(params: np.ndarray, n: int):
        attacks = params[:n]
        defenses = params[n:2 * n]
        home_adv = params[2 * n]
        rho = params[2 * n + 1]
        return attacks, defenses, home_adv, rho

    def _neg_ll(self, params, e1_idx, e2_idx, g1, g2, is_home_e1, weights, n):
        attacks, defenses, home_adv, rho = self._unpack(params, n)

        a1 = attacks[e1_idx]
        d1 = defenses[e1_idx]
        a2 = attacks[e2_idx]
        d2 = defenses[e2_idx]

        lam = a1 * d2 * np.where(is_home_e1, home_adv, 1.0)
        mu = a2 * d1 * np.where(is_home_e1, 1.0, home_adv)

        if np.any(lam <= 0) or np.any(mu <= 0):
            return 1e10

        log_p1 = g1 * np.log(lam) - lam - gammaln(g1 + 1)
        log_p2 = g2 * np.log(mu) - mu - gammaln(g2 + 1)

        tau = np.ones_like(lam)
        m00 = (g1 == 0) & (g2 == 0)
        m01 = (g1 == 0) & (g2 == 1)
        m10 = (g1 == 1) & (g2 == 0)
        m11 = (g1 == 1) & (g2 == 1)
        tau = np.where(m00, 1 - lam * mu * rho, tau)
        tau = np.where(m01, 1 + lam * rho, tau)
        tau = np.where(m10, 1 + mu * rho, tau)
        tau = np.where(m11, 1 - rho, tau)
        if np.any(tau <= 0):
            return 1e10
        log_tau = np.log(tau)

        ll = np.sum(weights * (log_tau + log_p1 + log_p2))
        return -ll

    def fit(self, df: pd.DataFrame, xi: float = 0.0, ref_date=None) -> "DixonColesModel":
        """
        df columnas requeridas: Equipo1, Equipo2, EQUIPO1_GOLES, EQUIPO2_GOLES.
        Opcionales: Es_Local_E1 (default 1), Fecha (para decay temporal).
        xi: decay exponencial sobre días (0 = sin decay). 0.0019 ≈ half-life 1 año.
        """
        df = df.dropna(subset=['EQUIPO1_GOLES', 'EQUIPO2_GOLES']).copy()
        df['EQUIPO1_GOLES'] = df['EQUIPO1_GOLES'].astype(int)
        df['EQUIPO2_GOLES'] = df['EQUIPO2_GOLES'].astype(int)

        teams = sorted(set(df['Equipo1'].astype(str)).union(df['Equipo2'].astype(str)))
        self.teams_ = teams
        team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        e1_idx = df['Equipo1'].astype(str).map(team_idx).values
        e2_idx = df['Equipo2'].astype(str).map(team_idx).values
        g1 = df['EQUIPO1_GOLES'].values
        g2 = df['EQUIPO2_GOLES'].values

        if 'Es_Local_E1' in df.columns:
            is_home_e1 = df['Es_Local_E1'].fillna(1).astype(bool).values
        else:
            is_home_e1 = np.ones(len(df), dtype=bool)

        if xi > 0 and 'Fecha' in df.columns:
            dates = pd.to_datetime(df['Fecha'], errors='coerce')
            ref = pd.to_datetime(ref_date) if ref_date is not None else dates.max()
            days_ago = (ref - dates).dt.days.fillna(0).clip(lower=0).values
            weights = np.exp(-xi * days_ago)
        else:
            weights = np.ones(len(df))

        x0 = np.concatenate([np.ones(n), np.ones(n), [1.3], [-0.1]])
        bounds = [(0.05, 5.0)] * (2 * n) + [(1.0, 2.5), (-0.3, 0.3)]
        cons = {'type': 'eq', 'fun': lambda p: np.mean(p[:n]) - 1.0}

        result = minimize(
            self._neg_ll, x0,
            args=(e1_idx, e2_idx, g1, g2, is_home_e1, weights, n),
            method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 400, 'ftol': 1e-7},
        )

        attacks, defenses, home_adv, rho = self._unpack(result.x, n)
        self.params_ = {
            'attack': dict(zip(teams, attacks)),
            'defense': dict(zip(teams, defenses)),
            'home_adv': float(home_adv),
            'rho': float(rho),
        }
        self.fit_result_ = result
        return self

    def predict_proba(self, equipo1: str, equipo2: str, is_home_e1: bool = True) -> np.ndarray:
        """Devuelve [P(Win=E1), P(Draw), P(Loss=E1)] — orden W/D/L."""
        p = self.params_
        if equipo1 not in p['attack'] or equipo2 not in p['attack']:
            return np.array([1 / 3, 1 / 3, 1 / 3])

        a1, d1 = p['attack'][equipo1], p['defense'][equipo1]
        a2, d2 = p['attack'][equipo2], p['defense'][equipo2]
        ha = p['home_adv']

        if is_home_e1:
            lam = a1 * d2 * ha
            mu = a2 * d1
        else:
            lam = a1 * d2
            mu = a2 * d1 * ha

        mg = self.max_goals
        x = np.arange(mg + 1)
        log_p_lam = x * np.log(lam) - lam - gammaln(x + 1)
        log_p_mu = x * np.log(mu) - mu - gammaln(x + 1)
        mat = np.outer(np.exp(log_p_lam), np.exp(log_p_mu))

        rho = p['rho']
        mat[0, 0] *= (1 - lam * mu * rho)
        mat[0, 1] *= (1 + lam * rho)
        mat[1, 0] *= (1 + mu * rho)
        mat[1, 1] *= (1 - rho)
        mat = mat / mat.sum()

        i_grid, j_grid = np.meshgrid(np.arange(mg + 1), np.arange(mg + 1), indexing='ij')
        p_win = float(mat[i_grid > j_grid].sum())
        p_draw = float(np.trace(mat))
        p_loss = float(mat[i_grid < j_grid].sum())
        return np.array([p_win, p_draw, p_loss])

    def predict(self, equipo1: str, equipo2: str, is_home_e1: bool = True) -> str:
        probs = self.predict_proba(equipo1, equipo2, is_home_e1)
        return ['Win', 'Draw', 'Loss'][int(np.argmax(probs))]

    def expected_score(self, equipo1: str, equipo2: str, is_home_e1: bool = True) -> tuple[float, float]:
        p = self.params_
        if equipo1 not in p['attack'] or equipo2 not in p['attack']:
            return float('nan'), float('nan')
        a1, d1 = p['attack'][equipo1], p['defense'][equipo1]
        a2, d2 = p['attack'][equipo2], p['defense'][equipo2]
        ha = p['home_adv']
        if is_home_e1:
            return a1 * d2 * ha, a2 * d1
        return a1 * d2, a2 * d1 * ha
