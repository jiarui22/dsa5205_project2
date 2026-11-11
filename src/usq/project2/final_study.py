"""Compact final study: rebuild six signals and replay representative comparisons.

The nominated strategy is retrospective. Forecast-model parameters and historical
predictions remain frozen; this module never tunes them on evaluation returns.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from usq.database.fundamentals import expand_events
from .factors import rank, residual, slog, surprise_events, price_inputs, wq
from .extensions import digest, progress
from .research_cache import atomic_json, bundle_complete, finish_bundle
from .research_portfolio import ResearchMarket
from .late_fusion import fuse_scores
from .supplement import paired_bootstrap

PHASES = {'validation': ('20220101', '20221231'),
          'evaluation': ('20230101', '20251231')}


def composite(parts, exclude=None):
    """Mean of available signed-rank components with missing signals neutral zero."""
    selected = [v for k, v in parts.items() if k != exclude]
    if not selected:
        raise ValueError('At least one factor required')
    return sum(selected) / len(selected)


class FinalStudy:
    phases = PHASES

    def __init__(self, config, output, standalone=False):
        self.cfg = json.loads(Path(config).read_text())
        self.standalone = standalone
        if standalone:
            self.cfg['execution_mode'] = 'independent research experiment'
        if standalone:
            self.paths = {'database': Path(self.cfg['database'])}
            self.db = self.paths['database']
            self.base = self.exp = self.deep = self.late = None
        else:
            self.paths = {k: Path(self.cfg[k]) for k in ['database', 'baseline', 'exploration', 'deep', 'late_fusion']}
            self.db, self.base, self.exp, self.deep, self.late = self.paths.values()
        self.out = Path(output)
        if any(self.out.resolve() == p.resolve() or p.resolve() in self.out.resolve().parents
               for p in self.paths.values()):
            raise ValueError('Use independent output directory')
        self.out.mkdir(parents=True, exist_ok=True)
        for sub in ['audit', 'tables', 'portfolios']:
            (self.out/sub).mkdir(exist_ok=True)
        for key in ['top_n', 'buffer_n', 'rebalance_every', 'cost_bps', 'max_target_weight', 'tie_seed']:
            assert self.cfg[key] == dict(top_n=100, buffer_n=150, rebalance_every=10,
                cost_bps=10, max_target_weight=.02, tie_seed=5205)[key]
        assert self.cfg.get('baseline_case', self.cfg.get('nominated_strategy')) == 'fundamental_sqrt_cap'
        p = self.out/'protocol.json'
        if p.exists() and json.loads(p.read_text()) != self.cfg:
            raise ValueError('Protocol changed; use new output')
        atomic_json(p, self.cfg)  # Record this experiment before computation
        self.selection = self.cfg['signals'] if standalone else json.loads((self.base/'factor_selection.json').read_text())
        self.market = ResearchMarket(self.db)
        self.u = pd.read_pickle(self.db/'panels/universe.pkl')
        self.dates = self.u.loc[self.phases['validation'][0]:self.phases['evaluation'][1]].index
        observed = np.zeros(self.u.shape, dtype=bool)
        inputs = {'config': digest(config), 'snapshot': digest(self.db/'checksums.sha256'),
                  'factor_selection': digest(config if standalone else self.base/'factor_selection.json')}
        prices = price_inputs(self.load, self.u) if standalone else None
        for name in self.selection['wq']:
            if standalone:
                f = wq(int(name[2:]), prices)
            else:
                p = self.base/'factors'/f'{name}.pkl'
                f = pd.read_pickle(p)
            assert f.index.equals(self.u.index) and f.columns.equals(self.u.columns)
            observed |= np.isfinite(f)
            if not standalone:
                inputs['WQ/'+name] = digest(p)
        del prices
        if not standalone:
            np.testing.assert_array_equal(observed, np.load(self.base/'design/observed.npy', mmap_mode='r'))
        self.rank_eligible = self.u.eq(1) & observed
        self.allowed = self.rank_eligible.reindex(self.dates)
        self.cap = pd.DataFrame(self.market.cap, index=self.u.index, columns=self.u.columns).reindex(self.dates)
        self.common = self.allowed & np.isfinite(self.cap) & self.cap.gt(0)
        self.rows = []
        self.inputs = inputs
        self.parts = {}
        self.rebuild_fundamentals()
        self.score = composite(self.parts).where(self.common)
        cp = pd.read_pickle(self.db/'panels/close_adj.pkl').astype(float)
        seg = pd.read_pickle(self.db/'panels/adjustment_segment.pkl')
        assets = (cp.ffill()/cp.ffill().shift()-1).where(
            seg.eq(seg.where(seg > 0).ffill().shift()) & seg.gt(0))
        terminal = pd.read_pickle(self.db/'panels/delisting_event.pkl').eq(1)
        vendor = pd.read_pickle(self.db/'panels/daily_return.pkl')
        self.assets = assets.where(~(terminal & np.isfinite(vendor) & vendor.ge(-1)), vendor).fillna(0)
        self.final_weights = None
        self.signature = None

    def load(self, name):
        x = pd.read_pickle(self.db/'panels'/f'{name}.pkl')
        assert x.index.equals(self.u.index) and x.columns.equals(self.u.columns)
        return x

    def rebuild_fundamentals(self):
        cap = self.load('market_cap')
        specs = [('earnings_ttm', 'market_cap'), ('book_equity', 'market_cap'),
                 ('revenue_q', 'market_cap'), ('earnings_q', 'book_equity')]
        checks = []
        for i, name in enumerate(self.selection['fundamental']):
            if i < 4:
                a, b = specs[i]
                f = residual(slog(self.load(a)), slog(cap if b == 'market_cap' else self.load(b)), self.u)
            elif i == 4:
                ev = cap + self.load('total_liabilities') - self.load('cash')
                f = (self.load('operating_cash_flow_ttm')-self.load('capital_expenditures_ttm'))/ev.where(ev > 0)
                industry = self.load('industry').apply(lambda s: pd.to_numeric(
                    s.astype(str).str.replace('SIC2:', '', regex=False), errors='coerce'))
                f = f.where(~(industry.ge(60) & industry.le(69)) & self.u.eq(1))
            else:
                q = pd.read_parquet(self.db/'metadata/fundamental_quarterly.parquet')
                events = surprise_events(q, cap)
                f = expand_events(events, self.u.index, self.u.columns).where(self.u.eq(1))
                del q, events
            f = f.replace([np.inf, -np.inf], np.nan).astype('float32')
            observer = getattr(self, 'factor_observer', None)
            if observer is not None:
                observer(name, f)
            if not self.standalone:
                reference = pd.read_pickle(self.base/'factors'/f'{name}.pkl')
                np.testing.assert_array_equal(f, reference)
                del reference
            checks.append({'factor': name, 'rebuilt_exactly': not self.standalone,
                           'direction': self.selection['directions'][name]})
            signed = f.reindex(self.dates) * self.selection['directions'][name]
            self.parts[name] = (rank(signed.where(self.allowed))-.5).fillna(0)
            if not self.standalone:
                self.inputs['fundamental/'+name] = digest(self.base/'factors'/f'{name}.pkl')
            progress('final_factor_rebuilt', factor=name)
            del f
            gc.collect()
        pd.DataFrame(checks).to_csv(self.out/'audit/rebuilt_factors.csv', index=False)

    def save(self, case, phase, score, *, cost=10, buffer=150, every=10,
             weighting='sqrt_cap', maximum=.02, all_eligible=False, retain=False):
        folder = self.out/'portfolios'/case/phase
        folder.mkdir(parents=True, exist_ok=True)
        # Score hashes guard interruption recovery; all inputs are also bound.
        import hashlib
        score_hash = hashlib.sha256(score.to_numpy().tobytes()).hexdigest()
        sig = {'case': case, 'phase': phase, 'cost': cost, 'buffer': buffer, 'every': every,
               'weighting': weighting, 'maximum': maximum, 'all_eligible': all_eligible,
               'retain': retain, 'score': score_hash, 'input_binding': self.inputs,
               'code': digest(Path(__file__))}
        required = ['daily.csv', 'summary.json', 'events.csv', 'monthly_holdings.parquet']
        if retain:
            required.append('weights.pkl')
        (folder/'complete.json').unlink(missing_ok=True)
        d, w, events, summary = self.market.backtest(score, *self.phases[phase], weights=True,
            cost_bps=cost, buffer_n=buffer, rebalance_every=every, weighting=weighting,
            max_weight=maximum, all_eligible=all_eligible)
        independent = (w.astype(float)*self.assets.reindex(w.index)).sum(axis=1)
        error = float((independent-d.gross_return).abs().max())
        assert error < 2e-7
        np.testing.assert_allclose(d.net_return, d.gross_return-d.cost, atol=1e-12)
        np.testing.assert_allclose(d.cost, d.traded_notional*cost/10000, atol=1e-12)
        summary.update(case=case, phase=phase, mean_cash=float(d.cash_weight.mean()),
            gross_cagr=float((1+d.gross_return).prod()**(252/len(d))-1),
            max_held_weight=float(w.to_numpy().max()), independent_gross_error=error)
        d.to_csv(folder/'daily.csv')
        events.to_csv(folder/'events.csv', index=False)
        m = w.groupby(w.index.str[:6]).tail(1)
        ii, jj = np.where(m.to_numpy() > 0)
        pd.DataFrame({'date': m.index.to_numpy()[ii], 'sid': m.columns.to_numpy()[jj],
                      'weight': m.to_numpy()[ii, jj]}).to_parquet(folder/'monthly_holdings.parquet', index=False)
        if retain:
            w.to_pickle(folder/'weights.pkl')
            self.final_weights = w
        atomic_json(folder/'summary.json', summary)
        finish_bundle(folder, 'complete.json', sig, required)
        self.rows = [r for r in self.rows if (r['case'], r['phase']) != (case, phase)] + [summary]
        pd.DataFrame(self.rows).to_csv(self.out/'portfolio_summary.csv', index=False)
        progress('final_portfolio', case=case, phase=phase, CAGR=summary['annual_return'])
        return d

    def predictions(self, family, extra=False):
        if extra:
            paths = [self.exp/'predictions'/f'extra_raw_{family}_{y}H{h}.pkl'
                     for y in [2022, 2023, 2024, 2025] for h in [1, 2]]
        else:
            paths = [self.base/'predictions'/f'ridge_{family}_2023_2025.pkl']
        for p in paths:
            self.inputs['prediction/'+p.name] = digest(p)
        return pd.concat([pd.read_pickle(p) for p in paths]).sort_index()

    def run(self):
        if self.standalone:
            return self.run_standalone()
        # Bind all external prediction inputs before any bundle can be reused.
        extra = {f: self.predictions(f, True) for f in ['A', 'B']}
        original = {f: self.predictions(f) for f in ['A', 'B']}
        for lag in [5, 20]:
            for name in self.parts:
                p = self.deep/'financial_delay'/f'lag{lag}'/f'{name}.pkl'
                self.inputs[f'lag{lag}/'+name] = digest(p)
        atomic_json(self.out/'input_binding.json', self.inputs)
        pool = pd.DataFrame(1., index=self.dates, columns=self.u.columns).where(self.common)
        for phase in PHASES:
            for weighting in ['equal', 'sqrt_cap', 'cap']:
                self.save('pool_'+weighting, phase, pool, buffer=None, weighting=weighting,
                          maximum=None, all_eligible=True)
                self.save('fundamental_'+weighting, phase, self.score, weighting=weighting,
                          retain=phase == 'evaluation' and weighting == 'sqrt_cap')
            for name in self.parts:
                self.save('without_'+name[:3], phase, composite(self.parts, name).where(self.common))
            for weight in [.25, .5]:
                self.save(f'fusion_{int(weight*100)}', phase,
                          fuse_scores(self.score, extra['A'].reindex(self.dates), weight))
        for family, score in original.items():
            self.save('ridge_'+family, 'evaluation', score, buffer=None, every=5,
                      weighting='equal', maximum=None)
            self.save('buffered_ridge_'+family, 'evaluation', score, weighting='equal', maximum=None)
        for family, score in extra.items():
            self.save('extra_'+family, 'evaluation', score, weighting='equal', maximum=None)
        for cost in [20, 30]:
            self.save('cost_'+str(cost), 'evaluation', self.score, cost=cost)
        for lag in [5, 20]:
            delayed = {}
            for name in self.parts:
                raw = pd.read_pickle(self.deep/'financial_delay'/f'lag{lag}'/f'{name}.pkl').reindex(self.dates)
                delayed[name] = (rank((raw*self.selection['directions'][name]).where(self.allowed))-.5).fillna(0)
            self.save('lag_'+str(lag), 'evaluation', composite(delayed).where(self.common))
            del delayed
        self.analyze()

    def run_standalone(self):
        """Rebuild the nominated score and minimal baselines from snapshot only."""
        # Check actual panel bytes, not only the checksum file's own hash.
        checked = 0
        for line in (self.db/'checksums.sha256').read_text().splitlines():
            expected, name = line.split('  ', 1)
            assert digest(self.db/name) == expected, name
            checked += 1
        atomic_json(self.out/'input_binding.json', self.inputs)
        pool = pd.DataFrame(1., index=self.dates, columns=self.u.columns).where(self.common)
        for phase in PHASES:
            for weighting in ['equal', 'sqrt_cap', 'cap']:
                self.save('pool_'+weighting, phase, pool, buffer=None, weighting=weighting,
                          maximum=None, all_eligible=True)
                self.save('fundamental_'+weighting, phase, self.score, weighting=weighting,
                          retain=phase == 'evaluation' and weighting == 'sqrt_cap')
        for cost in [20, 30]:
            self.save('cost_'+str(cost), 'evaluation', self.score, cost=cost)
        selection = {'nominated': self.cfg['nominated_strategy'],
            'status': self.cfg['nomination_status'], 'selection_uses_previously_observed_evaluation': True,
            'forecast_refits_in_this_run': 0, 'fundamental_factors_rebuilt': 6}
        atomic_json(self.out/'final_strategy.json', selection)
        atomic_json(self.out/'audit/verification.json', {'passed': True, 'portfolios': len(self.rows),
            'snapshot_files_verified': checked, 'archive_inputs_used': False,
            'maximum_independent_gross_error': max(r['independent_gross_error'] for r in self.rows)})
        progress('standalone_completed', portfolios=len(self.rows), archive_inputs_used=False)

    def analyze(self):
        def daily(case, phase='evaluation'):
            return pd.read_csv(self.out/'portfolios'/case/phase/'daily.csv', dtype={'date': str}).set_index('date')
        reference = daily('fundamental_sqrt_cap')
        pairs, annual = [], []
        names = ['fundamental_equal', 'fundamental_sqrt_cap', 'fundamental_cap',
                 'pool_equal', 'pool_sqrt_cap', 'pool_cap', 'extra_A', 'extra_B',
                 'buffered_ridge_A', 'buffered_ridge_B', 'fusion_25', 'fusion_50']
        for name in names:
            d = daily(name)
            for year, g in d.groupby(d.index.str[:4]):
                annual.append({'case': name, 'year': year,
                    'net_return': float((1+g.net_return).prod()-1),
                    'SPY_return': float((1+g.benchmark_return).prod()-1)})
        for name in ['SPY', 'pool_cap', 'fundamental_equal', 'fusion_25', 'fusion_50']:
            a = reference.benchmark_return if name == 'SPY' else daily(name).net_return
            for block in [10, 20, 40]:
                pairs.append({'a': name, 'b': 'fundamental_sqrt_cap', 'block': block,
                              **paired_bootstrap(a, reference.net_return, block=block)})
        pd.DataFrame(annual).to_csv(self.out/'tables/annual.csv', index=False)
        pd.DataFrame(pairs).to_csv(self.out/'tables/paired_intervals.csv', index=False)
        previous = pd.read_csv(self.exp/'portfolios/weights/simple_C_sqrt_cap/evaluation/daily.csv')
        np.testing.assert_allclose(reference.net_return, previous.net_return, atol=1e-12)
        previous = pd.read_csv(self.late/'portfolios/benchmark/pool_cap_cost10/evaluation/daily.csv')
        np.testing.assert_allclose(daily('pool_cap').net_return, previous.net_return, atol=1e-12)
        # Dollar P&L, not a compounded return assigned independently to each asset.
        w = self.final_weights.astype(float)
        pnl = w*self.assets.reindex(w.index)
        pnl = pnl.mul(reference.net_nav.shift(fill_value=1), axis=0)
        net = float(pnl.to_numpy().sum()-(reference.cost*reference.net_nav.shift(fill_value=1)).sum())
        np.testing.assert_allclose(net, reference.net_nav.iloc[-1]-1, atol=2e-7)
        pd.DataFrame({'sid': pnl.columns, 'dollar_PnL': pnl.sum().to_numpy()}).sort_values(
            'dollar_PnL', ascending=False).to_csv(self.out/'tables/stock_PnL.csv', index=False)
        effective = 1/(w.pow(2).sum(axis=1)/w.sum(axis=1).pow(2))
        atomic_json(self.out/'tables/concentration.json', {
            'mean_effective_positions': float(effective.mean()),
            'stock_PnL': float(pnl.to_numpy().sum()),
            'fee_PnL': float((reference.cost*reference.net_nav.shift(fill_value=1)).sum()),
            'net_cumulative_return': net})
        source_tables = {
            'factor_statistics.csv': self.base.parent/'project2_nextclose10_v2_extensions/statistics/factor_summary.csv',
            'factor_screen.csv': self.base/'evaluation/factor_screen.csv',
            'original_models.csv': self.base/'backtests/summary.csv',
            'trading_validation.csv': self.exp/'portfolio_summary.csv',
            'style_exposures.csv': self.exp/'analysis/mean_style_exposures.csv',
            'training_windows.csv': self.base/'training_windows.csv',
            'factor_selection.json': self.base/'factor_selection.json',
            'correlation.csv': self.base.parent/'project2_nextclose10_v2_extensions/correlation/stability_spearman.csv',
            'reconstruction_vs_ratio.csv': self.base/'supplement/reconstruction_vs_ratio.csv'}
        for name, p in source_tables.items():
            shutil.copy2(p, self.out/'tables'/name)
        self.u.loc['20150101':'20251231'].eq(1).sum(axis=1).rename('count').to_csv(
            self.out/'tables/universe_counts.csv', index_label='date')
        selection = {'nominated': self.cfg['nominated_strategy'],
            'status': self.cfg['nomination_status'], 'evaluation_replay_matches_previous': True,
            'forecast_refits_in_this_run': 0, 'fundamental_factors_rebuilt': 6,
            'selection_uses_previously_observed_evaluation': True}
        atomic_json(self.out/'final_strategy.json', selection)
        for r in json.loads((self.base/'run_manifest.json').read_text())['files']:
            assert digest(self.base/r['path']) == r['sha256']
        atomic_json(self.out/'audit/verification.json', {'passed': True,
            'portfolios': len(self.rows), 'maximum_independent_gross_error': max(
                r['independent_gross_error'] for r in self.rows), 'final_strategy': selection,
            'frozen_baseline_files_unchanged': 535})
        progress('final_study_completed', portfolios=len(self.rows), nominated=selection['nominated'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/project2_final.json')
    p.add_argument('--output', default='runs/project2_final_v1')
    p.add_argument('--standalone', action='store_true', help='Snapshot-only nominated strategy; use an independent output')
    a = p.parse_args()
    FinalStudy(a.config, a.output, standalone=a.standalone).run()


if __name__ == '__main__':
    main()
