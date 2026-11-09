from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from .common import calendar, dump, save_parquet, digest
from .fundamentals import ingest, normalize_records, primary_registry


def prepare_stage(config):
    stage=Path(config['stage_dir']);stage.mkdir(parents=True,exist_ok=True)
    expected=config.get('expected_quarterly_sha256')
    if expected and digest(config['quarterly_source'])!=expected:
        raise ValueError('Different quarterly download version; source hash does not match the team snapshot')
    dates=calendar(config['calendar_start'],config['calendar_end'])
    raw,source=ingest(config['quarterly_source'],stage/'quarterly_selected.parquet')
    annotated,features=normalize_records(raw,dates,allow_final_date_fallback=config.get('allow_final_date_fallback',True))
    master,features,identity=primary_registry(raw,features,research_start=config['research_start'],research_end=config['research_end'])
    minimum=config.get('minimum_security_candidates',4000)
    if len(master)<minimum:
        raise ValueError(f'Only {len(master)} primary-issue candidates; minimum {minimum}. Never pad with funds.')
    save_parquet(master,stage/'security_candidates.parquet')
    save_parquet(features,stage/'financial_features.parquet')
    # Audit-only exception tables do not belong to the downstream stock database.
    save_parquet(annotated.loc[annotated.period_conflict],stage/'fiscal_period_conflicts.parquet')
    save_parquet(annotated.loc[~annotated.date_usable],stage/'unusable_financial_dates.parquet')
    dump({'source':source,'identity':identity,'calendar_sessions':len(dates),
          'eligible_candidates':len(master),'period_conflict_rows':int(annotated.period_conflict.sum()),
          'date_unusable_rows':int((~annotated.date_usable).sum()),
          'availability_basis':annotated.availability_basis.value_counts().to_dict(),
          'research_accepted':False},stage/'stage_report.json')
    print(json.dumps({'stage':'financial_ready','candidates':len(master),'features':len(features),'identity':identity}),flush=True)
    return dates,master,features,source


def main():
    p=argparse.ArgumentParser(description='Reproducible local database, explicit prototype/acceptance status')
    p.add_argument('command',choices=['stage','fetch','build','audit'])
    p.add_argument('--config',default='configs/database.local.json')
    a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
    if a.command=='stage': prepare_stage(cfg)
    elif a.command=='fetch':
        from .market import fetch_snapshot
        fetch_snapshot(cfg)
    elif a.command=='build':
        from .build import build_snapshot
        build_snapshot(cfg)
    else:
        from .build import audit_snapshot
        print(json.dumps(audit_snapshot(Path(cfg['output'])),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
