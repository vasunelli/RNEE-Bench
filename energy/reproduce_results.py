"""Reproduce the reported results from released predictions and resampling data.

Run: python reproduce_results.py
Only this directory is used for scientific inputs. No fitted model is loaded.
No prediction or bootstrap sample is generated. Saved resample multiplicities
are evaluated deterministically. Output is stdout; package files are unchanged.
"""
import os, sys, json, hashlib, math, re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parent
RESULTS = []
WARNINGS = []

def safe(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise RuntimeError('External scientific input forbidden: ' + str(path))
    return p

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def check(name, condition, **details):
    row = {'name': name, 'passed': bool(condition), **details}
    RESULTS.append(row)
    if not condition:
        raise AssertionError(name + ' ' + str(details))

def warning(text):
    WARNINGS.append(text)
    print('WARNING ' + text, flush=True)

def read_json(path):
    return json.loads(safe(path).read_text(encoding='utf8'))

def csv(path, **kw):
    return pd.read_csv(safe(path), float_precision='round_trip', **kw)

def pq(path):
    return pd.read_parquet(safe(path))

def close(name, a, b, raw=False):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    tol = 1e-08 if raw else 1e-09
    ok = aa.shape == bb.shape and np.allclose(aa, bb, atol=tol, rtol=1e-12, equal_nan=True)
    diff = np.abs(aa - bb) if aa.shape == bb.shape else np.array([np.inf])
    finite = diff[np.isfinite(diff)]
    check(name, ok, max_absolute_error=float(finite.max()) if finite.size else 0.0)

def verify_residual_cancellation(S, T, C0, CR, label):
    """Dedicated block/prefix/complete/cohort check of T = S + CR - C0."""
    close('Residual identity ' + label, T, S + CR - C0, raw=True)
    return float(np.max(np.abs(np.asarray(T) - np.asarray(S) - np.asarray(CR) + np.asarray(C0))))

def means(frame, cols, target='target_l', factor=1000):
    return np.abs(frame[cols].to_numpy() - frame[target].to_numpy()[:, None]).mean(axis=0) * factor

def weighted_errors(d, cols, target, weights, keys, factor):
    ids = pd.Categorical(d.VehId.astype(str), categories=[str(x) for x in keys]).codes
    check('Bootstrap cluster keys cover observations', np.all(ids >= 0))
    e = np.abs(d[cols].to_numpy() - d[target].to_numpy()[:, None]) * factor
    counts = np.bincount(ids, minlength=len(keys))
    sums = np.stack([np.bincount(ids, weights=e[:, i], minlength=len(keys)) for i in range(len(cols))], axis=1)
    check('Saved multiplicities are 2000 complete vehicle draws', len(weights) == 2000 and np.issubdtype(weights.dtype, np.integer) and np.all(weights >= 0) and np.all(weights.sum(axis=1) == len(keys)))
    return weights @ sums / (weights @ counts)[:, None]
CORE = ['cold_trip', 'unseen_vehicle', 'cold_month']
SIX = ['1', '2', '4', '8', 'full_prefix', 'complete']
STATES = ['B', 'BR', 'BK', 'BKR', 'BZ', 'BZR', 'BKZ', 'BKZR']

def fuel_stats(a):
    gains = {s: a[..., STATES.index(s)] - a[..., STATES.index(s + 'R')] for s in ['B', 'BK', 'BZ', 'BKZ']}
    return {**{'G_' + s: v for s, v in gains.items()}, 'A_K': gains['B'] - gains['BK'], 'A_Z_given_K': gains['BK'] - gains['BKZ'], 'A_K_given_Z': gains['BZ'] - gains['BKZ']}

def primary():
    t = csv('tables/TABLE3_SOURCE.csv')
    oo = pq(source('/UNSEEN_VEHICLE_OOF_PREDICTIONS.parquet'))
    for x in t.itertuples():
        rpm = x.telemetry_spec == 'with_RPM'
        if x.setting == 'unseen_vehicle':
            d = oo
            cols = ['p_rpm_base', 'p_rpm_road'] if rpm else ['p_lower_base', 'p_lower_road']
        else:
            token = ('rpm_only_test_predictions/' if rpm else 'ensemble_test_predictions/') + x.setting + '__ICE_fuel_L.parquet'
            d = pq(source(token))
            cols = ['p__hgb__rpm__baseline', 'p__hgb__rpm__real'] if rpm else ['p__hgb__low__baseline', 'p__hgb__low__real']
        ma = means(d, cols)
        gain = 100 * (ma[0] - ma[1]) / ma[0]
        key = 'Table 3 ' + x.setting + ' ' + x.telemetry_spec
        check(key + ' counts', len(d) == x.n_segments and d.segment_id.nunique() == x.n_segments and (d.trip_uid.nunique() == x.n_trips) and (d.VehId.nunique() == x.n_vehicles))
        close(key + ' MAEs and gain', [*ma, gain], [x.telemetry_only_mae_mL, x.road_augmented_mae_mL, x.gain_percent])
        membership = source('/target_memberships/' + ('cold_vehicle' if x.setting == 'unseen_vehicle' else x.setting) + '/ICE_fuel_L/model_membership.parquet')
        mem = pq(membership)
        ids = set(mem.segment_id) if x.setting == 'unseen_vehicle' else set(mem.loc[mem.split_role.eq('test'), 'segment_id'])
        check(key + ' exact evaluation membership', set(d.segment_id) == ids)
        if x.setting == 'unseen_vehicle':
            draws = pq(source('/UNSEEN_VEHICLE_BOOTSTRAP_DRAWS.parquet'))
            tel = 'lower_plus_rpm' if rpm else 'lower_information'
            v = draws[draws.telemetry.eq(tel)].relative_mae_reduction_percent.to_numpy()
            close(key + ' saved percentile interval', np.quantile(v, [0.025, 0.975]), [x.lower_percent, x.upper_percent])
        else:
            effect = csv(source('rpm_only_effects.csv' if rpm else 'compact_effects.csv'))
            mask = effect.environment.eq(x.setting) & effect.target_id.eq('ICE_fuel_L') & effect.effect_type.eq('gain_real')
            mask = mask & (effect.family.eq('model_real_gain') & effect.model.eq('hgb') & effect.sensor.eq('lower_plus_rpm') if rpm else effect.sensor.eq('low') & effect.family.eq('road_gain'))
            rr = effect[mask].iloc[0]
            close(key + ' authoritative simultaneous interval', [100 * rr.simultaneous_lcb, 100 * rr.simultaneous_ucb], [x.lower_percent, x.upper_percent])
    compare_display('tables/table3_values.csv')
    warning('Earlier primary, paired-RPM, target-decile and speed interval outputs are retained and hash-checked. Their resample matrices are not all stored; seeds are provenance, and no missing matrices or samples are regenerated.')

def compare_display(path):
    cache = {}
    for i, row in csv(path, dtype={'manuscript_display': str, 'expected_display': str}).iterrows():
        s = row.manuscript_display.replace('−', '-').replace(',', '')
        digits = len(s.split('.')[1]) if '.' in s else 0
        expected = Decimal(str(row.source_full_precision)).quantize(Decimal('1').scaleb(-digits), rounding=ROUND_HALF_UP)
        check(path + ' cell ' + str(i), Decimal(s) == expected)
        if row.source_path not in cache:
            cache[row.source_path] = csv(row.source_path, dtype={'scale': str, 'H': str})
        frame = cache[row.source_path]
        parts = dict((x.split('=', 1) for x in row.source_key.split(';') if '=' in x))
        for key in ['setting', 'telemetry_spec', 'training_loss', 'scale', 'state', 'H']:
            if key in parts:
                frame = frame[frame[key].astype(str).eq(parts[key])]
        check(path + ' source row ' + str(i), len(frame) == 1)
        field = parts.get('source_column', parts.get('column'))
        value = float(frame.iloc[0][field])
        conversion = parts.get('conversion', 'identity')
        if conversion.startswith('multiply by 1000'):
            value *= 1000
        elif conversion.startswith('multiply by 100'):
            value *= 100
        close(path + ' source precision ' + str(i), value, row.source_full_precision, raw=True)
        if path == 'tables/table4_values.csv':
            canon = csv('tables/TABLE4_SOURCE.csv')
            rr = canon[canon.quantity.eq(row.quantity) & canon.setting.eq(row.setting)].iloc[0]
            close('Table 4 canonical source ' + str(i), rr[row.field], value, raw=True)

def selection():
    base = 'data/selection/'
    config = read_json(base + 'CONFIG.json')
    oof = pq(base + 'OOF_PREDICTIONS.parquet')
    historical = pq(source('/UNSEEN_VEHICLE_OOF_PREDICTIONS.parquet'))
    cols = ['segment_id', 'trip_uid', 'VehId', 'target_l', 'outer_fold']
    check('Independent selection preserves primary cohort/targets/folds', oof[cols].sort_values('segment_id').reset_index(drop=True).equals(historical[cols].sort_values('segment_id').reset_index(drop=True)))
    scores = csv(base + 'SELECTION_SCORES.csv')
    choices = read_json(base + 'SELECTED_ESTIMATORS.json')
    seed = read_json(base + 'SEED_POLICY.json')
    fits = read_json(base + 'MODEL_SPECIFICATIONS.json')
    hc = config['historical']
    check('Candidate scores: exact 150 cells', len(scores) == 150 and (not scores.duplicated(['outer_fold', 'split_family', 'target_id', 'model']).any()))
    close('Selection normalized MAE', scores.normalized_mae, scores.mae_L / scores.train_iqr)
    for fold in range(5):
        ev = pq(base + f'roles/fold{fold}_evaluation.parquet')
        outer = set(ev.VehId.astype(str))
        check('Outer fold ' + str(fold) + ' exact OOF identity', set(ev.segment_id) == set(oof.loc[oof.outer_fold.eq(fold), 'segment_id']))
        for role in ['train', 'validation', 'rpm_check']:
            d = pq(base + f'roles/fold{fold}_{role}.parquet')
            check(f'Outer fold {fold}: {role} excludes evaluation vehicles', not outer & set(d.VehId.astype(str)))
        for cell in [x for x in config['cells'] if x['outer_fold'] == fold]:
            name = cell['name']
            full = pq(base + 'roles/' + name + '_all_safe_development.parquet')
            d = pq(base + 'selection_data/' + name + '.parquet')
            check('Selection exclusion before sampling ' + name, not outer & set(full.VehId.astype(str)) and (not outer & set(d.VehId.astype(str))))
            for role, cap in [('train', hc['max_train_rows']), ('validation', hc['max_validation_rows'])]:
                z = full[full.split_role.eq(role)].copy()
                if len(z) > cap:
                    z['sortkey'] = [hashlib.sha256(f"{sid}|{hc['sampling_seed']}|{role}".encode()).hexdigest() for sid in z.segment_id]
                    expected = z.sort_values('sortkey', kind='stable').head(cap).segment_id.tolist()
                else:
                    expected = z.sort_values('segment_id').segment_id.tolist()
                check('Frozen deterministic sample ' + name + ' ' + role, expected == d.loc[d.split_role.eq(role), 'segment_id'].tolist())
            tr = d[d.split_role.eq('train')]
            xx = tr[cell['features']].replace([np.inf, -np.inf], np.nan)
            active = [col for col in xx if xx[col].dropna().min() < xx[col].dropna().max()]
            check('Training-only feature filter ' + name, active == cell['active_features'])
            close('Training-only IQR ' + name, np.quantile(tr[cell['target_column']], [0.75, 0.25]) @ np.array([1, -1]), cell['train_iqr'])
        ranking = scores[scores.outer_fold.eq(fold)].groupby('model').normalized_mae.mean().sort_values()
        choice = choices[fold]['selected_estimator']
        check('Selected estimator fold ' + str(fold), ranking.index[0] == choice)
        expected_seeds = [0] if seed[fold]['max_prediction_difference_L'] == 0 else [0, 1, 2]
        check('Fold-local seed policy ' + str(fold), seed[fold]['seeds'] == expected_seeds and all((x['seeds'] == expected_seeds and x['estimator'] == choice for x in fits if x['outer_fold'] == fold)))
        check('Saved fold predictions ' + str(fold), pq(base + f'OOF_FOLD{fold}.parquet').equals(oof[oof.outer_fold.eq(fold)].reset_index(drop=True)))
    check('Four HGB and one LightGBM', [x['selected_estimator'] for x in choices] == ['hist_gradient_boosting_l1'] * 4 + ['lightgbm_l1'])
    pooled = csv(base + 'POOLED_RESULTS.csv')
    folds = csv(base + 'FOLD_RESULTS.csv')
    draws = pq('data/bootstrap/estimator_selection/BOOTSTRAP_DRAWS.parquet')
    weights = np.load(safe('data/bootstrap/estimator_selection/BOOTSTRAP_MULTIPLICITIES.npz'), allow_pickle=False)
    for row in pooled.itertuples():
        tel = row.telemetry
        cols = ['p_lower_base', 'p_lower_road'] if tel == 'lower_information' else ['p_rpm_base', 'p_rpm_road']
        ma = means(oof, cols)
        close('Independent-selection pooled ' + tel, [*ma, 100 * (ma[0] - ma[1]) / ma[0]], [row.mae_telemetry_mL, row.mae_road_mL, row.relative_mae_reduction_percent])
        w = weights[tel + '_weights']
        keys = weights[tel + '_vehicles']
        boot = weighted_errors(oof, cols, 'target_l', w, keys, 1000)
        gain = 100 * (boot[:, 0] - boot[:, 1]) / boot[:, 0]
        close('Independent-selection saved replicates ' + tel, gain, draws[draws.telemetry.eq(tel)].sort_values('draw').relative_mae_reduction_percent)
        close('Independent-selection interval ' + tel, np.quantile(gain, [0.025, 0.975]), [row.relative_mae_reduction_lcb, row.relative_mae_reduction_ucb])
        for fold in range(5):
            q = oof[oof.outer_fold.eq(fold)]
            mm = means(q, cols)
            rr = folds[folds.outer_fold.eq(fold) & folds.telemetry.eq(tel)].iloc[0]
            close(f'Independent-selection fold gain {fold} {tel}', 100 * (mm[0] - mm[1]) / mm[0], rr.relative_mae_reduction_percent)
    check('Five positive without-RPM sensitivity folds', (folds[folds.telemetry.eq('lower_information')].relative_mae_reduction_percent > 0).all())

def overlap():
    base = 'data/overlap/'
    inf = csv(base + 'BOOTSTRAP_INFERENCE.csv')
    cfg = read_json(base + 'analysis.json')
    k = pq(base + 'kinematics/KINEMATIC_FEATURES.parquet')
    check('Exact five K variables and full cohort', k.columns.tolist() == ['segment_id'] + cfg['K'] and len(k) == 112787 and (k.segment_id.nunique() == 112787))
    for task, name, cols, factor in [('fuel', 'FUEL_PREDICTIONS.parquet', STATES, 1000), ('kplus', 'KPLUS_PREDICTIONS.parquet', ['B', 'BR'], 1), ('rpm', 'CONDITIONAL_RPM_PREDICTIONS.parquet', ['BK', 'BKR'], 1)]:
        full = pq(base + name)
        for setting, d in full.groupby('setting', sort=False):
            z = np.load(safe('data/bootstrap/predictive_overlap/' + task + '__' + setting + '.npz'), allow_pickle=False)
            boot = weighted_errors(d, cols, 'target', z['vehicle_multiplicities'], z['vehicle_keys'], factor)
            close('Overlap saved MAEs ' + task + ' ' + setting, boot, z['mae'])
            theta = means(d, cols, 'target', factor)
            obs = fuel_stats(theta) if task == 'fuel' else {'G_Kplus' if task == 'kplus' else 'G_RPM_given_K': theta[0] - theta[1]}
            reps = fuel_stats(boot) if task == 'fuel' else {next(iter(obs)): boot[:, 0] - boot[:, 1]}
            critical = None
            sd = {}
            if task == 'fuel' and setting in CORE:
                sd = {key: np.std(reps[key], ddof=1) for key in cfg['core_fuel_family']}
                critical = np.quantile(np.max([np.abs((reps[key] - obs[key]) / sd[key]) for key in sd], axis=0), 0.95)
            for key, value in obs.items():
                row = inf[inf.setting.eq(setting) & inf.statistic.eq(key)].iloc[0]
                if task == 'fuel' and setting in CORE and (key in cfg['core_fuel_family']):
                    lo, hi = (value - critical * sd[key], value + critical * sd[key])
                else:
                    lo, hi = np.quantile(reps[key], cfg['intermediate_core_quantiles'] if task != 'fuel' and setting in CORE else [0.025, 0.975])
                close('Predictive overlap ' + setting + ' ' + key, [value, lo, hi], [row.estimate, row.lower, row.upper])
            if task == 'fuel':
                close('Sequential overlap identity ' + setting, obs['G_B'], obs['A_K'] + obs['A_Z_given_K'] + obs['G_BKZ'])
                baseline = pq(base + 'FROZEN_BASELINE_PREDICTIONS.parquet')
                bb = baseline[baseline.setting.eq(setting)].set_index('segment_id').loc[d.segment_id]
                close('Frozen B/Z prediction binding ' + setting, d[['B', 'BR', 'BZ', 'BZR']], bb[['B', 'BR', 'BZ', 'BZR']])
            if task == 'kplus':
                kk = k.set_index('segment_id').loc[d.segment_id, 'K_plus']
                close('K+ target binding ' + setting, d.target, kk)
    f5 = csv('figure_data/figure05.csv')
    for i, row in f5.iterrows():
        if row.display_role == 'sequential_share':
            rr = next((x for x in read_json(row.source_path) if x['setting'] == row.setting))
            close('Figure 5 share ' + str(i), row.estimate, rr[row.statistic])
            continue
        rr = inf[inf.setting.eq(row.setting) & inf.statistic.eq(row.statistic)].iloc[0]
        close('Figure 5 endpoint/interval ' + str(i), [row.estimate, row.lower, row.upper], [rr.estimate, rr.lower, rr.upper])
    shares = read_json('figure_data/figure05/RETENTION_SHARES.json')
    for s in shares:
        rr = inf[inf.setting.eq(s['setting'])].set_index('statistic').estimate
        close('Figure 5 sequential shares ' + s['setting'], [s['kinematic_share'], s['rpm_share'], s['remaining_share'], s['sum']], [rr.A_K / rr.G_B, rr.A_Z_given_K / rr.G_B, rr.G_BKZ / rr.G_B, 1])

def accounting():
    base = 'data/accounting/'
    defs = pq(base + 'SIX_SCALE_BLOCK_DEFINITIONS.parquet')
    trips = pq(base + 'TRIP_MEMBERSHIPS.parquet')
    co = csv(base + 'COMMON8_COHORT.csv', dtype={'vehicle_id': str})
    sc = csv(base + 'SIX_SCALE_RESULTS.csv', dtype={'scale': str})
    comp = csv(base + 'ATTENUATION_COMPONENTS.csv')
    draws = pq(base + 'bootstrap/SIX_SCALE_DRAWS.parquet')
    prefix = pq(base + 'PREFIX_MEMBERSHIP.parquet')
    tail = pq(base + 'TAIL_MEMBERSHIP.parquet')
    complete = pq(base + 'COMPLETE_MEMBERSHIP.parquet')
    groups = {key: q.sort_values('block_index') for key, q in defs.groupby(['setting', 'physical_trip_id', 'H'])}
    for row in trips.itertuples():
        pp = list(row.ordered_prefix_ids)
        tt = list(row.ordered_tail_ids)
        cc = list(row.ordered_complete_ids)
        check('Complete = ordered prefix + tail ' + row.setting + ' ' + row.trip_uid, pp + tt == cc and (not set(pp) & set(tt)) and (len(cc) == len(set(cc))))
        for h in ['1', '2', '4', '8', 'FULL_COMMON_PREFIX']:
            found = [sid for a in groups[row.setting, row.trip_uid, h].ordered_segment_ids for sid in a]
            check('Identical ordered observations ' + row.setting + ' ' + row.trip_uid + ' ' + h, found == pp)
    check('Frozen common-eight cohort', set(zip(trips.setting, trips.trip_uid)) == set(zip(co.setting, co.physical_trip_id)))
    maximum_identity = {}
    for loss, file in [('absolute_error', 'ABSOLUTE_ERROR_PREDICTIONS.parquet'), ('squared_error', 'SQUARED_ERROR_PREDICTIONS.parquet')]:
        predictions = pq('data/diagnostics/loss/' + file)
        m = pq(base + loss + '/SIX_SCALE_BLOCK_METRICS.parquet')
        computed = []
        cached = {s: ({sid: i for i, sid in enumerate(q.segment_id)}, q[['target_l', 'B', 'BR']].to_numpy()) for s, q in predictions.groupby('setting')}
        for b in defs.itertuples():
            index, data = cached[b.setting]
            q = data[[index[sid] for sid in b.ordered_segment_ids]]
            y = q[:, 0]
            r0 = 1000 * (q[:, 1] - y)
            rr = 1000 * (q[:, 2] - y)
            a0 = math.fsum(abs(r0))
            ar = math.fsum(abs(rr))
            e0 = abs(math.fsum(r0))
            er = abs(math.fsum(rr))
            computed.append([b.block_id, a0, ar, e0, er, a0 - ar, e0 - er, a0 - e0, ar - er, 1000 * math.fsum(y)])
        calc = pd.DataFrame(computed, columns=['block_id', 'A0', 'AR', 'E0', 'ER', 'S', 'T', 'C0', 'CR', 'target_mL']).set_index('block_id')
        ref = m.set_index('block_id').loc[calc.index]
        close('Saved predictions reproduce all block metrics ' + loss, calc.to_numpy(), ref[calc.columns].to_numpy(), raw=True)
        for scale, q in m.groupby('scale'):
            maximum_identity[loss + ' ' + scale] = verify_residual_cancellation(q.S, q['T'], q.C0, q.CR, 'every block ' + loss + ' ' + scale)
        for setting in CORE:
            selected = sc[sc.setting.eq(setting) & sc.training_loss.eq(loss)].set_index('scale')
            component = comp[comp.setting.eq(setting) & comp.training_loss.eq(loss)].iloc[0]
            wdf = pq(base + 'bootstrap/VEHICLE_MULTIPLICITIES_' + setting + '.parquet')
            w = wdf.to_numpy()
            keys = list(wdf.columns)
            check('Same registered weights across losses ' + setting + ' ' + loss, np.array_equal(w, pq('data/diagnostics/loss/' + loss + '/BOOTSTRAP_WEIGHTS_' + setting + '.parquet').to_numpy()))
            check('Accounting bootstrap multiplicity validity ' + setting, len(w) == 2000 and np.issubdtype(w.dtype, np.integer) and (w >= 0).all() and (w.sum(axis=1) == len(keys)).all())
            boots = []
            obs = []
            for h in SIX:
                q = m[m.setting.eq(setting) & m.scale.eq(h)]
                row = selected.loc[h]
                target = q.target_mL.sum()
                raw = q.E0.sum() - q.ER.sum()
                D = 100 * raw / target
                check('Accounting support ' + setting + ' ' + loss + ' ' + h, len(q) == row.n_blocks and q.physical_trip_id.nunique() == row.n_trips and (q.vehicle_id.nunique() == row.n_vehicles))
                close('Accounting endpoint ' + setting + ' ' + loss + ' ' + h, [D, target, raw], [row.D, row.target_volume, row.raw_gain_mL], raw=True)
                verify_residual_cancellation(q.S.sum(), q['T'].sum(), q.C0.sum(), q.CR.sum(), 'cohort ' + setting + ' ' + loss + ' ' + h)
                g = q.assign(vehicle_id=q.vehicle_id.astype(str)).groupby('vehicle_id')[['E0', 'ER', 'target_mL']].sum().reindex(keys).to_numpy()
                a = w @ g
                bs = 100 * (a[:, 0] - a[:, 1]) / a[:, 2]
                theta = 100 * (g[:, 0].sum() - g[:, 1].sum()) / g[:, 2].sum()
                boots.append(bs)
                obs.append(theta)
                saved = draws[draws.setting.eq(setting) & draws.training_loss.eq(loss) & draws.scale.eq(h)].sort_values('replicate').D
                close('Six-scale saved draws ' + setting + ' ' + loss + ' ' + h, bs, saved)
            boot = np.stack(boots, axis=1)
            obs = np.asarray(obs)
            sd = np.std(boot, axis=0, ddof=1)
            crit = np.quantile(np.max(abs((boot - obs) / sd), axis=1), 0.95)
            for i, h in enumerate(SIX):
                row = selected.loc[h]
                close('Six-endpoint simultaneous interval ' + setting + ' ' + loss + ' ' + h, [obs[i] - crit * sd[i], obs[i] + crit * sd[i], crit], [row.CI_lower, row.CI_upper, row.critical])
            for label, i, j in [('Laggregation', 0, 4), ('Ltail', 4, 5), ('Ltrip', 0, 5)]:
                vec = boot[:, i] - boot[:, j]
                saved = pq(base + f'bootstrap/COMPONENT_DRAWS_{setting}_{loss}.parquet')[label]
                close('Saved component draws ' + setting + ' ' + loss + ' ' + label, vec, saved)
                lo, hi = np.quantile(vec, [0.025, 0.975])
                close('Attenuation component ' + setting + ' ' + loss + ' ' + label, [obs[i] - obs[j], lo, hi], [component[label], component[label + '_CI_lower'], component[label + '_CI_upper']])
            close('Ltrip = Laggregation + Ltail ' + setting + ' ' + loss, component.Ltrip, component.Laggregation + component.Ltail)
    print('MAXIMUM_RESIDUAL_IDENTITY_ERRORS ' + json.dumps(maximum_identity), flush=True)
    RESULTS.append({'name': 'Maximum residual identity errors', 'passed': True, 'values': maximum_identity})
    old = csv(base + 'original/SCALE_RESULTS.csv', dtype={'H': str})
    blocks = pq(base + 'original/BLOCK_METRICS.parquet')
    for row in old.itertuples():
        q = blocks[blocks.state.eq(row.state) & blocks.setting.eq(row.setting) & blocks.H.eq(row.H)]
        check('Original scale exact block count ' + row.state + ' ' + row.setting + ' ' + row.H, len(q) == row.n_blocks)
        pos = q.S > 0
        n = int(pos.sum())
        persist = int(((q['T'] > 0) & pos).sum())
        pi = persist / n if n else np.nan
        check('Persistence counts ' + row.state + ' ' + row.setting + ' ' + row.H, n == row.n_S_positive and persist == row.n_persist)
        close('Persistence fraction ' + row.state + ' ' + row.setting + ' ' + row.H, pi, row.Pi_H)
    compare_display('tables/table4_values.csv')
    figures(sc, old)

def figures(sc, old):
    f3 = csv('figure_data/figure03.csv')
    t3 = csv('tables/TABLE3_SOURCE.csv')
    for i, row in f3.iterrows():
        if row.panel == 'a':
            tel = 'without_RPM' if row.statistic == 'gain_lower_information' else 'with_RPM'
            r = t3[t3.setting.eq(row.setting) & t3.telemetry_spec.eq(tel)].iloc[0]
            close('Figure 3 primary ' + str(i), [row.estimate, row.lower, row.upper], [r.gain_percent, r.lower_percent, r.upper_percent])
        else:
            d = t3[t3.setting.eq(row.setting)].set_index('telemetry_spec')
            close('Figure 3 paired attenuation ' + row.setting, row.estimate, d.loc['without_RPM', 'gain_percent'] - d.loc['with_RPM', 'gain_percent'])
            s = csv(row.source_path)
            q = s[s.environment.eq(row.setting)].iloc[0]
            close('Figure 3 paired interval ' + row.setting, [row.lower, row.upper], [q.lcb_pp, q.ucb_pp])
    f11 = csv('figure_data/figure11.csv', dtype={'object_id': str})
    cross = csv('figure_data/figure11/TRIP_SCATTER_DATA.csv')
    heat = pq('figure_data/figure11/HEATMAP_VALUES.parquet')
    blocks = pq('data/accounting/absolute_error/SIX_SCALE_BLOCK_METRICS.parquet')
    cm = blocks[blocks.scale.eq('complete')].set_index(['setting', 'physical_trip_id'])
    for row in cross.itertuples():
        rr = cm.loc[row.setting, row.trip_id]
        fields = ['A0', 'AR', 'S', 'T', 'C0', 'CR']
        close('Figure 11 cancellation quantities ' + row.setting + ' ' + row.trip_id, [getattr(row, k) for k in fields], [rr[k] for k in fields], raw=True)
        close('Figure 11 normalized scatter/terminal ' + row.setting + ' ' + row.trip_id, [row.x, row.y, row.terminal], [100 * row.S / row.A0, 100 * (row.C0 - row.CR) / row.A0, 100 * row.T / row.A0])
        z = f11[f11.setting.eq(row.setting) & f11.object_id.eq(row.trip_id)].set_index('statistic')
        for field in ['S', 'T', 'C0', 'CR', 'A0', 'AR', 'x', 'y', 'terminal']:
            close('Figure 11 source cell ' + row.setting + ' ' + row.trip_id + ' ' + field, z.loc[field, 'estimate'], getattr(row, field), raw=True)
        hh = heat[heat.setting.eq(row.setting) & heat.trip_id.eq(row.trip_id)].iloc[0]
        cols = [c for c in heat if c.startswith('g_')]
        close('Figure 11 heatmap all 51 bins ' + row.setting + ' ' + row.trip_id, z.loc[cols, 'estimate'], hh[cols].to_numpy())
        close('Figure 11 heatmap terminal ' + row.setting + ' ' + row.trip_id, hh.terminal, row.terminal)
    for row in f11[f11.panel.eq('a')].itertuples():
        rr = sc[sc.setting.eq(row.setting) & sc.training_loss.eq('absolute_error') & sc.scale.eq(row.object_id)].iloc[0]
        close('Figure 11 D endpoint ' + row.setting + ' ' + row.object_id, [row.estimate, row.lower, row.upper], [rr.D, rr.CI_lower, rr.CI_upper])
    for row in f11[f11.statistic.eq('persistence')].itertuples():
        rr = old[old.state.eq('B') & old.setting.eq(row.setting) & old.H.eq(row.object_id)].iloc[0]
        close('Figure 11 persistence ' + row.setting + ' ' + row.object_id, row.estimate, rr.Pi_H)
        check('Figure 11 persistence counts ' + row.setting + ' ' + row.object_id, row.n_positive == rr.n_S_positive and row.n_persist == rr.n_persist)
        if row.object_id == '1':
            expected = [1.0, 1.0]
        else:
            inf = csv('data/accounting/original/BOOTSTRAP_INFERENCE.csv', dtype={'H': str})
            v = inf[inf.state.eq('B') & inf.setting.eq(row.setting) & inf.H.eq(row.object_id) & inf.metric.eq('Pi')].iloc[0]
            expected = [v.lower, v.upper]
        close('Figure 11 persistence interval ' + row.setting + ' ' + row.object_id, [row.lower, row.upper], expected)

def original_bootstrap_and_sensitivity():
    base = 'data/accounting/original/'
    blocks = pq(base + 'BLOCK_METRICS.parquet')
    scales = ['1', '2', '4', '8', 'COMPLETE_TRIP_COMMON8']
    for state in ['B', 'BKZ']:
        inf = csv(base + ('BOOTSTRAP_INFERENCE.csv' if state == 'B' else 'RICH_TELEMETRY_INFERENCE.csv'), dtype={'H': str})
        saved = pq(base + 'BOOTSTRAP_DRAWS_' + state + '.parquet')
        for setting in CORE:
            wdf = pq(base + 'BOOTSTRAP_WEIGHTS_' + setting + '.parquet')
            w = wdf.to_numpy()
            obs = []
            boots = []
            for h in scales:
                q = blocks[blocks.state.eq(state) & blocks.setting.eq(setting) & blocks.H.eq(h)].copy()
                q['cond'] = (q.S > 0).astype(int)
                q['persist'] = ((q.S > 0) & (q['T'] > 0)).astype(int)
                q.vehicle_id = q.vehicle_id.astype(str)
                cols = ['E0', 'ER', 'A0', 'AR', 'C0', 'CR', 'target_mL', 'cond', 'persist']
                a = q.groupby('vehicle_id')[cols].sum().reindex(wdf.columns).to_numpy()

                def statistics(x):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return np.stack([100 * (x[..., 0] - x[..., 1]) / x[..., 6], x[..., 5] / x[..., 3] - x[..., 4] / x[..., 2], x[..., 8] / x[..., 7]], axis=-1)
                ob = statistics(a.sum(axis=0))
                bs = statistics(w @ a)
                obs.append(ob)
                boots.append(bs)
                for j, k in enumerate(['D', 'Delta_Q', 'Pi']):
                    d = saved[saved.setting.eq(setting) & saved.H.eq(h) & saved.metric.eq(k)].sort_values('replicate').estimate
                    close('Original saved bootstrap ' + state + ' ' + setting + ' ' + h + ' ' + k, bs[:, j], d)
            ob = np.array(obs)
            bs = np.stack(boots, axis=1)
            for j, k in enumerate(['D', 'Delta_Q', 'Pi']):
                inds = list(range(5)) if j == 0 else list(range(1, 5))
                joint = np.isfinite(bs[:, inds, j]).all(axis=1)
                sd = np.nanstd(bs[:, inds, j], axis=0, ddof=1)
                deg = np.array([np.ptp(bs[np.isfinite(bs[:, ii, j]), ii, j]) == 0 for ii in inds])
                active = (sd > 0) & ~deg
                critical = np.quantile(np.abs((bs[joint][:, inds, j][:, active] - ob[inds, j][active]) / sd[active]).max(axis=1), 0.95) if active.any() else 0
                for ii, index in enumerate(inds):
                    d = bs[:, index, j]
                    d = d[np.isfinite(d)]
                    row = inf[inf.setting.eq(setting) & inf.H.eq(scales[index]) & inf.metric.eq(k)].iloc[0]
                    if deg[ii]:
                        lo = hi = ob[index, j]
                    elif state == 'B':
                        lo, hi = (ob[index, j] - critical * sd[ii], ob[index, j] + critical * sd[ii])
                    else:
                        lo, hi = np.quantile(d, [0.025, 0.975])
                    close('Original interval family ' + state + ' ' + setting + ' ' + scales[index] + ' ' + k, [lo, hi], [row.lower, row.upper])
    pairs = pq('data/diagnostics/loss/PAIRED_LOSS_BOOTSTRAP_DRAWS.parquet')
    close('Squared-minus-absolute saved loss contrasts', pairs.Delta, pairs.estimate_squared - pairs.estimate_absolute)
    comp = csv('data/diagnostics/loss/LOSS_COMPARISON.csv', dtype={'scale': str})
    for row in comp.itertuples():
        isD = row.metric == 'D'
        h = row.scale if isD else '1-minus-complete'
        metric = 'D' if isD else 'L_trip'
        h = 'COMPLETE_TRIP_COMMON8' if h == 'complete' else h
        d = pairs[pairs.setting.eq(row.setting) & pairs.H.eq(h) & pairs.metric.eq(metric)].Delta
        if len(d):
            close('Loss sensitivity interval ' + row.setting + ' ' + h, np.quantile(d, [0.025, 0.975]), [row.Delta_D_CI_lower, row.Delta_D_CI_upper] if isD else [row.Delta_L_CI_lower, row.Delta_L_CI_upper])
    bias = csv('data/diagnostics/loss/SIGNED_BIAS.csv')
    for row in bias.itertuples():
        file = 'ABSOLUTE_ERROR_PREDICTIONS.parquet' if row.training_loss == 'absolute_error' else 'SQUARED_ERROR_PREDICTIONS.parquet'
        p = pq('data/diagnostics/loss/' + file)
        mem = pq('data/accounting/' + ('PREFIX_MEMBERSHIP.parquet' if row.population == 'COMMON8_H1_PREFIX' else 'COMPLETE_MEMBERSHIP.parquet'))
        ids = set(mem.loc[mem.setting.eq(row.setting), 'segment_id'])
        q = p[p.setting.eq(row.setting) & p.segment_id.isin(ids)]
        e = 1000 * (q[row.model] - q.target_l)
        target = 1000 * q.target_l.sum()
        check('Signed-bias support ' + row.setting + ' ' + row.training_loss + ' ' + row.model + ' ' + row.population, len(q) == row.n_segments and q.trip_uid.nunique() == row.n_trips and (q.VehId.nunique() == row.n_vehicles))
        close('Signed-bias quantities ' + row.setting + ' ' + row.training_loss + ' ' + row.model + ' ' + row.population, [e.mean(), e.sum(), target, 100 * e.sum() / target], [row.mean_signed_segment_residual_mL, row.total_signed_residual_mL, row.total_observed_target_mL, row.total_signed_residual_over_target_percent], raw=True)

def diagnostic_results():
    spat = 'data/diagnostics/spatial/'
    s = csv(spat + 'simultaneous_inference_statistics.csv')
    draws = pq(spat + 'vehicle_bootstrap_draws.parquet')
    m = draws.pivot(index='replicate', columns='stat_id', values='estimate_mL').reindex(columns=s.stat_id).to_numpy()
    sd = np.std(m, axis=0, ddof=1)
    theta = s.point_estimate_mL.to_numpy()
    crit = np.quantile(np.max(abs((m - theta) / sd), axis=1), 0.95)
    close('Spatial 135-statistic simultaneous intervals', np.stack([theta - crit * sd, theta + crit * sd], axis=1), s[['simultaneous_lcb_mL', 'simultaneous_ucb_mL']])
    check('Spatial 44 cells in each contrast include zero', len(s) == 135 and (s.simultaneous_lcb_mL <= 0).all() and (s.simultaneous_ucb_mL >= 0).all())
    grid = csv(spat + 'grid_sensitivity_summary.csv')
    check('Spatial grid sizes and offsets retained', set(grid.cell_m) == {100, 200, 250} and set(grid.shift_e_m) == {0, 100} and (set(grid.shift_n_m) == {0, 100}))
    target = 'data/diagnostics/target_decile/'
    d = csv(target + 'SOURCE_DATA.csv')
    stats = csv(target + 'CLUSTERED_STATS.csv')
    close('Target-decile paired errors', d.absolute_error_telemetry_only_mL - d.absolute_error_road_aware_mL, d.paired_road_aware_benefit_mL)
    for row in stats.itertuples():
        q = d[d.population_id.eq(row.population_id) & d.target_decile.eq(row.target_decile)]
        ma = [q.absolute_error_telemetry_only_mL.mean(), q.absolute_error_road_aware_mL.mean()]
        check('Target-decile membership ' + row.population_id + ' ' + str(row.target_decile), len(q) == row.n_segments)
        close('Target-decile MAE and gain ' + row.population_id + ' ' + str(row.target_decile), [*ma, 100 * (ma[0] - ma[1]) / ma[0]], [row.telemetry_only_mae_mL, row.road_aware_mae_mL, row.relative_mae_reduction_pct])
    speed = 'data/diagnostics/speed/'
    d = csv(speed + 'fig9_conditional_records.csv')
    repeated = int(d.groupby('segment_id').size().gt(1).sum())
    check('Speed evaluation/physical identity counts', len(d) == 43143 and d.segment_id.nunique() == 38211 and (repeated == 4808), evaluation_records=43143, unique_segments=38211, segments_in_multiple_settings=repeated, extra_evaluation_records=len(d) - d.segment_id.nunique())
    close('Speed paired errors', d.error_improvement, d.abs_error_telemetry - d.abs_error_road)
    contributions = csv(speed + 'fig9_aggregate_contributions.csv')
    den = d.abs_error_telemetry.sum()
    close('Speed pooled decomposition', contributions.contribution_pct_points.sum(), 100 * d.error_improvement.sum() / den)
    exposure = csv(speed + 'fig9_exposure_share.csv')
    check('Speed exposure count', int(exposure.n_evaluation_records.sum()) == 43143)
    close('Speed exposure shares', exposure.evaluation_record_share_fraction, exposure.n_evaluation_records / 43143)
    for row in csv(speed + 'fig9_panel_a_profiles.csv').itertuples():
        q = d[d.operating_state.eq(row.state) & d.mean_speed_kmh.ge(row.speed_lower_kmh) & d.mean_speed_kmh.le(row.speed_upper_kmh)]
        support = len(q) >= 100 and q.trip_uid.nunique() >= 30 and (q.VehId.nunique() >= 15)
        if row.supported:
            close('Speed-conditioned gain ' + row.state + ' ' + str(row.speed_center_kmh), 100 * q.error_improvement.sum() / q.abs_error_telemetry.sum(), row.conditional_gain_pct)
    controls = csv('results/correspondence_control_draws.csv')
    pri = csv('tables/TABLE3_SOURCE.csv')
    for name, setting in [('Unseen trip', 'cold_trip'), ('October 2018', 'cold_month')]:
        q = controls[controls.population.eq(name) & controls.telemetry_order.eq(1)]
        real = float(pri[pri.setting.eq(setting) & pri.telemetry_spec.eq('without_RPM')].gain_percent.iloc[0])
        check('Authentic pairing exceeds every control ' + setting, len(q) == 40 and real > q.effect_pct.max())
    overlap = csv('results/road_attribute_overlap.csv')
    check('Distribution separation direction', overlap.loc[overlap.population.str.contains('area|road', case=False), 'domain_classifier_roc_auc'].min() > 0.99)

def setup():
    global np, pd
    import numpy as np
    import pandas as pd

def source(name):
    return read_json('config/source_keys.json')[name]

def hashes():
    index = read_json('checksums.json')
    for name, digest in index.items():
        check('File integrity ' + name, sha(safe(name)) == digest)

def main():
    hashes()
    setup()
    primary()
    selection()
    overlap()
    accounting()
    original_bootstrap_and_sensitivity()
    diagnostic_results()
    print('All reported-result calculations passed.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
