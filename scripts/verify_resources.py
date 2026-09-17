"""Check the current paper's compact resources using the Python standard library."""
import csv
import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with (ROOT / path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    index = json.loads((ROOT / 'metadata/resources.json').read_text(encoding='utf8'))
    for name, expected in index['files'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT), name
        assert path.is_file(), 'Missing resource: ' + name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Changed resource: ' + name
    payload_index = json.loads((ROOT / 'data/checksums.json').read_text(encoding='utf8'))
    release = json.loads((ROOT / 'data/release.json').read_text(encoding='utf8'))
    assert hashlib.sha256((ROOT / 'data/checksums.json').read_bytes()).hexdigest() == release['checksums_sha256']
    for path in list((ROOT / 'tables').glob('*_values.csv')) + list((ROOT / 'figure_data').glob('*.csv')) + list((ROOT / 'results').glob('*.csv')):
        for record in read_csv(path.relative_to(ROOT).as_posix()):
            source = record.get('source_path')
            if source:
                if source.startswith('data/energy/'):
                    assert source.removeprefix('data/energy/') in payload_index, 'Unresolved data source: ' + source
                else:
                    assert (ROOT / source).is_file(), 'Unresolved resource source: ' + source
    settings = json.loads((ROOT / 'config/evaluation.json').read_text(encoding='utf8'))['settings']
    primary = read_csv('results/primary_effects.csv')
    assert len(settings) == 5 and len(primary) == 10
    expected = {(s['id'], t) for s in settings for t in ['without_RPM', 'with_RPM']}
    assert {(r['setting'], r['telemetry_spec']) for r in primary} == expected
    support = {s['id']: s for s in settings}
    for row in primary:
        s = support[row['setting']]
        assert (int(row['n_segments']), int(row['n_trips']), int(row['n_vehicles'])) == (s['evaluation_segments'], s['evaluation_trips'], s['evaluation_vehicles'])
        b, r = float(row['telemetry_only_mae_mL']), float(row['road_augmented_mae_mL'])
        assert abs(100 * (b - r) / b - float(row['gain_percent'])) < 1e-9
    motorway = support['cold_functional_road_class']
    assert motorway['selection_unit'] == 'segment'
    features = json.loads((ROOT / 'config/features.json').read_text(encoding='utf8'))
    assert (len(features['B']), len(features['K']), len(features['Z'])) == (8, 5, 2)
    assert len(read_csv('metadata/road_attributes.csv')) == features['R']['candidate_count'] == 142
    model = json.loads((ROOT / 'config/analysis.json').read_text(encoding='utf8'))
    assert model['primary_estimator']['nonvehicle_seeds'] == [0, 1, 2]
    assert model['primary_estimator']['unseen_vehicle_seeds'] == [0]
    assert model['estimator_selection_sensitivity']['separate_from_primary']
    for number, count in [(3, 60), (4, 102)]:
        records = read_csv(f'tables/table{number}_values.csv')
        assert len(records) == count
        displays = []
        for record in records:
            text = record['manuscript_display'].replace('−', '-').replace(',', '')
            digits = len(text.split('.')[1]) if '.' in text else 0
            expected_value = Decimal(record['source_full_precision']).quantize(Decimal('1').scaleb(-digits), rounding=ROUND_HALF_UP)
            assert Decimal(text) == expected_value
            displays.append(Decimal(text))
        with (ROOT / f'tables/table{number}.csv').open(encoding='utf8', newline='') as stream:
            table = list(csv.reader(stream))
        width = 5 if number == 3 else 3
        cells = ' '.join(' '.join(row[-width:]) for row in table[1:] if len([v for v in row if v.strip()]) > width)
        numbers = [Decimal(v) for v in re.findall(r'[-+]?\d+(?:\.\d+)?', cells.replace('−', '-').replace(',', ''))]
        if number == 3:
            displays = [displays[i + j] for i in range(0, len(displays), 6) for j in [0, 1, 2, 4, 5, 3]]
        assert numbers == displays, f'Table {number} displayed values differ from their sources'
    assert {p.name for p in (ROOT / 'figures').glob('*.png')} == {f'figure_{i:02d}.png' for i in range(1, 12)}
    for number in range(1, 12):
        assert (ROOT / f'figures/figure_{number:02d}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    print(f'PASS: {len(index["files"])} public files; five settings; ten primary rows; four tables; eleven figures; all 162 displayed numerical values.')
    print('Use scripts/reproduce_results.py after data installation for complete saved-result calculations.')


if __name__ == '__main__':
    main()
