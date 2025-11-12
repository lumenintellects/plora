import pandas as pd
from plora.notebook_utils import extract_value_add_metrics


def test_extract_value_add_metrics_basic():
    # Simulated experiment data with edge cases (None placebo, missing cross)
    experiment_data = {
        'value_add': [
            {
                'config': {
                    'domain': 'arithmetic',
                    'rank': 4,
                    'scheme': 'all',
                    'seed': 42,
                    'eval_split': 'validation',
                },
                'trained': {'delta_mean': -0.5, 'wilcoxon_p': 0.01, 'ci': [-0.7, -0.3]},
                'placebo_a': None,  # should be treated as empty
                'placebo_b': {'delta_mean': -0.1, 'wilcoxon_p': 0.5, 'ci': [-0.2, 0.1]},
                'cross_domain': {'legal': {'delta_mean': -0.2}},
                'latency_ms': 123.4,
            },
            {
                'config': {
                    'domain': 'legal',
                    'rank': 8,
                    'scheme': 'attention',
                    'seed': 7,
                    'eval_split': 'test',
                },
                'trained': {'delta_mean': 0.2, 'wilcoxon_p': 0.2, 'ci': [0.1, 0.3]},
                'placebo_a': {'delta_mean': 0.05, 'wilcoxon_p': 0.9},
                'placebo_b': None,
                'cross_domain': {},
                'latency_ms': 87.0,
            },
        ]
    }

    df = extract_value_add_metrics(experiment_data)
    assert not df.empty, 'DataFrame should not be empty when records present'
    # Check required columns exist
    for col in [
        'domain', 'rank', 'scheme', 'seed', 'trained_delta_mean', 'trained_wilcoxon_p',
        'placebo_a_delta_mean', 'placebo_b_delta_mean', 'latency_ms'
    ]:
        assert col in df.columns, f'Missing column {col}'

    # Row count matches input
    assert len(df) == 2

    # Value checks
    row_arith = df[df['domain'] == 'arithmetic'].iloc[0]
    assert row_arith['trained_delta_mean'] == -0.5
    assert row_arith['placebo_a_delta_mean'] == 0  # None -> default 0
    assert 'cross_legal_delta_mean' in df.columns
    assert row_arith['cross_legal_delta_mean'] == -0.2

    row_legal = df[df['domain'] == 'legal'].iloc[0]
    assert row_legal['placebo_b_delta_mean'] == 0  # None -> default 0
    assert pd.isna(row_legal.get('cross_arithmetic_delta_mean', float('nan'))) or 'cross_arithmetic_delta_mean' not in df.columns

