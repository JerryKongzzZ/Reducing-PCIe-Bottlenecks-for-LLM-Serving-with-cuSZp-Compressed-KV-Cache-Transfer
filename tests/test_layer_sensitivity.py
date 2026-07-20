from benchmarks.layer_sensitivity_sweep import assign_sensitivity_categories


def test_assign_sensitivity_categories_creates_three_bins():
    results = {idx: {'score': float(idx)} for idx in range(12)}

    assign_sensitivity_categories(results)

    categories = {entry['category'] for entry in results.values()}
    assert categories == {'shallow', 'mid', 'deep'}
    assert results[11]['category'] == 'shallow'
    assert results[0]['category'] == 'deep'


def _profile(prompt_scale=1.0, top1_at_loose=1.0):
    layers = {}
    for layer_id in range(2):
        details = []
        for eps, multiplier, top1 in (
            (1e-5, 1e-4, 1.0),
            (1e-4, 1e-3, top1_at_loose),
        ):
            details.append({
                'eps': eps,
                'kl': prompt_scale * (layer_id + 1) * multiplier,
                'top1_match': top1,
                'k_size': 100,
                'v_size': 120,
                'k_actual_eb': eps,
                'v_actual_eb': eps,
            })
        layers[str(layer_id)] = {
            'score': 0.0,
            'max_safe_eps': 1e-4,
            'category': 'deep',
            'details': details,
        }
    return {'_metadata': {'model': 'test/model'}, 'layers': layers}


def test_multi_prompt_merge_uses_worst_kl_and_minimum_top1():
    from benchmarks.build_multi_prompt_sensitivity_profile import merge_profiles

    merged = merge_profiles(
        [('easy', _profile()), ('hard', _profile(2.0, top1_at_loose=0.75))],
        kl_threshold=1e-2,
        min_top1_match=1.0,
    )

    assert merged['_metadata']['prompt_count'] == 2
    assert merged['layers']['0']['details'][0]['kl'] == 2e-4
    assert merged['layers']['0']['details'][1]['top1_match'] == 0.75
    assert merged['layers']['0']['max_safe_eps'] == 1e-5


def test_mixed_profile_selector_ranks_by_quality_before_kl():
    from benchmarks.build_multi_prompt_sensitivity_profile import merge_profiles
    from benchmarks.select_mixed_bound_profile import select_mixed_profile

    merged = merge_profiles(
        [('easy', _profile()), ('hard', _profile(2.0, top1_at_loose=0.75))],
        kl_threshold=1e-2,
        min_top1_match=0.75,
    )
    mixed = select_mixed_profile(
        merged, tight_bound=1e-5, loose_bound=1e-4, loose_layers=1
    )

    assert mixed['_metadata']['loose_layers'] == [0]
    assert mixed['layers']['0']['max_safe_eps'] == 1e-4
    assert mixed['layers']['1']['max_safe_eps'] == 1e-5
