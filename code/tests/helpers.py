"""Synthetic unit-test fixtures only; not collected memes or human reviews."""
def family(fid='f00', split='dev', editable=True):
    def variant(name, gold, fact):
        return {'image_path': f'images/{fid}-{name}.png', 'sha256': 'a'*64, 'gold_option': gold, 'gold_fact': fact}
    variants = {'original': variant('original', 'o1', True)}
    if editable: variants.update(edited=variant('edited', 'o2', False), sham=variant('sham', 'o1', True))
    return {'family_id': fid, 'template_group': fid, 'split': split,
            'source': {'url': 'https://example.invalid/test-only', 'dataset': 'unit-test-fixture', 'item_id': fid, 'rights_status': 'research_only', 'attribution': 'Synthetic test'},
            'caption_zh': '仅用于测试的标题', 'options': {'o1': '解释一', 'o2': '解释二', 'o3': '解释三', 'none': '以上均不成立'},
            'option_order': ['o1','o2','o3','none'], 'fact_question': '手中有奖杯吗？',
            'backgrounds': {'relevant': '测试相关背景', 'irrelevant': '测试无关背景', 'false': '测试错误背景'},
            'variants': variants, 'reviews': []}

def full_design():
    rows = [family(f'f{i:02d}', 'dev' if i<20 else 'test', i<10 or 20<=i<40) for i in range(60)]
    for i, row in enumerate(rows):
        order = row['option_order']
        row['option_order'] = order[i % 4:] + order[:i % 4]
    return rows
