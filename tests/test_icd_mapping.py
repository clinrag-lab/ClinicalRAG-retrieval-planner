"""ICD category mapping fixes: each block covers the same conditions in ICD-9 and ICD-10."""

import pytest

from clinrag.workload.icd_categories import CATEGORIES

# (block, ICD-9 code, ICD-10 code) for the same condition; both must be in the block.
SAME_CONDITION = [
    ("diabetes", "24900", "E0800"),        # secondary diabetes
    ("diabetes", "25000", "E119"),
    ("copd", "4912", "J449"),              # obstructive chronic bronchitis
    ("copd", "4910", "J41"),               # simple chronic bronchitis
    ("copd", "4919", "J42"),               # unspecified chronic bronchitis
    ("copd", "4928", "J439"),              # emphysema
    ("copd", "496", "J449"),
    ("hypothyroidism", "2449", "E039"),
    ("hypothyroidism", "2430", "E031"),    # congenital hypothyroidism
    ("hypothyroidism", "2440", "E890"),    # postsurgical / postprocedural
    ("hypothyroidism", "2442", "E02"),     # iodine-deficiency hypothyroidism
]


@pytest.mark.parametrize("block,icd9,icd10", SAME_CONDITION)
def test_block_covers_condition_in_both_versions(block, icd9, icd10):
    category = CATEGORIES[block]
    assert category.contains(icd9, 9) and category.contains(icd10, 10)
    chapter = CATEGORIES[category.parent]
    assert chapter.contains(icd9, 9) and chapter.contains(icd10, 10)


@pytest.mark.parametrize(
    "block,code,version",
    [
        ("copd", "J40", 10),       # bronchitis not specified as acute or chronic (ICD-9 490, also excluded)
        ("copd", "J45909", 10),    # asthma
        ("copd", "4940", 9),       # bronchiectasis
        ("copd", "J479", 10),
        ("diabetes", "2510", 9),   # hypoglycemia, not diabetes
        ("diabetes", "E160", 10),
        ("hypothyroidism", "E891", 10),  # postprocedural hypoinsulinemia
        ("hypothyroidism", "2420", 9),   # thyrotoxicosis
        ("hypothyroidism", "E050", 10),
    ],
)
def test_fixes_do_not_overreach(block, code, version):
    assert not CATEGORIES[block].contains(code, version)
