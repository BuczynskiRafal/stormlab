"""The twelve-feature input of the catchment classifier.

Ported from cell 9 of ``models/catchment_classification_model/classifier-final.ipynb``,
the canonical notebook. Eight columns come straight from the INP/RPT frame; the
four runoff columns are coerced to numeric (swmmio returns report values as
strings) and divided by ``Area``, each guarded by its own ``sum() != 0`` test so
an all-zero column stays zero instead of turning into NaN.

The guard is per column on purpose: cell 34 of the notebook applies the
``TotalRunoffMG`` division under a guard that tests ``TotalRunoffIn`` instead, and
that typo is not reproduced here — it is a bug in a one-off inline copy, not part
of the trained pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd

from stormlab.contracts.catchment_features import (
    AREA_COLUMN,
    CATCHMENT_AREA_NORMALIZED,
    CATCHMENT_FEATURE_COLUMNS,
    CatchmentContractError,
    build_catchment_features,
    validate_catchment_features,
)

logger = logging.getLogger(__name__)


def extract_catchment_features(dfs: pd.DataFrame) -> pd.DataFrame:
    """Build the classifier input from a subcatchment frame.

    Args:
        dfs: subcatchment frame from :func:`stormlab.io.inp.read_frames`, read
            with the report attached so the runoff columns are present.

    Returns:
        A frame with exactly the twelve contract columns, in contract order, as
        ``float64``, indexed like ``dfs``. The input frame is not modified.

    Raises:
        CatchmentContractError: if a source column or ``Area`` is missing, or the
            result does not satisfy the contract (for example because a runoff
            column contains values swmmio could not parse as numbers).
    """
    missing = [column for column in (*CATCHMENT_FEATURE_COLUMNS, AREA_COLUMN) if column not in dfs.columns]
    if missing:
        raise CatchmentContractError(
            f"Subcatchment frame is missing {len(missing)} column(s) required by the catchment contract: {missing}"
        )

    source = dfs.copy()
    for column in CATCHMENT_AREA_NORMALIZED:
        source[column] = pd.to_numeric(source[column], errors="coerce")

    zeroed = [column for column in CATCHMENT_AREA_NORMALIZED if source[column].sum() == 0]
    if zeroed:
        logger.info("Runoff columns %s sum to zero and are kept unnormalised, as the training notebook did", zeroed)

    features = build_catchment_features(source)
    validate_catchment_features(features)
    return features


__all__ = ["extract_catchment_features"]
