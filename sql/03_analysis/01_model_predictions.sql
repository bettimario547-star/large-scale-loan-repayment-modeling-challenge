-- Paths are bound by the Python runner to the frozen local Parquet files.
-- The key includes dataset: train/test both have IDs beginning at zero.
CREATE TABLE model_registry AS SELECT * FROM read_parquet('__REGISTRY_PARQUET__');
CREATE TABLE model_predictions (
    dataset VARCHAR NOT NULL CHECK (dataset IN ('train', 'test')),
    id BIGINT NOT NULL,
    model_version VARCHAR NOT NULL,
    model_role VARCHAR NOT NULL CHECK (model_role IN ('champion', 'challenger')),
    prediction_kind VARCHAR NOT NULL,
    label TINYINT,
    fold TINYINT,
    stress_feature_492_upper_tail BOOLEAN,
    stress_feature_1039_observed BOOLEAN,
    p_repay DOUBLE NOT NULL CHECK (isfinite(p_repay) AND p_repay BETWEEN 0 AND 1),
    p_default DOUBLE NOT NULL CHECK (isfinite(p_default) AND p_default BETWEEN 0 AND 1),
    PRIMARY KEY (dataset, id, model_version),
    CHECK (
        (dataset = 'train' AND prediction_kind = 'oof'
         AND label IS NOT NULL AND label IN (0, 1)
         AND fold IS NOT NULL AND fold BETWEEN 0 AND 4
         AND stress_feature_492_upper_tail IS NOT NULL
         AND stress_feature_1039_observed IS NOT NULL)
        OR
        (dataset = 'test' AND prediction_kind = 'test_ensemble'
         AND label IS NULL AND fold IS NULL
         AND stress_feature_492_upper_tail IS NULL
         AND stress_feature_1039_observed IS NULL)
    ),
    CHECK (abs(p_repay + p_default - 1.0) < 1e-12)
);
INSERT INTO model_predictions SELECT * FROM read_parquet('__PREDICTIONS_PARQUET__');
