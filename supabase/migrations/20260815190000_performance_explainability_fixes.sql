-- Performance + explainability fixes for SegUX-SSPANet
-- Safe to run against an existing database.

ALTER TABLE predictions
  ADD COLUMN IF NOT EXISTS feature_explanation jsonb;

CREATE INDEX IF NOT EXISTS idx_predictions_user_created_at
  ON predictions(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_reports_user_created_at
  ON reports(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_reports_user_prediction
  ON reports(user_id, prediction_id);
