/*
# Brain Tumor Diagnosis System — Core Schema

## Purpose
Creates the database schema for the SegUX-SSPANet Brain Tumor Diagnosis System.
This is a multi-user app with authentication: each clinician/user sees only
their own patients and predictions.

## New Tables

### patients
Stores patient demographic information.
- id (uuid, PK)
- user_id (uuid, FK auth.users, defaults to auth.uid()) — owner
- name (text, not null)
- age (int, nullable)
- gender (text, nullable, check male/female/other)
- mrn (text, not null) — Medical Record Number
- notes (text, nullable)
- created_at (timestamptz, default now())

### predictions
Stores AI inference results for each MRI scan.
- id (uuid, PK)
- user_id (uuid, FK auth.users, defaults to auth.uid()) — owner
- patient_id (uuid, FK patients, cascade delete)
- image_base64 (text, not null) — MRI scan as base64 data URL
- predicted_class (text, not null) — glioma/meningioma/pituitary/no_tumor
- predicted_class_display (text, not null)
- probabilities (jsonb) — all class probabilities
- uncertainty (jsonb) — MC dropout uncertainty metrics
- segmentation (jsonb) — segmentation mask/overlay data
- gradcam_results (jsonb) — gradcam/gradcam++/eigengradcam results
- model_version (text)
- inference_time_ms (int)
- notes (text, nullable)
- created_at (timestamptz, default now())

### reports
Stores generated PDF report metadata.
- id (uuid, PK)
- user_id (uuid, FK auth.users, defaults to auth.uid()) — owner
- prediction_id (uuid, FK predictions, cascade delete)
- report_type (text, default 'full') — full/summary
- created_at (timestamptz, default now())

## Security
- RLS enabled on all tables.
- All tables are owner-scoped: each authenticated user can only access
  rows where user_id = auth.uid().
- 4 policies per table (select/insert/update/delete), scoped TO authenticated.
- user_id columns default to auth.uid() so inserts from the frontend
  succeed even when the client omits user_id.

## Notes
1. The app has a sign-in/sign-up screen, so all policies are scoped to
   authenticated users with ownership checks.
2. image_base64 stores the MRI scan as a data URL (base64-encoded PNG/JPG).
   For production, this would use Supabase Storage, but base64 in a text
   column works for this demo.
3. JSONB columns store structured AI results (probabilities, uncertainty,
   segmentation, gradcam) so the frontend can reconstruct full results.
4. Cascade deletes ensure that deleting a patient also deletes their
   predictions, and deleting a prediction deletes its reports.
*/
CREATE TABLE IF NOT EXISTS patients (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
  name text NOT NULL,
  age integer,
  gender text CHECK (gender IN ('male', 'female', 'other')),
  mrn text NOT NULL,
  notes text,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE patients ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_own_patients" ON patients;
CREATE POLICY "select_own_patients" ON patients FOR SELECT
  TO authenticated USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "insert_own_patients" ON patients;
CREATE POLICY "insert_own_patients" ON patients FOR INSERT
  TO authenticated WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "update_own_patients" ON patients;
CREATE POLICY "update_own_patients" ON patients FOR UPDATE
  TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "delete_own_patients" ON patients;
CREATE POLICY "delete_own_patients" ON patients FOR DELETE
  TO authenticated USING (auth.uid() = user_id);

CREATE TABLE IF NOT EXISTS predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
  patient_id uuid NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  image_base64 text NOT NULL,
  predicted_class text NOT NULL CHECK (predicted_class IN ('glioma', 'meningioma', 'pituitary', 'no_tumor')),
  predicted_class_display text NOT NULL,
  probabilities jsonb NOT NULL,
  uncertainty jsonb NOT NULL,
  segmentation jsonb NOT NULL,
  gradcam_results jsonb NOT NULL,
  model_version text NOT NULL DEFAULT 'SegUX-SSPANet-v1.0.0',
  inference_time_ms integer NOT NULL DEFAULT 0,
  notes text,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_own_predictions" ON predictions;
CREATE POLICY "select_own_predictions" ON predictions FOR SELECT
  TO authenticated USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "insert_own_predictions" ON predictions;
CREATE POLICY "insert_own_predictions" ON predictions FOR INSERT
  TO authenticated WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "update_own_predictions" ON predictions;
CREATE POLICY "update_own_predictions" ON predictions FOR UPDATE
  TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "delete_own_predictions" ON predictions;
CREATE POLICY "delete_own_predictions" ON predictions FOR DELETE
  TO authenticated USING (auth.uid() = user_id);

CREATE TABLE IF NOT EXISTS reports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
  prediction_id uuid NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
  report_type text NOT NULL DEFAULT 'full',
  created_at timestamptz DEFAULT now()
);

ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "select_own_reports" ON reports;
CREATE POLICY "select_own_reports" ON reports FOR SELECT
  TO authenticated USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "insert_own_reports" ON reports;
CREATE POLICY "insert_own_reports" ON reports FOR INSERT
  TO authenticated WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "update_own_reports" ON reports;
CREATE POLICY "update_own_reports" ON reports FOR UPDATE
  TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "delete_own_reports" ON reports;
CREATE POLICY "delete_own_reports" ON reports FOR DELETE
  TO authenticated USING (auth.uid() = user_id);

CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_patients_user_id ON patients(user_id);
CREATE INDEX IF NOT EXISTS idx_reports_prediction_id ON reports(prediction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
