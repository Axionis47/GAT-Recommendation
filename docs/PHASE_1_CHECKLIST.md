# Phase 1: GCP Bootstrap - Checklist

**Status**: In Progress  
**Started**: 2025-01-20  
**Completed**: TBD

---

## Objective

Bootstrap GCP infrastructure for ETP-GT project:
- Enable required APIs
- Create GCS bucket with proper structure
- Create Artifact Registry for Docker images
- Create Service Account with minimal IAM roles
- Document OIDC setup for GitHub Actions

---

## Prerequisites

### Local Environment
- [x] `.env` file created from `.env.example`
- [ ] GCP project ID configured in `.env`
- [ ] GCP region configured in `.env`
- [ ] Bucket name configured in `.env`

### GCP Account
- [ ] GCP project exists
- [ ] Billing enabled on project
- [ ] `gcloud` CLI installed
- [ ] Authenticated with `gcloud auth login`
- [ ] Sufficient IAM permissions (Owner or Editor + Service Account Admin)

---

## Deliverables

### 1. Scripts

- [x] `scripts/gcp/00_validate_env.sh` - Environment validation
- [x] `scripts/gcp/01_bootstrap.sh` - Infrastructure bootstrap

### 2. GCP Resources

- [ ] **APIs Enabled**:
  - [ ] `storage.googleapis.com` (Cloud Storage)
  - [ ] `artifactregistry.googleapis.com` (Artifact Registry)
  - [ ] `aiplatform.googleapis.com` (Vertex AI)
  - [ ] `compute.googleapis.com` (Compute Engine)
  - [ ] `run.googleapis.com` (Cloud Run)
  - [ ] `iam.googleapis.com` (IAM)
  - [ ] `iamcredentials.googleapis.com` (IAM Credentials)

- [ ] **GCS Bucket** (`gs://${GCS_BUCKET}`):
  - [ ] Bucket created in specified region
  - [ ] Versioning enabled
  - [ ] Lifecycle policy set (delete old versions after 30 days)
  - [ ] Directory structure:
    - [ ] `data/raw/`
    - [ ] `data/interim/`
    - [ ] `data/processed/`
    - [ ] `artifacts/baselines/`
    - [ ] `artifacts/etpgt/`
    - [ ] `artifacts/ablations/`
    - [ ] `artifacts/exports/`
    - [ ] `logs/`

- [ ] **Artifact Registry** (`${AR_REPO}`):
  - [ ] Repository created in specified region
  - [ ] Format: Docker
  - [ ] Description set

- [ ] **Service Account** (`${SA_NAME}`):
  - [ ] Service account created
  - [ ] Display name: "ETP-GT Training and Serving SA"
  - [ ] IAM roles granted:
    - [ ] `roles/storage.objectAdmin` (GCS read/write)
    - [ ] `roles/artifactregistry.reader` (Pull Docker images)
    - [ ] `roles/aiplatform.user` (Vertex AI jobs)
    - [ ] `roles/logging.logWriter` (Write logs)
    - [ ] `roles/monitoring.metricWriter` (Write metrics)

### 3. Documentation

- [x] `docs/GCP_OIDC_SETUP.md` - GitHub Actions OIDC guide (from Phase 0)
- [ ] `docs/PHASE_1_CHECKLIST.md` - This file
- [ ] Update `README.md` with GCP setup instructions

---

## Execution Steps

### Step 1: Validate Environment

```bash
# Run validation script
bash scripts/gcp/00_validate_env.sh
```

**Expected Output**:
- ✓ All required environment variables set
- ✓ gcloud CLI installed and authenticated
- ✓ Project accessible
- ✓ Billing enabled
- ✓ Region valid

**If validation fails**:
1. Review error messages
2. Fix issues (install gcloud, authenticate, enable billing, etc.)
3. Re-run validation

### Step 2: Run Bootstrap

```bash
# Run bootstrap script
make gcp-bootstrap

# Or manually:
bash scripts/gcp/01_bootstrap.sh
```

**Expected Output**:
- ✓ All APIs enabled
- ✓ GCS bucket created with structure
- ✓ Artifact Registry repository created
- ✓ Service Account created with IAM roles
- ✓ Docker authentication configured

**Script is idempotent**: Safe to re-run if interrupted.

### Step 3: Verify Resources

```bash
# Verify GCS bucket
gsutil ls -r gs://${GCS_BUCKET}

# Verify Artifact Registry
gcloud artifacts repositories describe ${AR_REPO} --location=${GCP_REGION}

# Verify Service Account
gcloud iam service-accounts describe ${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com

# Verify IAM roles
gcloud projects get-iam-policy ${GCP_PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
```

### Step 4: Configure GitHub OIDC (Optional for Phase 1)

**Note**: This can be done later when setting up GitHub Actions in Phase 8.

Follow `docs/GCP_OIDC_SETUP.md` to:
1. Create Workload Identity Pool
2. Create Workload Identity Provider
3. Grant Service Account Workload Identity User role
4. Configure GitHub Secrets

### Step 5: Update Configuration Files

Update `configs/*.yaml` with actual GCS paths:

```yaml
# Example: configs/yoochoose_baselines.yaml
data:
  raw_events: "gs://${GCS_BUCKET}/data/raw/yoochoose-clicks.dat"
  processed_dir: "gs://${GCS_BUCKET}/data/processed/yoochoose"

artifacts:
  output_dir: "gs://${GCS_BUCKET}/artifacts/baselines"
  logs_dir: "gs://${GCS_BUCKET}/logs/baselines"
```

---

## Validation

### Manual Checks

- [ ] Can list bucket contents: `gsutil ls gs://${GCS_BUCKET}`
- [ ] Can upload test file: `echo "test" | gsutil cp - gs://${GCS_BUCKET}/test.txt`
- [ ] Can download test file: `gsutil cat gs://${GCS_BUCKET}/test.txt`
- [ ] Can list AR repositories: `gcloud artifacts repositories list`
- [ ] Service Account has correct roles (see Step 3 above)

### Automated Checks

```bash
# Run validation script again
bash scripts/gcp/00_validate_env.sh

# Should show all APIs enabled
```

---

## Gate Criteria

- [ ] **GCS Bucket exists** with proper structure and lifecycle policy
- [ ] **Artifact Registry exists** and is accessible
- [ ] **Service Account exists** with minimal required IAM roles
- [ ] **All required APIs enabled**
- [ ] **Documentation updated** with actual setup steps
- [ ] **Can authenticate Docker** to Artifact Registry

**Gate Status**: 🔴 Not Met (resources not created yet)

---

## Troubleshooting

### Issue: "Permission denied" errors

**Solution**:
- Ensure you have sufficient IAM permissions
- Required roles: Owner, Editor, or custom role with:
  - `storage.buckets.create`
  - `artifactregistry.repositories.create`
  - `iam.serviceAccounts.create`
  - `resourcemanager.projects.setIamPolicy`

### Issue: "Billing not enabled"

**Solution**:
- Enable billing at: https://console.cloud.google.com/billing
- Link billing account to project

### Issue: "API not enabled"

**Solution**:
- Bootstrap script will enable APIs automatically
- Or manually: `gcloud services enable <api-name>`

### Issue: "Bucket name already taken"

**Solution**:
- GCS bucket names are globally unique
- Choose a different name in `.env`
- Recommended format: `${PROJECT_ID}-etpgt-data`

### Issue: "Region not available"

**Solution**:
- List available regions: `gcloud compute regions list`
- Choose a region with L4 GPU availability (for Phase 4+)
- Recommended: `us-central1`, `us-west1`, `europe-west4`

---

## Next Steps (Phase 2)

After Phase 1 gate is met:

1. Download RetailRocket dataset
2. Implement sessionization logic
3. Create temporal splits (70/15/15)
4. Build co-event graph
5. Upload processed data to GCS

**Phase 2 Gate**: Unit tests green for data prep and splits

---

## Resources

- [GCS Documentation](https://cloud.google.com/storage/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [Service Accounts Documentation](https://cloud.google.com/iam/docs/service-accounts)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)

---

## Notes

- **Idempotency**: All scripts are idempotent and safe to re-run
- **Cost**: GCS and AR have minimal costs for storage; no compute costs in Phase 1
- **Security**: Service Account follows principle of least privilege
- **Cleanup**: To delete resources, see `scripts/gcp/99_cleanup.sh` (to be created if needed)

---

**Last Updated**: 2025-01-20

