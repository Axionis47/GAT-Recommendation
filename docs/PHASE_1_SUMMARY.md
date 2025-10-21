# Phase 1: GCP Bootstrap - Summary

**Status**: ✅ **READY FOR EXECUTION**  
**Date**: 2025-01-20  
**Gate**: Infrastructure ready (bucket + AR + SA + OIDC docs)

---

## Executive Summary

Phase 1 provides complete automation for bootstrapping GCP infrastructure. All scripts are idempotent, validated, and ready to execute once you configure your GCP project.

### Key Deliverables

✅ **Environment validation script** (`00_validate_env.sh`)  
✅ **Infrastructure bootstrap script** (`01_bootstrap.sh`)  
✅ **Comprehensive setup guide** (`GCP_SETUP_GUIDE.md`)  
✅ **Phase 1 checklist** (`PHASE_1_CHECKLIST.md`)  
✅ **Makefile targets** (`gcp-validate`, `gcp-bootstrap`)  
✅ **Updated README** with GCP setup instructions

---

## What's Been Built

### 1. Scripts

**`scripts/gcp/00_validate_env.sh`** (165 lines):
- Validates all required environment variables
- Checks gcloud CLI installation and authentication
- Verifies project access and billing status
- Lists required APIs (will be enabled by bootstrap)
- Checks IAM permissions
- Validates region availability
- **Status**: ✅ Syntactically valid, ready to run

**`scripts/gcp/01_bootstrap.sh`** (230 lines):
- Enables 7 required GCP APIs
- Creates GCS bucket with:
  - Versioning enabled
  - Lifecycle policy (delete old versions after 30 days)
  - 8 subdirectories (data/raw, artifacts/baselines, etc.)
- Creates Artifact Registry repository (Docker format)
- Creates Service Account with 5 minimal IAM roles
- Configures Docker authentication
- **Idempotent**: Safe to re-run
- **Status**: ✅ Syntactically valid, ready to run

### 2. Documentation

**`docs/GCP_SETUP_GUIDE.md`** (300+ lines):
- Complete step-by-step setup guide
- Prerequisites checklist
- Environment variable configuration
- Validation and bootstrap instructions
- Cost estimation ($20-50 for full project)
- Troubleshooting section
- Next steps for Phase 2+

**`docs/PHASE_1_CHECKLIST.md`** (250+ lines):
- Detailed checklist for Phase 1 execution
- Prerequisites validation
- Resource creation checklist
- Verification commands
- Gate criteria
- Troubleshooting guide

**`docs/GCP_OIDC_SETUP.md`** (from Phase 0):
- GitHub Actions OIDC configuration
- Workload Identity Federation setup
- Will be used in Phase 8

### 3. Makefile Updates

Added two new targets:
```makefile
make gcp-validate    # Validate GCP environment
make gcp-bootstrap   # Bootstrap infrastructure
```

Updated help text and `.PHONY` declarations.

### 4. README Updates

Added validation step to GCP setup section:
```bash
1. make gcp-validate    # NEW
2. make gcp-bootstrap
3. make data
4. make gcp-train
5. make gcp-deploy
```

---

## Resources Created (After Execution)

### GCS Bucket
- **Name**: `gs://${GCS_BUCKET}`
- **Location**: `${GCP_REGION}`
- **Features**:
  - Versioning enabled
  - Lifecycle policy (30-day retention for old versions)
- **Structure**:
  ```
  gs://${GCS_BUCKET}/
  ├── data/
  │   ├── raw/
  │   ├── interim/
  │   └── processed/
  ├── artifacts/
  │   ├── baselines/
  │   ├── etpgt/
  │   ├── ablations/
  │   └── exports/
  └── logs/
  ```

### Artifact Registry
- **Name**: `${AR_REPO}`
- **Location**: `${GCP_REGION}`
- **Format**: Docker
- **Full path**: `${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}`

### Service Account
- **Name**: `${SA_NAME}`
- **Email**: `${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com`
- **IAM Roles**:
  - `roles/storage.objectAdmin` - GCS read/write
  - `roles/artifactregistry.reader` - Pull Docker images
  - `roles/aiplatform.user` - Submit Vertex AI jobs
  - `roles/logging.logWriter` - Write logs
  - `roles/monitoring.metricWriter` - Write metrics

### APIs Enabled
1. `storage.googleapis.com` - Cloud Storage
2. `artifactregistry.googleapis.com` - Artifact Registry
3. `aiplatform.googleapis.com` - Vertex AI
4. `compute.googleapis.com` - Compute Engine
5. `run.googleapis.com` - Cloud Run
6. `iam.googleapis.com` - IAM
7. `iamcredentials.googleapis.com` - IAM Credentials

---

## Execution Instructions

### Prerequisites

1. **GCP Project**:
   - Project created
   - Billing enabled
   - You have Owner or Editor role

2. **Local Tools**:
   - gcloud CLI installed
   - Authenticated: `gcloud auth login`

3. **Environment**:
   - `.env` file created from `.env.example`
   - All required variables configured

### Step-by-Step

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your GCP project details

# 2. Validate environment
make gcp-validate
# Expected: ✓ All prerequisites met!

# 3. Bootstrap infrastructure
make gcp-bootstrap
# Expected: ✓ All resources created

# 4. Verify resources
gsutil ls -r gs://${GCS_BUCKET}
gcloud artifacts repositories list
gcloud iam service-accounts list
```

### Verification Commands

```bash
# Check bucket
gsutil ls gs://${GCS_BUCKET}

# Check Artifact Registry
gcloud artifacts repositories describe ${AR_REPO} --location=${GCP_REGION}

# Check Service Account
gcloud iam service-accounts describe ${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com

# Check IAM roles
gcloud projects get-iam-policy ${GCP_PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
```

---

## Gate Criteria

Phase 1 is complete when:

- [x] **Scripts created** and syntactically valid ✅
- [ ] **GCS Bucket exists** with proper structure
- [ ] **Artifact Registry exists** and is accessible
- [ ] **Service Account exists** with minimal IAM roles
- [ ] **All required APIs enabled**
- [x] **Documentation complete** ✅
- [ ] **Can authenticate Docker** to Artifact Registry

**Current Status**: 🟡 **READY FOR EXECUTION** (scripts ready, awaiting user to run)

---

## Cost Estimate

### Phase 1 Only
- **GCS**: ~$0.02/GB/month (minimal, empty bucket)
- **Artifact Registry**: ~$0.10/GB/month (minimal, no images yet)
- **Total**: < $1/month

### Full Project (Phases 1-9)
- **Storage**: ~$2/month (data + artifacts)
- **Training**: ~$14 (10 runs × 2 hours × $0.70/hour)
- **Serving**: ~$5/month (1M requests)
- **Total**: ~$20-50 (depending on experimentation)

---

## Next Steps

### Immediate (User Action Required)

1. **Configure `.env`**:
   ```bash
   cp .env.example .env
   # Edit with your GCP project details
   ```

2. **Run validation**:
   ```bash
   make gcp-validate
   ```

3. **Run bootstrap**:
   ```bash
   make gcp-bootstrap
   ```

4. **Verify resources** (see commands above)

### After Phase 1 Gate

**Phase 2: Data Prep & Splits**
- Download RetailRocket dataset
- Implement sessionization (30-min gap, min length 3)
- Create temporal splits (70/15/15 with blackout)
- Build co-event graph (±5 steps)
- Upload to GCS
- **Gate**: Unit tests green

---

## Troubleshooting

### Common Issues

1. **"Permission denied"**
   - Ensure you have Owner or Editor role
   - Check: `gcloud projects get-iam-policy ${GCP_PROJECT_ID}`

2. **"Bucket name already taken"**
   - GCS bucket names are globally unique
   - Use format: `${PROJECT_ID}-etpgt-data`

3. **"Billing not enabled"**
   - Enable at: https://console.cloud.google.com/billing

4. **"API not enabled"**
   - Bootstrap script will enable automatically
   - Or manually: `gcloud services enable <api-name>`

### Getting Help

- Review `docs/GCP_SETUP_GUIDE.md` for detailed instructions
- Review `docs/PHASE_1_CHECKLIST.md` for step-by-step checklist
- Check GCP Console: https://console.cloud.google.com

---

## Files Modified/Created

### Created (6 files)
- `scripts/gcp/00_validate_env.sh` (165 lines)
- `scripts/gcp/01_bootstrap.sh` (230 lines)
- `docs/GCP_SETUP_GUIDE.md` (300+ lines)
- `docs/PHASE_1_CHECKLIST.md` (250+ lines)
- `docs/PHASE_1_SUMMARY.md` (this file)

### Modified (2 files)
- `Makefile` (added `gcp-validate` target)
- `README.md` (added validation step)

---

## Validation Results

### Script Syntax Check
```bash
bash -n scripts/gcp/00_validate_env.sh
bash -n scripts/gcp/01_bootstrap.sh
# ✓ Scripts are syntactically valid
```

### Idempotency
Both scripts are designed to be idempotent:
- Check if resource exists before creating
- Skip if already exists
- Safe to re-run multiple times

### Security
- Service Account follows principle of least privilege
- Only 5 minimal IAM roles granted
- No long-lived keys created (will use OIDC in Phase 8)

---

## Sign-off

**Phase Owner**: ML Team  
**Status**: 🟡 **READY FOR EXECUTION**  
**Scripts Ready**: ✅ YES  
**Documentation Complete**: ✅ YES  
**Awaiting**: User to configure `.env` and run scripts

---

**End of Phase 1 Summary**

