# GCP Setup Guide for ETP-GT

This guide walks you through setting up Google Cloud Platform infrastructure for the ETP-GT project.

---

## Overview

The ETP-GT project uses the following GCP services:

- **Cloud Storage (GCS)**: Store raw data, processed datasets, model artifacts, and logs
- **Artifact Registry**: Store Docker images for training and inference
- **Vertex AI**: Train models on GPUs (n1-standard-8 + L4)
- **Cloud Run**: Serve inference API with auto-scaling
- **IAM**: Service accounts for secure access

---

## Prerequisites

### 1. GCP Account and Project

- [ ] Google Cloud account created
- [ ] GCP project created (or use existing)
- [ ] Billing enabled on project
- [ ] Note your **Project ID** (e.g., `my-project-123`)

**Create a project**:
```bash
gcloud projects create my-etpgt-project --name="ETP-GT Recommendations"
```

**Enable billing**:
- Visit: https://console.cloud.google.com/billing
- Link billing account to your project

### 2. Local Tools

- [ ] **gcloud CLI** installed
  - Download: https://cloud.google.com/sdk/docs/install
  - Verify: `gcloud --version`

- [ ] **Python 3.11+** installed
  - Verify: `python3 --version`

- [ ] **Git** installed
  - Verify: `git --version`

### 3. IAM Permissions

You need one of the following:
- **Owner** role on the project (recommended for initial setup)
- **Editor** role + **Service Account Admin** role
- Custom role with these permissions:
  - `storage.buckets.create`
  - `artifactregistry.repositories.create`
  - `iam.serviceAccounts.create`
  - `resourcemanager.projects.setIamPolicy`
  - `serviceusage.services.enable`

---

## Step-by-Step Setup

### Step 1: Authenticate with GCP

```bash
# Login to GCP
gcloud auth login

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Verify authentication
gcloud auth list
gcloud config list
```

### Step 2: Clone Repository and Configure

```bash
# Clone repository
git clone https://github.com/your-org/etp-gt.git
cd etp-gt

# Setup Python environment
make setup
source .venv/bin/activate

# Create .env file
cp .env.example .env
```

### Step 3: Configure Environment Variables

Edit `.env` with your project details:

```bash
# Required GCP Configuration
GCP_PROJECT_ID=your-project-id-here        # Your GCP project ID
GCP_REGION=us-central1                     # Choose region with L4 GPU availability
GCS_BUCKET=your-project-id-etpgt-data      # Globally unique bucket name
AR_REPO=etpgt                              # Artifact Registry repository name
SA_NAME=etpgt-sa                           # Service account name

# Vertex AI Configuration
VAI_MACHINE=n1-standard-8                  # Machine type for training
VAI_ACCELERATOR=L4                         # GPU type (L4 recommended)
VAI_ACCELERATOR_COUNT=2                    # Number of GPUs

# Model Configuration
CANDIDATES_K=200                           # Number of candidates for inference
LAST_N=10                                  # Last N items to consider
LATENCY_P95_MS=120                         # Target p95 latency in ms

# Python Version
PYTHON=3.12                                # Your Python version
```

**Important Notes**:
- `GCS_BUCKET` must be globally unique across all GCP projects
- Recommended format: `${PROJECT_ID}-etpgt-data`
- Choose a region with L4 GPU availability (e.g., `us-central1`, `us-west1`, `europe-west4`)

### Step 4: Validate Environment

```bash
# Run validation script
make gcp-validate
```

**Expected Output**:
```
=== ETP-GT GCP Environment Validation ===

Checking required environment variables...
✓ GCP_PROJECT_ID = your-project-id
✓ GCP_REGION = us-central1
✓ GCS_BUCKET = your-project-id-etpgt-data
✓ AR_REPO = etpgt
✓ SA_NAME = etpgt-sa

Checking gcloud CLI...
✓ gcloud CLI installed (version: 456.0.0)

Checking gcloud authentication...
✓ Authenticated as: you@example.com

Checking GCP project access...
✓ Project accessible: your-project-id

Checking billing status...
✓ Billing enabled

Checking required APIs (will be enabled if needed)...
✓ storage.googleapis.com (enabled)
○ artifactregistry.googleapis.com (will be enabled)
○ aiplatform.googleapis.com (will be enabled)
...

=== Validation Summary ===
✓ All prerequisites met!

You can now run the bootstrap script:
  make gcp-bootstrap
```

**If validation fails**:
- Review error messages carefully
- Fix issues (install gcloud, authenticate, enable billing, etc.)
- Re-run `make gcp-validate`

### Step 5: Bootstrap GCP Infrastructure

```bash
# Run bootstrap script
make gcp-bootstrap
```

**What this does**:
1. Enables required GCP APIs (Storage, Artifact Registry, Vertex AI, Cloud Run, IAM)
2. Creates GCS bucket with:
   - Versioning enabled
   - Lifecycle policy (delete old versions after 30 days)
   - Directory structure for data and artifacts
3. Creates Artifact Registry repository for Docker images
4. Creates Service Account with minimal IAM roles:
   - `roles/storage.objectAdmin` - Read/write GCS
   - `roles/artifactregistry.reader` - Pull Docker images
   - `roles/aiplatform.user` - Submit Vertex AI jobs
   - `roles/logging.logWriter` - Write logs
   - `roles/monitoring.metricWriter` - Write metrics
5. Configures Docker authentication for Artifact Registry

**Expected Output**:
```
=== ETP-GT GCP Bootstrap ===

Project:           your-project-id
Region:            us-central1
GCS Bucket:        your-project-id-etpgt-data
Artifact Registry: etpgt
Service Account:   etpgt-sa

Setting GCP project...

Enabling required APIs...
✓ storage.googleapis.com already enabled
Enabling artifactregistry.googleapis.com...
✓ artifactregistry.googleapis.com enabled
...

Creating GCS bucket...
Creating bucket: gs://your-project-id-etpgt-data
✓ Bucket created with versioning and lifecycle policy

Creating bucket structure...
✓ Created data/raw/
✓ Created data/interim/
✓ Created data/processed/
✓ Created artifacts/baselines/
✓ Created artifacts/etpgt/
✓ Created artifacts/ablations/
✓ Created artifacts/exports/
✓ Created logs/

Creating Artifact Registry repository...
Creating repository: etpgt
✓ Repository created

Creating Service Account...
Creating service account: etpgt-sa
✓ Service Account created

Granting IAM roles to Service Account...
Granting roles/storage.objectAdmin...
✓ roles/storage.objectAdmin granted
...

Configuring Docker authentication...
✓ Docker authentication configured

=== Bootstrap Complete ===

✓ GCS Bucket:        gs://your-project-id-etpgt-data
✓ Artifact Registry: us-central1-docker.pkg.dev/your-project-id/etpgt
✓ Service Account:   etpgt-sa@your-project-id.iam.gserviceaccount.com

Next steps:
  1. Review docs/GCP_OIDC_SETUP.md for GitHub Actions OIDC setup
  2. Update configs/*.yaml with actual GCS paths
  3. Proceed to Phase 2: Data Prep & Splits
```

**Script is idempotent**: Safe to re-run if interrupted or if you need to verify resources.

### Step 6: Verify Resources

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

---

## Cost Estimation

### Phase 1 (Bootstrap)
- **GCS**: ~$0.02/GB/month for storage (minimal during setup)
- **Artifact Registry**: ~$0.10/GB/month for storage (minimal during setup)
- **Total**: < $1/month for empty infrastructure

### Phase 4+ (Training)
- **Vertex AI**: ~$0.70/hour for n1-standard-8 + 2x L4 GPUs
- **Example**: 10 training runs × 2 hours = $14

### Phase 7+ (Serving)
- **Cloud Run**: Pay-per-request, ~$0.40/million requests
- **Example**: 1M requests/month = $0.40

**Total estimated cost for full project**: $20-50 (depending on experimentation)

---

## Troubleshooting

### "Permission denied" errors

**Cause**: Insufficient IAM permissions

**Solution**:
```bash
# Check your roles
gcloud projects get-iam-policy ${GCP_PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:user:$(gcloud config get-value account)"

# Request Owner or Editor role from project admin
```

### "Bucket name already taken"

**Cause**: GCS bucket names are globally unique

**Solution**:
- Choose a different name in `.env`
- Recommended: `${PROJECT_ID}-etpgt-data-${RANDOM_SUFFIX}`

### "API not enabled"

**Cause**: Required API not enabled

**Solution**:
```bash
# Bootstrap script will enable automatically, or manually:
gcloud services enable storage.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### "Billing not enabled"

**Cause**: Project doesn't have billing enabled

**Solution**:
- Visit: https://console.cloud.google.com/billing
- Link billing account to project

---

## Next Steps

After successful bootstrap:

1. **Phase 2**: Data Prep & Splits
   - Download RetailRocket dataset
   - Implement sessionization
   - Create temporal splits
   - Upload to GCS

2. **Phase 3**: Sampler & Encodings
   - Implement TemporalPathSampler
   - Implement LapPE and HybridPE
   - Write unit tests

3. **Phase 4**: Baselines on Vertex AI
   - Train GraphSAGE, GAT, GraphTransformer
   - Establish performance baseline

---

## Resources

- [GCP Console](https://console.cloud.google.com)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [IAM Documentation](https://cloud.google.com/iam/docs)

---

**Last Updated**: 2025-01-20

