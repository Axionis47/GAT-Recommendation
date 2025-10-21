#!/usr/bin/env bash
# scripts/gcp/01_bootstrap.sh
# Idempotent GCP infrastructure bootstrap for ETP-GT
# Creates: GCS bucket, Artifact Registry, Service Account with minimal IAM roles

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo -e "${RED}ERROR: .env file not found. Copy .env.example to .env and configure.${NC}"
    exit 1
fi

# Required environment variables
REQUIRED_VARS=(
    "GCP_PROJECT_ID"
    "GCP_REGION"
    "GCS_BUCKET"
    "AR_REPO"
    "SA_NAME"
)

# Validate required variables
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo -e "${RED}ERROR: $var is not set in .env${NC}"
        exit 1
    fi
done

# Derived variables
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
AR_LOCATION="${GCP_REGION}"

echo "=== ETP-GT GCP Bootstrap ==="
echo ""
echo "Project:           $GCP_PROJECT_ID"
echo "Region:            $GCP_REGION"
echo "GCS Bucket:        $GCS_BUCKET"
echo "Artifact Registry: $AR_REPO"
echo "Service Account:   $SA_NAME"
echo ""

# Set project
echo -e "${BLUE}Setting GCP project...${NC}"
gcloud config set project "$GCP_PROJECT_ID"

# Enable required APIs
echo ""
echo -e "${BLUE}Enabling required APIs...${NC}"
REQUIRED_APIS=(
    "storage.googleapis.com"
    "artifactregistry.googleapis.com"
    "aiplatform.googleapis.com"
    "compute.googleapis.com"
    "run.googleapis.com"
    "iam.googleapis.com"
    "iamcredentials.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" 2>/dev/null | grep -q "$api"; then
        echo -e "${GREEN}✓${NC} $api already enabled"
    else
        echo -e "${YELLOW}Enabling $api...${NC}"
        gcloud services enable "$api"
        echo -e "${GREEN}✓${NC} $api enabled"
    fi
done

# Create GCS bucket
echo ""
echo -e "${BLUE}Creating GCS bucket...${NC}"
if gsutil ls -b "gs://$GCS_BUCKET" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Bucket gs://$GCS_BUCKET already exists"
else
    echo "Creating bucket: gs://$GCS_BUCKET"
    gsutil mb -p "$GCP_PROJECT_ID" -l "$GCP_REGION" -b on "gs://$GCS_BUCKET"
    
    # Enable versioning
    gsutil versioning set on "gs://$GCS_BUCKET"
    
    # Set lifecycle policy (delete old versions after 30 days)
    cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "numNewerVersions": 3,
          "daysSinceNoncurrentTime": 30
        }
      }
    ]
  }
}
EOF
    gsutil lifecycle set /tmp/lifecycle.json "gs://$GCS_BUCKET"
    rm /tmp/lifecycle.json
    
    echo -e "${GREEN}✓${NC} Bucket created with versioning and lifecycle policy"
fi

# Create bucket subdirectories (using marker objects)
echo "Creating bucket structure..."
BUCKET_DIRS=(
    "data/raw"
    "data/interim"
    "data/processed"
    "artifacts/baselines"
    "artifacts/etpgt"
    "artifacts/ablations"
    "artifacts/exports"
    "logs"
)

for dir in "${BUCKET_DIRS[@]}"; do
    if ! gsutil ls "gs://$GCS_BUCKET/$dir/" &> /dev/null; then
        echo "" | gsutil cp - "gs://$GCS_BUCKET/$dir/.gitkeep"
        echo -e "${GREEN}✓${NC} Created $dir/"
    else
        echo -e "${GREEN}✓${NC} $dir/ already exists"
    fi
done

# Create Artifact Registry repository
echo ""
echo -e "${BLUE}Creating Artifact Registry repository...${NC}"
if gcloud artifacts repositories describe "$AR_REPO" \
    --location="$AR_LOCATION" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Repository $AR_REPO already exists"
else
    echo "Creating repository: $AR_REPO"
    gcloud artifacts repositories create "$AR_REPO" \
        --repository-format=docker \
        --location="$AR_LOCATION" \
        --description="ETP-GT Docker images for training and inference"
    echo -e "${GREEN}✓${NC} Repository created"
fi

# Create Service Account
echo ""
echo -e "${BLUE}Creating Service Account...${NC}"
if gcloud iam service-accounts describe "$SA_EMAIL" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Service Account $SA_NAME already exists"
else
    echo "Creating service account: $SA_NAME"
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="ETP-GT Training and Serving SA" \
        --description="Service account for Vertex AI training and Cloud Run serving"
    echo -e "${GREEN}✓${NC} Service Account created"
fi

# Grant IAM roles to Service Account
echo ""
echo -e "${BLUE}Granting IAM roles to Service Account...${NC}"

# Minimal required roles
ROLES=(
    "roles/storage.objectAdmin"           # GCS read/write
    "roles/artifactregistry.reader"       # Pull Docker images
    "roles/aiplatform.user"               # Vertex AI jobs
    "roles/logging.logWriter"             # Write logs
    "roles/monitoring.metricWriter"       # Write metrics
)

for role in "${ROLES[@]}"; do
    # Check if binding already exists
    if gcloud projects get-iam-policy "$GCP_PROJECT_ID" \
        --flatten="bindings[].members" \
        --filter="bindings.role:$role AND bindings.members:serviceAccount:$SA_EMAIL" \
        --format="value(bindings.role)" 2>/dev/null | grep -q "$role"; then
        echo -e "${GREEN}✓${NC} $role already granted"
    else
        echo "Granting $role..."
        gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$role" \
            --condition=None \
            --quiet
        echo -e "${GREEN}✓${NC} $role granted"
    fi
done

# Configure Docker authentication for Artifact Registry
echo ""
echo -e "${BLUE}Configuring Docker authentication...${NC}"
gcloud auth configure-docker "${AR_LOCATION}-docker.pkg.dev" --quiet
echo -e "${GREEN}✓${NC} Docker authentication configured"

# Summary
echo ""
echo "=== Bootstrap Complete ==="
echo ""
echo -e "${GREEN}✓ GCS Bucket:${NC}        gs://$GCS_BUCKET"
echo -e "${GREEN}✓ Artifact Registry:${NC} $AR_LOCATION-docker.pkg.dev/$GCP_PROJECT_ID/$AR_REPO"
echo -e "${GREEN}✓ Service Account:${NC}   $SA_EMAIL"
echo ""
echo "Next steps:"
echo "  1. Review docs/GCP_OIDC_SETUP.md for GitHub Actions OIDC setup"
echo "  2. Update configs/*.yaml with actual GCS paths"
echo "  3. Proceed to Phase 2: Data Prep & Splits"
echo ""
echo "Useful commands:"
echo "  # List bucket contents"
echo "  gsutil ls -r gs://$GCS_BUCKET"
echo ""
echo "  # List Docker images"
echo "  gcloud artifacts docker images list $AR_LOCATION-docker.pkg.dev/$GCP_PROJECT_ID/$AR_REPO"
echo ""
echo "  # Test service account"
echo "  gcloud iam service-accounts get-iam-policy $SA_EMAIL"
echo ""

