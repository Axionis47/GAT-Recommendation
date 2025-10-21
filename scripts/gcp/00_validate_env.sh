#!/usr/bin/env bash
# scripts/gcp/00_validate_env.sh
# Validate GCP environment and prerequisites before bootstrap

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

echo "=== ETP-GT GCP Environment Validation ==="
echo ""

# Check required variables
echo "Checking required environment variables..."
MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING_VARS+=("$var")
        echo -e "${RED}✗${NC} $var is not set"
    else
        echo -e "${GREEN}✓${NC} $var = ${!var}"
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "\n${RED}ERROR: Missing required environment variables:${NC}"
    printf '%s\n' "${MISSING_VARS[@]}"
    echo "Please configure these in your .env file."
    exit 1
fi

echo ""

# Check gcloud CLI
echo "Checking gcloud CLI..."
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}✗ gcloud CLI not found${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
else
    GCLOUD_VERSION=$(gcloud version --format="value(version)")
    echo -e "${GREEN}✓${NC} gcloud CLI installed (version: $GCLOUD_VERSION)"
fi

# Check gcloud authentication
echo "Checking gcloud authentication..."
CURRENT_ACCOUNT=$(gcloud config get-value account 2>/dev/null || echo "")
if [ -z "$CURRENT_ACCOUNT" ]; then
    echo -e "${RED}✗ Not authenticated with gcloud${NC}"
    echo "Run: gcloud auth login"
    exit 1
else
    echo -e "${GREEN}✓${NC} Authenticated as: $CURRENT_ACCOUNT"
fi

# Check project access
echo "Checking GCP project access..."
if ! gcloud projects describe "$GCP_PROJECT_ID" &> /dev/null; then
    echo -e "${RED}✗ Cannot access project: $GCP_PROJECT_ID${NC}"
    echo "Ensure the project exists and you have access."
    exit 1
else
    echo -e "${GREEN}✓${NC} Project accessible: $GCP_PROJECT_ID"
fi

# Check current project setting
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
if [ "$CURRENT_PROJECT" != "$GCP_PROJECT_ID" ]; then
    echo -e "${YELLOW}⚠${NC} Current project ($CURRENT_PROJECT) differs from GCP_PROJECT_ID ($GCP_PROJECT_ID)"
    echo "Setting project to: $GCP_PROJECT_ID"
    gcloud config set project "$GCP_PROJECT_ID"
fi

# Check billing
echo "Checking billing status..."
BILLING_ENABLED=$(gcloud beta billing projects describe "$GCP_PROJECT_ID" \
    --format="value(billingEnabled)" 2>/dev/null || echo "false")
if [ "$BILLING_ENABLED" != "True" ]; then
    echo -e "${RED}✗ Billing not enabled for project${NC}"
    echo "Enable billing at: https://console.cloud.google.com/billing"
    exit 1
else
    echo -e "${GREEN}✓${NC} Billing enabled"
fi

# Check required APIs (will be enabled by bootstrap script)
echo ""
echo "Checking required APIs (will be enabled if needed)..."
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
        echo -e "${GREEN}✓${NC} $api (enabled)"
    else
        echo -e "${YELLOW}○${NC} $api (will be enabled)"
    fi
done

# Check IAM permissions
echo ""
echo "Checking IAM permissions..."
REQUIRED_ROLES=(
    "roles/storage.admin"
    "roles/artifactregistry.admin"
    "roles/iam.serviceAccountAdmin"
    "roles/iam.serviceAccountKeyAdmin"
    "roles/serviceusage.serviceUsageAdmin"
)

echo -e "${YELLOW}Note: Permission check requires elevated privileges. Skipping detailed check.${NC}"
echo "Ensure you have the following roles or equivalent permissions:"
for role in "${REQUIRED_ROLES[@]}"; do
    echo "  - $role"
done

# Check region validity
echo ""
echo "Checking region..."
if gcloud compute regions list --filter="name=$GCP_REGION" --format="value(name)" 2>/dev/null | grep -q "$GCP_REGION"; then
    echo -e "${GREEN}✓${NC} Region valid: $GCP_REGION"
else
    echo -e "${RED}✗ Invalid region: $GCP_REGION${NC}"
    echo "List valid regions: gcloud compute regions list"
    exit 1
fi

# Summary
echo ""
echo "=== Validation Summary ==="
echo -e "${GREEN}✓ All prerequisites met!${NC}"
echo ""
echo "You can now run the bootstrap script:"
echo "  make gcp-bootstrap"
echo ""
echo "Or manually:"
echo "  bash scripts/gcp/01_bootstrap.sh"
echo ""

