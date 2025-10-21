# GCP Workload Identity Federation Setup

This guide provides step-by-step instructions to configure GitHub Actions OIDC authentication with GCP, eliminating the need for long-lived service account keys.

## Prerequisites

- GCP project with billing enabled
- GitHub repository
- `gcloud` CLI installed and authenticated
- Owner or Security Admin role in GCP project

## Architecture

```
GitHub Actions → OIDC Token → Workload Identity Pool → Service Account → GCP Resources
```

**Benefits**:
- No long-lived credentials
- Automatic token rotation
- Fine-grained access control
- Audit trail via Cloud Logging

## Step 1: Enable Required APIs

```bash
gcloud services enable \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project=${GCP_PROJECT_ID}
```

## Step 2: Create Workload Identity Pool

```bash
# Create pool
gcloud iam workload-identity-pools create "github-pool" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Get pool ID
POOL_ID=$(gcloud iam workload-identity-pools describe "github-pool" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --format="value(name)")

echo "Pool ID: ${POOL_ID}"
```

## Step 3: Create Workload Identity Provider

```bash
# Create provider for GitHub
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Get provider ID
PROVIDER_ID=$(gcloud iam workload-identity-pools providers describe "github-provider" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)")

echo "Provider ID: ${PROVIDER_ID}"
```

## Step 4: Create Service Account (if not already created)

```bash
# Create service account
gcloud iam service-accounts create "${SA_NAME}" \
  --project="${GCP_PROJECT_ID}" \
  --display-name="ETP-GT Service Account"

# Get service account email
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
echo "Service Account: ${SA_EMAIL}"
```

## Step 5: Grant Service Account Permissions

```bash
# Grant roles for training and serving
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser"
```

## Step 6: Allow GitHub to Impersonate Service Account

```bash
# Replace <ORG> and <REPO> with your GitHub org/username and repo name
ORG_NAME="<your-github-org-or-username>"
REPO_NAME="etp-gt"

# Allow specific repository to impersonate service account
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --project="${GCP_PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${ORG_NAME}/${REPO_NAME}"
```

**Note**: This grants access to ALL workflows in the repository. For finer control, use:

```bash
# Restrict to specific branch (e.g., main)
--member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${ORG_NAME}/${REPO_NAME}/ref/refs/heads/main"

# Restrict to specific environment (e.g., production)
--member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${ORG_NAME}/${REPO_NAME}/environment/production"
```

## Step 7: Get Workload Identity Provider Resource Name

```bash
# Full provider resource name for GitHub Actions
WORKLOAD_IDENTITY_PROVIDER="projects/${GCP_PROJECT_ID}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"

echo "Workload Identity Provider: ${WORKLOAD_IDENTITY_PROVIDER}"
```

## Step 8: Configure GitHub Secrets

Add the following secrets to your GitHub repository:

1. Go to: `https://github.com/<ORG>/<REPO>/settings/secrets/actions`
2. Click "New repository secret"
3. Add each secret:

| Secret Name | Value | Example |
|-------------|-------|---------|
| `GCP_PROJECT_ID` | Your GCP project ID | `my-project-123` |
| `GCP_REGION` | GCP region | `us-central1` |
| `GCS_BUCKET` | GCS bucket name | `gs://etp-gt-abc123` |
| `AR_REPO` | Artifact Registry repo | `us-central1-docker.pkg.dev/my-project-123/etp-gt` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Full provider resource name | `projects/123.../providers/github-provider` |
| `GCP_SERVICE_ACCOUNT` | Service account email | `etp-gt-sa@my-project-123.iam.gserviceaccount.com` |

## Step 9: Test Authentication

Create a test workflow (`.github/workflows/test-gcp-auth.yaml`):

```yaml
name: Test GCP Auth

on:
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write

    steps:
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Test GCS access
        run: |
          gsutil ls gs://${{ secrets.GCS_BUCKET }}

      - name: Test Artifact Registry access
        run: |
          gcloud artifacts repositories list --location=${{ secrets.GCP_REGION }}
```

Run the workflow manually and verify it succeeds.

## Step 10: Update Production Workflows

Ensure all workflows use the OIDC authentication pattern:

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write  # Required for OIDC

    steps:
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      # ... rest of workflow
```

## Troubleshooting

### Error: "Permission denied"

**Cause**: Service account lacks required roles.

**Fix**:
```bash
# Check current roles
gcloud projects get-iam-policy ${GCP_PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SA_EMAIL}"

# Add missing role
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/<missing-role>"
```

### Error: "Workload identity pool does not exist"

**Cause**: Pool or provider not created correctly.

**Fix**:
```bash
# List pools
gcloud iam workload-identity-pools list --location=global --project=${GCP_PROJECT_ID}

# List providers
gcloud iam workload-identity-pools providers list \
  --workload-identity-pool=github-pool \
  --location=global \
  --project=${GCP_PROJECT_ID}
```

### Error: "Token request failed"

**Cause**: Incorrect attribute mapping or issuer URI.

**Fix**:
```bash
# Verify provider configuration
gcloud iam workload-identity-pools providers describe github-provider \
  --workload-identity-pool=github-pool \
  --location=global \
  --project=${GCP_PROJECT_ID}
```

### Error: "Subject does not match"

**Cause**: Repository filter too restrictive.

**Fix**:
```bash
# Remove existing binding
gcloud iam service-accounts remove-iam-policy-binding "${SA_EMAIL}" \
  --project="${GCP_PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${ORG_NAME}/${REPO_NAME}"

# Add broader binding (all repos in org)
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --project="${GCP_PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository_owner/${ORG_NAME}"
```

## Security Best Practices

1. **Least privilege**: Grant only required roles to service account
2. **Scope restrictions**: Use branch/environment filters in attribute mapping
3. **Audit logs**: Enable Cloud Audit Logs for IAM
4. **Rotation**: No rotation needed (tokens are short-lived)
5. **Monitoring**: Set up alerts for unusual authentication patterns

## References

- [GitHub OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [GCP Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
- [google-github-actions/auth](https://github.com/google-github-actions/auth)

## Outputs for Next Steps

After completing this setup, you should have:

- ✅ Workload Identity Pool: `github-pool`
- ✅ Workload Identity Provider: `github-provider`
- ✅ Service Account: `${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com`
- ✅ GitHub Secrets configured
- ✅ Test workflow passing

You can now proceed with Phase 1 (GCP Bootstrap) and Phase 8 (GitHub Actions).

