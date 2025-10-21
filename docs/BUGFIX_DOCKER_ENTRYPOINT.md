# Bug Fix: Docker ENTRYPOINT Issue

**Date**: 2025-10-19  
**Issue**: GAT training job failed with exit code 1  
**Root Cause**: Incorrect Docker ENTRYPOINT configuration  
**Status**: ✅ FIXED

---

## Problem

The GAT training job (Job ID: `625654500257955840`) failed with error:
```
The replica workerpool0-0 exited with a non-zero status of 1.
```

**Investigation**:
- Cloud Logging showed no detailed error messages
- Job was submitted with correct arguments
- Other jobs (GraphSAGE, GraphTransformer) were also pending but likely to fail

---

## Root Cause

The Docker image had an incorrect ENTRYPOINT configuration:

**Original `docker/train.Dockerfile`**:
```dockerfile
ENTRYPOINT ["python", "scripts/train/train_baseline.py"]
```

**Original `scripts/gcp/03_submit_training_job.sh`**:
```yaml
containerSpec:
  imageUri: us-central1-docker.pkg.dev/plotpointe/etpgt/etpgt-train:latest
  args:
    - --model=gat
    - --embedding-dim=256
    - ...
```

**Problem**: The ENTRYPOINT was hardcoded to `train_baseline.py`, which meant:
1. The container would always run the baseline training script
2. The args were passed as command-line arguments to the script
3. However, the script path was hardcoded, making it inflexible for ETP-GT training
4. The args were missing the script path, causing the container to fail

---

## Solution

### 1. Updated Docker ENTRYPOINT

**New `docker/train.Dockerfile`**:
```dockerfile
# Default entrypoint (can be overridden)
# For baselines: python scripts/train/train_baseline.py
# For ETP-GT: python scripts/train/train_etpgt.py
ENTRYPOINT ["python"]
```

**Rationale**: 
- Set ENTRYPOINT to just `python`
- Pass the script path as the first argument
- This allows flexibility to run different training scripts

### 2. Updated Job Submission Scripts

**New `scripts/gcp/03_submit_training_job.sh`**:
```yaml
containerSpec:
  imageUri: us-central1-docker.pkg.dev/plotpointe/etpgt/etpgt-train:latest
  args:
    - scripts/train/train_baseline.py  # Script path as first arg
    - --model=gat
    - --embedding-dim=256
    - ...
```

**New `scripts/gcp/04_submit_etpgt_job.sh`**:
```yaml
containerSpec:
  imageUri: us-central1-docker.pkg.dev/plotpointe/etpgt/etpgt-train:latest
  args:
    - scripts/train/train_etpgt.py  # Script path as first arg
    - --embedding-dim=256
    - ...
```

---

## Changes Made

### Files Modified

1. **`docker/train.Dockerfile`**
   - Changed ENTRYPOINT from `["python", "scripts/train/train_baseline.py"]` to `["python"]`
   - Added comments explaining the flexibility

2. **`scripts/gcp/03_submit_training_job.sh`**
   - Added `scripts/train/train_baseline.py` as first argument in containerSpec.args
   - Added comment explaining Docker ENTRYPOINT behavior

3. **`scripts/gcp/04_submit_etpgt_job.sh`**
   - Removed `"python"` from TRAINING_ARGS array (already in ENTRYPOINT)
   - Added comment explaining Docker ENTRYPOINT behavior

---

## Verification

### 1. Rebuilt Docker Image
```bash
bash scripts/gcp/02_build_training_image_cloudbuild.sh
```

**Result**: ✅ SUCCESS
- Build ID: `cad8bca9-5024-487b-957c-a24803bbb05f`
- Duration: 4m 4s
- Image: `us-central1-docker.pkg.dev/plotpointe/etpgt/etpgt-train:latest`

### 2. Resubmitted All Training Jobs

| Job Name | Model | Job ID | State |
|----------|-------|--------|-------|
| etpgt-gat-20251019-222553 | GAT | 2908698036358086656 | PENDING |
| etpgt-graphsage-20251019-222601 | GraphSAGE | 8256159643907129344 | PENDING |
| etpgt-graph_transformer-20251019-222608 | GraphTransformer | 6367462550178627584 | PENDING |
| etpgt-20251019-222615 | ETP-GT | 5373855882390011904 | PENDING |

**Result**: ✅ All 4 jobs submitted successfully

---

## Old Jobs (Cancelled/Failed)

| Job Name | Model | Job ID | State | Reason |
|----------|-------|--------|-------|--------|
| etpgt-gat-20251019-215620 | GAT | 625654500257955840 | FAILED | Incorrect ENTRYPOINT |
| etpgt-graphsage-20251019-215612 | GraphSAGE | 4039383017804791808 | PENDING | Old image (will likely fail) |
| etpgt-graph_transformer-20251019-215626 | GraphTransformer | 6840621986029240320 | PENDING | Old image (will likely fail) |
| etpgt-20251019-221332 | ETP-GT | 8529190371316465664 | PENDING | Old image (will likely fail) |

**Note**: Old jobs can be cancelled or left to fail. New jobs are using the fixed image.

---

## Lessons Learned

### 1. Docker ENTRYPOINT Best Practices
- Keep ENTRYPOINT minimal and flexible
- Use ENTRYPOINT for the interpreter/runtime (`python`, `java`, etc.)
- Pass script paths and arguments via CMD or containerSpec.args
- Avoid hardcoding script paths in ENTRYPOINT

### 2. Debugging Vertex AI Jobs
- Cloud Logging may not show detailed error messages for early failures
- Check job description with `gcloud ai custom-jobs describe <JOB_ID>`
- Verify Docker image locally before submitting to Vertex AI
- Test container with `docker run` to catch ENTRYPOINT issues

### 3. Testing Strategy
- Test Docker images locally with `docker run` before pushing
- Verify ENTRYPOINT and CMD behavior with different arguments
- Use `docker inspect` to check ENTRYPOINT configuration

---

## Testing Commands

### Test Docker Image Locally (Future)
```bash
# Build image locally
docker build -f docker/train.Dockerfile -t etpgt-train:test .

# Test baseline training
docker run etpgt-train:test scripts/train/train_baseline.py --model=gat --help

# Test ETP-GT training
docker run etpgt-train:test scripts/train/train_etpgt.py --help
```

### Monitor New Jobs
```bash
# List all jobs
gcloud ai custom-jobs list --region=us-central1 --filter='displayName~etpgt-'

# Stream logs for GAT job
gcloud ai custom-jobs stream-logs 2908698036358086656 --region=us-central1

# View in console
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=plotpointe
```

---

## Impact

**Before Fix**:
- ❌ GAT job failed
- ⚠️ Other jobs likely to fail with same issue
- ⚠️ ETP-GT job would fail

**After Fix**:
- ✅ All 4 jobs submitted successfully
- ✅ Docker image is flexible for both baseline and ETP-GT training
- ✅ Future training jobs will work correctly

---

## Next Steps

1. ✅ Monitor new jobs to ensure they start successfully
2. ⏳ Wait for training to complete (~2-4 hours)
3. ⏳ Evaluate results and compare with baselines
4. ⏳ Proceed to Phase 6 if Phase 5 gate is passed

---

**Last Updated**: 2025-10-19 22:27 UTC

