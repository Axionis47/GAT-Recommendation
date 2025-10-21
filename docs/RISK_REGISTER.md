# Risk Register

This document tracks risks, their impact, likelihood, and mitigation strategies for ETP-GT.

**Last Updated**: TBD

## Risk Matrix

| ID | Risk | Impact | Likelihood | Severity | Mitigation | Owner |
|----|------|--------|------------|----------|------------|-------|
| R1 | Temporal leakage in splits | High | Medium | **Critical** | Unit tests, blackout periods | ML Team |
| R2 | OOM during training | Medium | Medium | **High** | Sampler caps, gradient checkpointing | ML Team |
| R3 | Latency budget exceeded | High | Medium | **Critical** | 1-layer re-rank, profiling | MLOps Team |
| R4 | Custom model underperforms | Medium | Medium | **High** | Stop rule, baseline fallback | ML Team |
| R5 | GCP quota limits | Medium | Low | **Medium** | Pre-request increases, monitoring | MLOps Team |
| R6 | Data drift over time | Medium | High | **High** | Monitoring, retraining cadence | ML Team |
| R7 | Popularity bias | Medium | High | **High** | Stratified eval, diversity metrics | ML Team |
| R8 | Cold-start items | Medium | High | **High** | Hybrid with content features (future) | ML Team |
| R9 | Bot/spam sessions | Low | Medium | **Medium** | Session filtering, rate limiting | MLOps Team |
| R10 | Security vulnerabilities | High | Low | **High** | OIDC, no long-lived keys, audits | MLOps Team |
| R11 | Cost overruns | Medium | Medium | **High** | Budget alerts, auto-shutdown | MLOps Team |
| R12 | Model staleness | Medium | High | **High** | Automated retraining pipeline | MLOps Team |

**Severity**: Critical > High > Medium > Low

## Detailed Risk Analysis

### R1: Temporal Leakage in Splits

**Description**: Training data contains information from validation/test periods, leading to overly optimistic metrics.

**Impact**: High - Invalidates all evaluation results; production performance will be much worse.

**Likelihood**: Medium - Easy to introduce bugs in temporal filtering.

**Mitigation**:
1. **Unit tests**: `tests/test_splits.py` enforces:
   - No `ts` in val/test earlier than max train `ts`
   - Blackout periods respected (1-3 days)
   - No session spans multiple splits
2. **Code review**: All data prep changes require review
3. **Documentation**: Boundary timestamps logged in `docs/data_stats.md`

**Detection**:
- CI fails if tests don't pass
- Manual inspection of split boundaries

**Contingency**:
- Re-run data prep with corrected logic
- Re-train all models

---

### R2: OOM During Training

**Description**: GPU runs out of memory during training, causing job failures.

**Impact**: Medium - Delays training; wastes compute budget.

**Likelihood**: Medium - Large graphs can exceed VRAM limits.

**Mitigation**:
1. **Sampler caps**: Max 10,000 edges per batch
2. **Gradient checkpointing**: Trade compute for memory
3. **Mixed precision**: FP16 reduces memory by ~50%
4. **Batch size tuning**: Start small, increase gradually
5. **Monitoring**: Log VRAM usage per step

**Detection**:
- CUDA OOM errors in logs
- Vertex AI job failures

**Contingency**:
- Reduce batch size
- Use smaller model variant
- Request larger GPU (A100 instead of L4)

---

### R3: Latency Budget Exceeded

**Description**: Inference latency exceeds p95 ≤ 120ms target.

**Impact**: High - Fails production acceptance criteria.

**Likelihood**: Medium - Complex models often exceed latency budgets.

**Mitigation**:
1. **1-layer re-ranker**: Lightweight model for serving
2. **Profiling**: `scripts/profiling/measure_latency.py` identifies bottlenecks
3. **Pre-warming**: Min 1 Cloud Run instance to avoid cold starts
4. **Optimizations**:
   - Batch ANN queries
   - Cache embeddings in memory
   - Reduce fanout: [12, 8, 4]
5. **Fallback**: Popularity-based recommendations if timeout

**Detection**:
- Smoke tests measure p50/p95
- Production monitoring

**Contingency**:
- Deploy simpler model (e.g., GAT)
- Increase Cloud Run resources (4 CPU, 8GB RAM)
- Use dedicated GPU instance (Cloud Run GPU preview)

---

### R4: Custom Model Underperforms

**Description**: ETP-GT fails to beat baselines by ≥1.5% Recall@20.

**Impact**: Medium - Wasted development effort; must ship baseline.

**Likelihood**: Medium - Research models often don't transfer to production.

**Mitigation**:
1. **Stop rule**: Explicit threshold (≥1.5% gain) to avoid sunk cost fallacy
2. **Baseline quality**: Ensure strong baselines (GAT, GraphTransformer)
3. **Ablations**: Identify which components contribute
4. **Hyperparameter tuning**: Grid search on val set

**Detection**:
- Validation metrics after Phase 5

**Contingency**:
- Ship best baseline (e.g., GraphTransformer)
- Document findings in `docs/ISSUES_TUNING.md`
- Re-export embeddings from baseline
- Deploy baseline re-ranker

---

### R5: GCP Quota Limits

**Description**: Insufficient quota for L4 GPUs, Cloud Run instances, or GCS storage.

**Impact**: Medium - Blocks training/deployment.

**Likelihood**: Low - Can request increases proactively.

**Mitigation**:
1. **Pre-request increases**: Before Phase 1, request:
   - 2x L4 GPUs in us-central1
   - Cloud Run: 100 instances
   - GCS: 1TB storage
2. **Monitoring**: Track quota usage
3. **Documentation**: `docs/GCP_OIDC_SETUP.md` includes quota requirements

**Detection**:
- Quota exceeded errors in GCP console
- Vertex AI job failures

**Contingency**:
- Use T4 GPUs (more widely available)
- Use different region (us-west1, europe-west1)
- Request emergency quota increase (24-48h)

---

### R6: Data Drift Over Time

**Description**: User behavior changes; model performance degrades.

**Impact**: Medium - Gradual decline in production metrics.

**Likelihood**: High - E-commerce trends shift seasonally.

**Mitigation**:
1. **Monitoring**: Track Recall@20, NDCG@20 weekly
2. **Retraining cadence**: Monthly or when metrics drop >2%
3. **Automated pipeline**: GitHub Actions for retraining
4. **Alerts**: Slack/email if metrics degrade

**Detection**:
- Production A/B tests
- Offline evaluation on recent data

**Contingency**:
- Trigger immediate retraining
- Investigate data distribution shifts
- Update temporal buckets if needed

---

### R7: Popularity Bias

**Description**: Model over-recommends popular items; ignores long-tail.

**Impact**: Medium - Reduces diversity; poor user experience.

**Likelihood**: High - Common in collaborative filtering.

**Mitigation**:
1. **Stratified evaluation**: Track cold item performance
2. **Diversity metrics**: Coverage, Gini coefficient
3. **Negative sampling**: Include hard negatives (popular but irrelevant)
4. **Post-processing**: Re-rank to boost diversity (future)

**Detection**:
- Low coverage (<50% of catalog)
- High Gini coefficient (>0.8)

**Contingency**:
- Add diversity loss term
- Hybrid with content-based recommendations

---

### R8: Cold-Start Items

**Description**: New items with <5 interactions receive poor recommendations.

**Impact**: Medium - Limits catalog coverage.

**Likelihood**: High - E-commerce catalogs change frequently.

**Mitigation**:
1. **Hybrid approach**: Combine with content features (future)
2. **Popularity fallback**: Recommend trending items for cold-start
3. **Monitoring**: Track cold item coverage

**Detection**:
- Stratified evaluation shows low Recall@20 for cold items

**Contingency**:
- Use item metadata (category, price, brand) for cold-start
- Implement content-based fallback

---

### R9: Bot/Spam Sessions

**Description**: Automated sessions skew training data.

**Impact**: Low - Noise in training; minor metric degradation.

**Likelihood**: Medium - E-commerce sites attract bots.

**Mitigation**:
1. **Session filtering**:
   - Remove sessions with >100 events
   - Remove sessions with <1s average gap
   - Remove sessions with single item repeated >10 times
2. **Rate limiting**: In production API

**Detection**:
- Anomalous session length distribution
- Spike in traffic from single IP

**Contingency**:
- Re-run data prep with stricter filters
- Implement CAPTCHA for high-volume users

---

### R10: Security Vulnerabilities

**Description**: Credential leakage, unauthorized access, or code injection.

**Impact**: High - Data breach; service disruption.

**Likelihood**: Low - Mitigated by OIDC and best practices.

**Mitigation**:
1. **OIDC**: No long-lived service account keys
2. **Secrets management**: GitHub Secrets, never in code
3. **IAM**: Least-privilege service account roles
4. **Code review**: All PRs reviewed for security issues
5. **Dependency scanning**: Dependabot alerts

**Detection**:
- GCP audit logs
- GitHub security alerts

**Contingency**:
- Rotate credentials immediately
- Revoke compromised service accounts
- Incident response plan

---

### R11: Cost Overruns

**Description**: Training/serving costs exceed budget.

**Impact**: Medium - Financial waste; project cancellation.

**Likelihood**: Medium - Easy to overspend on GPUs.

**Mitigation**:
1. **Budget alerts**: GCP billing alerts at 50%, 80%, 100%
2. **Auto-shutdown**: Vertex AI jobs timeout after max runtime
3. **Cost tracking**: Log costs per experiment
4. **Optimization**:
   - Use preemptible VMs for training (50% discount)
   - Scale Cloud Run to zero when idle

**Detection**:
- GCP billing dashboard
- Budget alert emails

**Contingency**:
- Pause non-critical experiments
- Optimize model size/training time
- Request budget increase

---

### R12: Model Staleness

**Description**: Model not retrained; performance degrades over time.

**Impact**: Medium - Gradual decline in user satisfaction.

**Likelihood**: High - Manual retraining is error-prone.

**Mitigation**:
1. **Automated pipeline**: GitHub Actions workflow for retraining
2. **Scheduled runs**: Monthly cron job
3. **Monitoring**: Alert if model age >30 days
4. **Documentation**: Retraining SOP in `docs/GOVERNANCE.md`

**Detection**:
- Model version endpoint shows old timestamp
- Performance metrics degrade

**Contingency**:
- Trigger manual retraining
- Investigate pipeline failures

---

## Risk Review Cadence

- **Weekly**: Review critical/high risks during standup
- **Monthly**: Full risk register review; update mitigations
- **Post-incident**: Add new risks; update likelihood/impact

## Escalation

- **Critical risks**: Immediate escalation to project lead
- **High risks**: Escalate if mitigation fails
- **Medium/Low risks**: Track in issue tracker

## References

- [NIST Risk Management Framework](https://csrc.nist.gov/projects/risk-management)
- [Google SRE Book - Risk Management](https://sre.google/sre-book/embracing-risk/)

