# Governance & Operations

This document defines processes for maintaining and operating ETP-GT in production.

## Versioning

### Semantic Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

```
v<MAJOR>.<MINOR>.<PATCH>
```

- **MAJOR**: Breaking API changes, major architecture changes
- **MINOR**: New features, backward-compatible changes
- **PATCH**: Bug fixes, performance improvements

### Version Tracking

Every deployment includes:
- **Git SHA**: Exact commit
- **Model version**: Training date + config hash
- **Index version**: ANN index build date
- **Data version**: Training data snapshot date

Example:
```json
{
  "git_sha": "a1b2c3d4",
  "model": "etpgt-small@2025-01-15-abc123",
  "index": "ivfpq@2025-01-15",
  "data": "retailrocket-2025-01-01"
}
```

## Branching Strategy

### Git Flow

- **main**: Production-ready code; protected branch
- **develop**: Integration branch for next release
- **feature/<name>**: Feature development
- **bugfix/<name>**: Bug fixes
- **hotfix/<name>**: Emergency production fixes

### Branch Protection

**main** branch requires:
- ≥1 approving review
- CI passing (lint, typecheck, tests)
- Up-to-date with base branch
- No force pushes

## Release Process

### 1. Prepare Release

```bash
# From develop branch
git checkout develop
git pull origin develop

# Update version in pyproject.toml
# Update CHANGELOG.md with release notes
git add pyproject.toml CHANGELOG.md
git commit -m "chore: prepare v0.1.0 release"
```

### 2. Create Release PR

```bash
git checkout -b release/v0.1.0
git push origin release/v0.1.0
# Open PR: release/v0.1.0 → main
```

### 3. Merge and Tag

```bash
# After PR approval and merge
git checkout main
git pull origin main
git tag v0.1.0
git push origin v0.1.0
```

### 4. Automated Release

GitHub Actions `release.yaml` workflow:
- Builds package
- Creates GitHub release
- Attaches artifacts (metrics, configs, docs)
- (Optional) Publishes to PyPI

### 5. Deploy to Production

```bash
# Trigger deployment workflow
# Or manually:
make docker-infer-build
make docker-infer-push
make gcp-deploy
make gcp-smoke
```

## Retraining Cadence

### Scheduled Retraining

**Frequency**: Monthly (1st of each month)

**Trigger**: GitHub Actions cron job
```yaml
on:
  schedule:
    - cron: '0 0 1 * *'  # 00:00 UTC on 1st of month
```

**Process**:
1. Download latest data snapshot
2. Run data prep with updated date range
3. Train ETP-GT with same config
4. Evaluate on held-out test set
5. If metrics ≥ current model - 2%, deploy
6. Otherwise, investigate and alert

### Event-Driven Retraining

**Triggers**:
- Performance degradation >2% (weekly monitoring)
- Major data distribution shift (detected by monitoring)
- New feature deployment (manual)

**Process**:
1. Create issue documenting trigger
2. Run retraining pipeline
3. A/B test new model vs. current
4. Gradual rollout (10% → 50% → 100%)

## Monitoring

### Model Performance

**Metrics** (tracked weekly):
- Recall@20
- NDCG@20
- Coverage (% of catalog recommended)
- Diversity (Gini coefficient)

**Alerts**:
- Recall@20 drops >2%: Warning
- Recall@20 drops >5%: Critical
- Coverage <50%: Warning

### System Performance

**Metrics** (tracked real-time):
- Latency (p50, p95, p99)
- Error rate
- Throughput (RPS)
- Cold start rate

**Alerts**:
- p95 latency >120ms: Warning
- p95 latency >200ms: Critical
- Error rate >1%: Warning
- Error rate >5%: Critical

### Cost

**Metrics** (tracked daily):
- Training cost per job
- Serving cost per 1M requests
- Storage cost (GCS)

**Alerts**:
- Monthly cost >$X: Warning
- Daily cost spike >2x average: Warning

## Incident Response

### Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| P0 | Service down | <15 min | API returning 500s |
| P1 | Degraded performance | <1 hour | p95 latency >200ms |
| P2 | Minor issue | <1 day | Metrics drop 2-5% |
| P3 | Cosmetic | <1 week | Logging errors |

### Response Process

1. **Detect**: Monitoring alert or user report
2. **Triage**: Assess severity; assign owner
3. **Mitigate**: Rollback or hotfix
4. **Investigate**: Root cause analysis
5. **Resolve**: Permanent fix
6. **Document**: Post-mortem (for P0/P1)

### Rollback Procedure

```bash
# Identify last known good version
LAST_GOOD_SHA=<commit-sha>

# Rebuild and deploy
git checkout ${LAST_GOOD_SHA}
make docker-infer-build
make docker-infer-push
make gcp-deploy

# Verify
make gcp-smoke
```

## Code Review

### Requirements

All PRs require:
- ≥1 approving review from CODEOWNERS
- CI passing (lint, typecheck, tests)
- No unresolved comments
- Description of changes and testing

### Review Checklist

- [ ] Code follows style guidelines (Black, Ruff)
- [ ] Type hints present and correct
- [ ] Tests added/updated (≥80% coverage)
- [ ] Documentation updated (if API changes)
- [ ] No secrets or credentials in code
- [ ] Performance implications considered
- [ ] Backward compatibility maintained (or MAJOR version bump)

## Testing

### Unit Tests

**Coverage**: ≥80% for `etpgt/` modules

**Run**:
```bash
make test
```

**CI**: Runs on every PR

### Integration Tests

**Scope**: End-to-end workflows (data prep → train → eval → serve)

**Run**:
```bash
pytest tests/ -m integration
```

**CI**: Runs on `main` branch only (slow)

### Smoke Tests

**Scope**: Production deployment health checks

**Run**:
```bash
make gcp-smoke
```

**CI**: Runs after deployment

## Documentation

### Required Documentation

All features require:
- **Code comments**: Docstrings for public functions/classes
- **README updates**: If user-facing changes
- **CHANGELOG**: Entry for every release
- **ADR**: For architectural decisions (in DECISIONS.md)

### Documentation Review

- Technical accuracy: ML team
- Clarity: Product team
- Completeness: MLOps team

## Security

### Secrets Management

**Never commit**:
- API keys
- Service account keys
- Passwords
- `.env` files

**Use**:
- GitHub Secrets for CI/CD
- GCP Secret Manager for production
- `.env.example` for templates

### Dependency Updates

**Frequency**: Monthly

**Process**:
1. Dependabot creates PRs for updates
2. Review changelog for breaking changes
3. Run full test suite
4. Merge if tests pass

**Security patches**: Immediate (within 24h)

## Compliance

### Data Privacy

- **No PII**: Session data is anonymous
- **Retention**: Delete raw data after 90 days
- **GDPR**: Support data deletion requests

### Audit Trail

All production changes logged:
- Who: GitHub user
- What: Commit SHA, config changes
- When: Timestamp
- Why: PR description, issue link

**Retention**: 1 year

## Contacts

- **ML Team**: ml-team@example.com
- **MLOps Team**: mlops-team@example.com
- **On-call**: pagerduty.com/etp-gt

## References

- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [Semantic Versioning](https://semver.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)

