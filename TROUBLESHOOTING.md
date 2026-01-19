# Model Not Loading - Troubleshooting Guide

## Current Status
- API is running at http://98.80.216.214:8000
- Health endpoint shows: `{"status":"unhealthy", "model_loaded":false}`
- Model info endpoint shows: `{"detail":"Model not loaded"}`

## Quick Fix Steps

### Step 1: Trigger the Fix Workflow

**URL**: https://github.com/anasxxx/banking_loan_default-/actions/workflows/fix-model-now.yml

1. Click **"Run workflow"** button (top right)
2. Select branch: **main**
3. Click **"Run workflow"** button

Wait 2-3 minutes for completion.

### Step 2: Check Workflow Logs

After workflow completes, click on the workflow run to see logs. Look for:

**✅ Success indicators:**
- `Uploaded` messages for each file
- `Model file exists: -rw-r--r--` (showing file size)
- `✓ Model file accessible in container`
- Container logs showing: `Model loaded successfully`

**❌ Error indicators:**
- `file not found` during upload
- `Model file NOT accessible in container`
- `FileNotFoundError` in container logs
- `Permission denied` errors

### Step 3: Verify Container Status

If workflow completes but model still not loaded, the logs will show you exactly where it failed.

## Common Issues Found in Logs

### Issue 1: Files Not Uploaded
**Log shows**: `scp: file not found`
**Solution**: Files must be in repository. Verify:
```bash
git ls-files models/production/
```

### Issue 2: Files Uploaded But Container Can't Access
**Log shows**: `✗ Model file NOT accessible in container`
**Solution**: Volume mount issue. Check if container was started with:
```bash
-v ~/models:/app/models
```

### Issue 3: Container Not Restarting
**Log shows**: `Failed to start container` or container not running
**Solution**: Check Docker daemon and image exists:
```bash
docker ps -a
docker images | grep loan-api
```

### Issue 4: Permission Denied
**Log shows**: `Permission denied` when accessing file
**Solution**: Files need 755 permissions:
```bash
chmod -R 755 ~/models/production/
```

### Issue 5: Model File Path Issue
**Log shows**: `FileNotFoundError: [Errno 2] No such file or directory`
**Solution**: Path mismatch. API looks for `models/production/lightgbm_smote_high_performance.pkl`
- On EC2: `~/models/production/lightgbm_smote_high_performance.pkl`
- In container: `/app/models/production/lightgbm_smote_high_performance.pkl`

## What to Share if Still Not Working

If the workflow completes but model still doesn't load, please share:

1. **Workflow log snippet** showing:
   - File upload step results
   - File verification step results  
   - Container logs (last 50 lines)
   - Volume mount verification result

2. **Specific error message** from:
   - Container logs
   - API health endpoint response
   - Any error shown in workflow output

## Manual Verification (If You Have SSH Access)

If you can SSH into EC2, you can verify manually:

```bash
# SSH into EC2
ssh ubuntu@98.80.216.214

# Check files exist
ls -lh ~/models/production/lightgbm_smote_high_performance.pkl

# Check permissions
chmod -R 755 ~/models/production/

# Check container status
docker ps | grep loan-api

# Check container logs
docker logs loan-api --tail 50

# Check volume mount
docker inspect loan-api | grep -A 10 Mounts

# Check file inside container
docker exec loan-api ls -la /app/models/production/

# Restart container
docker restart loan-api
sleep 20
docker logs loan-api --tail 30
```

## Expected Container Logs

When model loads successfully, you should see in container logs:

```
Loading model from: models/production/lightgbm_smote_high_performance.pkl
Model loaded successfully at 2024-01-XX...
Model type: LGBMClassifier
Model features: 343
Load time: 1.23s
Loaded model metrics from: models/production/metrics_smote_high_performance.json
  AUC-ROC: 0.916
  Precision: 0.874
  Recall: 0.9217
  F1-Score: 0.90
```

## Next Steps

1. **Trigger the fix workflow** from GitHub Actions
2. **Wait for it to complete** (2-3 minutes)
3. **Check the workflow logs** to see what happened
4. **Share the logs** if model still doesn't load
5. I'll identify the exact issue from the logs and fix it

The workflow has extensive logging - it will show you exactly where the problem is!
