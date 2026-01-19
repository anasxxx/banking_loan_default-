# Model Loading Issue - Diagnostic Steps

## Problem
API shows "Model not loaded" error at `/model/info` endpoint.

## Root Cause Analysis

The model file needs to be:
1. ✅ In the repository (DONE - we added it)
2. ✅ Uploaded to EC2 at `~/models/production/` (Needs to be done)
3. ✅ Accessible inside Docker container at `/app/models/production/` (Volume mount needed)

## Immediate Fix

### Step 1: Trigger the Fix Workflow

Go to GitHub Actions and trigger the fix workflow:
- **URL**: https://github.com/anasxxx/banking_loan_default-/actions/workflows/fix-model-now.yml
- Click **"Run workflow"** button
- Click **"Run workflow"** again

This workflow will:
- Upload all model files
- Verify files exist on EC2
- Stop old container
- Start new container with proper volume mounts
- Show detailed logs to diagnose any remaining issues

### Step 2: Check Workflow Logs

After the workflow runs, check the logs for:
1. **File upload success**: Should see "Uploaded" messages
2. **File verification**: Should see files listed with sizes
3. **Container logs**: Should see model loading messages
4. **Volume mount verification**: Should confirm files accessible in container

## Common Issues and Solutions

### Issue 1: Files not uploaded
**Symptom**: Workflow shows "file not found" during upload
**Solution**: Make sure model files are in the repository. Check:
```bash
git ls-files models/production/
```

### Issue 2: Files uploaded but not accessible in container
**Symptom**: Files exist on EC2 but container can't access them
**Solution**: Check volume mount in docker run command:
```bash
docker run -d -p 8000:8000 \
  -v ~/models:/app/models \  # This should mount ~/models to /app/models
  --name loan-api \
  YOUR_IMAGE
```

### Issue 3: Container can't find model file
**Symptom**: Container logs show "FileNotFoundError"
**Solution**: Verify path inside container:
```bash
docker exec loan-api ls -la /app/models/production/
```

### Issue 4: Model file too large
**Symptom**: Upload fails or times out
**Solution**: The model file is 50MB. SCP should handle it, but if it fails, upload manually or use Git LFS.

## Manual Verification Steps

If the workflow completes but model still not loaded:

### On EC2 (SSH):
```bash
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
```

### Expected Container Logs:
```
Loading model from: models/production/lightgbm_smote_high_performance.pkl
Model loaded successfully at 2024-01-XX...
Model type: LGBMClassifier
Model features: 343
Load time: X.XXs
```

## Quick Fix Command (if you have SSH access)

If you can SSH into EC2, run this:

```bash
# Create directory
mkdir -p ~/models/production
chmod -R 755 ~/models

# Upload files from local machine (run this from your local machine):
scp models/production/lightgbm_smote_high_performance.pkl \
    models/production/metrics_smote_high_performance.json \
    models/production/optimal_threshold_smote.json \
    models/production/feature_names.json \
    ubuntu@98.80.216.214:~/models/production/

# Then on EC2:
docker restart loan-api
sleep 15
docker logs loan-api --tail 30
curl http://localhost:8000/model/info
```

## After Fix

Once model loads successfully, test these endpoints:
- Health: http://98.80.216.214:8000/health
- Model Info: http://98.80.216.214:8000/model/info
- Docs: http://98.80.216.214:8000/docs

You should see:
- `"model_loaded": true` in health check
- Model info with your metrics (91.6% AUC, 92.17% Recall, etc.)
