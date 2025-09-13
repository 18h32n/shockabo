# Weights & Biases Integration Setup Guide

This guide walks you through setting up Weights & Biases (W&B) integration for the ARC Prize 2025 evaluation framework.

## Table of Contents

1. [Account Creation](#account-creation)
2. [Environment Setup](#environment-setup)
3. [Configuration](#configuration)
4. [Usage Monitoring](#usage-monitoring)
5. [Platform-Specific Setup](#platform-specific-setup)
6. [Troubleshooting](#troubleshooting)

## Account Creation

### Step 1: Create a W&B Account

1. Visit [https://wandb.ai/site](https://wandb.ai/site)
2. Click "Sign up" and create a free account
3. Verify your email address

### Step 2: Generate API Key

1. Log in to your W&B account
2. Go to your account settings: [https://wandb.ai/settings](https://wandb.ai/settings)
3. Scroll to the "API keys" section
4. Click "New key" to generate an API key
5. **Important**: Copy and save this key securely - you won't be able to see it again

### Step 3: Verify Free Tier Details

The W&B free tier includes:
- **100GB storage** for artifacts, models, and logs
- **Unlimited experiments** and runs
- **Unlimited users** for personal projects
- **Basic integrations** with popular ML frameworks

## Environment Setup

### Secure Credential Storage

**Never commit your API key to version control!**

#### Option 1: Environment Variable (Recommended)

```bash
# Linux/Mac
export WANDB_API_KEY="your-api-key-here"

# Windows Command Prompt
set WANDB_API_KEY=your-api-key-here

# Windows PowerShell
$env:WANDB_API_KEY="your-api-key-here"
```

#### Option 2: .env File (Local Development)

Create a `.env` file in your project root:

```bash
WANDB_API_KEY=your-api-key-here
WANDB_PROJECT=arc-prize-2025
WANDB_ENTITY=your-username-or-team  # Optional
```

**Important**: Add `.env` to your `.gitignore` file:

```gitignore
# Environment variables
.env
.env.local
```

### Install W&B Package

```bash
pip install wandb
```

Or add to your `requirements.txt`:

```
wandb>=0.16.0
```

## Configuration

### Environment Variables

Configure W&B behavior using these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_API_KEY` | Your W&B API key | Required |
| `WANDB_PROJECT` | Project name | `arc-prize-2025` |
| `WANDB_ENTITY` | Team/organization name | Your username |
| `WANDB_MODE` | Run mode: `online`, `offline`, `disabled` | `online` |
| `WANDB_DIR` | Local directory for W&B files | `./wandb` |
| `WANDB_TAGS` | Comma-separated tags | `evaluation,arc` |

### Python Configuration

```python
import os
from src.adapters.external.wandb_client import get_wandb_client

# Set up environment (if not already done)
os.environ["WANDB_API_KEY"] = "your-api-key"
os.environ["WANDB_PROJECT"] = "arc-prize-2025"

# Initialize client
client = get_wandb_client()

# Start an experiment
from src.domain.evaluation_models import ExperimentRun
experiment = ExperimentRun(
    run_id="exp_001",
    experiment_name="Baseline Evaluation",
    task_ids=["task_1", "task_2"],
    strategy_config={"strategy": "pattern_match"},
    metrics={},
    status=TaskStatus.IN_PROGRESS,
    started_at=datetime.now()
)

client.start_experiment(experiment, config={"learning_rate": 0.001})
```

## Usage Monitoring

### Understanding the 100GB Limit

The 100GB storage limit includes:
- Model checkpoints and artifacts
- Logged images and media
- Large log files
- Dataset artifacts

The limit does **not** include:
- Basic metrics and scalar logs
- System metrics
- Small configuration files

### Monitoring Your Usage

The framework automatically monitors your W&B storage usage:

```python
from src.adapters.external.wandb_client import get_wandb_client

client = get_wandb_client()
usage_gb = client.usage_monitor.get_current_usage_gb()
print(f"Current usage: {usage_gb:.2f} GB / 100 GB")
```

### Automatic Alerts

The system will alert you at:
- **80% usage (80GB)**: Warning alert
- **95% usage (95GB)**: Critical alert, new artifacts blocked

### Managing Storage

To stay within the free tier:

1. **Delete old runs**:
   ```python
   import wandb
   api = wandb.Api()
   runs = api.runs("your-entity/arc-prize-2025")
   for run in runs:
       if run.state != "running" and run.created_at < cutoff_date:
           run.delete()
   ```

2. **Use artifact aliases**:
   ```python
   # Only keep best models
   artifact.aliases.append("best")
   # Delete old versions without aliases
   ```

3. **Log efficiently**:
   ```python
   # Don't log large tensors every step
   if step % 100 == 0:
       wandb.log({"large_histogram": data})
   ```

## Platform-Specific Setup

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export WANDB_API_KEY="your-api-key"
export WANDB_MODE="online"

# Run evaluation
python src/main.py evaluate --use-wandb
```

### Google Colab

```python
# In a Colab cell
import os
from google.colab import userdata

# Set API key from Colab secrets
os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY')

# Install W&B
!pip install wandb

# Import and use
from src.adapters.external.wandb_client import get_wandb_client
client = get_wandb_client()
```

### Kaggle Notebooks

1. Add W&B API key as a Kaggle secret:
   - Go to notebook settings
   - Add secret named `WANDB_API_KEY`

2. In your notebook:
```python
# Access Kaggle secret
import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")

# Use offline mode if no internet
os.environ["WANDB_MODE"] = "offline"
```

### Docker

Add to your `Dockerfile`:
```dockerfile
# Don't include API key in image!
ENV WANDB_PROJECT=arc-prize-2025
ENV WANDB_DIR=/app/wandb

# Mount .env file or pass environment variables at runtime
```

Run with:
```bash
docker run -e WANDB_API_KEY=$WANDB_API_KEY your-image
```

## Troubleshooting

### Common Issues

1. **"wandb: ERROR Unable to log in"**
   - Verify your API key is correct
   - Check internet connectivity
   - Try `wandb login --relogin`

2. **"Storage limit exceeded"**
   - Check usage: `client.usage_monitor.get_current_usage_gb()`
   - Delete old runs or artifacts
   - Consider upgrading to a paid plan

3. **"Module 'wandb' not found"**
   - Install wandb: `pip install wandb`
   - Verify it's in your requirements.txt

4. **Offline mode issues**
   - Set `WANDB_MODE=offline` for environments without internet
   - Sync runs later with `wandb sync wandb/offline-run-*`

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger("wandb").setLevel(logging.DEBUG)
```

### Getting Help

- W&B Documentation: [https://docs.wandb.ai](https://docs.wandb.ai)
- W&B Community: [https://community.wandb.ai](https://community.wandb.ai)
- Project Issues: [GitHub Issues](https://github.com/your-repo/issues)

## Security Best Practices

1. **Never commit API keys**
   - Use environment variables
   - Add `.env` to `.gitignore`
   - Use platform-specific secrets management

2. **Rotate keys periodically**
   - Generate new keys every 90 days
   - Delete old keys after rotation

3. **Use least privilege**
   - Create project-specific API keys if possible
   - Limit access to sensitive projects

4. **Monitor access**
   - Check W&B audit logs regularly
   - Review team member access

## Next Steps

1. Complete the setup following this guide
2. Run the example evaluation to test the integration
3. Monitor your usage to stay within limits
4. Explore W&B features like sweeps and reports

For more information, see the [W&B documentation](https://docs.wandb.ai) or the framework's [evaluation guide](../evaluation-guide.md).