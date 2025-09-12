# 10. Infrastructure

## Platform Deployment Strategy

**Multi-Platform Rotation System**
```python
class PlatformRotator:
    """Manages rotation between free GPU platforms"""
    PLATFORMS = {
        'kaggle': {
            'gpu_hours': 30,
            'reset_day': 'weekly',
            'setup_script': 'scripts/platform_deploy/kaggle_setup.py'
        },
        'colab': {
            'gpu_hours': 12,
            'reset_day': 'daily',
            'setup_script': 'scripts/platform_deploy/colab_setup.py'
        },
        'paperspace': {
            'gpu_hours': 6,
            'reset_day': 'daily',
            'setup_script': 'scripts/platform_deploy/paperspace_setup.py'
        }
    }
    
    def get_available_platform(self) -> str:
        """Select platform with most remaining GPU hours"""
        # Check usage tracking database
        # Return platform with most availability
```

## Containerization

**Dockerfile**
```dockerfile
FROM python:3.12.7-slim
