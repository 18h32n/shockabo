# Email Notifier Usage Guide

The `email_notifier.py` module provides a secure, rate-limited email notification system for the ARC Prize 2025 platform rotation system.

## Features

- **Secure TLS-encrypted SMTP connections**
- **Multiple notification types** (experiment completion, failures, platform failures, quota warnings, queue status)
- **Rate limiting** (configurable, default 10 emails per hour)
- **Queue-based email sending** with retry logic
- **Template-based HTML and plain text email formatting**
- **Asynchronous operation** to avoid blocking
- **Integration with secure credential storage**

## Configuration

### Environment Variables

Set these environment variables or store them securely using the credential manager:

```bash
# Required
export EMAIL_SMTP_SERVER="smtp.gmail.com:587"
export EMAIL_USERNAME="your-email@gmail.com"
export EMAIL_PASSWORD="your-app-password"
export EMAIL_FROM_ADDRESS="your-email@gmail.com"
```

### For Gmail Users

1. Enable 2-factor authentication on your Gmail account
2. Generate an app password: https://myaccount.google.com/apppasswords
3. Use the app password as `EMAIL_PASSWORD`

## Basic Usage

### Start the Email Notifier

```python
import asyncio
from src.infrastructure.components.email_notifier import get_email_notifier, EmailConfig

async def setup_email_system():
    # Get notifier with default configuration
    notifier = get_email_notifier()
    
    # Or with custom configuration
    config = EmailConfig(
        rate_limit=RateLimitConfig(max_emails_per_hour=20),
        timeout_seconds=30
    )
    notifier = get_email_notifier(config)
    
    # Start the notification system
    if await notifier.start():
        print("Email notifier started successfully")
    else:
        print("Failed to start email notifier")
```

### Send Notifications

#### Experiment Completion

```python
success = notifier.send_experiment_completion(
    experiment_name="Arc Solution Model Training",
    job_id="job-12345",
    platform="kaggle",
    duration_minutes=45.3,
    results_summary="Training completed with 92.5% validation accuracy",
    gpu_utilization=87.2,
    tasks_completed=400,
    queue_size=3
)
```

#### Experiment Failure

```python
success = notifier.send_experiment_failure(
    experiment_name="Arc Solution Model Training",
    job_id="job-12345",
    platform="colab",
    duration_minutes=23.1,
    error_message="CUDA out of memory",
    retry_count=1,
    max_retries=3,
    stack_trace="Full stack trace here..."
)
```

#### Platform Failure

```python
success = notifier.send_platform_failure(
    platform="paperspace",
    failure_type="Session Timeout",
    error_details="Session terminated unexpectedly after 2 hours",
    suspended_jobs=2,
    queue_status="3 jobs pending",
    next_platform="kaggle",
    recovery_estimate="5 minutes"
)
```

#### Quota Warning

```python
success = notifier.send_quota_warning(
    platform="colab",
    resource_type="GPU Hours",
    current_usage=9,
    quota_limit=12,
    warning_threshold=75.0,
    time_to_limit="3 hours",
    recommended_actions="Switch to Kaggle or pause non-critical experiments"
)
```

#### Queue Status Update

```python
success = notifier.send_queue_status(
    total_jobs=25,
    pending_jobs=5,
    running_jobs=2,
    completed_jobs=15,
    failed_jobs=3,
    platform_status="Kaggle: Active, Colab: GPU Quota Reached, Paperspace: Available",
    jobs_per_hour=3.2,
    avg_runtime_minutes=28.5,
    queue_completion_estimate="4 hours"
)
```

#### System Alert

```python
success = notifier.send_system_alert(
    alert_type="High Memory Usage",
    severity="WARNING",
    component="experiment_queue",
    alert_description="Memory usage has exceeded 85% for the past 10 minutes",
    system_impact="Potential performance degradation",
    manual_action_required="Monitor system resources and consider reducing concurrent jobs"
)
```

#### Daily Summary

```python
success = notifier.send_daily_summary(
    date="2025-01-15",
    jobs_completed=12,
    jobs_failed=2,
    total_runtime_hours=6.5,
    platform_distribution="Kaggle: 8, Colab: 4, Paperspace: 2",
    gpu_hours_used=5.2,
    cost_estimate=15.60,
    efficiency_rating=8,
    notable_events="Platform rotation occurred 2 times",
    queued_jobs=3,
    estimated_runtime_hours=2.5
)
```

### Custom Notifications

For custom notifications, use the generic `send_notification` method:

```python
from src.infrastructure.components.email_notifier import NotificationType, Priority

success = notifier.send_notification(
    NotificationType.SYSTEM_ALERT,
    Priority.HIGH,
    alert_type="Custom Alert",
    severity="HIGH",
    component="custom_component",
    alert_description="Custom alert description",
    # ... other context variables
)
```

### Rate Limiting

The system includes built-in rate limiting to prevent email spam:

- **Hourly limit**: Configurable (default 10 emails per hour)
- **Burst limit**: Configurable (default 3 emails in 5 minutes)
- **Cooldown period**: Configurable (default 15 minutes after burst limit exceeded)

Rate-limited emails are logged but not sent. Configure limits based on your email provider's restrictions.

### Template Customization

You can update email templates for different notification types:

```python
success = notifier.update_template(
    NotificationType.EXPERIMENT_COMPLETION,
    subject_template="[ARC] Experiment Done: {experiment_name}",
    body_template="Your experiment {experiment_name} finished in {duration_minutes:.1f} minutes.",
    html_template="<h2>Experiment Complete</h2><p>Duration: {duration_minutes:.1f} min</p>"
)
```

### Error Handling and Retries

The system includes automatic retry logic:

- **Retry attempts**: Configurable (default 3)
- **Retry delays**: Exponential backoff (default: 1min, 5min, 15min)
- **Timeout**: Configurable (default 30 seconds)

Failed emails are queued for retry and will be attempted again with increasing delays.

### Monitoring and Statistics

Get system statistics:

```python
stats = notifier.get_statistics()
print(f"Emails sent: {stats['emails_sent']}")
print(f"Emails failed: {stats['emails_failed']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Queue size: {stats['queue_size']}")
```

### Testing

Test your email configuration:

```python
# Test connection
if notifier.test_connection():
    print("Email configuration is working")
else:
    print("Email configuration has issues")

# Run comprehensive tests
python scripts/test_email_notifier.py
```

### Cleanup

Always stop the notifier when your application shuts down:

```python
async def shutdown():
    await notifier.stop()
    print("Email notifier stopped and remaining emails sent")
```

## Integration with Platform Rotation System

The email notifier integrates seamlessly with the existing infrastructure components:

```python
from src.infrastructure.components import (
    get_email_notifier,
    get_platform_detector,
    get_availability_checker
)

async def setup_monitoring():
    # Start email notifier
    notifier = get_email_notifier()
    await notifier.start()
    
    # Monitor platform events
    detector = get_platform_detector()
    availability = get_availability_checker()
    
    # Send notifications based on platform events
    current_platform = detector.get_current_platform()
    if current_platform:
        await notifier.send_system_alert(
            alert_type="System Startup",
            severity="INFO",
            component="platform_detector",
            alert_description=f"System started on {current_platform.name}"
        )
```

## Security Considerations

- **Credential Storage**: Uses encrypted credential storage via `SecureCredentialManager`
- **TLS Encryption**: All SMTP connections use TLS encryption
- **Rate Limiting**: Prevents abuse and respects email provider limits
- **Input Validation**: All template variables are validated before rendering
- **Error Logging**: Detailed logging without exposing sensitive information

## Troubleshooting

### Common Issues

1. **"Email credentials missing"**
   - Verify environment variables are set
   - Check credential manager has stored the values

2. **"SMTP connection failed"**
   - Verify SMTP server and port
   - Check network connectivity
   - Ensure TLS is enabled for Gmail/Outlook

3. **"Rate limited"**
   - Reduce notification frequency
   - Increase rate limits in configuration
   - Check if burst limit was exceeded

4. **"Template rendering failed"**
   - Verify all required context variables are provided
   - Check template syntax for typos

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
import structlog

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
structlog.configure(log_level="DEBUG")
```

This will provide detailed information about email sending attempts, rate limiting decisions, and error conditions.