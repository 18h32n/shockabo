#!/usr/bin/env python3
"""Test script for the email notifier system.

This script demonstrates how to use the email notification system and
can be used to verify that email credentials and configuration are working properly.

Usage:
    python scripts/test_email_notifier.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.components.email_notifier import (
    EmailConfig,
    EmailNotifier,
    NotificationType,
    Priority,
    RateLimitConfig,
    get_email_notifier,
)


async def test_email_connection():
    """Test email connection and credentials."""
    print("Testing email connection...")

    # Get email notifier with test configuration
    config = EmailConfig(
        rate_limit=RateLimitConfig(max_emails_per_hour=50),  # Higher limit for testing
        timeout_seconds=15
    )

    notifier = get_email_notifier(config)

    # Test connection
    if notifier.test_connection():
        print("✓ Email connection test successful")
        return True
    else:
        print("✗ Email connection test failed")
        return False


async def test_basic_notification():
    """Test sending a basic notification."""
    print("\nTesting basic notification...")

    notifier = get_email_notifier()

    # Start notifier
    if not await notifier.start():
        print("✗ Failed to start email notifier")
        return False

    print("✓ Email notifier started")

    try:
        # Send test notification
        success = notifier.send_notification(
            NotificationType.SYSTEM_ALERT,
            Priority.NORMAL,
            alert_type="Email System Test",
            severity="INFO",
            component="email_notifier",
            alert_description="This is a test notification to verify the email system is working properly.",
            system_impact="No impact - this is a test",
            automated_response="None required",
            manual_action_required="Verify that this email was received successfully"
        )

        if success:
            print("✓ Test notification queued successfully")

            # Wait a bit for processing
            await asyncio.sleep(5)

            stats = notifier.get_statistics()
            print(f"✓ Notifier statistics: {stats['emails_sent']} sent, {stats['emails_failed']} failed")

        else:
            print("✗ Failed to queue test notification")
            return False

    finally:
        # Stop notifier
        await notifier.stop()
        print("✓ Email notifier stopped")

    return success


async def test_experiment_notifications():
    """Test experiment-related notifications."""
    print("\nTesting experiment notifications...")

    notifier = get_email_notifier()

    if not await notifier.start():
        print("✗ Failed to start email notifier")
        return False

    try:
        # Test experiment completion
        success1 = notifier.send_experiment_completion(
            experiment_name="Test Experiment",
            job_id="test-job-123",
            platform="kaggle",
            duration_minutes=15.5,
            results_summary="Test completed successfully with 95% accuracy",
            gpu_utilization=87.5,
            tasks_completed=100,
            queue_size=5
        )

        # Test experiment failure
        success2 = notifier.send_experiment_failure(
            experiment_name="Test Failed Experiment",
            job_id="test-job-456",
            platform="colab",
            duration_minutes=8.2,
            error_message="CUDA out of memory error",
            retry_count=1,
            max_retries=3,
            stack_trace="Traceback (most recent call last):\n  File test.py, line 42, in train\n    RuntimeError: CUDA out of memory"
        )

        # Test quota warning
        success3 = notifier.send_quota_warning(
            platform="paperspace",
            resource_type="GPU Hours",
            current_usage=24,
            quota_limit=30,
            warning_threshold=80.0,
            time_to_limit="2 hours",
            recommended_actions="Consider switching to Kaggle or Colab",
            pending_jobs=3,
            running_jobs=1,
            estimated_completion="45 minutes"
        )

        if all([success1, success2, success3]):
            print("✓ All experiment notifications queued successfully")

            # Wait for processing
            await asyncio.sleep(10)

            stats = notifier.get_statistics()
            print(f"✓ Final statistics: {stats['emails_sent']} sent, {stats['emails_failed']} failed")

            return True
        else:
            print(f"✗ Some notifications failed: completion={success1}, failure={success2}, quota={success3}")
            return False

    finally:
        await notifier.stop()


async def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\nTesting rate limiting...")

    # Configure with very low rate limit for testing
    config = EmailConfig(
        rate_limit=RateLimitConfig(
            max_emails_per_hour=3,
            burst_limit=2,
            burst_window_minutes=1
        )
    )

    notifier = EmailNotifier(config)

    if not await notifier.start():
        print("✗ Failed to start email notifier with test config")
        return False

    try:
        sent_count = 0
        rate_limited_count = 0

        # Try to send more emails than the limit allows
        for i in range(5):
            success = notifier.send_system_alert(
                alert_type=f"Rate Limit Test {i+1}",
                severity="INFO",
                component="rate_limiter",
                alert_description=f"Test notification {i+1} to verify rate limiting"
            )

            if success:
                sent_count += 1
                print(f"✓ Notification {i+1} queued")
            else:
                rate_limited_count += 1
                print(f"⚠ Notification {i+1} rate limited")

            # Small delay between attempts
            await asyncio.sleep(0.5)

        print(f"✓ Rate limiting test completed: {sent_count} queued, {rate_limited_count} rate limited")

        # Wait for processing
        await asyncio.sleep(5)

        stats = notifier.get_statistics()
        print(f"✓ Rate limiting statistics: {stats['emails_rate_limited']} rate limited")

        return rate_limited_count > 0  # Success if some were rate limited

    finally:
        await notifier.stop()


def display_configuration_help():
    """Display help for configuring email credentials."""
    print("\n" + "="*60)
    print("EMAIL CONFIGURATION HELP")
    print("="*60)
    print("\nTo use the email notifier, you need to set up email credentials.")
    print("The system supports the following environment variables:")
    print("\nRequired:")
    print("  EMAIL_SMTP_SERVER    - SMTP server with port (e.g., 'smtp.gmail.com:587')")
    print("  EMAIL_USERNAME       - Your email address")
    print("  EMAIL_PASSWORD       - Your email password or app password")
    print("  EMAIL_FROM_ADDRESS   - Email address to send from (can be same as username)")
    print("\nFor Gmail:")
    print("  1. Enable 2-factor authentication")
    print("  2. Generate an app password: https://myaccount.google.com/apppasswords")
    print("  3. Use the app password as EMAIL_PASSWORD")
    print("\nExample setup:")
    print("  export EMAIL_SMTP_SERVER='smtp.gmail.com:587'")
    print("  export EMAIL_USERNAME='your-email@gmail.com'")
    print("  export EMAIL_PASSWORD='your-app-password'")
    print("  export EMAIL_FROM_ADDRESS='your-email@gmail.com'")
    print("\nAlternatively, you can store credentials securely using the platform")
    print("credential manager, which encrypts and stores them locally.")
    print("="*60)


async def main():
    """Main test function."""
    print("ARC Prize 2025 - Email Notifier Test Suite")
    print("="*50)

    # Check if email credentials are available
    notifier = get_email_notifier()
    if not notifier._validate_setup():
        print("✗ Email configuration is not valid")
        display_configuration_help()
        return False

    print("✓ Email configuration appears valid")

    # Test connection
    connection_ok = await test_email_connection()
    if not connection_ok:
        print("\n✗ Email connection failed. Please check your configuration.")
        display_configuration_help()
        return False

    # Run tests
    tests = [
        ("Basic Notification", test_basic_notification),
        ("Experiment Notifications", test_experiment_notifications),
        ("Rate Limiting", test_rate_limiting)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                print(f"✓ {test_name} test passed")
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name:<25} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Email notifier is working correctly.")
        return True
    else:
        print("⚠ Some tests failed. Please review the configuration and logs.")
        return False


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
