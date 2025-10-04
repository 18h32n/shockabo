#!/usr/bin/env python3
"""
Platform rotation script for automated switching between platforms.

This script manages the automated rotation between Kaggle, Colab, Paperspace,
and local platforms to maximize GPU utilization.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infrastructure.components import (
    AvailabilityStatus,
    Platform,
    get_availability_checker,
    get_platform_detector,
)


class PlatformRotationManager:
    """Manages automated platform rotation for experiment execution."""

    def __init__(self, config_path: str | None = None):
        self.detector = get_platform_detector()
        self.availability_checker = get_availability_checker()
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self._setup_state_file()

    def _load_config(self, config_path: str | None) -> dict:
        """Load rotation configuration."""
        default_config = {
            "rotation_strategy": "best_available",  # "best_available", "round_robin", "quota_based"
            "min_experiment_hours": 0.5,
            "max_experiment_hours": 8.0,
            "rotation_interval_minutes": 30,
            "platform_priorities": {
                "kaggle": 1,
                "colab": 2,
                "paperspace": 3,
                "local": 0
            },
            "quota_safety_margin": 0.2,  # 20% safety margin
            "email_notifications": False,
            "state_file": "platform_rotation_state.json",
            "max_rotation_attempts": 3
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for rotation manager."""
        logger = logging.getLogger('platform_rotation')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'platform_rotation.log')
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    def _setup_state_file(self):
        """Setup state file for tracking rotation state."""
        self.state_file = Path(self.config['state_file'])
        if not self.state_file.exists():
            self._save_state({
                'current_platform': None,
                'rotation_history': [],
                'last_rotation_time': None,
                'experiment_queue': [],
                'total_rotations': 0
            })

    def _load_state(self) -> dict:
        """Load rotation state from file."""
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return {}

    def _save_state(self, state: dict):
        """Save rotation state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    async def check_rotation_needed(self, estimated_runtime_hours: float = 2.0) -> tuple[bool, str]:
        """
        Check if platform rotation is needed.
        
        Args:
            estimated_runtime_hours: Expected runtime for next experiment
            
        Returns:
            Tuple of (rotation_needed, reason)
        """
        current_platform = self.detector.detect_platform().platform
        current_check = await self.availability_checker.check_availability(
            current_platform, estimated_runtime_hours
        )

        # Check if current platform can handle the experiment
        if current_check.can_start_experiment:
            # Check quota safety margin
            if current_check.quota_info.gpu_hours_remaining != -1:  # Not unlimited
                safety_hours = estimated_runtime_hours * (1 + self.config['quota_safety_margin'])
                if current_check.quota_info.gpu_hours_remaining < safety_hours:
                    return True, f"Approaching quota limit on {current_platform.value}"

            # Check session timeout
            if (current_check.quota_info.session_timeout_hours > 0 and
                current_check.quota_info.session_duration_hours + estimated_runtime_hours >
                current_check.quota_info.session_timeout_hours * 0.9):  # 90% of timeout
                return True, f"Approaching session timeout on {current_platform.value}"

            return False, f"Current platform {current_platform.value} is sufficient"

        # Current platform can't handle experiment
        if current_check.status == AvailabilityStatus.QUOTA_EXCEEDED:
            return True, f"Quota exceeded on {current_platform.value}"
        elif current_check.status == AvailabilityStatus.TIMEOUT_APPROACHING:
            return True, f"Session timeout approaching on {current_platform.value}"
        else:
            return True, f"Platform {current_platform.value} unavailable"

    async def find_best_platform(self, estimated_runtime_hours: float = 2.0) -> Platform | None:
        """
        Find the best available platform for experiment execution.
        
        Args:
            estimated_runtime_hours: Expected runtime for experiment
            
        Returns:
            Best platform or None if none available
        """
        strategy = self.config['rotation_strategy']

        if strategy == "best_available":
            return await self._find_best_available_platform(estimated_runtime_hours)
        elif strategy == "round_robin":
            return await self._find_round_robin_platform(estimated_runtime_hours)
        elif strategy == "quota_based":
            return await self._find_quota_based_platform(estimated_runtime_hours)
        else:
            self.logger.error(f"Unknown rotation strategy: {strategy}")
            return await self._find_best_available_platform(estimated_runtime_hours)

    async def _find_best_available_platform(self, estimated_runtime_hours: float) -> Platform | None:
        """Find best available platform based on quota and priority."""
        best_check = await self.availability_checker.get_best_platform(
            estimated_runtime_hours, prefer_current=False
        )
        return best_check.platform if best_check else None

    async def _find_round_robin_platform(self, estimated_runtime_hours: float) -> Platform | None:
        """Find next platform in round-robin order."""
        state = self._load_state()
        current_platform = state.get('current_platform')

        # Platform order for round-robin
        platforms = [Platform.KAGGLE, Platform.COLAB, Platform.PAPERSPACE, Platform.LOCAL]

        # Find current index
        try:
            current_index = platforms.index(Platform(current_platform)) if current_platform else -1
        except (ValueError, TypeError):
            current_index = -1

        # Try platforms in round-robin order
        for i in range(len(platforms)):
            next_index = (current_index + 1 + i) % len(platforms)
            platform = platforms[next_index]

            check = await self.availability_checker.check_availability(platform, estimated_runtime_hours)
            if check.can_start_experiment:
                return platform

        return None

    async def _find_quota_based_platform(self, estimated_runtime_hours: float) -> Platform | None:
        """Find platform with most available quota."""
        checks = await self.availability_checker.check_all_platforms(estimated_runtime_hours)
        available_checks = [c for c in checks if c.can_start_experiment]

        if not available_checks:
            return None

        # Sort by remaining quota (descending)
        available_checks.sort(
            key=lambda c: c.quota_info.gpu_hours_remaining if c.quota_info.gpu_hours_remaining != -1 else float('inf'),
            reverse=True
        )

        return available_checks[0].platform

    async def rotate_to_platform(self, target_platform: Platform) -> bool:
        """
        Rotate to a specific platform.
        
        Args:
            target_platform: Platform to rotate to
            
        Returns:
            True if rotation successful, False otherwise
        """
        current_platform = self.detector.detect_platform().platform

        if current_platform == target_platform:
            self.logger.info(f"Already on target platform: {target_platform.value}")
            return True

        self.logger.info(f"Rotating from {current_platform.value} to {target_platform.value}")

        try:
            # Save current state before rotation
            await self._save_experiment_state()

            # Execute platform-specific rotation logic
            success = await self._execute_platform_rotation(current_platform, target_platform)

            if success:
                # Update state
                state = self._load_state()
                state['current_platform'] = target_platform.value
                state['last_rotation_time'] = datetime.now().isoformat()
                state['total_rotations'] += 1
                state['rotation_history'].append({
                    'from': current_platform.value,
                    'to': target_platform.value,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
                self._save_state(state)

                # Start session tracking on new platform
                self.availability_checker.start_session_tracking(target_platform)

                self.logger.info(f"Successfully rotated to {target_platform.value}")
                return True
            else:
                self.logger.error(f"Failed to rotate to {target_platform.value}")
                return False

        except Exception as e:
            self.logger.error(f"Error during rotation: {e}")
            return False

    async def _save_experiment_state(self):
        """Save current experiment state before rotation."""
        try:
            # Import necessary modules
            from utils.gcs_integration import CheckpointMetadata, create_gcs_manager_from_env

            # Get GCS manager
            gcs_manager = create_gcs_manager_from_env()
            if not gcs_manager:
                self.logger.warning("GCS manager not available - skipping cloud backup")
                return

            # Get current platform
            current_platform = self.detector.detect_platform().platform

            # Get state file path
            state = self._load_state()
            experiment_queue = state.get('experiment_queue', [])

            if experiment_queue:
                # Save queue state to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump({
                        'platform': current_platform.value,
                        'queue': experiment_queue,
                        'timestamp': datetime.now().isoformat(),
                        'rotation_count': state.get('total_rotations', 0)
                    }, f, indent=2)
                    temp_path = f.name

                # Upload to GCS
                metadata = CheckpointMetadata(
                    name=f"rotation_state_{current_platform.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    version="1.0",
                    created_at=datetime.now(),
                    size_bytes=Path(temp_path).stat().st_size,
                    platform=current_platform.value,
                    experiment_id="platform_rotation",
                    tags=["rotation", "state", "queue"]
                )

                success = gcs_manager.upload_checkpoint(
                    temp_path,
                    metadata.name,
                    metadata
                )

                # Clean up temp file
                Path(temp_path).unlink()

                if success:
                    self.logger.info(f"Saved experiment state to GCS: {metadata.name}")
                else:
                    self.logger.error("Failed to save experiment state to GCS")
            else:
                self.logger.info("No experiments in queue to save")

        except Exception as e:
            self.logger.error(f"Error saving experiment state: {e}")
            # Don't fail rotation if state save fails
            pass

    async def _execute_platform_rotation(self, from_platform: Platform, to_platform: Platform) -> bool:
        """Execute the actual platform rotation."""
        try:
            # Platform-specific rotation scripts
            script_map = {
                Platform.KAGGLE: 'kaggle_setup.py',
                Platform.COLAB: 'colab_setup.py',
                Platform.PAPERSPACE: 'paperspace_setup.py'
            }

            if to_platform in script_map:
                script_name = script_map[to_platform]
                script_path = Path(__file__).parent / 'platform_deploy' / script_name

                if script_path.exists():
                    # Execute platform setup script
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300
                    )

                    if result.returncode == 0:
                        self.logger.info("Platform setup script completed successfully")
                        return True
                    else:
                        self.logger.error(f"Platform setup failed: {result.stderr}")
                        return False
                else:
                    self.logger.error(f"Setup script not found: {script_path}")
                    return False
            elif to_platform == Platform.LOCAL:
                # Local platform doesn't need special setup
                return True
            else:
                self.logger.error(f"Unknown target platform: {to_platform}")
                return False

        except Exception as e:
            self.logger.error(f"Rotation execution failed: {e}")
            return False

    async def automatic_rotation_loop(self, estimated_runtime_hours: float = 2.0):
        """Run automatic rotation loop."""
        self.logger.info("Starting automatic rotation loop...")

        rotation_interval = timedelta(minutes=self.config['rotation_interval_minutes'])
        last_check = datetime.now() - rotation_interval  # Force immediate check

        while True:
            try:
                now = datetime.now()

                # Check if it's time for rotation check
                if now - last_check >= rotation_interval:
                    last_check = now

                    # Check if rotation is needed
                    rotation_needed, reason = await self.check_rotation_needed(estimated_runtime_hours)

                    if rotation_needed:
                        self.logger.info(f"Rotation needed: {reason}")

                        # Find best platform
                        best_platform = await self.find_best_platform(estimated_runtime_hours)

                        if best_platform:
                            # Attempt rotation
                            success = await self.rotate_to_platform(best_platform)
                            if not success:
                                self.logger.error("Rotation failed, continuing on current platform")
                        else:
                            self.logger.warning("No suitable platform found for rotation")
                    else:
                        self.logger.debug(f"No rotation needed: {reason}")

                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                self.logger.info("Rotation loop interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in rotation loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def get_status_report(self) -> dict:
        """Get current rotation status report."""
        current_platform = self.detector.detect_platform()
        current_availability = await self.availability_checker.check_availability(
            current_platform.platform, 2.0
        )

        all_platforms = await self.availability_checker.check_all_platforms(2.0)
        state = self._load_state()

        return {
            'current_platform': {
                'name': current_platform.platform.value,
                'gpu_available': current_platform.gpu_available,
                'gpu_count': current_platform.gpu_count,
                'memory_gb': current_platform.memory_gb,
                'availability': {
                    'status': current_availability.status.value,
                    'can_start_experiment': current_availability.can_start_experiment,
                    'quota_remaining': current_availability.quota_info.gpu_hours_remaining,
                    'warnings': current_availability.warnings or []
                }
            },
            'all_platforms': [
                {
                    'name': check.platform.value,
                    'status': check.status.value,
                    'can_start_experiment': check.can_start_experiment,
                    'quota_remaining': check.quota_info.gpu_hours_remaining,
                    'next_available': check.next_available_time.isoformat() if check.next_available_time else None
                }
                for check in all_platforms
            ],
            'rotation_stats': {
                'total_rotations': state.get('total_rotations', 0),
                'last_rotation': state.get('last_rotation_time'),
                'rotation_history_count': len(state.get('rotation_history', []))
            },
            'config': self.config
        }


async def main():
    """Main entry point for the rotation script."""
    parser = argparse.ArgumentParser(description='Platform rotation management')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--runtime-hours', type=float, default=2.0,
                       help='Estimated runtime hours for experiments')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check rotation status, don\'t execute')
    parser.add_argument('--auto-rotate', action='store_true',
                       help='Run automatic rotation loop')
    parser.add_argument('--rotate-to', choices=['kaggle', 'colab', 'paperspace', 'local'],
                       help='Manually rotate to specific platform')
    parser.add_argument('--status', action='store_true',
                       help='Show current status report')

    args = parser.parse_args()

    # Create rotation manager
    manager = PlatformRotationManager(args.config)

    try:
        if args.status:
            # Show status report
            report = await manager.get_status_report()
            print(json.dumps(report, indent=2, default=str))

        elif args.check_only:
            # Check if rotation is needed
            needed, reason = await manager.check_rotation_needed(args.runtime_hours)
            print(f"Rotation needed: {needed}")
            print(f"Reason: {reason}")

            if needed:
                best_platform = await manager.find_best_platform(args.runtime_hours)
                if best_platform:
                    print(f"Recommended platform: {best_platform.value}")
                else:
                    print("No suitable platform found")

        elif args.rotate_to:
            # Manual rotation
            target_platform = Platform(args.rotate_to)
            success = await manager.rotate_to_platform(target_platform)
            if success:
                print(f"Successfully rotated to {target_platform.value}")
            else:
                print(f"Failed to rotate to {target_platform.value}")
                sys.exit(1)

        elif args.auto_rotate:
            # Automatic rotation loop
            await manager.automatic_rotation_loop(args.runtime_hours)

        else:
            # Default: show help
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
