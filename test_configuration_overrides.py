#!/usr/bin/env python3
"""
Configuration Override Testing for Platform Compatibility
========================================================

This script tests configuration overrides across different platforms
to ensure platform-specific settings are correctly applied.

Run with: python test_configuration_overrides.py
"""

import json
import yaml
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ConfigOverrideTestResult:
    """Result of configuration override test."""
    platform: str
    override_category: str
    test_name: str
    success: bool
    expected_value: Any
    actual_value: Any
    error_message: Optional[str] = None


class ConfigurationOverrideTester:
    """Test configuration overrides across platforms."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize configuration override tester."""
        self.config_dir = config_dir or Path("configs")
        self.results: List[ConfigOverrideTestResult] = []
        
        # Define test platforms
        self.test_platforms = ["kaggle", "colab", "paperspace", "local"]
        
        # Load base configurations
        self.base_configs = {}
        self.platform_configs = {}
        self.strategy_configs = {}
        
        self._load_configurations()

    def _load_configurations(self):
        """Load all configuration files."""
        # Load base development config
        dev_config_path = self.config_dir / "development.yaml"
        if dev_config_path.exists():
            with open(dev_config_path, 'r', encoding='utf-8') as f:
                self.base_configs['development'] = yaml.safe_load(f) or {}
        
        # Load platform-specific configs
        for platform in self.test_platforms:
            config_path = self.config_dir / f"{platform}.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.platform_configs[platform] = yaml.safe_load(f) or {}
        
        # Load strategy configs
        strategies_dir = self.config_dir / "strategies"
        if strategies_dir.exists():
            for strategy_file in strategies_dir.glob("*.yaml"):
                strategy_name = strategy_file.stem
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    self.strategy_configs[strategy_name] = yaml.safe_load(f) or {}

    def _deep_get(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation."""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

    def _record_test_result(self, platform: str, category: str, test_name: str, 
                           expected: Any, actual: Any, error: str = None) -> bool:
        """Record test result."""
        success = expected == actual and error is None
        
        result = ConfigOverrideTestResult(
            platform=platform,
            override_category=category,
            test_name=test_name,
            success=success,
            expected_value=expected,
            actual_value=actual,
            error_message=error
        )
        
        self.results.append(result)
        
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {platform} - {category} - {test_name}")
        if not success:
            print(f"    Expected: {expected}, Got: {actual}")
            if error:
                print(f"    Error: {error}")
        
        return success

    def test_platform_detection_overrides(self) -> Dict[str, int]:
        """Test platform detection and automatic overrides."""
        print("\nTesting platform detection overrides...")
        
        results = {'passed': 0, 'failed': 0}
        
        # Test each platform configuration
        for platform in self.test_platforms:
            if platform not in self.platform_configs:
                continue
            
            config = self.platform_configs[platform]
            
            # Test platform name override
            platform_name = self._deep_get(config, 'platform.name')
            success = self._record_test_result(
                platform, 'platform_detection', 'platform_name',
                platform, platform_name
            )
            results['passed' if success else 'failed'] += 1
            
            # Test memory limits
            memory_limit = self._deep_get(config, 'platform.memory_limit_gb')
            expected_memory = {
                'kaggle': 32,
                'colab': 16,
                'paperspace': None,  # May not be set
                'local': None  # May not be set
            }.get(platform)
            
            if expected_memory is not None:
                success = self._record_test_result(
                    platform, 'platform_detection', 'memory_limit',
                    expected_memory, memory_limit
                )
                results['passed' if success else 'failed'] += 1
            
            # Test GPU hours limit
            gpu_hours = self._deep_get(config, 'platform.gpu_hours_limit')
            expected_gpu_hours = {
                'kaggle': 30,
                'colab': 12,
                'paperspace': None,
                'local': None
            }.get(platform)
            
            if expected_gpu_hours is not None:
                success = self._record_test_result(
                    platform, 'platform_detection', 'gpu_hours_limit',
                    expected_gpu_hours, gpu_hours
                )
                results['passed' if success else 'failed'] += 1
            
            # Test persistent storage
            has_persistent = self._deep_get(config, 'platform.has_persistent_storage')
            expected_persistent = {
                'kaggle': True,
                'colab': False,
                'paperspace': True,
                'local': True
            }.get(platform)
            
            if expected_persistent is not None:
                success = self._record_test_result(
                    platform, 'platform_detection', 'persistent_storage',
                    expected_persistent, has_persistent
                )
                results['passed' if success else 'failed'] += 1
        
        return results

    def test_path_overrides(self) -> Dict[str, int]:
        """Test path configuration overrides."""
        print("\nTesting path overrides...")
        
        results = {'passed': 0, 'failed': 0}
        
        # Expected path patterns for each platform
        expected_paths = {
            'kaggle': {
                'data_dir': '/kaggle/input',
                'working_dir': '/kaggle/working',
                'cache_dir': '/kaggle/working/cache',
                'logs_dir': '/kaggle/working/logs',
                'models_dir': '/kaggle/working/models'
            },
            'colab': {
                'data_dir': '/content/data',
                'working_dir': '/content',
                'cache_dir': '/content/cache',
                'logs_dir': '/content/logs',
                'models_dir': '/content/models'
            }
        }
        
        for platform, expected_platform_paths in expected_paths.items():
            if platform not in self.platform_configs:
                continue
            
            config = self.platform_configs[platform]
            
            for path_type, expected_path in expected_platform_paths.items():
                actual_path = self._deep_get(config, f'paths.{path_type}')
                success = self._record_test_result(
                    platform, 'path_overrides', path_type,
                    expected_path, actual_path
                )
                results['passed' if success else 'failed'] += 1
        
        return results

    def test_model_configuration_overrides(self) -> Dict[str, int]:
        """Test model configuration overrides."""
        print("\nTesting model configuration overrides...")
        
        results = {'passed': 0, 'failed': 0}
        
        # Expected model configurations for platforms
        expected_model_configs = {
            'kaggle': {
                'device': 'cuda',
                'batch_size': 32,
                'use_fp16': True
            },
            'colab': {
                'device': 'cuda',
                'batch_size': 16,
                'use_fp16': True,
                'max_length': 1024
            }
        }
        
        for platform, expected_config in expected_model_configs.items():
            if platform not in self.platform_configs:
                continue
            
            config = self.platform_configs[platform]
            
            for setting, expected_value in expected_config.items():
                actual_value = self._deep_get(config, f'model.{setting}')
                success = self._record_test_result(
                    platform, 'model_config', setting,
                    expected_value, actual_value
                )
                results['passed' if success else 'failed'] += 1
        
        return results

    def test_training_overrides(self) -> Dict[str, int]:
        """Test training configuration overrides."""
        print("\nTesting training configuration overrides...")
        
        results = {'passed': 0, 'failed': 0}
        
        expected_training_configs = {
            'kaggle': {
                'batch_size': 16,
                'gradient_accumulation_steps': 1,
                'use_gpu': True
            },
            'colab': {
                'batch_size': 8,
                'gradient_accumulation_steps': 2,
                'use_gpu': True
            }
        }
        
        for platform, expected_config in expected_training_configs.items():
            if platform not in self.platform_configs:
                continue
            
            config = self.platform_configs[platform]
            
            for setting, expected_value in expected_config.items():
                actual_value = self._deep_get(config, f'training.{setting}')
                success = self._record_test_result(
                    platform, 'training_config', setting,
                    expected_value, actual_value
                )
                results['passed' if success else 'failed'] += 1
        
        return results

    def test_resource_limits_overrides(self) -> Dict[str, int]:
        """Test resource limits overrides."""
        print("\nTesting resource limits overrides...")
        
        results = {'passed': 0, 'failed': 0}
        
        expected_resource_configs = {
            'kaggle': {
                'max_memory_gb': 28,
                'max_concurrent_tasks': 4,
                'gpu_memory_fraction': 0.9
            },
            'colab': {
                'max_memory_gb': 12,
                'max_concurrent_tasks': 2,
                'gpu_memory_fraction': 0.8
            }
        }
        
        for platform, expected_config in expected_resource_configs.items():
            if platform not in self.platform_configs:
                continue
            
            config = self.platform_configs[platform]
            
            for setting, expected_value in expected_config.items():
                actual_value = self._deep_get(config, f'resources.{setting}')
                success = self._record_test_result(
                    platform, 'resource_limits', setting,
                    expected_value, actual_value
                )
                results['passed' if success else 'failed'] += 1
        
        return results

    def test_ttt_strategy_overrides(self) -> Dict[str, int]:
        """Test TTT strategy platform-specific overrides."""
        print("\nTesting TTT strategy overrides...")
        
        results = {'passed': 0, 'failed': 0}
        
        if 'ttt' not in self.strategy_configs:
            print("  TTT strategy config not found, skipping...")
            return results
        
        ttt_config = self.strategy_configs['ttt']
        platform_overrides = ttt_config.get('platform_overrides', {})
        
        # Expected TTT overrides
        expected_ttt_overrides = {
            'kaggle': {
                'training.batch_size': 1,
                'training.gradient_accumulation_steps': 2,
                'training.per_instance_epochs': 1,
                'lora.rank': 32,
                'adaptation.use_chain_augmentation': False
            },
            'colab': {
                'training.batch_size': 1,
                'training.mixed_precision': True,
                'training.gradient_checkpointing': True,
                'model.quantization': True,
                'lora.rank': 32,
                'adaptation.permute_n': 1
            }
        }
        
        for platform, expected_overrides in expected_ttt_overrides.items():
            if platform not in platform_overrides:
                continue
            
            actual_overrides = platform_overrides[platform]
            
            for setting_path, expected_value in expected_overrides.items():
                actual_value = self._deep_get(actual_overrides, setting_path)
                success = self._record_test_result(
                    platform, 'ttt_strategy', setting_path,
                    expected_value, actual_value
                )
                results['passed' if success else 'failed'] += 1
        
        return results

    def test_feature_flags_overrides(self) -> Dict[str, int]:
        """Test feature flags overrides."""
        print("\nTesting feature flags overrides...")
        
        results = {'passed': 0, 'failed': 0}
        
        expected_feature_configs = {
            'kaggle': {
                'hot_reload': False,
                'experiment_tracking': True,
                'ttt_training': True,
                'program_synthesis': True
            },
            'colab': {
                'hot_reload': False,
                'experiment_tracking': True,
                'ttt_training': False,  # Disabled due to memory constraints
                'program_synthesis': True
            }
        }
        
        for platform, expected_config in expected_feature_configs.items():
            if platform not in self.platform_configs:
                continue
            
            config = self.platform_configs[platform]
            
            for feature, expected_value in expected_config.items():
                actual_value = self._deep_get(config, f'features.{feature}')
                success = self._record_test_result(
                    platform, 'feature_flags', feature,
                    expected_value, actual_value
                )
                results['passed' if success else 'failed'] += 1
        
        return results

    def test_configuration_manager_integration(self) -> Dict[str, int]:
        """Test integration with ConfigManager."""
        print("\nTesting ConfigManager integration...")
        
        results = {'passed': 0, 'failed': 0}
        
        try:
            from src.infrastructure.config import ConfigManager
            
            # Test each platform configuration loading
            for platform in self.test_platforms:
                if platform not in self.platform_configs:
                    continue
                
                try:
                    # Mock platform detection by using platform-specific config
                    config_manager = ConfigManager(config_dir=self.config_dir)
                    
                    # Test basic configuration access
                    all_config = config_manager.get_all()
                    success = self._record_test_result(
                        platform, 'config_manager', 'load_success',
                        True, isinstance(all_config, dict)
                    )
                    results['passed' if success else 'failed'] += 1
                    
                    # Test platform info access
                    platform_info = config_manager.get_platform_info()
                    success = self._record_test_result(
                        platform, 'config_manager', 'platform_info',
                        True, platform_info is not None
                    )
                    results['passed' if success else 'failed'] += 1
                    
                except Exception as e:
                    success = self._record_test_result(
                        platform, 'config_manager', 'integration',
                        True, False, str(e)
                    )
                    results['failed'] += 1
        
        except ImportError as e:
            print(f"  Error importing ConfigManager: {e}")
            results['failed'] += 1
        
        return results

    def run_all_override_tests(self) -> Dict[str, Dict[str, int]]:
        """Run all configuration override tests."""
        print("Running configuration override tests...")
        print(f"Found configurations for platforms: {list(self.platform_configs.keys())}")
        print(f"Found strategy configurations: {list(self.strategy_configs.keys())}")
        
        test_results = {}
        
        test_methods = [
            ('Platform Detection', self.test_platform_detection_overrides),
            ('Path Overrides', self.test_path_overrides),
            ('Model Configuration', self.test_model_configuration_overrides),
            ('Training Configuration', self.test_training_overrides),
            ('Resource Limits', self.test_resource_limits_overrides),
            ('TTT Strategy Overrides', self.test_ttt_strategy_overrides),
            ('Feature Flags', self.test_feature_flags_overrides),
            ('ConfigManager Integration', self.test_configuration_manager_integration)
        ]
        
        for test_name, test_method in test_methods:
            test_results[test_name] = test_method()
        
        return test_results

    def generate_override_report(self, output_file: Path = None) -> Dict[str, Any]:
        """Generate configuration override test report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"configuration_override_report_{timestamp}.json")
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Group results by platform and category
        platform_summary = {}
        category_summary = {}
        
        for result in self.results:
            # Platform summary
            if result.platform not in platform_summary:
                platform_summary[result.platform] = {'passed': 0, 'failed': 0}
            platform_summary[result.platform]['passed' if result.success else 'failed'] += 1
            
            # Category summary
            if result.override_category not in category_summary:
                category_summary[result.override_category] = {'passed': 0, 'failed': 0}
            category_summary[result.override_category]['passed' if result.success else 'failed'] += 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'platform_summary': platform_summary,
            'category_summary': category_summary,
            'test_results': [asdict(result) for result in self.results],
            'configuration_files': {
                'platforms': list(self.platform_configs.keys()),
                'strategies': list(self.strategy_configs.keys())
            },
            'recommendations': self._generate_override_recommendations()
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nConfiguration override report saved to: {output_file}")
        return report

    def _generate_override_recommendations(self) -> List[str]:
        """Generate configuration override recommendations."""
        recommendations = []
        
        # Analyze failed tests
        failed_tests = [r for r in self.results if not r.success]
        
        if failed_tests:
            failed_platforms = set(r.platform for r in failed_tests)
            failed_categories = set(r.override_category for r in failed_tests)
            
            recommendations.append(f"Failed tests detected in {len(failed_tests)} configurations")
            recommendations.append(f"Affected platforms: {', '.join(failed_platforms)}")
            recommendations.append(f"Affected categories: {', '.join(failed_categories)}")
        
        # Platform-specific recommendations
        for platform in self.platform_configs:
            platform_failures = [r for r in failed_tests if r.platform == platform]
            if platform_failures:
                recommendations.append(f"Platform {platform}: {len(platform_failures)} configuration issues detected")
        
        # Strategy-specific recommendations
        ttt_failures = [r for r in failed_tests if 'ttt' in r.override_category]
        if ttt_failures:
            recommendations.append("TTT strategy configuration issues detected - verify platform overrides")
        
        # General recommendations
        if not failed_tests:
            recommendations.append("All configuration overrides working correctly")
        else:
            recommendations.append("Review configuration files for missing or incorrect platform overrides")
        
        return recommendations


def main():
    """Main function for configuration override testing."""
    try:
        # Initialize tester
        tester = ConfigurationOverrideTester()
        
        # Run tests
        test_results = tester.run_all_override_tests()
        
        # Generate report
        report = tester.generate_override_report()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"CONFIGURATION OVERRIDE TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        
        print(f"\nPlatform Summary:")
        for platform, stats in report['platform_summary'].items():
            total = stats['passed'] + stats['failed']
            success_rate = stats['passed'] / total if total > 0 else 0
            print(f"  {platform}: {stats['passed']}/{total} ({success_rate:.1%})")
        
        print(f"\nCategory Summary:")
        for category, stats in report['category_summary'].items():
            total = stats['passed'] + stats['failed']
            success_rate = stats['passed'] / total if total > 0 else 0
            print(f"  {category}: {stats['passed']}/{total} ({success_rate:.1%})")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        # Exit with appropriate code
        if report['summary']['failed_tests'] > 0:
            print(f"\nWARNING: {report['summary']['failed_tests']} tests failed!")
            return 1
        else:
            print(f"\nSUCCESS: All configuration override tests passed!")
            return 0
            
    except Exception as e:
        print(f"Error running configuration override tests: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())