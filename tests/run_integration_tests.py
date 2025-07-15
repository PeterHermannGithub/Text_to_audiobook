#!/usr/bin/env python3
"""
Integration test runner for the distributed text-to-audiobook pipeline.

This script provides a comprehensive test runner for all integration tests
with various options for different test scenarios.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_command(cmd: List[str], cwd: Optional[str] = None, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        raise
    except Exception as e:
        print(f"Command failed: {e}")
        raise


def check_prerequisites() -> bool:
    """Check if all prerequisites are installed."""
    print("Checking prerequisites...")
    
    requirements = [
        ('python', '--version'),
        ('pip', '--version'),
        ('pytest', '--version')
    ]
    
    missing = []
    for req, version_flag in requirements:
        try:
            result = run_command([req, version_flag], timeout=10)
            if result.returncode != 0:
                missing.append(req)
            else:
                print(f"✓ {req} is available")
        except Exception:
            missing.append(req)
    
    if missing:
        print(f"❌ Missing prerequisites: {', '.join(missing)}")
        return False
    
    print("✓ All prerequisites are available")
    return True


def install_test_dependencies() -> bool:
    """Install test dependencies."""
    print("Installing test dependencies...")
    
    test_requirements = [
        'pytest>=7.0.0',
        'pytest-mock>=3.6.0',
        'pytest-cov>=4.0.0',
        'pytest-benchmark>=4.0.0',
        'pytest-xdist>=3.0.0',
        'pytest-timeout>=2.1.0',
        'pytest-html>=3.1.0',
        'pytest-json-report>=1.5.0'
    ]
    
    try:
        for requirement in test_requirements:
            result = run_command(['pip', 'install', requirement])
            if result.returncode != 0:
                print(f"❌ Failed to install {requirement}")
                print(result.stderr)
                return False
        
        print("✓ Test dependencies installed successfully")
        return True
    
    except Exception as e:
        print(f"❌ Failed to install test dependencies: {e}")
        return False


def run_unit_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run unit tests."""
    print("Running unit tests...")
    
    cmd = ['pytest', 'tests/unit/', '-v' if verbose else '-q']
    
    if coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])
    
    result = run_command(cmd)
    
    if result.returncode == 0:
        print("✓ Unit tests passed")
        return True
    else:
        print("❌ Unit tests failed")
        print(result.stdout)
        print(result.stderr)
        return False


def run_integration_tests(
    verbose: bool = False,
    test_pattern: str = "*",
    parallel: bool = False,
    timeout: int = 300,
    markers: Optional[List[str]] = None
) -> bool:
    """Run integration tests."""
    print("Running integration tests...")
    
    cmd = ['pytest', 'tests/integration/', '-v' if verbose else '-q']
    
    # Add test pattern
    if test_pattern != "*":
        cmd.extend(['-k', test_pattern])
    
    # Add markers
    if markers:
        for marker in markers:
            cmd.extend(['-m', marker])
    
    # Add parallel execution
    if parallel:
        cmd.extend(['-n', 'auto'])
    
    # Add timeout
    cmd.extend(['--timeout', str(timeout)])
    
    # Add HTML report
    cmd.extend(['--html=reports/integration_test_report.html', '--self-contained-html'])
    
    # Add JSON report
    cmd.extend(['--json-report', '--json-report-file=reports/integration_test_report.json'])
    
    result = run_command(cmd, timeout=timeout + 60)
    
    if result.returncode == 0:
        print("✓ Integration tests passed")
        return True
    else:
        print("❌ Integration tests failed")
        print(result.stdout)
        print(result.stderr)
        return False


def run_performance_tests(verbose: bool = False) -> bool:
    """Run performance tests."""
    print("Running performance tests...")
    
    cmd = [
        'pytest', 'tests/integration/', '-v' if verbose else '-q',
        '-m', 'performance',
        '--benchmark-only',
        '--benchmark-html=reports/performance_report.html',
        '--benchmark-json=reports/performance_report.json'
    ]
    
    result = run_command(cmd, timeout=600)
    
    if result.returncode == 0:
        print("✓ Performance tests passed")
        return True
    else:
        print("❌ Performance tests failed")
        print(result.stdout)
        print(result.stderr)
        return False


def run_smoke_tests(verbose: bool = False) -> bool:
    """Run smoke tests for basic functionality."""
    print("Running smoke tests...")
    
    cmd = [
        'pytest', 'tests/integration/', '-v' if verbose else '-q',
        '-k', 'test_dag_structure or test_spark_integration or test_kafka_integration',
        '--timeout=60'
    ]
    
    result = run_command(cmd)
    
    if result.returncode == 0:
        print("✓ Smoke tests passed")
        return True
    else:
        print("❌ Smoke tests failed")
        print(result.stdout)
        print(result.stderr)
        return False


def run_end_to_end_tests(verbose: bool = False) -> bool:
    """Run end-to-end tests."""
    print("Running end-to-end tests...")
    
    cmd = [
        'pytest', 'tests/integration/', '-v' if verbose else '-q',
        '-k', 'test_end_to_end',
        '--timeout=300'
    ]
    
    result = run_command(cmd)
    
    if result.returncode == 0:
        print("✓ End-to-end tests passed")
        return True
    else:
        print("❌ End-to-end tests failed")
        print(result.stdout)
        print(result.stderr)
        return False


def setup_test_environment() -> bool:
    """Set up the test environment."""
    print("Setting up test environment...")
    
    # Create necessary directories
    directories = [
        'reports',
        'logs',
        'tmp/test_input',
        'tmp/test_output',
        'tmp/test_airflow'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Set environment variables
    test_env = {
        'PYTHONPATH': f"{project_root}/src:{project_root}/tests",
        'AIRFLOW_HOME': f"{project_root}/tmp/test_airflow",
        'TESTING': 'true',
        'LOG_LEVEL': 'DEBUG'
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    print("✓ Test environment set up successfully")
    return True


def generate_test_report(results: Dict[str, bool]) -> None:
    """Generate a comprehensive test report."""
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    # Generate JSON report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_suites': total_tests,
        'passed_suites': passed_tests,
        'failed_suites': failed_tests,
        'success_rate': passed_tests/total_tests*100,
        'results': results
    }
    
    with open('reports/test_summary.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed reports available in: reports/")
    print("="*60)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Integration test runner for text-to-audiobook pipeline"
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Run all test suites'
    )
    parser.add_argument(
        '--unit', action='store_true',
        help='Run unit tests'
    )
    parser.add_argument(
        '--integration', action='store_true',
        help='Run integration tests'
    )
    parser.add_argument(
        '--performance', action='store_true',
        help='Run performance tests'
    )
    parser.add_argument(
        '--smoke', action='store_true',
        help='Run smoke tests'
    )
    parser.add_argument(
        '--e2e', action='store_true',
        help='Run end-to-end tests'
    )
    parser.add_argument(
        '--pattern', default='*',
        help='Test pattern to match (default: *)'
    )
    parser.add_argument(
        '--markers', nargs='+',
        help='Test markers to run (e.g., integration, performance)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--parallel', '-p', action='store_true',
        help='Run tests in parallel'
    )
    parser.add_argument(
        '--coverage', action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--timeout', type=int, default=300,
        help='Test timeout in seconds (default: 300)'
    )
    parser.add_argument(
        '--install-deps', action='store_true',
        help='Install test dependencies'
    )
    parser.add_argument(
        '--setup-only', action='store_true',
        help='Only set up test environment'
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            sys.exit(1)
    
    # Set up test environment
    if not setup_test_environment():
        sys.exit(1)
    
    if args.setup_only:
        print("Test environment setup complete")
        return
    
    # Determine which tests to run
    test_results = {}
    
    if args.all or args.unit:
        test_results['Unit Tests'] = run_unit_tests(
            verbose=args.verbose,
            coverage=args.coverage
        )
    
    if args.all or args.smoke:
        test_results['Smoke Tests'] = run_smoke_tests(verbose=args.verbose)
    
    if args.all or args.integration:
        test_results['Integration Tests'] = run_integration_tests(
            verbose=args.verbose,
            test_pattern=args.pattern,
            parallel=args.parallel,
            timeout=args.timeout,
            markers=args.markers
        )
    
    if args.all or args.performance:
        test_results['Performance Tests'] = run_performance_tests(verbose=args.verbose)
    
    if args.all or args.e2e:
        test_results['End-to-End Tests'] = run_end_to_end_tests(verbose=args.verbose)
    
    # If no specific tests were requested, run smoke tests
    if not test_results:
        test_results['Smoke Tests'] = run_smoke_tests(verbose=args.verbose)
    
    # Generate test report
    generate_test_report(test_results)
    
    # Exit with appropriate code
    if all(test_results.values()):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()