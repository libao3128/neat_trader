#!/usr/bin/env python3
"""
Test runner for NEAT Trader package.

This script provides different ways to run the test suite:
1. Run all tests with pytest
2. Run specific test categories
3. Run tests with different verbosity levels

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit            # Run only unit tests
    python run_tests.py --integration     # Run only integration tests
    python run_tests.py --quick           # Run quick tests only
    python run_tests.py --verbose         # Run with verbose output
"""

import sys
import os
import subprocess
import argparse

def run_pytest_tests(test_path="tests", verbose=False, markers=None):
    """Run tests using pytest."""
    if not check_pytest():
        print("pytest not found. Install it with: pip install pytest")
        return False
    
    cmd = [sys.executable, "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    if markers:
        cmd.extend(["-m", markers])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except FileNotFoundError:
        print("pytest not found. Install it with: pip install pytest")
        return False

def run_basic_functionality_test():
    """Run the basic functionality test."""
    print("Running basic functionality test...")
    try:
        result = subprocess.run([sys.executable, "tests/test_basic_functionality.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except FileNotFoundError:
        print("tests/test_basic_functionality.py not found")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ["pandas", "numpy"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print(f"Install them with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_pytest():
    """Check if pytest is installed."""
    try:
        __import__("pytest")
        return True
    except ImportError:
        return False

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run NEAT Trader package tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", action="store_true", help="Run with verbose output")
    parser.add_argument("--basic", action="store_true", help="Run basic functionality test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    print("NEAT Trader Package Test Runner")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    success = True
    
    if args.basic or not any([args.unit, args.integration, args.quick, args.all]):
        # Run basic functionality test by default
        success = run_basic_functionality_test()
    
    if args.unit:
        print("\nRunning unit tests...")
        success = run_pytest_tests("tests/test_*.py", args.verbose, "unit") and success
    
    if args.integration:
        print("\nRunning integration tests...")
        success = run_pytest_tests("tests/test_integration.py", args.verbose, "integration") and success
    
    if args.quick:
        print("\nRunning quick tests...")
        success = run_pytest_tests("tests", args.verbose, "not slow") and success
    
    if args.all:
        print("\nRunning all tests...")
        success = run_pytest_tests("tests", args.verbose) and success
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
