#!/usr/bin/env python3
"""
Basic functionality test for NEAT Trader package.
"""

import sys
import os

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    from neat_trader import NeatTrader, Evaluator, DataHandler
    print("[OK] Main imports successful")

def test_config():
    """Test configuration."""
    print("Testing configuration...")
    from neat_trader import get_config, NODE_NAMES
    config = get_config()
    assert isinstance(config, dict)
    assert isinstance(NODE_NAMES, dict)
    print("[OK] Configuration working")

def main():
    """Run tests."""
    print("NEAT Trader Package - Basic Test")
    print("=" * 40)
    
    tests = [test_imports, test_config]
    passed = sum(1 for test in tests if test())
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
