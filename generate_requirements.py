"""
GENERATE REQUIREMENTS FOR DOCKER
=================================
This script generates a requirements.txt file with exact versions
from your current Python environment.

Usage:
    python generate_requirements.py

Author: Rafael
Date: 2025-11-10
"""

import sys
import subprocess
import pkg_resources

print("="*80)
print("GENERATING REQUIREMENTS.TXT FOR DOCKER")
print("="*80)
print()

# ============================================================================
# Show Current Environment Info
# ============================================================================

print("Current Environment:")
print("-" * 40)
print(f"Python version: {sys.version}")
print()

# ============================================================================
# Get Installed Packages
# ============================================================================

print("Detecting installed packages...")
print()

# Essential packages for the API
ESSENTIAL_PACKAGES = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'scikit-learn',
    'xgboost',
    'numpy',
    'pandas',
    'joblib',
    'python-multipart',
    'requests',
]

# Optional but recommended
OPTIONAL_PACKAGES = [
    'nltk',
    'imbalanced-learn',
    'scipy',
    'tqdm',
    'lightgbm',
]

installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# ============================================================================
# Generate requirements.txt
# ============================================================================

requirements = []

print("Essential Packages:")
print("-" * 40)
for package in ESSENTIAL_PACKAGES:
    if package in installed_packages:
        version = installed_packages[package]
        requirements.append(f"{package}=={version}")
        print(f"✓ {package:20s} {version}")
    else:
        print(f"✗ {package:20s} NOT INSTALLED")

print()
print("Optional Packages:")
print("-" * 40)
for package in OPTIONAL_PACKAGES:
    if package in installed_packages:
        version = installed_packages[package]
        requirements.append(f"{package}=={version}")
        print(f"✓ {package:20s} {version}")
    else:
        print(f"  {package:20s} (not installed)")

# Add uvicorn extras
if 'uvicorn' in installed_packages:
    # Remove plain uvicorn and add with extras
    requirements = [r for r in requirements if not r.startswith('uvicorn==')]
    requirements.append(f"uvicorn[standard]=={installed_packages['uvicorn']}")

print()

# ============================================================================
# Save to File
# ============================================================================

output_file = 'requirements_docker.txt'

with open(output_file, 'w') as f:
    f.write("# Requirements for Docker Deployment\n")
    f.write(f"# Generated from Python {sys.version.split()[0]}\n")
    f.write("# Date: 2025-11-10\n")
    f.write("\n")
    f.write("# API Framework\n")
    
    api_packages = ['fastapi', 'uvicorn[standard]', 'pydantic', 'python-multipart', 'requests']
    for req in requirements:
        for pkg in api_packages:
            if req.startswith(pkg.split('[')[0]):
                f.write(f"{req}\n")
    
    f.write("\n# Machine Learning\n")
    ml_packages = ['scikit-learn', 'xgboost', 'numpy', 'pandas', 'scipy', 'imbalanced-learn']
    for req in requirements:
        for pkg in ml_packages:
            if req.startswith(pkg):
                f.write(f"{req}\n")
    
    f.write("\n# NLP\n")
    nlp_packages = ['nltk']
    for req in requirements:
        for pkg in nlp_packages:
            if req.startswith(pkg):
                f.write(f"{req}\n")
    
    f.write("\n# Utilities\n")
    util_packages = ['joblib', 'tqdm', 'lightgbm']
    for req in requirements:
        for pkg in util_packages:
            if req.startswith(pkg):
                f.write(f"{req}\n")

print("="*80)
print(f"✓ Requirements saved to: {output_file}")
print("="*80)
print()
print("Next steps:")
print(f"1. Review {output_file}")
print(f"2. Copy it to your project: cp {output_file} requirements.txt")
print("3. Rebuild Docker image: docker build -t book-classifier-api:v2 .")
print()

# ============================================================================
# Show Key Versions
# ============================================================================

print("KEY VERSIONS FOR DOCKER:")
print("-" * 40)

key_versions = {
    'Python': sys.version.split()[0],
    'scikit-learn': installed_packages.get('scikit-learn', 'NOT INSTALLED'),
    'xgboost': installed_packages.get('xgboost', 'NOT INSTALLED'),
    'numpy': installed_packages.get('numpy', 'NOT INSTALLED'),
    'pandas': installed_packages.get('pandas', 'NOT INSTALLED'),
}

for name, version in key_versions.items():
    print(f"{name:15s} {version}")

print()
print("⚠️  IMPORTANT: Your Dockerfile should use Python", sys.version.split()[0].rsplit('.', 1)[0])
print(f"   Update Dockerfile: FROM python:{sys.version.split()[0].rsplit('.', 1)[0]}-slim")
print()


'''
# En Windows (tu entorno conda)
conda activate mid-term-project

python -c "
import sys
from importlib.metadata import distributions

packages = {d.name: d.version for d in distributions()}
essential = ['fastapi', 'uvicorn', 'pydantic', 'scikit-learn', 'xgboost', 'numpy', 'pandas', 'joblib', 'requests', 'python-multipart']

print('=== VERSIONES CLAVE ===')
print(f'Python: {sys.version.split()[0]}')
for pkg in essential:
    ver = packages.get(pkg, packages.get(pkg.replace('-', '_'), 'NOT INSTALLED'))
    print(f'{pkg}: {ver}')
"

'''