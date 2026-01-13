#!/bin/bash

# TinyHAR GitHub Setup Script
# This script initializes git repository and prepares for GitHub upload

echo "================================================"
echo "TinyHAR - GitHub Repository Setup"
echo "================================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: git is not installed"
    echo "Please install git first: https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git is installed"
echo ""

# Get current directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "📁 Project directory: $PROJECT_DIR"
echo ""

# Initialize git repository
echo "🔧 Initializing git repository..."
cd "$PROJECT_DIR"

if [ -d ".git" ]; then
    echo "⚠️  Git repository already exists"
    read -p "Do you want to reinitialize? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .git
        git init
        echo "✅ Repository reinitialized"
    else
        echo "ℹ️  Using existing repository"
    fi
else
    git init
    echo "✅ Git repository initialized"
fi

echo ""

# Configure git user (if not already configured)
if [ -z "$(git config user.name)" ]; then
    echo "⚙️  Git user not configured"
    read -p "Enter your name: " git_name
    read -p "Enter your email: " git_email
    git config user.name "$git_name"
    git config user.email "$git_email"
    echo "✅ Git user configured"
else
    echo "✅ Git user already configured: $(git config user.name) <$(git config user.email)>"
fi

echo ""

# Add all files
echo "📦 Adding files to git..."
git add .

# Show status
echo ""
echo "📊 Git status:"
git status --short

echo ""

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Complete TinyHAR documentation package

- 2,786 lines of comprehensive documentation
- 5 main documents (Literature Review, Dataset Comparison, Technical Protocols, Implementation Plan)
- 70+ code samples and examples
- 50+ academic references
- Complete project structure
- Dataset download scripts
- Requirements and configuration files

Documentation Status: ✅ Complete
Version: 1.0
Date: January 2026"

echo ""
echo "✅ Initial commit created"
echo ""

# Instructions for GitHub
echo "================================================"
echo "📤 Next Steps: Upload to GitHub"
echo "================================================"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: TinyHAR"
echo "   - Description: Human Activity Recognition on ESP32 with TensorFlow Lite Micro"
echo "   - Public or Private: Your choice"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. After creating the repository, run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/TinyHAR.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "   Replace YOUR_USERNAME with your GitHub username"
echo ""
echo "3. Alternative: Use GitHub CLI (if installed):"
echo ""
echo "   gh repo create TinyHAR --public --source=. --remote=origin"
echo "   git push -u origin main"
echo ""
echo "================================================"
echo "✅ Git repository is ready for GitHub!"
echo "================================================"
echo ""
echo "📊 Repository Statistics:"
echo "   - Files: $(git ls-files | wc -l)"
echo "   - Documentation: $(find docs -name '*.md' | wc -l) files"
echo "   - Scripts: $(find scripts -name '*.py' | wc -l) files"
echo ""
echo "🎉 Setup complete! Follow the instructions above to push to GitHub."

