#!/usr/bin/env bash
# ==============================================================================
# Design Matcher - macOS Application & DMG Packaging Script
# Compiles native .app bundle and packages it into a distributable .dmg disk image.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="DesignMatcher"
APP_TITLE="Design Matcher"
VERSION="1.0.2"
BUNDLE_ID="com.designmatcher.app"

echo "=========================================================="
echo " Design Matcher - macOS Packaging Workflow (v${VERSION})"
echo "=========================================================="

# 1. Activate Virtual Environment (supports local .venv and CI environments)
if [ -d ".venv" ]; then
    echo "[1/5] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "[1/5] Using active virtual environment: $VIRTUAL_ENV"
else
    echo "[1/5] Using system/environment Python: $(which python3)"
fi

# 2. Verify / Install PyInstaller
echo "[2/5] Verifying packaging tools..."
pip install --upgrade pyinstaller pillow > /dev/null 2>&1

# 3. Ensure Icon Assets Exist
if [ ! -f "assets/icon.icns" ]; then
    echo "  Generating assets/icon.icns..."
    python3 -c "
import os, subprocess
from PIL import Image

iconset = 'assets/icon.iconset'
os.makedirs(iconset, exist_ok=True)
img = Image.open('assets/icon.png')
sizes = [
    (16, 'icon_16x16.png'), (32, 'icon_16x16@2x.png'),
    (32, 'icon_32x32.png'), (64, 'icon_32x32@2x.png'),
    (128, 'icon_128x128.png'), (256, 'icon_128x128@2x.png'),
    (256, 'icon_256x256.png'), (512, 'icon_256x256@2x.png'),
    (512, 'icon_512x512.png'), (1024, 'icon_512x512@2x.png')
]
for s, filename in sizes:
    resized = img.resize((s, s), Image.Resampling.LANCZOS)
    resized.save(os.path.join(iconset, filename))
subprocess.run(['iconutil', '-c', 'icns', iconset, '-o', 'assets/icon.icns'], check=True)
"
fi

# 4. PyInstaller Build (.app Bundle)
echo "[3/5] Building macOS Application Bundle via PyInstaller..."
rm -rf build "dist/${APP_NAME}.app" "dist/${APP_NAME}"
pyinstaller --clean -y design_matcher.spec

APP_PATH="dist/${APP_NAME}.app"
if [ ! -d "$APP_PATH" ]; then
    echo "[ERROR] Application bundle build failed: $APP_PATH was not created!"
    exit 1
fi
echo "  -> Created native application bundle: $APP_PATH"

# 5. Code Signing
echo "[4/5] Code signing application bundle..."
SIGN_IDENTITY="${APPLE_SIGN_IDENTITY:-}"
if [ -n "$SIGN_IDENTITY" ]; then
    echo "  Signing with identity: ${SIGN_IDENTITY}"
    codesign --force --deep --options runtime --sign "${SIGN_IDENTITY}" "$APP_PATH"
else
    echo "  Applying ad-hoc signature (local execution without Apple Developer cert)..."
    codesign --force --deep --sign - "$APP_PATH"
fi

# 6. Package into Distributable Drag-and-Drop .dmg Disk Image
echo "[5/5] Packaging into Drag-and-Drop Disk Image (.dmg)..."
DMG_NAME="DesignMatcher_v${VERSION}.dmg"
DMG_OUTPUT="dist/${DMG_NAME}"
DMG_STAGING="dist/dmg_staging"

rm -rf "$DMG_STAGING" "$DMG_OUTPUT"
mkdir -p "$DMG_STAGING"

# Copy .app bundle to staging
cp -R "$APP_PATH" "$DMG_STAGING/"

# Add symbolic link to /Applications for standard drag-to-install UX
ln -s /Applications "$DMG_STAGING/Applications"

# Create compressed read-only DMG using native hdiutil (fast compression)
hdiutil create \
    -volname "${APP_TITLE}" \
    -srcfolder "$DMG_STAGING" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=1 \
    "$DMG_OUTPUT"

# Optional: Sign DMG if identity provided
if [ -n "$SIGN_IDENTITY" ]; then
    codesign --force --sign "${SIGN_IDENTITY}" "$DMG_OUTPUT"
fi

# Clean up staging
rm -rf "$DMG_STAGING"

echo "=========================================================="
echo " [SUCCESS] macOS Setup Created Successfully!"
echo " App Bundle:  $APP_PATH"
echo " DMG Image:   $DMG_OUTPUT"
echo "=========================================================="
