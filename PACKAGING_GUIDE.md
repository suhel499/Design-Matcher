# Cross-Platform Packaging & Distribution Guide: Windows & macOS

This guide provides complete instructions and automated scripts to package **Design Matcher** into production installers for **Windows** (`.exe` setup wizard) and **macOS** (`.dmg` drag-and-drop disk image & `.app` bundle).

---

## 1. Quick Start (One-Click Automated Packaging)

### On Windows
Run either the Batch or PowerShell script from the repository root:

```cmd
:: Method A: Windows Command Prompt
package_windows.bat
```

```powershell
# Method B: PowerShell (with optional code-signing)
.\package_windows.ps1 -CertThumbprint "YOUR_AUTHENTICODE_THUMBPRINT"
```
**Output:** `Output\DesignMatcher_Setup_v1.0.0.exe`

---

### On macOS
Run the shell script from Terminal:

```bash
# Optional: Set your Apple Developer certificate ID if signing
# export APPLE_SIGN_IDENTITY="Developer ID Application: Your Company (TEAM_ID)"

./package_macos.sh
```
**Output:**
- `dist/DesignMatcher.app` (Native macOS Application Bundle)
- `dist/DesignMatcher_v1.0.0.dmg` (Drag-to-Applications Disk Image Installer)

---

## 2. Windows Packaging Details

### 2.1. Prerequisites
1. **Operating System:** Windows 10 (64-bit) or Windows 11 (64-bit).
2. **Python:** Python 3.11, 3.12, or 3.13 (64-bit).
3. **Inno Setup:** Version 6.x or newer from [jrsoftware.org](https://jrsoftware.org/isdl.php). (Installed to `C:\Program Files (x86)\Inno Setup 6\`).
4. **Visual C++ Redistributable:** VC++ 2015-2022 x64 installed.
5. *(Optional for release)* **Code Signing Certificate:** Standard or EV certificate with `signtool.exe` (from Windows SDK).

### 2.2. Step-by-Step Manual Build

```powershell
# 1. Activate virtual environment
. .venv\Scripts\Activate.ps1

# 2. Install build dependencies
python -m pip install --upgrade pip setuptools wheel pyinstaller pillow

# 3. Compile standalone folder with PyInstaller
pyinstaller --clean -y design_matcher.spec

# 4. (Optional) Sign executable and DLLs
& "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe" sign `
    /tr http://timestamp.digicert.com /td sha256 /fd sha256 `
    /sha1 "CERT_THUMBPRINT" dist\DesignMatcher\DesignMatcher.exe

# 5. Compile Inno Setup Installer
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss

# 6. (Optional) Sign the final setup executable
& "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe" sign `
    /tr http://timestamp.digicert.com /td sha256 /fd sha256 `
    /sha1 "CERT_THUMBPRINT" Output\DesignMatcher_Setup_v1.0.0.exe
```

### 2.3. Windows Installer Features (`installer.iss`)
- **Per-User or Machine-Wide:** Default `PrivilegesRequired=lowest` allows non-admin users to install into `%LOCALAPPDATA%\Programs\Design Matcher` without UAC elevation.
- **Shortcuts:** Automatically creates Start Menu and Desktop shortcuts with custom icon `assets\icon.ico`.
- **File Associations:**
  - `.dxf`: AutoCAD DXF linework drawings open directly with Design Matcher.
  - `.lic`: Double-clicking an exported signed license file installs it.
  - `.dmreq`: Offline license activation requests.
- **Uninstaller:** Leaves user configuration and license cache in `%LOCALAPPDATA%\DesignMatcher` intact so reinstalls or updates retain the user's active license.

---

## 3. macOS Packaging Details

### 3.1. Prerequisites
1. **Operating System:** macOS Monterey (12.0) or newer (Apple Silicon M1/M2/M3/M4 or Intel x86_64).
2. **Python:** Python 3.11, 3.12, or 3.13.
3. **Xcode Command Line Tools:** `xcode-select --install`
4. *(Optional for release)* **Apple Developer ID Certificate:** For Gatekeeper compliance and Apple Notarization.

### 3.2. Step-by-Step Manual Build

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install build dependencies
pip install --upgrade pip pyinstaller pillow

# 3. Generate icon bundle (if assets/icon.icns does not exist)
# Handled automatically by package_macos.sh using iconutil

# 4. Compile .app bundle with PyInstaller
pyinstaller --clean -y design_matcher.spec

# 5. Sign application bundle
# Ad-hoc signing (local test):
codesign --force --deep --sign - "dist/DesignMatcher.app"

# OR with Apple Developer ID:
# codesign --force --deep --options runtime --sign "Developer ID Application: Your Name (ID)" "dist/DesignMatcher.app"

# 6. Package into Drag-and-Drop .dmg
mkdir -p dist/dmg_staging
cp -R "dist/DesignMatcher.app" dist/dmg_staging/
ln -s /Applications dist/dmg_staging/Applications

hdiutil create \
    -volname "Design Matcher" \
    -srcfolder dist/dmg_staging \
    -ov \
    -format UDZO \
    dist/DesignMatcher_v1.0.0.dmg

rm -rf dist/dmg_staging
```

### 3.3. macOS Apple Notarization (For Public Distribution)
To eliminate the *"Apple could not verify this software"* Gatekeeper alert on client Macs:

```bash
# 1. Sign app with hardened runtime & secure timestamp
codesign --force --deep --options runtime \
    --sign "Developer ID Application: Your Company (TEAM_ID)" \
    "dist/DesignMatcher.app"

# 2. Package signed app into DMG
# (Run DMG creation step above)

# 3. Sign the DMG
codesign --force --sign "Developer ID Application: Your Company (TEAM_ID)" \
    "dist/DesignMatcher_v1.0.0.dmg"

# 4. Submit to Apple Notary Service
xcrun notarytool submit "dist/DesignMatcher_v1.0.0.dmg" \
    --keychain-profile "AC_NOTARY" \
    --wait

# 5. Staple notarization ticket to DMG
xcrun stapler staple "dist/DesignMatcher_v1.0.0.dmg"
```

---

## 4. Architecture & Security Checklist

> [!CRITICAL]
> **Private Key Exclusion Rule**
> The client desktop application bundle MUST NEVER contain private keys or server database files:
> 1. Verify `licensing_server/` is NOT bundled into the client package (it is completely excluded in `design_matcher.spec`).
> 2. Verify that `design_matcher/app/core/licensing/public_key.py` only contains the **Public Key** (`BEGIN PUBLIC KEY`).
> 3. Verify that `licensing.db`, `.pem`, `.env`, and private keys are in `.gitignore`.

---

## 5. File Structure Summary

| File / Folder | Platform | Purpose |
| :--- | :--- | :--- |
| `design_matcher.spec` | Cross-Platform | Master PyInstaller specification for both Windows & macOS |
| `design_matcher/build.spec`| Cross-Platform | Compatibility wrapper delegating to `design_matcher.spec` |
| `installer.iss` | Windows | Inno Setup 6 compilation script for `.exe` installer wizard |
| `package_windows.bat` | Windows | One-click command prompt packaging script |
| `package_windows.ps1` | Windows | One-click PowerShell packaging script with Authenticode signing |
| `package_macos.sh` | macOS | One-click shell script for `.app` bundle & `.dmg` image creation |
| `assets/icon.ico` | Windows | Multi-resolution application icon (16x16 to 256x256) |
| `assets/icon.icns` | macOS | High-resolution Apple icon bundle (16x16 to 1024x1024 Retina) |
| `assets/icon.png` | Cross-Platform | 1024x1024 master icon artwork |
