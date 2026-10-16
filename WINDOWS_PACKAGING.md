# Windows Packaging & Distribution Guide — Design Matcher

This guide details how to build, cryptographically sign, package, and distribute the **Design Matcher** Windows desktop application with integrated Ed25519 subscription licensing.

---

## 1. Prerequisites (Windows Build Environment)

1. **Operating System:** Windows 10 (64-bit) or Windows 11 (64-bit).
2. **Python:** Python 3.11 or 3.12 (64-bit) from [python.org](https://python.org).
3. **C++ Build Tools:** Microsoft Visual C++ 2015-2022 Redistributable installed.
4. **Inno Setup:** Version 6.x or newer (from [jrsoftware.org](https://jrsoftware.org/isdl.php)) for creating professional MSI/EXE installers.
5. **Code Signing Certificate:** Standard or Extended Validation (EV) Code Signing Certificate from a trusted CA (DigiCert, Sectigo, etc.) with `signtool.exe` (installed with Windows SDK).

---

## 2. Virtual Environment & Dependencies

On the Windows build machine, open PowerShell as administrator:

```powershell
# 1. Clone repository or pull latest release tag
cd C:\Build\DesignMatcher

# 2. Create isolated virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel pyinstaller

# 4. Install all project dependencies (including PyTorch, Transformers DINOv2, PySide6, FAISS)
pip install -r requirements.txt
```

---

## 3. Security Audit Prior to Packaging

> [!CRITICAL]
> **Checklist for Private Key Exclusion:**
> 1. Run the following command to ensure NO private keys exist in the `design_matcher` app directory:
>    ```powershell
>    Get-ChildItem -Path design_matcher -Recurse -Include *.pem,*.key,*.env
>    ```
>    The result MUST be empty!
> 2. Inspect `design_matcher/app/core/licensing/public_key.py`:
>    Verify that it contains the **Public Key** (`-----BEGIN PUBLIC KEY-----`) corresponding to your production licensing server, and NOT the private key.

---

## 4. Building with PyInstaller

We have provided a production-configured PyInstaller specification file: `design_matcher/build.spec`.

### 4.1. Run the Build
```powershell
.venv\Scripts\Activate.ps1
pyinstaller --clean design_matcher/build.spec
```

The compiled standalone executable and runtime assets will be generated in `dist/DesignMatcher/`.

### 4.2. Verify the Build Output
Check the directory structure in `dist/DesignMatcher/`:
- `DesignMatcher.exe`: The primary executable.
- `_internal/` (or side-by-side DLLs): PySide6 Qt libraries, OpenCV, FAISS, OpenSSL, and Python runtime.
- Verify that `license.json` or `.license_clock.json` are NOT bundled inside `dist/DesignMatcher` (they are created dynamically in the user's `%LOCALAPPDATA%\DesignMatcher\` at runtime).

---

## 5. Code Signing with Authenticode (`signtool.exe`)

Code signing prevents Windows SmartScreen warnings and guarantees binary integrity.

```powershell
# Set path to Windows SDK signtool.exe
$SignTool = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe"

# Sign all executable binaries and DLLs
& $SignTool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 `
    /sha1 "THUMBPRINT_OF_YOUR_CODE_SIGNING_CERT" `
    dist\DesignMatcher\DesignMatcher.exe

# Sign main PySide6 / DLL dependencies
Get-ChildItem -Path dist\DesignMatcher\*.dll -Recurse | ForEach-Object {
    & $SignTool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 `
        /sha1 "THUMBPRINT_OF_YOUR_CODE_SIGNING_CERT" $_.FullName
}

# Verify signature
& $SignTool verify /pa /v dist\DesignMatcher\DesignMatcher.exe
```

---

## 6. Creating a Professional Installer with Inno Setup

Save the following configuration as `installer.iss` in the project root:

```pascal
; Inno Setup Script for Design Matcher
#define MyAppName "Design Matcher"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Your Company Name"
#define MyAppURL "https://www.yourcompany.com"
#define MyAppExeName "DesignMatcher.exe"

[Setup]
AppId={{E6F74D42-B089-467D-891A-85472CFE8091}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE.txt
OutputDir=Output
OutputBaseFilename=DesignMatcher_Setup_v{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\DesignMatcher\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Do NOT include license files here. Licenses are stored in %LOCALAPPDATA%\DesignMatcher

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
```

### 6.1. Compile the Installer
Run the Inno Setup compiler:
```powershell
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```
This produces `Output\DesignMatcher_Setup_v1.0.0.exe`.

### 6.2. Sign the Final Installer
```powershell
& $SignTool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 `
    /sha1 "THUMBPRINT_OF_YOUR_CODE_SIGNING_CERT" `
    Output\DesignMatcher_Setup_v1.0.0.exe
```

---

## 7. Testing in Windows Sandbox

Before deploying to customers:
1. Open **Windows Sandbox** (isolated clean environment with no Python installed).
2. Copy `DesignMatcher_Setup_v1.0.0.exe` into the Sandbox.
3. Run the installer:
   - Ensure the installation succeeds without admin prompt (per-user install in `%LOCALAPPDATA%\Programs\Design Matcher`).
   - Launch `Design Matcher`.
   - Verify the **License Activation Screen** appears on first launch.
   - Test **Online Activation** with a test license key.
   - Test **Offline Activation**: export `.dmreq`, sign on admin portal, import `.lic`, verify license unlocks all DXF features.
   - Close and reopen application to confirm persistence.
   - Confirm DXF matching, indexing, search, and catalog features function properly.
