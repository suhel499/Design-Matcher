; Inno Setup 6 Script for Design Matcher (Windows x64)
; Generates a professional, signed setup wizard with file associations and desktop shortcuts.

#define MyAppName "Design Matcher"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Design Matcher Software"
#define MyAppURL "https://github.com/suhel499/Design-Matcher"
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
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=DesignMatcher_Setup_v{#MyAppVersion}
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
; Allows users without administrator privileges to install in %LOCALAPPDATA%\Programs
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
WizardResizable=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associate_dxf"; Description: "Associate AutoCAD DXF files (.dxf) with Design Matcher"; GroupDescription: "File Associations:"; Flags: unchecked
Name: "associate_license"; Description: "Associate License files (.lic, .dmreq) with Design Matcher"; GroupDescription: "File Associations:"

[Files]
; Copy all compiled binaries and assets from dist\DesignMatcher
Source: "dist\DesignMatcher\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Embed application icon
Source: "assets\icon.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
; NOTE: Do NOT package user license or clock state. Those are created dynamically in %LOCALAPPDATA%\DesignMatcher at runtime.

[Icons]
Name: "{autoprograms}\{#MyAppName}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\icon.ico"
Name: "{autoprograms}\{#MyAppName}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\icon.ico"; Tasks: desktopicon

[Registry]
; File association: .dxf
Root: HKA; Subkey: "Software\Classes\.dxf"; ValueType: string; ValueName: ""; ValueData: "DesignMatcher.DXF"; Flags: uninsdeletevalue; Tasks: associate_dxf
Root: HKA; Subkey: "Software\Classes\DesignMatcher.DXF"; ValueType: string; ValueName: ""; ValueData: "AutoCAD DXF Drawing"; Flags: uninsdeletekey; Tasks: associate_dxf
Root: HKA; Subkey: "Software\Classes\DesignMatcher.DXF\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\assets\icon.ico,0"; Tasks: associate_dxf
Root: HKA; Subkey: "Software\Classes\DesignMatcher.DXF\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: associate_dxf

; File association: .lic (Signed License)
Root: HKA; Subkey: "Software\Classes\.lic"; ValueType: string; ValueName: ""; ValueData: "DesignMatcher.License"; Flags: uninsdeletevalue; Tasks: associate_license
Root: HKA; Subkey: "Software\Classes\DesignMatcher.License"; ValueType: string; ValueName: ""; ValueData: "Design Matcher License File"; Flags: uninsdeletekey; Tasks: associate_license
Root: HKA; Subkey: "Software\Classes\DesignMatcher.License\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\assets\icon.ico,0"; Tasks: associate_license
Root: HKA; Subkey: "Software\Classes\DesignMatcher.License\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: associate_license

; File association: .dmreq (License Request)
Root: HKA; Subkey: "Software\Classes\.dmreq"; ValueType: string; ValueName: ""; ValueData: "DesignMatcher.Request"; Flags: uninsdeletevalue; Tasks: associate_license
Root: HKA; Subkey: "Software\Classes\DesignMatcher.Request"; ValueType: string; ValueName: ""; ValueData: "Design Matcher Activation Request"; Flags: uninsdeletekey; Tasks: associate_license
Root: HKA; Subkey: "Software\Classes\DesignMatcher.Request\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\assets\icon.ico,0"; Tasks: associate_license
Root: HKA; Subkey: "Software\Classes\DesignMatcher.Request\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: associate_license

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
