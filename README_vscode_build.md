# Build instructions (VS Code / Windows Win32)

This project contains a small UDP haptic client using OpenHaptics. You said you use Win32 and don't have "Properties" in your editor — these instructions help you build in VS Code or in Visual Studio.

1) Quick fix for IntelliSense include errors

- I added `.vscode/c_cpp_properties.json` with the OpenHaptics include directories:
  - `C:/OpenHaptics/Developer/3.5.0/include`
  - `C:/OpenHaptics/Developer/3.5.0/utilities/include`

This should remove the `cannot open source file "HD/hd.h"` squiggles.

2) Building options

Option A — Build the existing Visual Studio solution (recommended if you have VS installed):
- Make sure Visual Studio is installed (with C++ workload).
- Open `udp_client/udp_client.sln` in Visual Studio (double-click it).
- Set `Configuration = Debug` and `Platform = Win32` then build (Build > Rebuild Solution).
- If you open the solution in Visual Studio, add the include/lib paths under Project -> Properties -> C/C++ -> Additional Include Directories and Linker -> Additional Library Directories (point to `C:/OpenHaptics/Developer/3.5.0/lib/Win32`) and add `hd.lib;hdu.lib` to Linker -> Input -> Additional Dependencies.

Option B — Build from VS Code (msbuild task)
- In VS Code press Ctrl+Shift+B or use the command palette "Tasks: Run Build Task" and pick "Build udp_client (msbuild)".
- This calls `msbuild` on the `.sln`. You must have MSBuild on PATH (installed with Visual Studio). This relies on the project in the solution being configured correctly for Win32.

Option C — Build single file from VS Code (cl.exe task)
- There's another task "Build single file (cl.exe)" which:
  - Calls `vcvarsall.bat x86` to set up the environment
  - Invokes `cl` with include and library paths to compile `main.cpp` into `udp_client.exe`
- To use this you need Visual Studio (so the vcvars script and cl exist). If your VS installation path differs from the task, edit the `vcvarsall.bat` path in `.vscode/tasks.json`.

3) Runtime
- Ensure OpenHaptics runtime and drivers are installed and the device is connected.
- Ensure `C:/OpenHaptics/Developer/3.5.0/lib/Win32` contains `hd.lib` and `hdu.lib` and that the corresponding DLLs (if any) are discoverable at runtime (usually installed by OpenHaptics).

4) If squiggles persist
- Open `.vscode/c_cpp_properties.json` and update the `compilerPath` to the exact path of your `cl.exe` installation (or set it to `""` if unknown). The important part is `includePath`.

If you want, I can:
- Edit `.vscode/c_cpp_properties.json` to use your exact Visual Studio installation path for `compilerPath` if you tell me which VS version you have and where it's installed.
- Attempt to run the build task from here and paste the output (I can run msbuild/cl if you want me to invoke the terminal).