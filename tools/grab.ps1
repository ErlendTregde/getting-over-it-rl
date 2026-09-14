# Capture the Getting Over It window to a PNG.
param([string]$Out = "player_raw.png")

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W {
  [StructLayout(LayoutKind.Sequential)]
  public struct RECT { public int Left, Top, Right, Bottom; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, ref RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
}
"@

$p = Get-Process GettingOverIt -ErrorAction Stop | Select-Object -First 1
$h = $p.MainWindowHandle
[void][W]::ShowWindow($h, 9)          # SW_RESTORE
[void][W]::SetForegroundWindow($h)
Start-Sleep -Milliseconds 900

$r = New-Object W+RECT
$ok = [W]::GetWindowRect($h, [ref]$r)
$w = $r.Right - $r.Left
$ht = $r.Bottom - $r.Top
if (-not $ok -or $w -le 0 -or $ht -le 0) { throw "GetWindowRect failed: ok=$ok ${w}x${ht}" }

$bmp = New-Object System.Drawing.Bitmap($w, $ht)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($r.Left, $r.Top, 0, 0, (New-Object System.Drawing.Size($w, $ht)))
$g.Dispose()
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
"captured ${w}x${ht} at ($($r.Left),$($r.Top)) -> $Out"
