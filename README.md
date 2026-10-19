# RetroLens AR (Android, Kotlin + C++ NDK)
Hand-gesture AR filters, ported from the Python/OpenCV/MediaPipe version.

Gestures
- Both hands (thumb + index frame) -> PORTAL showing the filter
- Both hands pinch -> change portal filter (Cyber Neon / Sepia / Negative)
- Left hand peace + right hand pinch, then release -> "ketapel": whole screen takes the portal filter
- One open palm -> reset

Build: push to GitHub, open Actions, run "Build RetroLens APK", download the APK from Artifacts.
