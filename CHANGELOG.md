# Depthai Viewer changelog

## 0.1.4
- Added depdendency installer.
- Added auto device selection on startup:
    - On startup: If a preferred device is set, try to connect to that device. Else (if preferred device not found or not set) connect to the first available device.
    - Whenever a device is selected it is designated as the preferred device. A referred device is persisted between runs of depthai-viewer.

## 0.1.4-alpha.0

- Added dependency installer.
- Added auto device selection (+preferred device) on initial startup.

## 0.1.3

- Fix default_stereo_pairs for OAK-1 MAX

## 0.1.2

- Fix stereo depth creation for OAK-D-Pro-W 97, and potentially others

## 0.1.1

- Fixes Metal shader compilation on MacOS
- Performance improvement - only create a camera node when a camera is actually needed in any way.
- Added sentry.io error reporting
- Update Windows and MacOS app icons

## 0.1.0

Depthai Viewer official Beta release on pypi!

- Performance improvements
- Better Auto Layouts
- Tweaks to UI for a better UX

## 0.0.8-alpha.0

- Pre-release

## 0.0.7

- Install depthai_sdk from artifactory
- Change logos

## 0.0.6

- App startup bugfixes

## 0.0.5

- App startup bugfixes
- Better default focusing in 3d views

## 0.0.4

- Disable depth settings if intrinsics aren't available.
- App startup bugfixes.

## 0.0.3

- Added support for all devices.

## 0.0.2

Patch release.

## 0.0.1

Beta release of the new Depthai Viewer.
