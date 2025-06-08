[app]

# (str) Title of your application
title = Honey Detector

# (str) Package name
package.name = honeydetector

# (str) Package domain (reverse domain name style)
package.domain = org.example

# (str) Source code where the main.py or your app entry point lives
source.dir = .

# (list) Source files to include (e.g. your model and assets)
source.include_exts = py,png,jpg,kv,pt,mp3

# (str) Application version
version = 1.0.0

# (str) Requirements, separated by commas
requirements = python3,kivy,torch,torchvision,opencv-python

# (list) Permissions to request from the device
android.permissions = CAMERA,INTERNET,WAKE_LOCK,VIBRATE

# (bool) Indicate if the app should be fullscreen or not
fullscreen = 1

# (str) Icon of the app
# icon.filename = %(source.dir)s/icon.png

# (str) Supported orientation (landscape, portrait or all)
orientation = portrait

# (bool) Whether to copy library dependencies or not (generally True)
copy_libs = True

# (str) Android API level target
android.api = 33

# (str) Minimum Android API level supported
android.minapi = 21

# (str) Android SDK version to compile against
android.sdk = 33

# (int) Android NDK version to use
android.ndk = 25b

# (bool) Use --private data storage (recommended)
android.private_storage = True

# (str) Presplash image (optional)
# presplash.filename = %(source.dir)s/presplash.png

# (str) Custom Java class for Android (optional)
# android.entrypoint = org.kivy.android.PythonActivity

# (list) Android additional libraries to link
# android.gradle_dependencies =

# (bool) Enable Android camera2 API (useful for camera apps)
android.camera2 = True
