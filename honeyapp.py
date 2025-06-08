import torch
import cv2
import numpy as np
from PIL import Image
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.camera import Camera
from kivy.uix.dropdown import DropDown
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from torchvision import transforms


MODEL_PATH = "an_honey_detector_model.pt"
SONG_PATH = "assets/song.mp3"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')

        self.cam = Camera(play=True, resolution=(640, 480))
        self.cam.allow_stretch = True
        self.cam.keep_ratio = False
        self.cam.size_hint = (1, 1)
        layout.add_widget(self.cam)

        button_bar = BoxLayout(size_hint_y=0.1)
        self.status_label = Label(text="❌ Looking for Honey…")
        settings_button = Button(text="⚙️ Settings")
        settings_button.bind(on_release=self.open_settings)
        button_bar.add_widget(self.status_label)
        button_bar.add_widget(settings_button)

        layout.add_widget(button_bar)
        self.add_widget(layout)

        self.model = torch.jit.load(MODEL_PATH, map_location=torch.device('cpu')).eval()
        self.sound = SoundLoader.load(SONG_PATH)
        self.soundToggle = False  # 🔇 Default: Music Off
        self.detecting = True

        Clock.schedule_interval(self.detect_honey, 0.5)

    def open_settings(self, *args):
        popup = Popup(title='Settings', size_hint=(0.7, 0.5))
        settings_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        camera_label = Label(text="Choose Camera")
        camera_dropdown = DropDown()
        for i in range(2):
            btn = Button(text=f"Camera {i}", size_hint_y=None, height=40)
            btn.bind(on_release=lambda btn: camera_dropdown.select(btn.text))
            camera_dropdown.add_widget(btn)

        camera_main_button = Button(text='Camera 0')
        camera_main_button.bind(on_release=camera_dropdown.open)
        camera_dropdown.bind(on_select=lambda instance, x: camera_main_button.setter('text')(camera_main_button, x))

        music_toggle = ToggleButton(
            text="🔈 Music: OFF", state='normal' if not self.soundToggle else 'down'
        )

        def toggle_music(btn):
            self.soundToggle = btn.state == 'down'
            btn.text = "🔊 Music: ON" if self.soundToggle else "🔈 Music: OFF"
            if not self.soundToggle and self.sound and self.sound.state == 'play':
                self.sound.stop()

        music_toggle.bind(on_press=toggle_music)

        close_btn = Button(text="Close")
        close_btn.bind(on_release=popup.dismiss)

        settings_layout.add_widget(camera_label)
        settings_layout.add_widget(camera_main_button)
        settings_layout.add_widget(music_toggle)
        settings_layout.add_widget(close_btn)

        popup.content = settings_layout
        popup.open()

    def detect_honey(self, dt):
        if not self.cam.texture:
            return

        buffer = self.cam.texture.pixels
        w, h = self.cam.texture.size
        img = Image.frombytes(mode='RGBA', size=(w, h), data=buffer)
        img = img.convert('RGB')
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(probs, dim=0)

        if pred.item() == 0 and conf.item() > 0.7:
            self.status_label.text = "✅ Honey Detected!"
            if self.soundToggle and self.sound and self.sound.state != 'play':
                self.sound.play()
        else:
            self.status_label.text = "❌ Looking for Honey…"
            if self.sound and self.sound.state == 'play':
                self.sound.stop()


class HoneyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name="main"))
        return sm


if __name__ == "__main__":
    HoneyApp().run()
