from PIL import Image, ImageOps

class ImageLoader:
    def __init__(self, filename):
        self.filename = filename
        try:
            self.img = Image.open(self.filename)
        except FileNotFoundError as e:
            print(e)
        pass

    def get_pixel_values(self, invert=True):
        # In Graustufen konvertieren ('L' steht für Luma/Luminance → 0-255)
        grayscale_img = self.img.convert('L')
        # Testdaten sind Weiß auf Schwarz, daher müssen wir invertieren
        if invert:
            grayscale_img = ImageOps.invert(grayscale_img)

        # Das Bild als Liste der Werte zurückgeben
        return list(grayscale_img.getdata())
