""" Stellt die Klasse ImageLoader bereit. """

from PIL import Image, ImageOps


class ImageLoader:
    """
    Klasse zum Laden und umwandeln einer Bilddatei.
    """

    def __init__(self, filename):
        self.filename = filename
        try:
            self.img = Image.open(self.filename)
        except FileNotFoundError as e:
            print(e)

    def get_pixel_values(self, invert=True):
        """
        Umrechnung der Pixelwerte des Bildes in
        Luma-Werte (schwarz-weiß) mit optionaler Invertierung.
        Da der Testdatensatz weiß auf schwarz ist, wird standardmäßig invertiert.

        :param invert: ob das Bild invertiert werden soll (schwarz und weiß tauschen)
        :return: ein sequenzierbares Objekt (ImagingCore)
        """
        # In Graustufen konvertieren ('L' steht für Luma/Luminance → 0-255)
        grayscale_img = self.img.convert('L')
        # Testdaten sind Weiß auf Schwarz, daher müssen wir invertieren
        if invert:
            grayscale_img = ImageOps.invert(grayscale_img)

        # Das Bild als Liste der Werte zurückgeben
        return grayscale_img.getdata()

    def get_filename(self):
        """
        :return: den gespeicherten Dateinamen
        """
        return self.filename

    def set_filename(self, new_filename):
        """
        Überschreibt den gespeicherten Dateinamen mit einem neuen Dateinamen

        :param new_filename: neuer Dateiname
        :return: None
        """
        self.filename = new_filename
