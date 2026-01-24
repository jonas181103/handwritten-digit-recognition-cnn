""" Stellt die Klasse ImageLoader bereit. """

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageLoader:
    """
    Klasse zum Laden und umwandeln einer Bilddatei.
    """

    def __init__(self, filename):
        self.img = None
        # Dateinamen aktualisieren lädt auch gleichzeitig das Bild
        self.filename = filename

    def _update_image(self):
        """
        Öffnet das Bild und kopiert den Inhalt in die self.img Variable

        :return: None
        """
        try:
            with Image.open(self.filename) as img:
                self.img = img.copy()
        except UnidentifiedImageError as exc:
            raise ValueError(f"File '{self.filename}' is not a valid image file'") from exc


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

    @property
    def filename(self):
        """
        :return: den gespeicherten Dateinamen
        """
        return self._filename

    @filename.setter
    def filename(self, new_filename):
        """
        Überschreibt den gespeicherten Dateinamen mit einem neuen Dateinamen

        :param new_filename: neuer Dateiname
        :return: None
        """
        if not new_filename:
            raise ValueError("Filename must not be empty")
        self._filename = new_filename
        self._update_image()
