from django.db import models
import pandas as pd
# Create your models here.
class LayerData(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    filedata = models.FileField(upload_to='layer_files/')

    def __str__(self):
        return self.name
    def parse_data(self):
        if self.filedata:
            # Assuming the file is a .xlsx file
            excel_file_path = self.filedata.path
            df = pd.read_excel(excel_file_path)
            # Assuming the Excel file has columns: Wavelength, n, k
            return df.values.tolist()  # Converts dataframe to a list of lists
        return []

    @property
    def parsed_data(self):
        return self.parse_data()
    @property
    def min_value(self):
        data = self.parse_data()
        if data:
            return min(row[0] for row in data)  # Assuming Wavelength is in the first column
        return None

    @property
    def max_value(self):
        data = self.parse_data()
        if data:
            return max(row[0] for row in data)  # Assuming Wavelength is in the first column
        return None

    @property
    def data_length(self):
        data = self.parse_data()
        return len(data)

from django.db import models
import pandas as pd

class LEDSpectrumData(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    filedata = models.FileField(upload_to='led_spectrum_files/')

    def __str__(self):
        return self.name

    def parse_data(self):
        if self.filedata:
            # Assuming the file is a .xlsx file containing LED spectrum data
            excel_file_path = self.filedata.path
            df = pd.read_excel(excel_file_path)
            # Assuming the Excel file has columns: Wavelength, Intensity
            return df.values.tolist()  # Converts dataframe to a list of lists
        return []

    @property
    def parsed_data(self):
        return self.parse_data()

    @property
    def min_wavelength(self):
        data = self.parse_data()
        if data:
            return min(row[0] for row in data)  # Assuming Wavelength is in the first column
        return None

    @property
    def max_wavelength(self):
        data = self.parse_data()
        if data:
            return max(row[0] for row in data)  # Assuming Wavelength is in the first column
        return None

    @property
    def data_length(self):
        data = self.parse_data()
        return len(data)

    def calculate_chromaticity_coordinates(self, x_bar, y_bar, z_bar):
        """
        Calculate the chromaticity coordinates (x, y) for the LED spectrum.
        Assumes x_bar, y_bar, z_bar are lists or arrays corresponding to the color matching functions.
        """
        data = self.parse_data()
        if not data:
            return None, None

        wavelengths = [row[0] for row in data]
        intensities = [row[1] for row in data]

        X = sum(i * x for i, x in zip(intensities, x_bar))
        Y = sum(i * y for i, y in zip(intensities, y_bar))
        Z = sum(i * z for i, z in zip(intensities, z_bar))

        if (X + Y + Z) == 0:
            return None, None

        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)

        return x, y

    def plot_chromaticity(self, x_bar, y_bar, z_bar):
        """
        Plot the chromaticity coordinates on a CIE 1931 diagram.
        """
        import matplotlib.pyplot as plt

        x, y = self.calculate_chromaticity_coordinates(x_bar, y_bar, z_bar)
        if x is None or y is None:
            return "Invalid data or calculation error."

        plt.figure(figsize=(6, 6))
        plt.plot(x, y, 'o', color='red')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Chromaticity Diagram')
        plt.grid(True)
        plt.show()

class MultilayerFilmData(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    filedata = models.FileField(upload_to='multilayer_files/')

    def __str__(self):
        return self.name

    def parse_data(self):
        if self.filedata:
            # Assuming the file is a .xlsx file containing multilayer film data
            excel_file_path = self.filedata.path
            df = pd.read_excel(excel_file_path)
            # Assuming the Excel file has columns: Wavelength (nm), Reflectance R_unpolarized, 
            # Transmittance T_unpolarized, Absorbance A__unpolarized
            return df.values.tolist()  # Converts dataframe to a list of lists
        return []

    @property
    def parsed_data(self):
        return self.parse_data()

    @property
    def min_wavelength(self):
        data = self.parse_data()
        if data:
            return min(row[0] for row in data)  # Assuming Wavelength is in the first column
        return None

    @property
    def max_wavelength(self):
        data = self.parse_data()
        if data:
            return max(row[0] for row in data)  # Assuming Wavelength is in the first column
        return None

    @property
    def data_length(self):
        data = self.parse_data()
        return len(data)

    def get_reflectance_data(self):
        data = self.parse_data()
        if data:
            return [(row[0], row[1]) for row in data]  # Returns Wavelength and Reflectance pairs
        return []

    def get_transmittance_data(self):
        data = self.parse_data()
        if data:
            return [(row[0], row[2]) for row in data]  # Returns Wavelength and Transmittance pairs
        return []

    def get_absorbance_data(self):
        data = self.parse_data()
        if data:
            return [(row[0], row[3]) for row in data]  # Returns Wavelength and Absorbance pairs
        return []

    def calculate_reflected_light(self, light_spectrum_data):
        """
        Calculate the reflected light spectrum by multiplying the input light spectrum
        with the reflectance data of the multilayer film.
        """
        reflectance_data = self.get_reflectance_data()
        if not reflectance_data or not light_spectrum_data:
            return None

        wavelengths = [row[0] for row in reflectance_data]
        reflectance = [row[1] for row in reflectance_data]
        interpolated_reflectance = np.interp([w[0] for w in light_spectrum_data], wavelengths, reflectance)

        reflected_light = [(wavelength, intensity * r) 
                           for (wavelength, intensity), r in zip(light_spectrum_data, interpolated_reflectance)]
        return reflected_light

    def calculate_transmitted_light(self, light_spectrum_data):
        """
        Calculate the transmitted light spectrum by multiplying the input light spectrum
        with the transmittance data of the multilayer film.
        """
        transmittance_data = self.get_transmittance_data()
        if not transmittance_data or not light_spectrum_data:
            return None

        wavelengths = [row[0] for row in transmittance_data]
        transmittance = [row[1] for row in transmittance_data]
        interpolated_transmittance = np.interp([w[0] for w in light_spectrum_data], wavelengths, transmittance)

        transmitted_light = [(wavelength, intensity * t) 
                             for (wavelength, intensity), t in zip(light_spectrum_data, interpolated_transmittance)]
        return transmitted_light

    def calculate_absorbed_light(self, light_spectrum_data):
        """
        Calculate the absorbed light spectrum by multiplying the input light spectrum
        with the absorbance data of the multilayer film.
        """
        absorbance_data = self.get_absorbance_data()
        if not absorbance_data or not light_spectrum_data:
            return None

        wavelengths = [row[0] for row in absorbance_data]
        absorbance = [row[1] for row in absorbance_data]
        interpolated_absorbance = np.interp([w[0] for w in light_spectrum_data], wavelengths, absorbance)

        absorbed_light = [(wavelength, intensity * a) 
                          for (wavelength, intensity), a in zip(light_spectrum_data, interpolated_absorbance)]
        return absorbed_light
