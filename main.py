from pore_size import pore_size
import os
import cv2
import csv
import math
with open("pore_data.csv", "w", newline="") as csvfile, open("pore_data_per_image.csv", "w", newline="") as diasfile:
    writer = csv.writer(csvfile)
    dias_writer = csv.writer(diasfile)
    writer.writerow(["image_id", "average pore area (microns^2)", "average pore diameter (microns)", "biggest pore area (microns^2)", "biggest pore diameter (microns)"])
    dias_writer.writerow(["image_id", "dias"])
    root_dir = os.getcwd() + '\ct-images'
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".bmp"):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)     
                image, blurred, edges, average_area, biggest_pore, areas = pore_size(img)
                dias = [2 * math.sqrt(area/math.pi) for area in areas]
                average_dia = 2 * math.sqrt(average_area/math.pi)
                biggest_dia = 2 * math.sqrt(biggest_pore/math.pi)
                writer.writerow([file, average_area, average_dia, biggest_pore, biggest_dia])
                for dia in dias:
                    dias_writer.writerow([file, dia])
                                
            