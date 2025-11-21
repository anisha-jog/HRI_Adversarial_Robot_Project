from skimage.morphology import skeletonize
from skimage import data
import svgwrite
import potrace
from potrace import Bitmap
import math
import cv2
import numpy as np

def skeletonize_bitmap(bitmap):
    binary_bitmap = bitmap > 0
    skeleton = skeletonize(binary_bitmap)
    return skeleton

def create_svg(lines, filename, width, height):
    """
    Generates an SVG file using svgwrite from a list of (x, y) points.
    """
    dwg = svgwrite.Drawing(filename, size=(f"{width}px", f"{height}px"), profile='tiny')

    # Add a polyline connecting the points
    for points in lines:
      dwg.add(dwg.polyline(points, stroke=svgwrite.rgb(0, 0, 0, '%'), fill='none', stroke_width=2))
    dwg.save()
    print(f"SVG file '{filename}' created.")
  
def convert_pixels_to_inches(lines, img_height, img_width, target_height_in=11, target_width_in=8.5):
    """
    Converts a list of lines with (x, y) points from pixels to inches.
    """
    scale_x = target_width_in / img_width
    scale_y = target_height_in / img_height
    lines_in_inches = []
    for points in lines:
        converted_points = [(x * scale_x, y * scale_y) for (x, y) in points]
        lines_in_inches.append(converted_points)
    return lines_in_inches

def convert_pixels_to_millimeters(lines, img_height, img_width, target_height_mm=279.4, target_width_mm=215.9):
    """
    Converts a list of lines with (x, y) points from pixels to millimeters.
    """
    scale_x = target_width_mm / img_width
    scale_y = target_height_mm / img_height
    lines_in_mm = []
    for points in lines:
        converted_points = [(x * scale_x, y * scale_y) for (x, y) in points]
        lines_in_mm.append(converted_points)
    return lines_in_mm

def convert_pixels_to_meters(lines, img_height, img_width, target_height_m=0.2794, target_width_m=0.2159):
    """
    Converts a list of lines with (x, y) points from pixels to meters.
    """
    scale_x = target_width_m / img_width
    scale_y = target_height_m / img_height
    lines_in_m = []
    for points in lines:
        converted_points = [(x * scale_x, y * scale_y) for (x, y) in points]
        lines_in_m.append(converted_points)
    return lines_in_m

def bezier_to_points(p1, c1, c2, p2, segments=10):
    """
    Approximates a cubic Bezier curve with a series of line segments.
    p1, p2: start and end points of the curve (tuples: (x, y))
    c1, c2: control points of the curve (tuples: (x, y))
    segments: number of line segments to use for approximation
    """
    acc = []
    for i in range(segments + 1):
        t = float(i / segments)
        x = (1.0 - t)**3.0 * float(p1[0])+3.0 * (1.0- t)**2.0 * t * float(c1[0]) + 3.0 * (1.0 - t) * t**2.0 * float(c2[0]) + t**3.0 * float(p2[0])
        y = (1 - t)**3 * p1[1]+ 3 * (1 - t)**2 * t * c1[1] + 3 * (1 - t) * t**2 * c2[1] + t**3 * p2[1]
        acc.append((x, y))
    return acc

def linearize_bezier(path, segments=10):
  lines = []
  for curve in path:
    # Add the starting point of each curve
    points = []
    points.append((curve.start_point.x, curve.start_point.y))
    for segment in curve:
      if segment.is_corner:
          points.append((segment.end_point.x, segment.end_point.y))
      else:
          p1 = (points[-1][0],points[-1][1])
          c1 = (segment.c1.x, segment.c1.y)
          c2 = (segment.c2.x, segment.c2.y)
          p2 = (segment.end_point.x, segment.end_point.y)
          bezier_points = bezier_to_points(p1, c1, c2, p2, segments=10)
          points.extend(bezier_points[1:]) 
    lines.append(points)
  return lines[1:]

# Create SVG lines from bitmap, also returns the points for each line
def bitmap_to_svg(bitmap, filename="out.svg", segments=10):
  skeleton = skeletonize(bitmap)
  vector = Bitmap(skeleton).trace()
  lines = linearize_bezier(vector)
  create_svg(lines, filename, bitmap.shape[1], bitmap.shape[0])
  return lines

def bitmap_to_lines(bitmap, segments=10):
  skeleton = skeletonize(bitmap)
  vector = Bitmap(skeleton).trace()
  lines = linearize_bezier(vector)
  return lines

# Create SVG from image (i.e. additional threshhold operation), also returns the points for each line
def image_to_svg(image, filename="out.svg", low=200, high=255, segments=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, low, high, cv2.THRESH_BINARY_INV)
    return bitmap_to_svg(binary, filename)

def image_to_lines(image, low=200, high=255, segments=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, low, high, cv2.THRESH_BINARY_INV)
    return bitmap_to_lines(binary)