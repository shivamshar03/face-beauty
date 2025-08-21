import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFilter
import colorsys
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import io
import base64
import os
import glob
from datetime import datetime
import threading
import queue

# ======================== Page Config & Styles ========================
st.set_page_config(
    page_title="Fashion AR Beauty Filter Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# WebRTC Configuration to fix setIn error
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .skin-tone-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .color-card {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.3rem;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .color-card:hover {
        transform: scale(1.05);
    }
    .season-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .filter-preview {
        max-width: 80px;
        max-height: 80px;
        border-radius: 8px;
        margin: 0.2rem;
        cursor: pointer;
        border: 2px solid transparent;
    }
    .filter-preview:hover {
        border: 2px solid #667eea;
    }
    .selected-filter {
        border: 3px solid #f093fb !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ======================== Updated 5-Season Color Analysis ========================

def get_seasonal_color_analysis(skin_tone, undertone, season_preference=None):
    """Comprehensive 5-season color analysis with professional recommendations"""

    seasonal_palettes = {
        "Spring": {
            "description": "Fresh, bright, and warm colors like blooming flowers",
            "characteristics": ["Light to medium skin", "Golden undertones", "Bright, clear features"],
            "best_colors": {
                "clothing": ["Coral", "Peach", "Golden Yellow", "Fresh Green", "Turquoise", "Clear Red", "Warm Pink",
                             "Light Navy"],
                "lipstick": ["Coral Pink", "Peach", "Bright Pink", "Orange Red", "Warm Berry"],
                "eyeshadow": ["Gold", "Peach", "Coral", "Light Brown", "Warm Green"],
                "blush": ["Peach", "Coral", "Warm Pink"],
                "nail": ["Coral", "Peach", "Bright Pink", "Orange", "Turquoise"]
            },
            "hex_codes": {
                "clothing": ["#FF7F7F", "#FFCBA4", "#FFD700", "#00FF7F", "#40E0D0", "#FF0000", "#FF69B4", "#1E3A8A"],
                "lipstick": ["#F08080", "#FFCBA4", "#FF1493", "#FF4500", "#8B008B"],
                "eyeshadow": ["#FFD700", "#FFCBA4", "#FF7F50", "#D2B48C", "#98FB98"],
                "blush": ["#FFCBA4", "#FF7F50", "#FF69B4"],
                "nail": ["#FF7F50", "#FFCBA4", "#FF1493", "#FFA500", "#40E0D0"]
            },
            "avoid": ["Black", "Pure White", "Burgundy", "Dark Navy", "Gray"],
            "metals": ["Gold", "Copper", "Warm Silver"],
            "season_match": ["Very Fair", "Fair", "Light"]
        },

        "Summer": {
            "description": "Soft, cool, and gentle colors like a summer breeze",
            "characteristics": ["Fair to medium skin", "Cool/pink undertones", "Soft, muted coloring"],
            "best_colors": {
                "clothing": ["Powder Blue", "Lavender", "Rose Pink", "Sage Green", "Soft Plum", "Navy", "Soft White",
                             "Light Gray"],
                "lipstick": ["Rose", "Mauve", "Berry", "Soft Pink", "Cool Red"],
                "eyeshadow": ["Taupe", "Rose", "Lavender", "Soft Brown", "Blue Gray"],
                "blush": ["Rose", "Pink", "Soft Coral"],
                "nail": ["Rose", "Mauve", "Soft Pink", "Lavender", "Navy"]
            },
            "hex_codes": {
                "clothing": ["#B0E0E6", "#E6E6FA", "#FF69B4", "#9CAF88", "#8E4585", "#000080", "#F8F8FF", "#D3D3D3"],
                "lipstick": ["#FF69B4", "#8B7355", "#8B008B", "#FFB6C1", "#DC143C"],
                "eyeshadow": ["#D2B48C", "#FF69B4", "#E6E6FA", "#A0522D", "#708090"],
                "blush": ["#FF69B4", "#FFB6C1", "#FF7F50"],
                "nail": ["#FF69B4", "#8B7355", "#FFB6C1", "#E6E6FA", "#000080"]
            },
            "avoid": ["Orange", "Gold", "Bright Yellow", "Warm Red"],
            "metals": ["Silver", "Platinum", "White Gold"],
            "season_match": ["Fair", "Light", "Very Fair"]
        },

        "Rainy": {
            "description": "Deep, muted, and sophisticated colors like stormy skies",
            "characteristics": ["Medium to deep skin", "Neutral undertones", "Sophisticated, muted coloring"],
            "best_colors": {
                "clothing": ["Charcoal", "Deep Teal", "Burgundy", "Forest Green", "Plum", "Navy", "Stone Gray",
                             "Cream"],
                "lipstick": ["Deep Rose", "Plum", "Burgundy", "Mauve", "Deep Berry"],
                "eyeshadow": ["Smoky Gray", "Deep Purple", "Forest Green", "Bronze", "Charcoal"],
                "blush": ["Deep Rose", "Mauve", "Soft Plum"],
                "nail": ["Deep Teal", "Burgundy", "Charcoal", "Plum", "Navy"]
            },
            "hex_codes": {
                "clothing": ["#36454F", "#004D40", "#800020", "#228B22", "#8E4585", "#000080", "#918E85", "#F5F5DC"],
                "lipstick": ["#8B0A50", "#8E4585", "#800020", "#8B7355", "#800080"],
                "eyeshadow": ["#708090", "#8A2BE2", "#228B22", "#CD7F32", "#36454F"],
                "blush": ["#8B0A50", "#8B7355", "#8E4585"],
                "nail": ["#004D40", "#800020", "#36454F", "#8E4585", "#000080"]
            },
            "avoid": ["Neon colors", "Bright Orange", "Hot Pink", "Electric Blue"],
            "metals": ["Oxidized Silver", "Gunmetal", "Antique Bronze"],
            "season_match": ["Medium", "Olive", "Deep"]
        },

        "Autumn": {
            "description": "Rich, warm, and earthy colors like fall leaves",
            "characteristics": ["Medium to deep skin", "Warm/golden undertones", "Rich, earthy coloring"],
            "best_colors": {
                "clothing": ["Rust", "Golden Brown", "Olive Green", "Burgundy", "Orange", "Mustard", "Camel",
                             "Chocolate"],
                "lipstick": ["Brick Red", "Brown", "Orange", "Deep Coral", "Burgundy"],
                "eyeshadow": ["Bronze", "Copper", "Golden Brown", "Olive", "Rust"],
                "blush": ["Peach", "Coral", "Bronze"],
                "nail": ["Rust", "Brown", "Burgundy", "Orange", "Gold"]
            },
            "hex_codes": {
                "clothing": ["#B7410E", "#8B4513", "#808000", "#800020", "#FFA500", "#FFDB58", "#C19A6B", "#D2691E"],
                "lipstick": ["#B22222", "#8B4513", "#FFA500", "#FF7F50", "#800020"],
                "eyeshadow": ["#CD7F32", "#B87333", "#8B4513", "#808000", "#B7410E"],
                "blush": ["#FFCBA4", "#FF7F50", "#CD7F32"],
                "nail": ["#B7410E", "#8B4513", "#800020", "#FFA500", "#FFD700"]
            },
            "avoid": ["Pink", "Blue", "Purple", "Black", "Pure White"],
            "metals": ["Gold", "Copper", "Bronze"],
            "season_match": ["Light", "Medium", "Olive"]
        },

        "Winter": {
            "description": "Bold, cool, and intense colors like winter ice",
            "characteristics": ["Fair to deep skin", "Cool undertones", "High contrast coloring"],
            "best_colors": {
                "clothing": ["Pure White", "Black", "Royal Blue", "Emerald", "Fuchsia", "Purple", "Silver", "True Red"],
                "lipstick": ["True Red", "Berry", "Plum", "Fuchsia", "Wine"],
                "eyeshadow": ["Silver", "Black", "Purple", "Navy", "Emerald"],
                "blush": ["Rose", "Berry", "Pink"],
                "nail": ["Red", "Black", "Purple", "Navy", "Silver"]
            },
            "hex_codes": {
                "clothing": ["#FFFFFF", "#000000", "#4169E1", "#50C878", "#FF1493", "#8A2BE2", "#C0C0C0", "#FF0000"],
                "lipstick": ["#FF0000", "#8B008B", "#8E4585", "#FF1493", "#722F37"],
                "eyeshadow": ["#C0C0C0", "#000000", "#8A2BE2", "#000080", "#50C878"],
                "blush": ["#FF69B4", "#8B008B", "#FFB6C1"],
                "nail": ["#FF0000", "#000000", "#8A2BE2", "#000080", "#C0C0C0"]
            },
            "avoid": ["Orange", "Yellow", "Peach", "Warm Brown"],
            "metals": ["Silver", "Platinum", "White Gold"],
            "season_match": ["Very Fair", "Deep", "Fair"]
        }
    }

    # Determine best season based on skin tone and undertone
    best_seasons = []
    for season, data in seasonal_palettes.items():
        if skin_tone in data["season_match"]:
            if (undertone in ["Warm", "Golden"] and season in ["Spring", "Autumn"]) or \
                    (undertone in ["Cool", "Pink"] and season in ["Summer", "Winter"]) or \
                    (undertone == "Neutral" and season == "Rainy") or \
                    (undertone == "Olive" and season in ["Autumn", "Rainy"]):
                best_seasons.append(season)

    # Default to most likely season if no match
    if not best_seasons:
        if undertone in ["Warm", "Golden"]:
            best_seasons = ["Spring", "Autumn"]
        elif undertone in ["Cool", "Pink"]:
            best_seasons = ["Summer", "Winter"]
        elif undertone == "Neutral":
            best_seasons = ["Rainy"]
        else:
            best_seasons = list(seasonal_palettes.keys())

    primary_season = season_preference if season_preference in best_seasons else best_seasons[0]

    return {
        "primary_season": primary_season,
        "suitable_seasons": best_seasons,
        "palette": seasonal_palettes[primary_season],
        "all_palettes": seasonal_palettes
    }


# ======================== Enhanced Skin Detection ========================

def detect_skin_tone_advanced(image, face_landmarks):
    """Advanced skin tone detection with seasonal analysis"""
    height, width = image.shape[:2]

    # Comprehensive facial regions for better sampling
    skin_regions = {
        'forehead': [10, 151, 9, 8, 107, 55, 285, 296, 334, 293, 300, 276, 283, 282, 295, 285],
        'cheeks': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147,
                   345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 436, 416, 376, 352],
        'nose_bridge': [6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102],
        'chin': [175, 199, 200, 3, 175, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 361, 340, 346]
    }

    all_skin_pixels = []
    region_colors = {}

    # Sample from each region
    for region_name, points in skin_regions.items():
        region_pixels = []
        for pid in points:
            if pid < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[pid]
                x = int(lm.x * width)
                y = int(lm.y * height)

                # Sample 9x9 area around each point
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            pixel = image[ny, nx]
                            region_pixels.append(pixel)
                            all_skin_pixels.append(pixel)

        if region_pixels:
            region_colors[region_name] = np.mean(region_pixels, axis=0)

    if not all_skin_pixels:
        return None, "Unknown", "Unknown", 0.0

    # Advanced outlier removal using robust statistics
    all_pixels = np.array(all_skin_pixels)

    # Calculate luminance for each pixel
    luminance = 0.299 * all_pixels[:, 2] + 0.587 * all_pixels[:, 1] + 0.114 * all_pixels[:, 0]

    # Remove outliers using modified Z-score
    median_lum = np.median(luminance)
    mad = np.median(np.abs(luminance - median_lum))
    modified_z_scores = 0.6745 * (luminance - median_lum) / (mad + 1e-10)

    outlier_mask = np.abs(modified_z_scores) < 3.5
    filtered_pixels = all_pixels[outlier_mask] if np.any(outlier_mask) else all_pixels

    # Calculate average skin color
    avg_color = np.mean(filtered_pixels, axis=0)

    # Enhanced classification using multiple methods

    # Method 1: ITA (Individual Typology Angle)
    L = 0.299 * avg_color[2] + 0.587 * avg_color[1] + 0.114 * avg_color[0]
    b_component = 0.5 * (avg_color[2] - avg_color[0])
    ITA = np.degrees(np.arctan2((L - 50), b_component)) if abs(b_component) > 1e-6 else 0

    # Method 2: HSV analysis
    rgb_norm = avg_color[[2, 1, 0]] / 255.0
    hsv = colorsys.rgb_to_hsv(*rgb_norm)

    # Method 3: Melanin index approximation
    melanin_index = 100 * np.log10(1 / (avg_color[1] / 255.0 + 1e-6))

    # Combined classification
    if ITA > 55 and hsv[2] > 0.8:
        tone = "Very Fair"
    elif ITA > 41 and hsv[2] > 0.7:
        tone = "Fair"
    elif ITA > 28 and hsv[2] > 0.55:
        tone = "Light"
    elif ITA > 10 and hsv[2] > 0.4:
        tone = "Medium"
    elif ITA > -30 and hsv[2] > 0.25:
        tone = "Olive"
    else:
        tone = "Deep"

    # Advanced undertone detection
    r, g, b = rgb_norm

    # Color temperature analysis
    color_temp = -3000 * (r - b) / (r + g + b + 1e-6)

    # Chromatic analysis
    red_green_diff = r - g
    blue_yellow_diff = b - (r + g) / 2

    # Combined undertone classification
    if red_green_diff > 0.02 and color_temp > -500:
        if melanin_index < 45:
            undertone = "Warm"
        else:
            undertone = "Golden"
    elif blue_yellow_diff > 0.01 and color_temp < -1000:
        if melanin_index < 40:
            undertone = "Cool"
        else:
            undertone = "Pink"
    elif abs(red_green_diff) < 0.015 and abs(blue_yellow_diff) < 0.01:
        undertone = "Neutral"
    elif g > max(r, b) and melanin_index > 35:
        undertone = "Olive"
    else:
        undertone = "Neutral"

    # Calculate confidence based on consistency across regions
    region_consistency = 0.8
    if len(region_colors) >= 2:
        color_vars = []
        for region1 in region_colors:
            for region2 in region_colors:
                if region1 != region2:
                    diff = np.linalg.norm(region_colors[region1] - region_colors[region2])
                    color_vars.append(diff)

        if color_vars:
            avg_variance = np.mean(color_vars)
            region_consistency = max(0.3, min(1.0, 1.0 - avg_variance / 100))

    confidence = region_consistency * (1.0 - np.std(filtered_pixels) / 255.0)
    confidence = max(0.1, min(1.0, confidence))

    return avg_color, tone, undertone, confidence


# ======================== Precise Lipstick Application ========================

def apply_precision_lipstick(image, face_landmarks, color, intensity=0.8, glossy=True):
    """Ultra-precise lipstick application with natural blending"""
    height, width = image.shape[:2]

    # Precise lip landmark mapping (468 face landmarks)
    lip_landmarks = {
        'outer_upper': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402],
        'inner_upper': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
        'outer_lower': [146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                        95, 78],
        'inner_lower': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318],
        'corners': [61, 291, 39, 269, 270, 267, 271, 272],
        'cupid_bow': [12, 15, 16, 17, 18]
    }

    # Extract all lip points
    all_lip_points = []
    lip_center_points = []

    for region, points in lip_landmarks.items():
        for pid in points:
            if pid < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[pid]
                x = int(lm.x * width)
                y = int(lm.y * height)
                all_lip_points.append((x, y))

                # Collect center points for gradient
                if region in ['outer_upper', 'outer_lower']:
                    lip_center_points.append((x, y))

    if len(all_lip_points) < 15:
        return image

    # Convert to PIL for advanced processing
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Create multiple layers for natural effect
    base_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    highlight_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    shadow_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))

    base_draw = ImageDraw.Draw(base_layer)
    highlight_draw = ImageDraw.Draw(highlight_layer)
    shadow_draw = ImageDraw.Draw(shadow_layer)

    # Parse color
    if color.startswith('#'):
        color_rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    else:
        color_rgb = (220, 20, 60)  # Default crimson

    try:
        # Create precise lip mask using convex hull
        points_array = np.array(all_lip_points, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        hull_points = [(int(point[0][0]), int(point[0][1])) for point in hull]

        # Base lipstick application
        base_alpha = max(100, int(255 * intensity))
        base_draw.polygon(hull_points, fill=color_rgb + (base_alpha,))

        # Add subtle shadow for depth
        shadow_color = tuple(max(0, c - 30) for c in color_rgb)
        shadow_points = []
        center_x = sum(p[0] for p in hull_points) // len(hull_points)
        center_y = sum(p[1] for p in hull_points) // len(hull_points)

        for x, y in hull_points:
            # Slightly offset shadow inward
            shadow_x = center_x + int((x - center_x) * 0.95)
            shadow_y = center_y + int((y - center_y) * 0.98) + 1
            shadow_points.append((shadow_x, shadow_y))

        shadow_draw.polygon(shadow_points, fill=shadow_color + (int(base_alpha * 0.3),))

        # Add glossy highlight if enabled
        if glossy:
            highlight_color = tuple(min(255, c + 40) for c in color_rgb)

            # Create highlight on upper lip
            if len(lip_center_points) > 5:
                # Find upper lip points
                upper_points = sorted(lip_center_points, key=lambda p: p[1])[:len(lip_center_points) // 2]
                if len(upper_points) >= 3:
                    # Create smaller highlight area
                    highlight_points = []
                    upper_center_x = sum(p[0] for p in upper_points) // len(upper_points)
                    upper_center_y = sum(p[1] for p in upper_points) // len(upper_points)

                    for x, y in upper_points:
                        new_x = upper_center_x + int((x - upper_center_x) * 0.6)
                        new_y = upper_center_y + int((y - upper_center_y) * 0.4)
                        highlight_points.append((new_x, new_y))

                    if len(highlight_points) >= 3:
                        highlight_draw.polygon(highlight_points, fill=highlight_color + (int(base_alpha * 0.4),))

        # Apply subtle blur for natural blending
        base_layer = base_layer.filter(ImageFilter.GaussianBlur(radius=0.5))
        highlight_layer = highlight_layer.filter(ImageFilter.GaussianBlur(radius=0.8))

        # Composite layers
        result = pil_img.convert('RGBA')
        result = Image.alpha_composite(result, shadow_layer)
        result = Image.alpha_composite(result, base_layer)
        result = Image.alpha_composite(result, highlight_layer)

        return cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR)

    except Exception as e:
        # Fallback to simple application
        simple_draw = ImageDraw.Draw(base_layer)
        simple_draw.polygon(all_lip_points, fill=color_rgb + (int(255 * intensity),))
        result = Image.alpha_composite(pil_img.convert('RGBA'), base_layer)
        return cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR)


# ======================== Local Filter Management ========================

def load_local_filters(filter_directory="filters"):
    """Load PNG filters from local directory"""
    filters = {
        'dresses': [],
        'jewelry': [],
        'accessories': [],
        'makeup': []
    }

    # Create filter directory structure if it doesn't exist
    for category in filters.keys():
        category_path = os.path.join(filter_directory, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path, exist_ok=True)

    # Load filters from each category
    for category in filters.keys():
        category_path = os.path.join(filter_directory, category)
        if os.path.exists(category_path):
            png_files = glob.glob(os.path.join(category_path, "*.png"))
            for png_file in png_files:
                try:
                    filter_name = os.path.basename(png_file).replace('.png', '')
                    filters[category].append({
                        'name': filter_name,
                        'path': png_file,
                        'image': cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
                    })
                except Exception as e:
                    st.warning(f"Could not load filter {png_file}: {e}")

    return filters


# ======================== Enhanced Filter Application ========================

def apply_enhanced_png_filter(image, face_landmarks, filter_data, filter_type, scale=1.0, offset=(0, 0),
                              blend_mode='normal'):
    """Enhanced PNG filter application with better positioning and blending"""
    if filter_data is None or 'image' not in filter_data:
        return image

    filter_image = filter_data['image']
    height, width = image.shape[:2]

    if face_landmarks:
        # Get facial feature points for better positioning
        face_points = []
        key_points = {
            'forehead': [10, 151, 9],
            'nose_tip': [1, 2],
            'chin': [175, 199, 200],
            'left_cheek': [116, 117],
            'right_cheek': [345, 346],
            'mouth': [61, 291],
            'left_eye': [33, 133],
            'right_eye': [362, 263]
        }

        facial_features = {}
        for feature, points in key_points.items():
            feature_points = []
            for pid in points:
                if pid < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[pid]
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    feature_points.append((x, y))
            if feature_points:
                facial_features[feature] = feature_points

        # Calculate face dimensions
        all_face_points = []
        face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
                        148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        for pid in face_outline:
            if pid < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[pid]
                x = int(lm.x * width)
                y = int(lm.y * height)
                all_face_points.append((x, y))

        if len(all_face_points) < 10:
            return image

        # Calculate positioning based on filter type
        min_x = min(p[0] for p in all_face_points)
        max_x = max(p[0] for p in all_face_points)
        min_y = min(p[1] for p in all_face_points)
        max_y = max(p[1] for p in all_face_points)

        face_width = max_x - min_x
        face_height = max_y - min_y
        face_center_x = (min_x + max_x) // 2
        face_center_y = (min_y + max_y) // 2

        # Smart positioning based on filter type
        if filter_type == "dress":
            filter_width = int(face_width * 2.8 * scale)
            filter_height = int(filter_image.shape[0] * filter_width / filter_image.shape[1])
            filter_x = face_center_x - filter_width // 2 + offset[0]
            filter_y = max_y + int(face_height * 0.15) + offset[1]

        elif filter_type == "jewelry":
            if "necklace" in filter_data['name'].lower():
                filter_width = int(face_width * 1.5 * scale)
                filter_height = int(filter_image.shape[0] * filter_width / filter_image.shape[1])
                filter_x = face_center_x - filter_width // 2 + offset[0]
                filter_y = max_y - int(face_height * 0.1) + offset[1]
            else:  # earrings
                filter_width = int(face_width * 0.8 * scale)
                filter_height = int(filter_image.shape[0] * filter_width / filter_image.shape[1])
                filter_x = face_center_x - filter_width // 2 + offset[0]
                filter_y = face_center_y - filter_height // 2 + offset[1]

        elif filter_type == "accessory":
            if "hat" in filter_data['name'].lower() or "crown" in filter_data['name'].lower():
                filter_width = int(face_width * 1.6 * scale)
                filter_height = int(filter_image.shape[0] * filter_width / filter_image.shape[1])
                filter_x = face_center_x - filter_width // 2 + offset[0]
                filter_y = min_y - int(filter_height * 0.7) + offset[1]
            else:  # glasses, masks, etc.
                filter_width = int(face_width * 1.2 * scale)
                filter_height = int(filter_image.shape[0] * filter_width / filter_image.shape[1])
                filter_x = face_center_x - filter_width // 2 + offset[0]
                filter_y = face_center_y - filter_height // 2 + offset[1]
        else:
            # Default positioning
            filter_width = int(face_width * scale)
            filter_height = int(filter_image.shape[0] * filter_width / filter_image.shape[1])
            filter_x = face_center_x - filter_width // 2 + offset[0]
            filter_y = face_center_y - filter_height // 2 + offset[1]

        # Ensure filter stays within image bounds
        filter_x = max(0, min(filter_x, width - filter_width))
        filter_y = max(0, min(filter_y, height - filter_height))

        # Apply filter with proper blending
        if filter_width > 0 and filter_height > 0:
            try:
                resized_filter = cv2.resize(filter_image, (filter_width, filter_height))

                # Convert to PIL for better alpha handling
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Handle different image formats
                if len(resized_filter.shape) == 4:  # RGBA
                    filter_pil = Image.fromarray(resized_filter, 'RGBA')
                elif len(resized_filter.shape) == 3:  # RGB
                    filter_pil = Image.fromarray(cv2.cvtColor(resized_filter, cv2.COLOR_BGR2RGB))
                    filter_pil = filter_pil.convert('RGBA')
                else:
                    return image

                # Create overlay
                overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                overlay.paste(filter_pil, (filter_x, filter_y))

                # Apply blend mode
                if blend_mode == 'multiply':
                    # Multiply blend for shadows
                    result = Image.blend(pil_img.convert('RGBA'), overlay, 0.7)
                elif blend_mode == 'overlay':
                    # Overlay blend for vibrant effects
                    result = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
                else:  # normal
                    result = Image.alpha_composite(pil_img.convert('RGBA'), overlay)

                return cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR)

            except Exception as e:
                st.error(f"Error applying filter: {e}")
                return image

    return image


# ======================== Enhanced Beauty Filters ========================

def apply_enhanced_blush(image, face_landmarks, color, intensity=0.6):
    """Enhanced blush with natural gradient"""
    height, width = image.shape[:2]

    # More precise cheek landmark detection
    cheek_regions = {
        'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147],
        'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 436, 416, 376, 352]
    }

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if color.startswith('#'):
        color_rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    else:
        color_rgb = (255, 182, 193)  # Light pink

    alpha = int(255 * intensity)

    for region_name, cheek_points in cheek_regions.items():
        cheek_coords = []
        for pid in cheek_points:
            if pid < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[pid]
                x = int(lm.x * width)
                y = int(lm.y * height)
                cheek_coords.append((x, y))

        if len(cheek_coords) > 8:
            # Calculate cheek center and bounds
            min_x = min(p[0] for p in cheek_coords)
            max_x = max(p[0] for p in cheek_coords)
            min_y = min(p[1] for p in cheek_coords)
            max_y = max(p[1] for p in cheek_coords)

            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            # Create gradient blush effect
            for i, radius in enumerate(range(35, 15, -4)):
                current_alpha = int(alpha * (0.2 + 0.6 * (35 - radius) / 20))
                ellipse_height = int(radius * 0.8)  # Make it more natural oval shape

                draw.ellipse([center_x - radius, center_y - ellipse_height,
                              center_x + radius, center_y + ellipse_height],
                             fill=color_rgb + (current_alpha,))

    # Apply subtle blur for natural blending
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.2))
    result = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
    return cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR)


def apply_enhanced_eyeshadow(image, face_landmarks, color, intensity=0.7):
    """Enhanced eyeshadow with proper eye contouring"""
    height, width = image.shape[:2]

    # Detailed eye regions for better application
    eye_regions = {
        'left_eye': {
            'upper_lid': [33, 7, 163, 144, 145, 153, 154, 155, 133],
            'crease': [246, 161, 160, 159, 158, 157, 173],
            'brow_bone': [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305]
        },
        'right_eye': {
            'upper_lid': [362, 382, 381, 380, 374, 373, 390, 249, 263],
            'crease': [466, 388, 387, 386, 385, 384, 398],
            'brow_bone': [276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276, 283]
        }
    }

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if color.startswith('#'):
        color_rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    else:
        color_rgb = (138, 43, 226)  # Blue violet

    alpha = int(255 * intensity)

    for eye_name, eye_parts in eye_regions.items():
        # Process each part of the eye
        for part_name, points in eye_parts.items():
            coords = []
            for pid in points:
                if pid < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[pid]
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    coords.append((x, y))

            if len(coords) > 4:
                min_x = min(p[0] for p in coords)
                max_x = max(p[0] for p in coords)
                min_y = min(p[1] for p in coords)
                max_y = max(p[1] for p in coords)

                # Different intensity for different parts
                if part_name == 'upper_lid':
                    part_alpha = alpha
                    offset_y = 0
                elif part_name == 'crease':
                    part_alpha = int(alpha * 0.7)
                    offset_y = -8
                else:  # brow_bone
                    part_alpha = int(alpha * 0.3)
                    offset_y = -15

                # Create layered eyeshadow effect
                center_y = min_y + offset_y
                for i in range(3):
                    current_alpha = int(part_alpha * (1.0 - i * 0.25))
                    expansion = i * 3

                    draw.ellipse([min_x - 5 - expansion, center_y - 10 - expansion,
                                  max_x + 5 + expansion, max_y + 5 + expansion],
                                 fill=color_rgb + (current_alpha,))

    # Apply blur for smooth blending
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.5))
    result = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
    return cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR)


# ======================== Enhanced WebRTC Transformer - NO BLINKING FIX ========================

class ProfessionalBeautyTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Initialize with thread-safe frame processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.processing_lock = threading.Lock()

        # Beauty filter settings - STABLE STORAGE
        self._lipstick_settings = {
            'enable': False,
            'color': "#FF1493",
            'intensity': 0.8,
            'glossy': True
        }

        self._blush_settings = {
            'enable': False,
            'color': "#FF69B4",
            'intensity': 0.6
        }

        self._eyeshadow_settings = {
            'enable': False,
            'color': "#8A2BE2",
            'intensity': 0.7
        }

        # PNG Filter settings - STABLE STORAGE
        self._selected_filters = {
            'dress': None,
            'jewelry': None,
            'accessory': None
        }

        self._filter_settings = {
            'dress': {'scale': 1.0, 'offset': (0, 0), 'blend': 'normal'},
            'jewelry': {'scale': 1.0, 'offset': (0, 0), 'blend': 'normal'},
            'accessory': {'scale': 1.0, 'offset': (0, 0), 'blend': 'normal'}
        }

        # Analysis settings
        self._frame_count = 0
        self.last_skin_data = None
        self.seasonal_analysis = None
        self._show_analysis_hud = True
        self._auto_suggest_colors = True

        # Stable color storage to prevent blinking
        self._stable_colors = {
            'lipstick': "#FF1493",
            'blush': "#FF69B4",
            'eyeshadow': "#8A2BE2"
        }

        # Analysis refresh counter
        self._analysis_refresh_counter = 0

    # Property setters to update stable storage
    def update_lipstick_settings(self, enable, color, intensity, glossy):
        """Update lipstick settings in stable storage"""
        self._lipstick_settings['enable'] = enable
        if not self._auto_suggest_colors:  # Only update if not auto-suggesting
            self._lipstick_settings['color'] = color
            self._stable_colors['lipstick'] = color
        self._lipstick_settings['intensity'] = intensity
        self._lipstick_settings['glossy'] = glossy

    def update_blush_settings(self, enable, color, intensity):
        """Update blush settings in stable storage"""
        self._blush_settings['enable'] = enable
        if not self._auto_suggest_colors:
            self._blush_settings['color'] = color
            self._stable_colors['blush'] = color
        self._blush_settings['intensity'] = intensity

    def update_eyeshadow_settings(self, enable, color, intensity):
        """Update eyeshadow settings in stable storage"""
        self._eyeshadow_settings['enable'] = enable
        if not self._auto_suggest_colors:
            self._eyeshadow_settings['color'] = color
            self._stable_colors['eyeshadow'] = color
        self._eyeshadow_settings['intensity'] = intensity

    def update_filter_selection(self, filter_type, filter_data):
        """Update filter selection in stable storage"""
        self._selected_filters[filter_type] = filter_data

    def update_filter_settings(self, filter_type, scale, offset, blend):
        """Update filter settings in stable storage"""
        self._filter_settings[filter_type] = {
            'scale': scale,
            'offset': offset,
            'blend': blend
        }

    def update_auto_suggest(self, auto_suggest):
        """Update auto-suggest setting"""
        self._auto_suggest_colors = auto_suggest

    def update_show_hud(self, show_hud):
        """Update HUD display setting"""
        self._show_analysis_hud = show_hud

    def get_seasonal_color_suggestion(self, makeup_type):
        """Get color suggestion based on seasonal analysis"""
        if not self.seasonal_analysis:
            return None

        palette = self.seasonal_analysis['palette']
        if makeup_type in palette['best_colors']:
            colors = palette['hex_codes'][makeup_type]
            if colors:
                return colors[0]  # Return primary recommended color
        return None

    def _update_stable_colors_from_analysis(self):
        """Update stable colors based on seasonal analysis - CONTROLLED UPDATE"""
        if not self._auto_suggest_colors or not self.seasonal_analysis:
            return

        # Only update every 60 frames to prevent blinking
        if self._analysis_refresh_counter % 60 != 0:
            return

        for makeup_type in ['lipstick', 'blush', 'eyeshadow']:
            suggested_color = self.get_seasonal_color_suggestion(makeup_type)
            if suggested_color:
                self._stable_colors[makeup_type] = suggested_color

                # Update internal settings
                if makeup_type == 'lipstick':
                    self._lipstick_settings['color'] = suggested_color
                elif makeup_type == 'blush':
                    self._blush_settings['color'] = suggested_color
                elif makeup_type == 'eyeshadow':
                    self._eyeshadow_settings['color'] = suggested_color

    def _draw_professional_hud(self, img, skin_data, seasonal_data):
        """Professional HUD with comprehensive analysis"""
        if not skin_data or not self._show_analysis_hud:
            return img

        h, w = img.shape[:2]
        overlay = img.copy()

        # Main HUD background
        cv2.rectangle(overlay, (10, 10), (320, 200), (20, 20, 20), -1)
        alpha = 0.9
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Title
        cv2.putText(img, "Professional Skin Analysis", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Skin analysis
        cv2.putText(img, f"Tone: {skin_data['tone']}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Undertone: {skin_data['undertone']}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Seasonal analysis
        if seasonal_data:
            cv2.putText(img, f"Season: {seasonal_data['primary_season']}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Confidence
        confidence = skin_data.get('confidence', 0.0)
        cv2.putText(img, f"Confidence: {confidence:.1%}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Color swatch
        b, g, r = [int(x) for x in skin_data['color']]
        cv2.rectangle(img, (20, 130), (70, 165), (int(b), int(g), int(r)), -1)
        cv2.rectangle(img, (20, 130), (70, 165), (255, 255, 255), 1)
        cv2.putText(img, "Skin", (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Active filters - STABLE INDICATORS
        filter_indicators = []
        if self._lipstick_settings['enable']:
            filter_indicators.append(("L", (255, 20, 147)))
        if self._blush_settings['enable']:
            filter_indicators.append(("B", (255, 105, 180)))
        if self._eyeshadow_settings['enable']:
            filter_indicators.append(("E", (138, 43, 226)))

        for filter_type, selected_filter in self._selected_filters.items():
            if selected_filter:
                if filter_type == 'dress':
                    filter_indicators.append(("D", (65, 105, 225)))
                elif filter_type == 'jewelry':
                    filter_indicators.append(("J", (255, 215, 0)))
                elif filter_type == 'accessory':
                    filter_indicators.append(("A", (255, 140, 0)))

        for i, (letter, color) in enumerate(filter_indicators):
            x_pos = 90 + (i * 20)
            cv2.putText(img, letter, (x_pos, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Auto-suggest indicator
        if self._auto_suggest_colors:
            cv2.putText(img, "Auto Color: ON", (180, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(img, "Auto Color: OFF", (180, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

        # Display current stable colors
        cv2.putText(img, "Active Colors:", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return img

    def recv(self, frame):
        with self.processing_lock:
            img = frame.to_ndarray(format="bgr24")
            self._analysis_refresh_counter += 1

            # Skip processing some frames for better performance
            if self._frame_count % 2 != 0:
                self._frame_count += 1
                return img

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_lm = results.multi_face_landmarks[0]

                # Enhanced skin analysis every 60 frames (reduced frequency for stability)
                if self._frame_count % 60 == 0:
                    avg_color, tone, undertone, confidence = detect_skin_tone_advanced(img, face_lm)
                    if avg_color is not None:
                        self.last_skin_data = {
                            'color': avg_color,
                            'tone': tone,
                            'undertone': undertone,
                            'confidence': confidence,
                        }

                        # Update seasonal analysis
                        self.seasonal_analysis = get_seasonal_color_analysis(tone, undertone)

                        # Update stable colors from analysis (controlled)
                        self._update_stable_colors_from_analysis()

                # Apply beauty filters using STABLE settings
                if self._lipstick_settings['enable']:
                    img = apply_precision_lipstick(img, face_lm, self._stable_colors['lipstick'],
                                                   self._lipstick_settings['intensity'],
                                                   self._lipstick_settings['glossy'])

                if self._blush_settings['enable']:
                    img = apply_enhanced_blush(img, face_lm, self._stable_colors['blush'],
                                               self._blush_settings['intensity'])

                if self._eyeshadow_settings['enable']:
                    img = apply_enhanced_eyeshadow(img, face_lm, self._stable_colors['eyeshadow'],
                                                   self._eyeshadow_settings['intensity'])

                # Apply PNG filters using STABLE settings
                for filter_type, filter_data in self._selected_filters.items():
                    if filter_data:
                        settings = self._filter_settings[filter_type]
                        img = apply_enhanced_png_filter(
                            img, face_lm, filter_data, filter_type,
                            settings['scale'], settings['offset'], settings['blend']
                        )

                # Professional HUD overlay
                img = self._draw_professional_hud(img, self.last_skin_data, self.seasonal_analysis)

            self._frame_count += 1
            return img


# ======================== Enhanced UI Components ========================

def display_seasonal_recommendations(seasonal_data, skin_data):
    """Display comprehensive seasonal color recommendations"""
    if not seasonal_data or not skin_data:
        return

    primary_season = seasonal_data['primary_season']
    palette = seasonal_data['palette']

    # Season-specific styling
    season_colors = {
        "Spring": "linear-gradient(135deg, #98FB98 0%, #FFD700 100%)",
        "Summer": "linear-gradient(135deg, #87CEEB 0%, #DDA0DD 100%)",
        "Rainy": "linear-gradient(135deg, #708090 0%, #2F4F4F 100%)",
        "Autumn": "linear-gradient(135deg, #D2691E 0%, #8B4513 100%)",
        "Winter": "linear-gradient(135deg, #4682B4 0%, #191970 100%)"
    }

    st.markdown(
        f"""
    <div style="background: {season_colors.get(primary_season, season_colors['Spring'])};
               padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0;">
        <h3> Your Color Season: {primary_season}</h3>
        <p>{palette['description']}</p>
        <p><strong>Best Metals:</strong> {', '.join(palette['metals'])}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create tabs for different categories
    tabs = st.tabs([" Clothing", " Makeup", " Accessories", " Avoid"])

    with tabs[0]:  # Clothing
        st.markdown("#### Perfect Clothing Colors")
        cols = st.columns(4)
        for i, (color_name, hex_code) in enumerate(zip(palette['best_colors']['clothing'],
                                                       palette['hex_codes']['clothing'])):
            with cols[i % 4]:
                st.markdown(
                    f"""
                <div class="color-card" onclick="navigator.clipboard.writeText('{hex_code}')">
                    <div style="width:50px;height:50px;border-radius:8px;
                               background-color:{hex_code};margin:0 auto 0.5rem;
                               border:2px solid #ddd;box-shadow:0 2px 8px rgba(0,0,0,0.1);"></div>
                    <strong>{color_name}</strong><br>
                    <small>{hex_code}</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    with tabs[1]:  # Makeup
        makeup_categories = ['lipstick', 'eyeshadow', 'blush', 'nail']
        for category in makeup_categories:
            if category in palette['best_colors']:
                st.markdown(f"##### {category.title()}")
                cols = st.columns(5)
                for i, (color_name, hex_code) in enumerate(zip(palette['best_colors'][category],
                                                               palette['hex_codes'][category])):
                    with cols[i % 5]:
                        st.markdown(
                            f"""
                        <div class="color-card">
                            <div style="width:40px;height:40px;border-radius:50%;
                                       background-color:{hex_code};margin:0 auto 0.3rem;
                                       border:2px solid #ddd;"></div>
                            <small><strong>{color_name}</strong></small>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

    with tabs[2]:  # Accessories
        st.markdown("#### Recommended Metals & Accessories")
        cols = st.columns(3)
        for i, metal in enumerate(palette['metals']):
            with cols[i % 3]:
                metal_colors = {
                    'Gold': '#FFD700',
                    'Silver': '#C0C0C0',
                    'Copper': '#B87333',
                    'Bronze': '#CD7F32',
                    'Platinum': '#E5E4E2',
                    'White Gold': '#F8F8F8',
                    'Oxidized Silver': '#696969',
                    'Gunmetal': '#2C3539',
                    'Antique Bronze': '#CD7F32',
                    'Warm Silver': '#E8E8E8'
                }
                color = metal_colors.get(metal, '#C0C0C0')
                st.markdown(
                    f"""
                <div class="color-card">
                    <div style="width:45px;height:45px;border-radius:50%;
                               background-color:{color};margin:0 auto 0.3rem;
                               border:2px solid #ddd;box-shadow:0 2px 8px rgba(0,0,0,0.1);"></div>
                    <strong>{metal}</strong>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    with tabs[3]:  # Avoid
        st.markdown("#### Colors to Avoid")
        avoid_text = ", ".join(palette['avoid'])
        st.markdown(f"**Avoid these colors:** {avoid_text}")
        st.markdown("*These colors may wash you out or clash with your natural coloring.*")


def display_filter_gallery(local_filters):
    """Improved filter gallery with working selection"""
    st.markdown("### 🎭 Local Filter Collection")

    if not any(filters for filters in local_filters.values()):
        st.info("""
         **No local filters found!**

        To add filters, create these folders in your app directory:
        - `filters/dresses/` - for dress PNG files
        - `filters/jewelry/` - for jewelry PNG files  
        - `filters/accessories/` - for accessory PNG files

        Then add PNG files with transparent backgrounds to each folder.
        """)
        return

    # Create tabs for each filter category
    filter_tabs = []
    tab_names = []

    for category, filters in local_filters.items():
        if filters:
            tab_names.append(f"{category.title()} ({len(filters)})")
            filter_tabs.append(category)

    if not tab_names:
        st.warning("No filter categories found with valid PNG files.")
        return

    tabs = st.tabs(tab_names)

    for tab_idx, category in enumerate(filter_tabs):
        with tabs[tab_idx]:
            filters = local_filters[category]

            # Display filters in a grid
            cols_per_row = 4
            rows = len(filters) // cols_per_row + (1 if len(filters) % cols_per_row else 0)

            for row in range(rows):
                cols = st.columns(cols_per_row)

                for col_idx in range(cols_per_row):
                    filter_idx = row * cols_per_row + col_idx
                    if filter_idx < len(filters):
                        filter_item = filters[filter_idx]

                        with cols[col_idx]:
                            # Create preview
                            if filter_item['image'] is not None:
                                try:
                                    # Create preview image
                                    preview_size = 80
                                    original_img = filter_item['image']

                                    # Resize maintaining aspect ratio
                                    h, w = original_img.shape[:2]
                                    if h > w:
                                        new_h, new_w = preview_size, int(w * preview_size / h)
                                    else:
                                        new_h, new_w = int(h * preview_size / w), preview_size

                                    preview_img = cv2.resize(original_img, (new_w, new_h))

                                    # Convert for display
                                    if len(preview_img.shape) == 4:  # RGBA
                                        preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGRA2RGB)
                                    else:  # RGB
                                        preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)

                                    # Convert to base64
                                    _, buffer = cv2.imencode('.png', cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))
                                    img_base64 = base64.b64encode(buffer).decode()

                                    # Check if currently selected
                                    current_selection = st.session_state.get(f'selected_{category.rstrip("s")}')
                                    is_selected = (current_selection and
                                                   current_selection.get('name') == filter_item['name'])

                                    border_style = "border: 3px solid #FF6B9D;" if is_selected else "border: 2px solid #ddd;"

                                    # Display preview
                                    st.markdown(f"""
                                    <div style="text-align: center; margin-bottom: 1rem;">
                                        <div style="background: white; padding: 0.5rem; border-radius: 10px; 
                                                   {border_style} box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                            <img src="data:image/png;base64,{img_base64}" 
                                                 style="max-width: {preview_size}px; max-height: {preview_size}px; 
                                                        border-radius: 5px;" />
                                            <br><small style="color: #333; font-weight: bold;">{filter_item['name']}</small>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # Selection button
                                    button_text = "✅ Selected" if is_selected else "Select"
                                    button_type = "secondary" if is_selected else "primary"

                                    if st.button(button_text, key=f"select_{category}_{filter_idx}",
                                                 type=button_type, use_container_width=True):
                                        if is_selected:
                                            # Deselect
                                            st.session_state[f'selected_{category.rstrip("s")}'] = None
                                            st.rerun()
                                        else:
                                            # Select
                                            st.session_state[f'selected_{category.rstrip("s")}'] = filter_item
                                            st.success(f"✅ Selected {filter_item['name']}")
                                            st.rerun()

                                except Exception as e:
                                    st.error(f"Error displaying {filter_item['name']}: {e}")
                            else:
                                st.warning(f"Could not load {filter_item['name']}")


# ======================== Main Application ========================

st.markdown(
    """
<div class="main-header">
    <h1>✨  Fashion AR Beauty Studio ✨</h1>
    <p>Advanced AI-Powered Beauty Analysis • 5-Season Color Theory • No Blinking Filters</p>
</div>
""",
    unsafe_allow_html=True,
)

# Initialize session state for stable filter management
if 'selected_dress' not in st.session_state:
    st.session_state.selected_dress = None
if 'selected_jewelry' not in st.session_state:
    st.session_state.selected_jewelry = None
if 'selected_accessory' not in st.session_state:
    st.session_state.selected_accessory = None

# Load local filters
local_filters = load_local_filters()

# Enhanced Sidebar Controls
st.sidebar.markdown(
    """
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
    <h2>🎛 Professional Controls</h2>
    <p><small>Stable • No Blinking • 5 Seasons</small></p>
</div>
""",
    unsafe_allow_html=True,
)

# Auto Color Suggestion
st.sidebar.markdown("###  Smart Color Suggestions")
auto_suggest_colors = st.sidebar.checkbox("Auto-Suggest Colors Based on Skin Tone", value=True)
st.sidebar.markdown("*Colors update based on Spring, Summer, Rainy, Autumn, Winter analysis*")

# Beauty Filters Section
st.sidebar.markdown("###  Enhanced Beauty Filters")

with st.sidebar.expander(" Precision Lipstick", expanded=True):
    enable_lipstick = st.checkbox("Enable Lipstick", value=False)
    if not auto_suggest_colors:
        lipstick_color = st.color_picker("Color", "#FF1493", key="lip_precise")
    else:
        lipstick_color = "#FF1493"  # Will be auto-updated by transformer
        st.info(" Color auto-suggested based on your season")
    lipstick_intensity = st.slider("Intensity", 0.3, 1.0, 0.8, 0.05, key="lip_intensity_precise")
    lipstick_glossy = st.checkbox("Glossy Finish", value=True)

with st.sidebar.expander(" Natural Blush"):
    enable_blush = st.checkbox("Enable Blush", value=False)
    if not auto_suggest_colors:
        blush_color = st.color_picker("Color", "#FF69B4", key="blush_natural")
    else:
        blush_color = "#FF69B4"  # Will be auto-updated by transformer
        st.info(" Color auto-suggested based on your season")
    blush_intensity = st.slider("Intensity", 0.2, 1.0, 0.6, 0.05, key="blush_intensity_natural")

with st.sidebar.expander("👁 Professional Eyeshadow"):
    enable_eyeshadow = st.checkbox("Enable Eyeshadow", value=False)
    if not auto_suggest_colors:
        eyeshadow_color = st.color_picker("Color", "#8A2BE2", key="eye_professional")
    else:
        eyeshadow_color = "#8A2BE2"  # Will be auto-updated by transformer
        st.info(" Color auto-suggested based on your season")
    eyeshadow_intensity = st.slider("Intensity", 0.3, 1.0, 0.7, 0.05, key="eye_intensity_professional")

# Local Filters Section
st.sidebar.markdown("###  Local PNG Filters")

# Filter selection for each category
for category in ['dress', 'jewelry', 'accessory']:
    category_filters = local_filters.get(category + 's', [])  # Convert to plural
    if category_filters:
        st.sidebar.markdown(f"#### {category.title()} Filters")
        filter_names = [f['name'] for f in category_filters]
        selected_name = st.sidebar.selectbox(f"Select {category.title()}",
                                             ['None'] + filter_names,
                                             key=f"select_{category}")

        if selected_name != 'None':
            selected_filter = next(f for f in category_filters if f['name'] == selected_name)
            st.session_state[f'selected_{category}'] = selected_filter

            # Filter adjustment controls
            with st.sidebar.expander(f"{category.title()} Settings"):
                scale = st.slider(f"{category.title()} Scale", 0.3, 3.0, 1.0, 0.1, key=f"{category}_scale")
                offset_x = st.slider(f"Horizontal Offset", -200, 200, 0, 10, key=f"{category}_offset_x")
                offset_y = st.slider(f"Vertical Offset", -200, 200, 0, 10, key=f"{category}_offset_y")
                blend_mode = st.selectbox(f"Blend Mode", ['normal', 'multiply', 'overlay'], key=f"{category}_blend")
        else:
            st.session_state[f'selected_{category}'] = None

# File Upload Section
st.sidebar.markdown("###  Upload Custom Filters")
uploaded_dress = st.sidebar.file_uploader("Upload Dress PNG", type=['png'], key="upload_dress")
uploaded_jewelry = st.sidebar.file_uploader("Upload Jewelry PNG", type=['png'], key="upload_jewelry")
uploaded_accessory = st.sidebar.file_uploader("Upload Accessory PNG", type=['png'], key="upload_accessory")

# Process uploaded files
uploaded_filters = {}
for upload_type, uploaded_file in [('dress', uploaded_dress), ('jewelry', uploaded_jewelry),
                                   ('accessory', uploaded_accessory)]:
    if uploaded_file:
        try:
            file_bytes = uploaded_file.getvalue()
            file_array = np.frombuffer(file_bytes, np.uint8)
            uploaded_image = cv2.imdecode(file_array, cv2.IMREAD_UNCHANGED)
            uploaded_filters[upload_type] = {
                'name': uploaded_file.name.replace('.png', ''),
                'image': uploaded_image,
                'path': 'uploaded'
            }
            st.sidebar.success(f"✅ {upload_type.title()} uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {upload_type}: {e}")

# Display Settings
st.sidebar.markdown("### ⚙ Display Settings")
show_analysis_hud = st.sidebar.checkbox("Show Professional HUD", value=True)
show_filter_gallery = st.sidebar.checkbox("Show Filter Gallery", value=True)

# Performance Settings
st.sidebar.markdown("### ⚡ Performance")
processing_quality = st.sidebar.selectbox("Processing Quality",
                                          ["High Quality (Slower)", "Balanced", "Fast (Lower Quality)"],
                                          index=1)

# Main Layout
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown("###  Live Professional Camera (Stable - No Blinking)")

    # Create WebRTC context with proper configuration
    ctx = webrtc_streamer(
        key="professional-beauty-webrtc-stable",
        video_transformer_factory=ProfessionalBeautyTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Bind all controls to transformer using STABLE methods
    if ctx and ctx.video_transformer:
        t = ctx.video_transformer

        # Update beauty filters using stable methods
        t.update_lipstick_settings(enable_lipstick, lipstick_color, lipstick_intensity, lipstick_glossy)
        t.update_blush_settings(enable_blush, blush_color, blush_intensity)
        t.update_eyeshadow_settings(enable_eyeshadow, eyeshadow_color, eyeshadow_intensity)

        # Update system settings
        t.update_auto_suggest(auto_suggest_colors)
        t.update_show_hud(show_analysis_hud)

        # Update PNG filters from session state and uploads
        if uploaded_filters.get('dress'):
            t.update_filter_selection('dress', uploaded_filters['dress'])
        elif st.session_state.get('selected_dress'):
            t.update_filter_selection('dress', st.session_state['selected_dress'])
        else:
            t.update_filter_selection('dress', None)

        if uploaded_filters.get('jewelry'):
            t.update_filter_selection('jewelry', uploaded_filters['jewelry'])
        elif st.session_state.get('selected_jewelry'):
            t.update_filter_selection('jewelry', st.session_state['selected_jewelry'])
        else:
            t.update_filter_selection('jewelry', None)

        if uploaded_filters.get('accessory'):
            t.update_filter_selection('accessory', uploaded_filters['accessory'])
        elif st.session_state.get('selected_accessory'):
            t.update_filter_selection('accessory', st.session_state['selected_accessory'])
        else:
            t.update_filter_selection('accessory', None)

        # Update filter settings
        for category in ['dress', 'jewelry', 'accessory']:
            if f'{category}_scale' in st.session_state:
                scale = st.session_state.get(f'{category}_scale', 1.0)
                offset_x = st.session_state.get(f'{category}_offset_x', 0)
                offset_y = st.session_state.get(f'{category}_offset_y', 0)
                blend = st.session_state.get(f'{category}_blend', 'normal')
                t.update_filter_settings(category, scale, (offset_x, offset_y), blend)

        # Adjust processing quality
        if processing_quality == "Fast (Lower Quality)":
            t.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.3,
            )
        elif processing_quality == "High Quality (Slower)":
            t.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.7,
            )

with col2:
    st.markdown("### Professional Analysis & Recommendations")

    if ctx and ctx.video_transformer and ctx.video_transformer.last_skin_data:
        skin_data = ctx.video_transformer.last_skin_data
        seasonal_data = ctx.video_transformer.seasonal_analysis

        # Display enhanced skin analysis
        display_seasonal_recommendations(seasonal_data, skin_data)

        # Color picker suggestions
        if seasonal_data:
            st.markdown("###  Quick Color Picks")
            palette = seasonal_data['palette']

            pick_tabs = st.tabs([" Lips", "👁 Eyes", " Cheeks"])

            with pick_tabs[0]:
                st.markdown("**Recommended Lipstick Colors:**")
                lip_cols = st.columns(3)
                for i, (color_name, hex_code) in enumerate(zip(palette['best_colors']['lipstick'][:3],
                                                               palette['hex_codes']['lipstick'][:3])):
                    with lip_cols[i]:
                        if st.button(f"{color_name}", key=f"lip_pick_{i}"):
                            # Force color update
                            if ctx.video_transformer:
                                ctx.video_transformer._stable_colors['lipstick'] = hex_code
                                ctx.video_transformer._lipstick_settings['color'] = hex_code

            with pick_tabs[1]:
                st.markdown("**Recommended Eyeshadow Colors:**")
                eye_cols = st.columns(3)
                for i, (color_name, hex_code) in enumerate(zip(palette['best_colors']['eyeshadow'][:3],
                                                               palette['hex_codes']['eyeshadow'][:3])):
                    with eye_cols[i]:
                        if st.button(f"{color_name}", key=f"eye_pick_{i}"):
                            # Force color update
                            if ctx.video_transformer:
                                ctx.video_transformer._stable_colors['eyeshadow'] = hex_code
                                ctx.video_transformer._eyeshadow_settings['color'] = hex_code

            with pick_tabs[2]:
                st.markdown("**Recommended Blush Colors:**")
                blush_cols = st.columns(3)
                for i, (color_name, hex_code) in enumerate(zip(palette['best_colors']['blush'][:3],
                                                               palette['hex_codes']['blush'][:3])):
                    with blush_cols[i]:
                        if st.button(f"{color_name}", key=f"blush_pick_{i}"):
                            # Force color update
                            if ctx.video_transformer:
                                ctx.video_transformer._stable_colors['blush'] = hex_code
                                ctx.video_transformer._blush_settings['color'] = hex_code

        # Show current active colors
        if ctx.video_transformer and hasattr(ctx.video_transformer, '_stable_colors'):
            st.markdown("###  Current Active Colors")
            current_colors = ctx.video_transformer._stable_colors

            color_display_cols = st.columns(3)
            with color_display_cols[0]:
                st.markdown(f"""
                <div style="text-align:center;">
                    <div style="width:50px;height:50px;border-radius:50%;
                               background-color:{current_colors['lipstick']};margin:0 auto 0.3rem;
                               border:2px solid #ddd;"></div>
                    <small><strong>Lipstick</strong><br>{current_colors['lipstick']}</small>
                </div>
                """, unsafe_allow_html=True)

            with color_display_cols[1]:
                st.markdown(f"""
                <div style="text-align:center;">
                    <div style="width:50px;height:50px;border-radius:50%;
                               background-color:{current_colors['eyeshadow']};margin:0 auto 0.3rem;
                               border:2px solid #ddd;"></div>
                    <small><strong>Eyeshadow</strong><br>{current_colors['eyeshadow']}</small>
                </div>
                """, unsafe_allow_html=True)

            with color_display_cols[2]:
                st.markdown(f"""
                <div style="text-align:center;">
                    <div style="width:50px;height:50px;border-radius:50%;
                               background-color:{current_colors['blush']};margin:0 auto 0.3rem;
                               border:2px solid #ddd;"></div>
                    <small><strong>Blush</strong><br>{current_colors['blush']}</small>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info(
            " Start the camera to see advanced 5-season skin analysis and professional makeup suggestions.\n\n"
            "**New 5-Season Analysis:**\n"
            "• * **Spring** - Fresh, bright, warm colors\n"
            "• ☀ **Summer** - Soft, cool, gentle colors\n"
            "• 🌧 **Rainy** - Deep, muted, sophisticated colors\n"
            "• :) **Autumn** - Rich, warm, earthy colors\n"
            "• ❄ **Winter** - Bold, cool, intense colors")

# Filter Gallery Section
if show_filter_gallery and local_filters:
    st.markdown("---")
    display_filter_gallery(local_filters)

# Performance Metrics & Status
st.sidebar.markdown("---")
st.sidebar.markdown("###  System Status")

if ctx and ctx.video_transformer:
    frame_count = getattr(ctx.video_transformer, '_frame_count', 0)
    skin_data_available = ctx.video_transformer.last_skin_data is not None
    seasonal_analysis_available = ctx.video_transformer.seasonal_analysis is not None

    st.sidebar.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.1);padding:1rem;border-radius:10px;margin:0.5rem 0;">
        <div style="display:flex;justify-content:space-between;margin:0.3rem 0;">
            <span> Face Detection</span>
            <span style="color:#4CAF50;">●</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin:0.3rem 0;">
            <span> Skin Analysis</span>
            <span style="color:{'#4CAF50' if skin_data_available else '#FF9800'};">●</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin:0.3rem 0;">
            <span> 5-Season Analysis</span>
            <span style="color:{'#4CAF50' if seasonal_analysis_available else '#FF9800'};">●</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin:0.3rem 0;">
            <span> Stable Filters</span>
            <span style="color:#4CAF50;">●</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin:0.3rem 0;">
            <span> PNG Filters</span>
            <span style="color:#4CAF50;">●</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin:0.3rem 0;">
            <span>Frames Processed</span>
            <span style="color:#FFF;">{frame_count}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin:0.3rem 0;">
            <span> Anti-Blink</span>
            <span style="color:#00FF00;">ACTIVE</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """
    <div style="background:rgba(255,255,255,0.1);padding:1rem;border-radius:10px;margin:0.5rem 0;">
        <div style="text-align:center;color:#FF9800;">
            <span> Camera Not Started</span><br>
            <small>Click "START" to begin analysis</small>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Instructions & Tips
with st.expander("Usage Guide & Tips"):
    st.markdown(
        """
## Getting Started

**Camera Setup**
1. Click "START" and allow camera permissions
2. Position your face centrally with good lighting
3. Wait for analysis to complete (5-10 seconds)

** Enhanced 5-Season Color Analysis**
- **Spring**: Fresh, bright, warm (Golden undertones)
- **Summer**: Soft, cool, gentle (Pink/Cool undertones)  
- **Rainy**: Deep, muted, sophisticated (Neutral undertones)
- **Autumn**: Rich, warm, earthy (Golden/Warm undertones)
- **Winter**: Bold, cool, intense (Cool undertones)

**Enhanced Lipstick Precision**
- Ultra-precise lip detection with 468 facial landmarks
- Natural gradient and glossy effects
- Adjustable intensity and finish options

**Smart Color Suggestions**
- Enable "Auto-Suggest Colors" for AI-recommended shades
- Colors update based on your 5-season analysis
- Manual override available by unchecking auto-suggest
- Use Quick Color Picks for instant color changes

**Local Filter System**
- Create a "filters" folder in your project directory
- Add subfolders: "dresses", "jewelry", "accessories"
- Drop PNG files with transparent backgrounds
- Filters appear automatically in the sidebar

**🎯 Best Practices**
- Use high-quality PNG files with clean transparent backgrounds
- Ensure even, natural lighting for accurate skin analysis
- Start with lower filter intensities and adjust upward
- Use Quick Color Picks for immediate color changes
- Try different seasons to see which suits you best

**⚙️ Performance Tips**
- Close other camera applications
- Use Chrome or Edge for best WebRTC support
- Adjust processing quality based on your device capabilities
- Enable HUD for real-time analysis feedback

**🎨 Color Theory Tips**
- Your seasonal analysis determines your best colors
- Spring/Autumn: Warm undertones (golds, oranges, warm reds)
- Summer/Winter: Cool undertones (silvers, blues, cool pinks)
- Rainy: Neutral undertones (sophisticated, muted colors)
- Experiment with recommended colors for best results
"""
    )



# Footer
st.markdown("---")
st.markdown(
    """
     <style>
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 1rem;
        font-size: 14px;
        border-radius: 15px 15px 0 0;
    }
    </style>

<div class="footer">
    <h3>✨ Fashion AR Beauty Studio ✨</h3>
    <h5> Beauty Filters | AI-Powered Beauty Analysis | Virtual Try-On | Jewelry AR | Color Analysis | 5-Season Color Theory </h5>
    <p style="margin-top: 1rem; font-size: 12px;">© 2025 Fashion AR Beauty Studio | Designed by AI developers of Carinasfotlabs</p>
</div>
""",
    unsafe_allow_html=True,
)