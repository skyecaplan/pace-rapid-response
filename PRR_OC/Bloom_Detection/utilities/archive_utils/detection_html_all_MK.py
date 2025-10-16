"""
Put images into a html page, which can be shared through FILE_SHARE

Meng Gao, Sep 25, 2025

Images are embeded into the html page. 
Use resolution and image quality to adjust file size.
"""

import os
import base64
from PIL import Image

def is_valid_image(file_path, valid_extensions=None):
    """
    Check if the file is a valid image.
    """
    if valid_extensions is None:
        valid_extensions = {".png", ".jpg", ".jpeg"}
    return os.path.isfile(file_path) and os.path.splitext(file_path.lower())[1] in valid_extensions


def resize_and_compress_image(image_path, factor, output_format="JPEG", quality=85):
    """
    Resize and compress the image to reduce file size.
    """
    try:
        with Image.open(image_path) as img:
            new_width = max(1, img.width // factor)
            new_height = max(1, img.height // factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save the resized image as compressed data
            from io import BytesIO
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format=output_format, quality=quality)
            return buffer.getvalue()
    except Exception as e:
        print(f"❌ Error resizing/compressing image '{image_path}': {e}")
        return None


def encode_image_to_base64(image_path, factor=1, output_format="JPEG", quality=85):
    """
    Encode an image to a Base64 string, optionally resizing/compressing it first.
    """
    try:
        if factor > 1:
            compressed_data = resize_and_compress_image(image_path, factor, output_format, quality)
            if compressed_data:
                return base64.b64encode(compressed_data).decode("utf-8")
        else:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ Error encoding image '{image_path}': {e}")
        return None


def get_images_from_subfolders(base_folder, valid_extensions=None):
    """
    Collect images from all subdirectories (timestamps).
    """
    if valid_extensions is None:
        valid_extensions = {".png", ".jpg", ".jpeg"}

    grouped_images = []
    for folder_name in sorted(os.listdir(base_folder)):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            images = [os.path.join(folder_path, img) for img in sorted(os.listdir(folder_path))
                      if is_valid_image(os.path.join(folder_path, img), valid_extensions)]
            if images:
                grouped_images.append({"timestamp": folder_name, "images": images})
    return grouped_images


def create_html_from_subfolders(image_groups, output_html, sequence, \
                                title="Combined Image Gallery", title2=None,\
                                 titlev=None, resolution_factor=1, quality=85):
    """
    Create an HTML file grouping images by timestamp, with `titlev` and shrink functionality.
    Assumes `titlev` format matches `sequence` (2D list).
    """
    with open(output_html, "w") as f:
        # Start the HTML file
        f.write("<!DOCTYPE html>\n<html>\n<head>\n<meta charset='UTF-8'>\n")
        f.write(f"<title>{title}</title>\n")
        f.write("<style>\nbody { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write(".gallery { display: flex; flex-direction: column; gap: 20px; }\n")
        f.write(".row { display: flex; justify-content: center; gap: 20px; }\n")
        f.write(".small { flex: 1; } /* Shrink the globe */\n")
        f.write(".large { flex: 2; } /* Full-size image */\n")
        f.write(".img-container { display: flex; flex-direction: column; align-items: center; border: 1px solid #ccc; ")
        f.write("padding: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background-color: #fafafa; }\n")
        f.write(".img-container img { max-width: 100%; height: auto; display: block; border-radius: 5px; }\n")
        f.write(".caption { margin-top: 10px; font-weight: bold; text-align: center; }\n")
        f.write("</style>\n</head>\n<body>\n")
        f.write(f"<h1>{title}</h1>\n")
        #f.write(f"<h1>{title2}</h1>\n")
        f.write(f"<p>{title2}</p>\n")

        # Write each group of images
        for group in image_groups:
            timestamp = group["timestamp"]
            images = group["images"]
            print("create html", timestamp)
            
            f.write(f"<h2 style='text-align: center; margin-top: 50px;'>Timestamp: {timestamp}</h2>\n")
            f.write("<div class='gallery'>\n")
            write_gallery_section(images, f, sequence, titlev, resolution_factor, quality)
            f.write("</div>\n")

        f.write("</body>\n</html>\n")
    print(f"✅ Combined HTML gallery written to {output_html}")


def write_gallery_section(image_list, file_handle, sequence, titlev, resolution_factor=1, quality=85):
    """
    Write images from a section into a gallery following the provided sequence and custom titles.
    """
    valid_extensions = {".png", ".jpg", ".jpeg"}
    filtered_images = [img for img in image_list if is_valid_image(img, valid_extensions)]

    for row_idx, row in enumerate(sequence):
        file_handle.write("<div class='row'>\n")
        for col_idx, col_key in enumerate(row):
            matched_image = next((img for img in filtered_images if f"_{col_key}." in img), None)
            if matched_image:
                css_class = "small" if col_key == "globe" else "large"
                encoded_image = encode_image_to_base64(matched_image, factor=resolution_factor, quality=quality)
                caption = titlev[row_idx][col_idx] if titlev and titlev[row_idx][col_idx] else ""
                
                if encoded_image:
                    file_handle.write(f"<div class='img-container {css_class}'>\n")
                    file_handle.write(f"<img src='data:image/jpeg;base64,{encoded_image}' alt='{caption}' />\n")
                    file_handle.write(f"<div class='caption'>{caption}</div>\n</div>\n")
        file_handle.write("</div>\n")


# Example usage
if __name__ == "__main__":
    base_folder = "./plot"
    output_file = "./combined_gallery.html"
    sequence = [['globe', 'rgb', 'aot', ], ['ssa', 'fvf', 'sph']]
    titlev_custom = [["", "", "AOD (550nm)"], ["Single Scattering Albedo (550nm)", 
                                               "Fine Mode Volume Fraction", "Spherical Fraction"]]

    image_groups = get_images_from_subfolders(base_folder)
    create_html_from_subfolders(image_groups, output_file, sequence, title="Image Gallery",
                                 titlev=titlev_custom, resolution_factor=2, quality=75)