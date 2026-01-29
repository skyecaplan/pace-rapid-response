'''
gt_html_utils.py

Utility functions for PACE Phytoplankton Bloom Detection and Visualization. Specifically, functions to
construct daily html files from directories of images produces by functions in detection_util_MK.py

This .py file provides tools for:
- converting images (either paths to png file or in-memory figures) to base64 strings
- writing html files
- writing mutliple images to an html file in a formatted manner

Authors:
    Graham Trolley, NASA/GSFC, 2025-10-24

Dependencies:
    - base64
    - os
    - io 
    - Pillow (PIL)
'''

import io
from io import BytesIO
import os
from PIL import Image
import base64

def fig_to_base64(input_source, factor=1, max_height=None, output_format='PNG', quality=100):
    '''function by gt
       A function to manipulate figures made by fig, ax = plt.subplots OR PNG file paths to prepare them for writing
       as base64 encoded strings to html files. Returns a base64 encoded string manipulated 
       with the desired factor and quality. factor can now be a float, factor=1.5 means size is reduced by 33%
       
       Args:
           input_source: Either a matplotlib figure object or a string path to a PNG file
           factor: Float resize factor (factor=1.5 means size is reduced by 33%, factor=2.0 means 50% reduction)
           max_height: If specified, resize image to this max height while maintaining aspect ratio
           output_format: 'PNG' or 'JPEG'
           quality: JPEG quality (1-100, ignored for PNG)
    '''
    
    # Check if input is a string (file path) or matplotlib figure
    if isinstance(input_source, str):
        # Handle file path input
        if not os.path.exists(input_source):
            raise FileNotFoundError(f"Image file not found: {input_source}")
        
        # Open the image directly from file
        img = Image.open(input_source)
        #print(f"Loaded image from file: {input_source}")
        original_size = img.size
        
    else:
        # Handle matplotlib figure input (original logic)
        buffer = io.BytesIO()
        input_source.savefig(buffer, format=output_format, dpi=300)
        buffer.seek(0)
        
        # Open image from buffer
        img = Image.open(buffer)
        original_size = img.size
    
    #print(f"Original size: {original_size}")
    
    # Resize image if factor > 1 (now supports floats!)
    if factor > 1:
        new_width = max(1, int(img.width / factor))    # Changed from // to / and added int()
        new_height = max(1, int(img.height / factor))  # Changed from // to / and added int()
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Additional resize based on max_height (this happens after factor resize)
    if max_height and img.height > max_height:
        aspect_ratio = img.width / img.height
        new_height = max_height
        new_width = int(max_height * aspect_ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    #print(f"Final size: {img.size}")

    # Save processed image to buffer
    output_buffer = io.BytesIO()
    
    if output_format.upper() == 'JPEG':
        img = img.convert("RGB")  # JPEG doesn't support transparency
        img.save(output_buffer, format=output_format, quality=quality)
    else:
        img.save(output_buffer, format=output_format)  # PNG ignores quality
    
    # Convert to base64
    base64_string = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    return base64_string

# def write_html_image_row(f, imgs, capts, sizes, fill_width=True, equal_height=False):

#     row_classes = ['row']
#     if fill_width:
#         row_classes.append('full-width')
#     if equal_height:
#         row_classes.append('equal-height-row')
#     if len(imgs) == 1:  # Add single-image class for single images
#         row_classes.append('single-image')
    
#     f.write(f"<div class='{' '.join(row_classes)}'>\n")

#     for i in range(len(imgs)):
#         f.write(f"<div class='img-container {sizes[i]}'>\n")
#         f.write(f"<img src='data:image/png;base64,{imgs[i]}' alt='{capts[i]}' />\n")
#         f.write(f"<div class='caption'>{capts[i]}</div>\n</div>\n")
    
#     f.write("</div>\n")

def write_html_image_row(f, imgs, capts, sizes, fill_width=True, equal_height=False):
    row_classes = ['row']
    if fill_width:
        row_classes.append('full-width')
    if equal_height:
        row_classes.append('equal-height-row')
    if len(imgs) == 1:
        row_classes.append('single-image')
    
    f.write(f"<div class='{' '.join(row_classes)}'>\n")

    for i in range(len(imgs)):
        f.write(f"<div class='img-container {sizes[i]}'>\n")
        # ADD onclick event to make image clickable
        f.write(f"<img src='data:image/png;base64,{imgs[i]}' alt='{capts[i]}' onclick='openModal(this)' />\n")
        f.write(f"<div class='caption'>{capts[i]}</div>\n</div>\n")
    
    f.write("</div>\n")



def write_full_html(day, ofilepath):
    '''
    function to write daily PRR bloom detection html, provided a date string in the format '20250916'
    This function does not download pace data or make images, it reads in the data in the directory '20250916'
    and organizes it into an html file, which is saved to a desired location

    GT 10/24/2025

        simple usage:
        day = '20250916' # choose the directory with the day of data you want
        ofilepath = day + '/html/OCI_chlor_a_anomaly_daily_'+day+'.html'
        write_full_html(day, ofilepath)
    '''

    granule_folders  = [item for item in os.listdir('figures/'+day+'/png/') if os.path.isdir(os.path.join('figures/'+day+'/png/', item))]# old
    granule_folders = sorted([item for item in os.listdir('figures/'+day+'/png/') if os.path.isdir(os.path.join('figures/'+day+'/png/', item))])#sorts ascending
    
    h1 = 'PACE OCI daily Chlorophyll-a Anomaly, '+day
    tab_title = day+'_chla_anom_oci'
    with open(ofilepath, "w") as f:
            # #Start the HTML file
            f.write("<!DOCTYPE html>\n<html>\n<head>\n<meta charset='UTF-8'>\n")
            f.write(f"<title>{tab_title}</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write(".gallery { display: flex; flex-direction: column; gap: 20px; }\n")

            # Base row styles
            f.write(".row { display: flex; width: 100%; gap: 10px; align-items: flex-start; }\n")

            # Base size classes
            f.write(".small { flex: 1; }\n")
            f.write(".large { flex: 1; min-width: 0; }\n")

            # Container styles
            f.write(".img-container { display: flex; flex-direction: column; align-items: center; border: 1px solid #ccc; ")
            f.write("padding: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background-color: #fafafa; height: min-content; }\n")

            # Image styles
            f.write(".img-container img { width: 100%; height: 300px; display: block; border-radius: 5px; object-fit: contain; }\n")
            f.write(".equal-height-row .img-container img { width: 100%; height: 300px; display: block; border-radius: 5px; object-fit: contain; }\n")

            # Full-width overrides (put AFTER base classes for higher specificity)
            f.write(".row.full-width.single-image { justify-content: center; }\n")
            f.write(".row.full-width.single-image .img-container { flex: 0 0 70%; max-width: 70%; }\n")
            #f.write(".row.full-width.single-image .img-container { flex: 1; max-width: 100%; }\n")
            #f.write(".row.full-width.single-image .img-container { flex: 1; }\n")
            f.write(".row.full-width.single-image .img-container img { width: 100%; height: auto; object-fit: fill; }\n")  # ADD THIS LINE

            #CODE FOR ZOOM
            # Add this to your existing f.write("<style>\n") section:
            f.write(".img-container img { width: 100%; height: 300px; display: block; border-radius: 5px; object-fit: contain; cursor: pointer; transition: transform 0.2s; }\n")
            f.write(".img-container img:hover { transform: scale(1.02); }\n")  # Slight hover effect
            f.write(".modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.9); }\n")
            f.write(".modal-content { margin: auto; display: block; width: 90%; max-width: 1200px; max-height: 90%; object-fit: contain; }\n")
            f.write(".close { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }\n")
            f.write(".close:hover { color: #ccc; }\n")
            
            # NEW: Navigation arrow styles
            f.write(".modal-nav { position: absolute; top: 50%; transform: translateY(-50%); color: #f1f1f1; font-size: 60px; font-weight: bold; cursor: pointer; padding: 0 20px; user-select: none; }\n")
            f.write(".modal-nav:hover { color: #ccc; }\n")
            f.write(".prev { left: 20px; }\n")
            f.write(".next { right: 20px; }\n")

            f.write(".caption { margin-top: 10px; font-weight: bold; text-align: center; }\n")
            f.write("</style>\n</head>\n<body>\n")
                    
            f.write(f"<h1 style='text-align: center; text-decoration: underline;'>{h1}</h1>\n")
            f.write(f"""<p>The Bloom Detection Dashboard identifies L2 PACE OCI granules that exhibit substantial
                    chlorophyll-a anomalies. These anomalies are calculated from L3 daily composites relative to
                    the preceding 30-day L3 composite mean (i.e., chlorophyll-a anomaly = daily chlorophyll-a - 
                    30-day running chlorophyll-a mean). Potential bloom regions are detected in the L3 data by 
                    partitioning the scene into 100x100 L3-pixel blocks and flagging blocks where at least 10% of 
                    pixels show a chlorophyll-a anomaly greater than 1 mg/m^3. Flagged blocks serve as bounding boxes 
                    to locate corresponding L2 granules for the same day. These L2 granules are shown with optical and 
                    biogeochemical parameters overlaid on true-color imagery with embedded red squares to highlight the 
                    L3 blocks that contain the large chlorophyll-a anomalies.</p>\n""")

            f.write(f"<div>Developed by Graham Trolley and Matthew Kehrli</div>\n")

            # imgs = [fig_to_base64(day+'/png/'+'L3_Chl_'+day+'_bboxes.png'),fig_to_base64(day+'/png/'+'L3_Chl_30dayMean_'+day+'_bboxes.png') ]
            # capts = ['Daily Chl_a '+day, 'Chl_a 30-day']
            # sizes = ['large', 'large']
            # write_html_image_row(f, imgs, capts, sizes)

            # imgs = [fig_to_base64(day+'/png/'+'L3_Chl_Anomaly_'+day+'_bboxes.png') ]
            # capts = ['Chl_a anomaly '+day+' vs 30-day mean']
            # sizes = ['large']
            # write_html_image_row(f, imgs, capts, sizes, fill_width=True, equal_height=False)# single-image row


            imgs = [fig_to_base64('figures/'+day+'/png/'+'L2_Grans_All_'+day+'.png') ]
            capts = ['Chl_a anomalous granules '+day+' vs 30-day mean']
            sizes = ['large']
            write_html_image_row(f, imgs, capts, sizes, fill_width=True, equal_height=False)# single-image row

            #header is done, now write a loop to plot all the granule data

            granule_images_shrink_factor = 1.7 # use a scale factor to reduce quality of saved images, greatly aids in keeping filesize reasonable
            for granule in granule_folders:
                #f.write(f"<div>______________________________________________________</div>\n")
                download_url = 'https://oceandata.sci.gsfc.nasa.gov/getfile/'+'PACE_OCI.'+granule+'.L2.OC_BGC.V3_1.nc'
                download_url_nrt = 'https://oceandata.sci.gsfc.nasa.gov/getfile/PACE_OCI.'+granule+'.L2.OC_BGC.V3_1.NRT.nc'
                f.write(f"<h2 style='text-align: center; text-decoration: underline;'>{'Granule '+granule}\n")
                f.write(f"<a href='{download_url}' class='title-link' target='_blank' title='Download Data'> Refined ~</a>")
                f.write(f"<a href='{download_url_nrt}' class='title-link' target='_blank' title='Download Data'> NRT  ~</a>")
                f.write("</h2>\n")

                # Load all images
                carbon_phyto_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_carbon_phyto_overlay.png', factor = granule_images_shrink_factor)
                chlor_a_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_chlor_a_overlay.png', factor = granule_images_shrink_factor)
                poc_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_poc_overlay.png', factor = granule_images_shrink_factor)
                avw_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_avw_overlay.png', factor = granule_images_shrink_factor)
                nflh_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_nflh_overlay.png', factor = granule_images_shrink_factor)
                outline_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_outline.png', factor = granule_images_shrink_factor)
                anom_img = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_L3_Chl_Anomaly_L2gran_bboxes.png', factor = granule_images_shrink_factor)
                tc_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_TC.png', factor = granule_images_shrink_factor)
                

                # First row - always visible
                imgs = [outline_im, chlor_a_im, anom_img, nflh_im]
                capts = ['&nbsp;', 'Chlorophyll-a', 'Chl-a anomaly', 'NFLH']
                sizes = ['large', 'large', 'large', 'large']
                write_html_image_row(f, imgs, capts, sizes)
                
                # Add toggle button for this specific granule
                toggle_id = f"toggle_{granule}"
                content_id = f"more_{granule}"
                f.write(f'<button id="{toggle_id}" onclick="toggleContent(\'{content_id}\', \'{toggle_id}\')" style="background-color: #3498db; color: white; padding: 8px 16px; border: none; cursor: pointer; border-radius: 4px; margin: 10px auto; display: block; font-size: 14px;">Show More Parameters</button>\n')
                
                # Second row - hidden by default
                f.write(f'<div id="{content_id}" style="display: none;">\n')
                imgs = [ tc_im, carbon_phyto_im, poc_im, avw_im]
                capts = ['True Color', 'Phyto Carbon', 'POC', 'AVW']
                sizes = ['large', 'large', 'large', 'large']
                write_html_image_row(f, imgs, capts, sizes)


                # sometimes sst's work, sometimes they dont based on server data download timouts. so, try to do it, but proceed if they dont work
                try:
                    sst_anom_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_GHRSSTL4_SST_Anomaly_L2gran_bboxes.png', factor = granule_images_shrink_factor)
                    sst_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_GHRSSTL4_SST_L2gran_bboxes.png', factor = granule_images_shrink_factor)
                    imgs = [ sst_im, sst_anom_im]
                    capts = ['SST', 'SST Anomaly']
                    sizes = ['large', 'large']

                    write_html_image_row(f, imgs, capts, sizes)# if you want a second or third row of images behind the show more button, do it here
                except:
                    print("skiped sst's for ", str(granule))
                f.write('</div>\n')
                
                # Add some spacing between granules
                f.write('<hr style="margin: 30px 0; border: 0; border-top: 1px solid #eee;">\n')

            # Add JavaScript function (right before closing body)
            # Add this just before f.write("</body>\n</html>\n")
            f.write("""
            <script>
            function toggleContent(contentId, buttonId) {
                var content = document.getElementById(contentId);
                var button = document.getElementById(buttonId);
                
                if (content.style.display === "none") {
                    content.style.display = "block";
                    button.textContent = "Hide Parameters";
                } else {
                    content.style.display = "none";
                    button.textContent = "Show More Parameters";
                }
            }
            </script>
            """)

            f.write(f"<br>\n")
            f.write(f"<div>Granule download note: recent L2 granules need to use the NRT (near-real time) download link, older ones need to use the refined link. If one link isnt working, try the other. If both links aren't working, there was probably a data version update; you can manually correct the end of the filename to the correct version to fix the download link, i.e. at the end of the file, .L2.OC_BGC.V3_1.NRT.nc the V3_1 is changed to reflect the newest version</div>\n")
            
            # UPDATED: Modal with navigation arrows
            f.write("""
            <!-- Modal for image zoom -->
            <div id="imageModal" class="modal">
                <span class="close" onclick="closeModal()">&times;</span>
                <span class="modal-nav prev" onclick="changeImage(-1); event.stopPropagation();">&#10094;</span>
                <span class="modal-nav next" onclick="changeImage(1); event.stopPropagation();">&#10095;</span>
                <img class="modal-content" id="modalImg">
                <div id="modalCaption" style="text-align: center; color: white; padding: 20px; font-size: 18px;"></div>
            </div>

            <script>
            var allImages = [];
            var currentImageIndex = 0;

            // Collect all images on page load
            window.onload = function() {
                allImages = Array.from(document.querySelectorAll('.img-container img'));
                
                // Add click handlers to all images
                allImages.forEach(function(img, index) {
                    img.onclick = function() {
                        currentImageIndex = index;
                        openModal(img);
                    };
                });
            };

            function openModal(img) {
                var modal = document.getElementById("imageModal");
                var modalImg = document.getElementById("modalImg");
                var caption = document.getElementById("modalCaption");
                
                modal.style.display = "block";
                modalImg.src = img.src;
                caption.innerHTML = img.alt;
            }

            function closeModal() {
                document.getElementById("imageModal").style.display = "none";
            }

            function changeImage(direction) {
                currentImageIndex += direction;
                
                // Loop around if at the end or beginning
                if (currentImageIndex >= allImages.length) {
                    currentImageIndex = 0;
                } else if (currentImageIndex < 0) {
                    currentImageIndex = allImages.length - 1;
                }
                
                var img = allImages[currentImageIndex];
                document.getElementById("modalImg").src = img.src;
                document.getElementById("modalCaption").innerHTML = img.alt;
            }

            // Close modal when pressing Escape key, navigate with arrow keys
            document.addEventListener('keydown', function(event) {
                var modal = document.getElementById("imageModal");
                if (modal.style.display === "block") {
                    if (event.key === 'Escape') {
                        closeModal();
                    } else if (event.key === 'ArrowRight') {
                        changeImage(1);
                    } else if (event.key === 'ArrowLeft') {
                        changeImage(-1);
                    }
                }
            });

            // Close modal when clicking the background (not the image or arrows)
            document.getElementById("imageModal").addEventListener('click', function(event) {
                if (event.target === this) {
                    closeModal();
                }
            });
            </script>
            """)
            
            f.write("</body>\n</html>\n")
# # convert above code into function
# def write_full_html(day, ofilepath):
#     '''
#     function to write daily PRR bloom detection html, provided a date string in the format '20250916'
#     This function does not download pace data or make images, it reads in the data in the directory '20250916'
#     and organizes it into an html file, which is saved to a desired location

#     GT 10/24/2025

#         simple usage:
#         day = '20250916' # choose the directory with the day of data you want
#         ofilepath = day + '/html/OCI_chlor_a_anomaly_daily_'+day+'.html'
#         write_full_html(day, ofilepath)
#     '''

#     granule_folders  = [item for item in os.listdir('figures/'+day+'/png/') if os.path.isdir(os.path.join('figures/'+day+'/png/', item))]# old
#     granule_folders = sorted([item for item in os.listdir('figures/'+day+'/png/') if os.path.isdir(os.path.join('figures/'+day+'/png/', item))])#sorts ascending
    
#     h1 = 'PACE OCI daily Chlorophyll-a Anomaly, '+day
#     tab_title = day+'_chla_anom_oci'
#     with open(ofilepath, "w") as f:
#             # #Start the HTML file
#             f.write("<!DOCTYPE html>\n<html>\n<head>\n<meta charset='UTF-8'>\n")
#             f.write(f"<title>{tab_title}</title>\n")
#             f.write("<style>\n")
#             f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
#             f.write(".gallery { display: flex; flex-direction: column; gap: 20px; }\n")

#             # Base row styles
#             f.write(".row { display: flex; width: 100%; gap: 10px; align-items: flex-start; }\n")

#             # Base size classes
#             f.write(".small { flex: 1; }\n")
#             f.write(".large { flex: 1; min-width: 0; }\n")

#             # Container styles
#             f.write(".img-container { display: flex; flex-direction: column; align-items: center; border: 1px solid #ccc; ")
#             f.write("padding: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background-color: #fafafa; height: min-content; }\n")

#             # Image styles
#             f.write(".img-container img { width: 100%; height: 300px; display: block; border-radius: 5px; object-fit: contain; }\n")
#             f.write(".equal-height-row .img-container img { width: 100%; height: 300px; display: block; border-radius: 5px; object-fit: contain; }\n")

#             # Full-width overrides (put AFTER base classes for higher specificity)
#             f.write(".row.full-width.single-image { justify-content: center; }\n")
#             f.write(".row.full-width.single-image .img-container { flex: 0 0 70%; max-width: 70%; }\n")
#             #f.write(".row.full-width.single-image .img-container { flex: 1; max-width: 100%; }\n")
#             #f.write(".row.full-width.single-image .img-container { flex: 1; }\n")
#             f.write(".row.full-width.single-image .img-container img { width: 100%; height: auto; object-fit: fill; }\n")  # ADD THIS LINE

#             #CODE FOR ZOOm
#             # Add this to your existing f.write("<style>\n") section:
#             f.write(".img-container img { width: 100%; height: 300px; display: block; border-radius: 5px; object-fit: contain; cursor: pointer; transition: transform 0.2s; }\n")
#             f.write(".img-container img:hover { transform: scale(1.02); }\n")  # Slight hover effect
#             f.write(".modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.9); }\n")
#             f.write(".modal-content { margin: auto; display: block; width: 90%; max-width: 1200px; max-height: 90%; object-fit: contain; }\n")
#             f.write(".close { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }\n")
#             f.write(".close:hover { color: #ccc; }\n")


#             f.write(".caption { margin-top: 10px; font-weight: bold; text-align: center; }\n")
#             f.write("</style>\n</head>\n<body>\n")
                    
#             f.write(f"<h1 style='text-align: center; text-decoration: underline;'>{h1}</h1>\n")
#             f.write(f"""<p>The Bloom Detection Dashboard identifies L2 PACE OCI granules that exhibit substantial
#                     chlorophyll-a anomalies. These anomalies are calculated from L3 daily composites relative to
#                     the preceding 30-day L3 composite mean (i.e., chlorophyll-a anomaly = daily chlorophyll-a - 
#                     30-day running chlorophyll-a mean). Potential bloom regions are detected in the L3 data by 
#                     partitioning the scene into 100x100 L3-pixel blocks and flagging blocks where at least 10% of 
#                     pixels show a chlorophyll-a anomaly greater than 1 mg/m^3. Flagged blocks serve as bounding boxes 
#                     to locate corresponding L2 granules for the same day. These L2 granules are shown with optical and 
#                     biogeochemical parameters overlaid on true-color imagery with embedded red squares to highlight the 
#                     L3 blocks that contain the large chlorophyll-a anomalies.</p>\n""")

#             f.write(f"<div>Developed by Graham Trolley and Matthew Kehrli</div>\n")

#             # imgs = [fig_to_base64(day+'/png/'+'L3_Chl_'+day+'_bboxes.png'),fig_to_base64(day+'/png/'+'L3_Chl_30dayMean_'+day+'_bboxes.png') ]
#             # capts = ['Daily Chl_a '+day, 'Chl_a 30-day']
#             # sizes = ['large', 'large']
#             # write_html_image_row(f, imgs, capts, sizes)

#             # imgs = [fig_to_base64(day+'/png/'+'L3_Chl_Anomaly_'+day+'_bboxes.png') ]
#             # capts = ['Chl_a anomaly '+day+' vs 30-day mean']
#             # sizes = ['large']
#             # write_html_image_row(f, imgs, capts, sizes, fill_width=True, equal_height=False)# single-image row


#             imgs = [fig_to_base64('figures/'+day+'/png/'+'L2_Grans_All_'+day+'.png') ]
#             capts = ['Chl_a anomalous granules '+day+' vs 30-day mean']
#             sizes = ['large']
#             write_html_image_row(f, imgs, capts, sizes, fill_width=True, equal_height=False)# single-image row

#             #header is done, now write a loop to plot all the granule data

#             granule_images_shrink_factor = 1.7 # use a scale factor to reduce quality of saved images, greatly aids in keeping filesize reasonable
#             for granule in granule_folders:
#                 #f.write(f"<div>______________________________________________________</div>\n")
#                 download_url = 'https://oceandata.sci.gsfc.nasa.gov/getfile/'+'PACE_OCI.'+granule+'.L2.OC_BGC.V3_1.nc'
#                 download_url_nrt = 'https://oceandata.sci.gsfc.nasa.gov/getfile/PACE_OCI.'+granule+'.L2.OC_BGC.V3_1.NRT.nc'
#                 f.write(f"<h2 style='text-align: center; text-decoration: underline;'>{'Granule '+granule}\n")
#                 f.write(f"<a href='{download_url}' class='title-link' target='_blank' title='Download Data'> Refined ~</a>")
#                 f.write(f"<a href='{download_url_nrt}' class='title-link' target='_blank' title='Download Data'> NRT  ~</a>")
#                 f.write("</h2>\n")

#                 # Load all images
#                 carbon_phyto_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_carbon_phyto_overlay.png', factor = granule_images_shrink_factor)
#                 chlor_a_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_chlor_a_overlay.png', factor = granule_images_shrink_factor)
#                 poc_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_poc_overlay.png', factor = granule_images_shrink_factor)
#                 avw_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_avw_overlay.png', factor = granule_images_shrink_factor)
#                 nflh_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_nflh_overlay.png', factor = granule_images_shrink_factor)
#                 outline_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_outline.png', factor = granule_images_shrink_factor)
#                 anom_img = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_L3_Chl_Anomaly_L2gran_bboxes.png', factor = granule_images_shrink_factor)
#                 tc_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_TC.png', factor = granule_images_shrink_factor)
                

#                 # First row - always visible
#                 imgs = [outline_im, chlor_a_im, anom_img, avw_im]
#                 capts = ['&nbsp;', 'Chlorophyll-a', 'Chl-a anomaly', 'AVW']
#                 sizes = ['large', 'large', 'large', 'large']
#                 write_html_image_row(f, imgs, capts, sizes)
                
#                 # Add toggle button for this specific granule
#                 toggle_id = f"toggle_{granule}"
#                 content_id = f"more_{granule}"
#                 f.write(f'<button id="{toggle_id}" onclick="toggleContent(\'{content_id}\', \'{toggle_id}\')" style="background-color: #3498db; color: white; padding: 8px 16px; border: none; cursor: pointer; border-radius: 4px; margin: 10px auto; display: block; font-size: 14px;">Show More Parameters</button>\n')
                
#                 # Second row - hidden by default
#                 f.write(f'<div id="{content_id}" style="display: none;">\n')
#                 imgs = [ tc_im, nflh_im, carbon_phyto_im, poc_im]
#                 capts = ['True Color', 'NFLH', 'Phyto Carbon', 'POC']
#                 sizes = ['large', 'large', 'large', 'large']
#                 write_html_image_row(f, imgs, capts, sizes)


#                 # sometimes sst's work, sometimes they dont based on server data download timouts. so, try to do it, but proceed if they dont work
#                 try:
#                     sst_anom_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_GHRSSTL4_SST_Anomaly_L2gran_bboxes.png', factor = granule_images_shrink_factor)
#                     sst_im = fig_to_base64('figures/'+day+'/png/'+granule+'/'+granule+'_GHRSSTL4_SST_L2gran_bboxes.png', factor = granule_images_shrink_factor)
#                     imgs = [ sst_im, sst_anom_im]
#                     capts = ['SST', 'SST Anomaly']
#                     sizes = ['large', 'large']

#                     write_html_image_row(f, imgs, capts, sizes)# if you want a second or third row of images behind the show more button, do it here
#                 except:
#                     print("skiped sst's for ", str(granule))
#                 f.write('</div>\n')
                
#                 # Add some spacing between granules
#                 f.write('<hr style="margin: 30px 0; border: 0; border-top: 1px solid #eee;">\n')

#             # Add JavaScript function (right before closing body)
#             # Add this just before f.write("</body>\n</html>\n")
#             f.write("""
#             <script>
#             function toggleContent(contentId, buttonId) {
#                 var content = document.getElementById(contentId);
#                 var button = document.getElementById(buttonId);
                
#                 if (content.style.display === "none") {
#                     content.style.display = "block";
#                     button.textContent = "Hide Parameters";
#                 } else {
#                     content.style.display = "none";
#                     button.textContent = "Show More Parameters";
#                 }
#             }
#             </script>
#             """)

#             f.write(f"<br>\n")
#             f.write(f"<div>Granule download note: recent L2 granules need to use the NRT (near-real time) download link, older ones need to use the refined link. If one link isnt working, try the other. If both links aren't working, there was probably a data version update; you can manually correct the end of the filename to the correct version to fix the download link, i.e. at the end of the file, .L2.OC_BGC.V3_1.NRT.nc the V3_1 is changed to reflect the newest version</div>\n")
#             # Add this before f.write("</body>\n</html>\n")
#             f.write("""
#             <!-- Modal for image zoom -->
#             <div id="imageModal" class="modal" onclick="closeModal()">
#                 <span class="close" onclick="closeModal()">&times;</span>
#                 <img class="modal-content" id="modalImg">
#                 <div id="modalCaption" style="text-align: center; color: white; padding: 20px; font-size: 18px;"></div>
#             </div>

#             <script>
#             function openModal(img) {
#                 var modal = document.getElementById("imageModal");
#                 var modalImg = document.getElementById("modalImg");
#                 var caption = document.getElementById("modalCaption");
                
#                 modal.style.display = "block";
#                 modalImg.src = img.src;
#                 caption.innerHTML = img.alt;
#             }

#             function closeModal() {
#                 document.getElementById("imageModal").style.display = "none";
#             }

#             // Close modal when pressing Escape key
#             document.addEventListener('keydown', function(event) {
#                 if (event.key === 'Escape') {
#                     closeModal();
#                 }
#             });
#             </script>
#             """)
            
#             f.write("</body>\n</html>\n")





#             # for granule in granule_folders:
#             #         #f.write(f"<div>______________________________________________________</div>\n")
#             #         download_url = 'https://oceandata.sci.gsfc.nasa.gov/getfile/'+'PACE_OCI.'+granule+'.L2.OC_BGC.V3_1.nc'
#             #         download_url_nrt = 'https://oceandata.sci.gsfc.nasa.gov/getfile/PACE_OCI.'+granule+'.L2.OC_BGC.V3_1.NRT.nc'
#             #         f.write(f"<h2 style='text-align: center; text-decoration: underline;'>{'Granule '+granule}\n")
#             #         f.write(f"<a href='{download_url}' class='title-link' target='_blank' title='Download Data'> Refined ~</a>")#⬇️
#             #         f.write(f"<a href='{download_url_nrt}' class='title-link' target='_blank' title='Download Data'> NRT  ~</a>")#⬇️
#             #         f.write("</h2>\n")
#             #         #f.write(f"<div>{'NRT L2 link: ' + 'https://oceandata.sci.gsfc.nasa.gov/getfile/PACE_OCI.'+granule+'.L2.OC_BGC.V3_1.NRT.nc'}</div>\n")
#             #         #f.write(f"<div>{'L2 link: ' + 'https://oceandata.sci.gsfc.nasa.gov/getfile/'+'PACE_OCI.'+granule+'.L2.OC_BGC.V3_1.nc'}</div>\n")

#             #         carbon_phyto_im = fig_to_base64(day+'/png/'+granule+'/'+granule+'_carbon_phyto_overlay.png', factor = granule_images_shrink_factor)
#             #         chlor_a_im = fig_to_base64(day+'/png/'+granule+'/'+granule+'_chlor_a_overlay.png', factor = granule_images_shrink_factor)
#             #         poc_im = fig_to_base64(day+'/png/'+granule+'/'+granule+'_poc_overlay.png', factor = granule_images_shrink_factor)
#             #         avw_im = fig_to_base64(day+'/png/'+granule+'/'+granule+'_avw_overlay.png', factor = granule_images_shrink_factor)
#             #         nflh_im = fig_to_base64(day+'/png/'+granule+'/'+granule+'_nflh_overlay.png', factor = granule_images_shrink_factor)
#             #         outline_im = fig_to_base64(day+'/png/'+granule+'/'+granule+'_outline.png', factor = granule_images_shrink_factor)
#             #         anom_img = fig_to_base64(day+'/png/'+granule+'/'+granule+'_L3_Chl_Anomaly_L2gran_bboxes.png', factor = granule_images_shrink_factor)
#             #         #20251029T163346_L3_Chl_Anomaly_L2gran_bboxes.png


#             #         imgs = [outline_im,anom_img,avw_im,poc_im]
#             #         capts = ['&nbsp;','Chl-a anomaly','AVW','POC'] #non-breaking space used to keep galleries aligned after adding captions for other plots
#             #         sizes = ['large', 'large','large', 'large']
#             #         write_html_image_row(f, imgs, capts, sizes)
                    
#             #         imgs = [chlor_a_im,nflh_im,carbon_phyto_im]
#             #         capts = ['Chl-a','NFLH','Phyto Carbon']
#             #         sizes = ['large', 'large', 'large']
#             #         write_html_image_row(f, imgs, capts, sizes)

#             # f.write(f"<br>\n")  # Creates 3 line breaks
            
#             # f.write(f"<div>Granule download note: recent L2 granules need to use the NRT (near-real time) download link, older ones need to use the refined link. If one link isnt working, try the other. If both links aren't working, there was probably a data version update; you can manually correct the end of the filename to the correct version to fix the download link, i.e. at the end of the file, .L2.OC_BGC.V3_1.NRT.nc the V3_1 is changed to reflect the newest version</div>\n")
#             # f.write("</body>\n</html>\n")

