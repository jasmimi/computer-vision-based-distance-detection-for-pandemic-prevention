# Imports
import cv2

# Function to draw the line and make parts inside the bounding boxes transparent while preserving the original content
def draw_line_with_transparency(image, start_point, end_point, box1, box2, color, thickness):
    # Copy the original image to overlay the line on
    overlay = image.copy()

    # Draw the line on the overlay
    cv2.line(overlay, start_point, end_point, color, thickness)

    # Save the original content within the bounding boxes
    box1_content = image[box1[1]:box1[3], box1[0]:box1[2]].copy()
    box2_content = image[box2[1]:box2[3], box2[0]:box2[2]].copy()

    # Apply the overlay with the line to the original image
    image = cv2.addWeighted(overlay, 1, image, 0, 0)

    # Restore the original content inside the bounding boxes
    image[box1[1]:box1[3], box1[0]:box1[2]] = box1_content
    image[box2[1]:box2[3], box2[0]:box2[2]] = box2_content

    return image

# Makes text resize with bounding box
def put_responsive_text(image, text, position, box_width, box_height, font=cv2.FONT_HERSHEY_DUPLEX, color=(255, 255, 255), thickness=1):
    # Initial font scale and thickness
    font_scale = 1.0
    max_scale_down = 0.1  # How much to reduce the font size if it doesn't fit
    min_font_scale = 0.3  # Minimum font scale to prevent overly small text
    
    # Measure text size with the initial scale
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Adjust the font scale so the text fits the box width
    while text_size[0] > box_width and font_scale > min_font_scale:
        font_scale -= max_scale_down
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Center the text within the bounding box
    text_x = position[0] + (box_width - text_size[0]) // 2
    text_y = position[1] + (box_height + text_size[1]) // 2
    
    # Draw the text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

# Draws distance between text on line
def draw_text(image, text, font = cv2.FONT_HERSHEY_SIMPLEX, pos = (0,0), font_scale=3, font_thickness=2, text_color=(0, 255,0), text_color_bg=(0, 0, 0)):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(image, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(image, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size